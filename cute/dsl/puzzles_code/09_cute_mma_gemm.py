"""Puzzle 09: Ampere tensor-core GEMM with a CuTe TiledMma.

This first MMA kernel deliberately skips shared memory and copies each thread's
global A/B partition directly into registers. That is not the fastest dataflow,
but it isolates the essential CuTe MMA hierarchy:

    hardware MMA operation
      -> TiledMma replicated across warps
      -> per-thread ThrMma slice
      -> per-thread partitions of A, B, and C
      -> register fragments
      -> cute.gemm

Puzzle 08 assigned one scalar C value to each thread. Here the MMA atom defines
a nontrivial thread/value mapping: every lane owns several distributed A, B,
and accumulator values dictated by the hardware ``mma.sync`` instruction.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import (
    DeviceArray,
    check,
    random,
    require_compute_capability,
    stream,
    synchronize,
    unfinished,
)


N = 64
THREADS = 128


@cute.kernel
def kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    tiled_mma: cute.TiledMma,
):
    tid, _, _ = cute.arch.thread_idx()

    # tiled_mma describes the ownership of the complete replicated MMA atom.
    # get_slice fixes its thread mode at this CUDA thread and returns ThrMma:
    # an object that knows exactly which logical A/B/C values this lane owns.
    thr_mma = tiled_mma.get_slice(tid)

    # Partition the global tensors according to the hardware MMA's operand
    # layouts. These are tensor *views*, not loads and not allocations:
    #
    #   thread_a: (MMA, MMA_M, MMA_K) -> A global-memory address
    #   thread_b: (MMA, MMA_N, MMA_K) -> B global-memory address
    #   thread_c: (MMA, MMA_M, MMA_N) -> C global-memory address
    #
    # B is logically shaped (N,K), matching CuTe GEMM's A(M,K)*B(N,K)
    # convention. Its stride makes that logical view alias the host's physical
    # row-major B[K,N] allocation.
    thread_a = thr_mma.partition_A(a)
    thread_b = thr_mma.partition_B(b)
    thread_c = thr_mma.partition_C(c)

    # Register fragments have layouts compatible with the MMA instruction's
    # register ABI. make_fragment_* allocates thread-private register tensors;
    # it does not preserve the global pointer engines of thread_a/b/c.
    fragment_a = tiled_mma.make_fragment_A(thread_a)
    fragment_b = tiled_mma.make_fragment_B(thread_b)
    accumulator = tiled_mma.make_fragment_C(thread_c)
    accumulator.fill(0.0)

    # Universal copy is used only to make the ownership transition explicit:
    # global partition -> register fragment. A high-performance SM80 kernel
    # stages through shared memory and uses ldmatrix-compatible tiled copies.
    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
    cute.copy(load_atom, thread_a, fragment_a)
    cute.copy(load_atom, thread_b, fragment_b)

    # cute.gemm walks the fragment's MMA_M/MMA_N/MMA_K modes and emits the
    # underlying SM80 mma.sync instructions. The final accumulator argument is
    # also the input C fragment, so this expresses D = A*B + C in place. Since
    # it was zero-filled above, the mathematical result is simply A*B.
    cute.gemm(
        tiled_mma,
        accumulator,
        fragment_a,
        fragment_b,
        accumulator,
    )

    # Every accumulator value has a unique lane/value owner, so no atomics or
    # CTA barrier are required. Store the register fragment through the C
    # partition's global-memory addresses.
    cute.copy(store_atom, accumulator, thread_c)


@cute.jit
def launch(ap: cute.Pointer, bp: cute.Pointer, cp: cute.Pointer, s: cuda.CUstream):
    # A is a conventional row-major logical (M,K) tensor.
    a = cute.make_tensor(ap, cute.make_layout((N, N), stride=(N, 1)))

    # The host allocation B is row-major [K,N], but CuTe's MMA partition_B
    # expects logical B[N,K]. This transposed *view* performs no data movement:
    #   b(n,k) -> bp + k*N + n
    b = cute.make_tensor(bp, cute.make_layout((N, N), stride=(1, N)))

    # Float32 row-major output C[M,N].
    c = cute.make_tensor(cp, cute.make_layout((N, N), stride=(N, 1)))

    # One SM80 warp-level instruction computes a 16x8x16 FP16*FP16->FP32 MMA.
    # MmaF16BF16Op describes its register-level thread/value contract.
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float32,
        (16, 8, 16),
    )

    # Replicate the instruction atom over a (2,2,1) warp layout: two warp
    # groups along M, two along N, one along K. Four warps = 128 threads.
    # The resulting fundamental MMA tile is 32x16x16; partition_A/B/C repeat
    # that tile as necessary across the fixed 64x64x64 tensors.
    atom_layout_mnk = cute.make_layout((2, 2, 1))
    tiled_mma = cute.make_tiled_mma(mma_op, atom_layout_mnk)

    kernel(a, b, c, tiled_mma).launch(grid=(1, 1, 1), block=(THREADS, 1, 1), stream=s)


def main():
    require_compute_capability(8, 0)

    # Tensor cores consume FP16 inputs and accumulate in FP32. Construct the
    # reference in FP32 so NumPy does not introduce an extra FP16 accumulation.
    a = random((N, N), 14, -0.25, 0.25).astype(np.float16)
    b = random((N, N), 15, -0.25, 0.25).astype(np.float16)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    output = unfinished((N, N))
    with DeviceArray(a) as da, DeviceArray(b) as db, DeviceArray(output) as dc:
        launch(da.ptr, db.ptr, dc.ptr, stream())
        synchronize()
        check(dc.download(), expected, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    main()
