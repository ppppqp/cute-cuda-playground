"""Puzzle 08: shared-memory tiled SIMT GEMM.

Compute C[M,N] = A[M,K] @ B[K,N] with a 16x16x16 CTA tile. Each of 256
threads owns one C element. For every K tile, the CTA cooperatively loads 256 A
and 256 B values into shared memory, synchronizes, and performs 16 scalar FMAs.

This is a pedagogical SIMT GEMM, not a tensor-core kernel. Its purpose is to
make the CTA/thread/value mappings and data-reuse argument concrete before
introducing CuTe MMA atoms in puzzle 09.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


TILE_M = 16
TILE_N = 16
TILE_K = 16
THREADS = TILE_M * TILE_N


@cute.kernel
def kernel(
    tiled_a: cute.Tensor,
    tiled_b: cute.Tensor,
    tiled_c: cute.Tensor,
    tiled_a_coord: cute.Tensor,
    tiled_b_coord: cute.Tensor,
    tiled_c_coord: cute.Tensor,
    a_shape: cute.Shape,
    b_shape: cute.Shape,
    c_shape: cute.Shape,
    k: cutlass.Int32,
    tv_layout: cute.Layout,
):
    tid, _, _ = cute.arch.thread_idx()
    block_n, block_m, _ = cute.arch.block_idx()

    # tv_layout's conceptual mapping is:
    #
    #   (thread,value) -> (tile_row,tile_col)
    #
    # T=256 and V=1. Thread IDs vary fastest over tile_col, so each warp owns
    # two consecutive 16-element rows. This is sufficient for the teaching
    # kernel, though a production copy layout would normally vectorize loads.
    tile_identity = cute.make_identity_tensor((TILE_M, TILE_N))
    tv_tile_coord = cute.composition(tile_identity, tv_layout)
    value_profile = cute.repeat_like(None, tv_tile_coord[1])
    thread_tile_coord = tv_tile_coord[tid, value_profile]
    local_coord = thread_tile_coord[0]
    local_m = local_coord[0]
    local_n = local_coord[1]

    # Select this CTA's C tile:
    #   tiled_c domain: ((16,16),(ceil(M/16),ceil(N/16))) -> address
    cta_c_slice = ((None, None), (block_m, block_n))
    block_c = tiled_c[cta_c_slice]
    block_c_coord = tiled_c_coord[cta_c_slice]

    # Compose the C tile with thread ownership and retain this thread's V mode.
    # thread_c and thread_c_coord both have size one in this kernel.
    tv_c = cute.composition(block_c, tv_layout)
    tv_c_coord = cute.composition(block_c_coord, tv_layout)
    thread_slice = (tid, cute.repeat_like(None, tv_c[1]))
    thread_c = tv_c[thread_slice]
    thread_c_coord = tv_c_coord[thread_slice]

    # Shared-memory tensors are ordinary CuTe tensors with an smem pointer
    # engine and a static row-major layout:
    #   shared_a(m,k) -> smem_a + m*16 + k
    #   shared_b(k,n) -> smem_b + k*16 + n
    smem = cutlass.utils.SmemAllocator()
    shared_layout = cute.make_layout((16, 16), stride=(16, 1))
    shared_a = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=shared_layout,
        byte_alignment=16,
        swizzle=None,
    )
    shared_b = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=shared_layout,
        byte_alignment=16,
        swizzle=None,
    )

    # Apply the same TV ownership to both shared tiles. Each thread obtains one
    # destination address in A and one in B for the cooperative load.
    tv_shared_a = cute.composition(shared_a, tv_layout)
    tv_shared_b = cute.composition(shared_b, tv_layout)
    shared_thread_slice = (tid, cute.repeat_like(None, tv_shared_a[1]))
    thread_shared_a = tv_shared_a[shared_thread_slice]
    thread_shared_b = tv_shared_b[shared_thread_slice]

    accumulator = cutlass.Float32(0.0)

    # K may not be divisible by 16, so the logical K-tile count is rounded up.
    # `cutlass.range` represents a runtime loop; it is not fully unrolled across
    # K tiles. The inner 16-FMA loop below is compile-time unrolled.
    for k_tile in cutlass.range(cute.ceil_div(k, TILE_K), unroll=1):
        # A tile coordinate: (M tile=block_m, K tile=k_tile)
        # B tile coordinate: (K tile=k_tile, N tile=block_n)
        block_a = tiled_a[((None, None), (block_m, k_tile))]
        block_b = tiled_b[((None, None), (k_tile, block_n))]
        block_a_coord = tiled_a_coord[((None, None), (block_m, k_tile))]
        block_b_coord = tiled_b_coord[((None, None), (k_tile, block_n))]

        # Compose global tiles with the same ownership used for shared memory.
        # The value owned by a thread is at matching local (row,col) positions
        # in its global and shared tensors.
        tv_a = cute.composition(block_a, tv_layout)
        tv_b = cute.composition(block_b, tv_layout)
        tv_a_coord = cute.composition(block_a_coord, tv_layout)
        tv_b_coord = cute.composition(block_b_coord, tv_layout)
        thread_a = tv_a[thread_slice]
        thread_b = tv_b[thread_slice]
        thread_a_coord = tv_a_coord[thread_slice]
        thread_b_coord = tv_b_coord[thread_slice]

        # Predicated cooperative global->shared load. Zero is the additive
        # identity, so zero-filling padded M/N/K coordinates preserves GEMM.
        if cute.elem_less(thread_a_coord[0], a_shape):
            thread_shared_a[0] = thread_a[0]
        else:
            thread_shared_a[0] = 0.0

        if cute.elem_less(thread_b_coord[0], b_shape):
            thread_shared_b[0] = thread_b[0]
        else:
            thread_shared_b[0] = 0.0

        # Every shared value must be visible before any thread begins using the
        # current K tile.
        cute.arch.sync_threads()

        # This thread owns C[local_m,local_n] within the CTA tile. It walks the
        # shared K dimension, reusing one A row and one B column:
        #
        #   accumulator += shared_a[local_m,kk] * shared_b[kk,local_n]
        for kk in cutlass.range_constexpr(TILE_K):
            accumulator += shared_a[local_m, kk] * shared_b[kk, local_n]

        # No thread may overwrite shared_a/shared_b with the next K tile until
        # all threads have finished reading the current one.
        cute.arch.sync_threads()

    # Only the final M/N edge tile needs output predication. The identity
    # coordinate tensor maps this thread's one C value back to global (m,n).
    if cute.elem_less(thread_c_coord[0], c_shape):
        thread_c[0] = accumulator


@cute.jit
def launch(
    ap: cute.Pointer,
    bp: cute.Pointer,
    cp: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    s: cuda.CUstream,
):
    a_shape = (m, k)
    b_shape = (k, n)
    c_shape = (m, n)

    # Row-major global tensors:
    #   A(m,k) -> ap + m*K + k
    #   B(k,n) -> bp + k*N + n
    #   C(m,n) -> cp + m*N + n
    a = cute.make_tensor(ap, cute.make_layout(a_shape, stride=(k, 1)))
    b = cute.make_tensor(bp, cute.make_layout(b_shape, stride=(n, 1)))
    c = cute.make_tensor(cp, cute.make_layout(c_shape, stride=(n, 1)))

    # A and B use 16x16 tiles, interpreted respectively as (M,K) and (K,N).
    # C uses the same numerical tile shape, interpreted as (M,N).
    a_tiler = (TILE_M, TILE_K)
    b_tiler = (TILE_K, TILE_N)
    c_tiler = (TILE_M, TILE_N)
    tiled_a = cute.zipped_divide(a, a_tiler)
    tiled_b = cute.zipped_divide(b, b_tiler)
    tiled_c = cute.zipped_divide(c, c_tiler)

    # Apply identical tiling to coordinate tensors for safe edge predicates.
    tiled_a_coord = cute.zipped_divide(cute.make_identity_tensor(a_shape), a_tiler)
    tiled_b_coord = cute.zipped_divide(cute.make_identity_tensor(b_shape), b_tiler)
    tiled_c_coord = cute.zipped_divide(cute.make_identity_tensor(c_shape), c_tiler)

    # Logically arrange 256 flat CUDA threads as a 16x16 tile. The singleton
    # value layout means each thread owns one coordinate in every A/B/C tile.
    thread_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    value_layout = cute.make_layout((1, 1))
    _, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    kernel(
        tiled_a,
        tiled_b,
        tiled_c,
        tiled_a_coord,
        tiled_b_coord,
        tiled_c_coord,
        a_shape,
        b_shape,
        c_shape,
        k,
        tv_layout,
    ).launch(
        grid=(cute.ceil_div(n, TILE_N), cute.ceil_div(m, TILE_M), 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
        stream=s,
    )


def main():
    m, n, k = 71, 58, 65
    a = random((m, k), 12)
    b = random((k, n), 13)
    o = unfinished((m, n))
    with DeviceArray(a) as da, DeviceArray(b) as db, DeviceArray(o) as dc:
        launch(da.ptr, db.ptr, dc.ptr, m, n, k, stream())
        synchronize()
        check(dc.download(), a @ b, atol=8e-4, rtol=8e-4)


if __name__ == "__main__":
    main()
