"""Puzzle 05: numerically stable fused row softmax.

One 256-thread CTA owns one row. For every row we compute:

    m = max(x)
    denominator = sum(exp(x - m))
    y = exp(x - m) / denominator

Subtracting ``m`` prevents overflow without changing the result. The kernel
never writes maxima or exponentials to global memory; only the final output is
materialized.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


THREADS = 256
LOG2_E = 1.4426950408889634


@cute.kernel
def kernel(
    tiled_x: cute.Tensor,
    tiled_y: cute.Tensor,
    tiled_coordinate: cute.Tensor,
    problem_shape: cute.Shape,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    # All arguments were tiled in the @jit launcher with cta_tiler=(1,256):
    #
    #   tiled_x/y:
    #     ((1,256),(rows,ceil(cols/256))) -> gmem address
    #
    #   tiled_coordinate:
    #     the same domain -> original (row,col) coordinate
    #
    # Fix the within-tile row to 0, within-tile column to this thread,
    # matrix row to this CTA, and retain all column chunks with None.
    thread_slice = ((0, tid), (row, None))
    thread_x = tiled_x[thread_slice]
    thread_y = tiled_y[thread_slice]
    thread_coordinate = tiled_coordinate[thread_slice]

    # Each thread owns ceil(cols/256) logical positions. For cols=777 that is
    # four positions: tid, tid+256, tid+512, tid+768. Some positions in the
    # final chunk are padding and are rejected by the coordinate predicate.

    # One shared Float32 slot per thread. We reuse the same allocation for the
    # max reduction and sum reduction because the phases do not overlap.
    smem = cutlass.utils.SmemAllocator()
    partial = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout(THREADS),
        byte_alignment=16,
        swizzle=None,
    )

    # ----------------------------------------------------------------------
    # Phase 1: maximum
    # ----------------------------------------------------------------------
    # Invalid positions contribute the identity for max: negative infinity.
    # fmax is preferable to a Python max because it lowers explicitly to the
    # device floating-point maximum operation.
    local_max = -cutlass.Float32.inf
    for chunk in cutlass.range(cute.size(thread_x), unroll=1):
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            local_max = cute.arch.fmax(local_max, thread_x[chunk])

    partial[tid] = local_max
    cute.arch.sync_threads()

    # Shared-memory CTA max tree:
    #   256 -> 128 -> 64 -> ... -> 1
    # Each barrier separates a write stage from the following read stage.
    for stage in cutlass.range_constexpr(8):
        offset = 128 >> stage
        if tid < offset:
            partial[tid] = cute.arch.fmax(partial[tid], partial[tid + offset])
        cute.arch.sync_threads()

    # After the last barrier every thread may safely read the broadcast value.
    row_max = partial[0]

    # ----------------------------------------------------------------------
    # Phase 2: exponential sum
    # ----------------------------------------------------------------------
    local_sum = cutlass.Float32(0.0)
    for chunk in cutlass.range(cute.size(thread_x), unroll=1):
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            # CuTe commonly exposes fast exp2. Use
            #   exp(z) = exp2(z * log2(e)).
            # Since thread_x <= row_max, the exponent is <= 0 and cannot
            # overflow. At least one row element produces exactly exp(0)=1.
            shifted = thread_x[chunk] - row_max
            local_sum += cute.math.exp2(shifted * LOG2_E, fastmath=True)

    # Reusing partial is safe because all threads completed the maximum phase
    # and crossed its final barrier.
    partial[tid] = local_sum
    cute.arch.sync_threads()

    # Shared-memory CTA sum tree with zero as the conceptual identity.
    for stage in cutlass.range_constexpr(8):
        offset = 128 >> stage
        if tid < offset:
            partial[tid] = partial[tid] + partial[tid + offset]
        cute.arch.sync_threads()

    row_sum = partial[0]

    # ----------------------------------------------------------------------
    # Phase 3: normalized output
    # ----------------------------------------------------------------------
    # Recompute exp rather than storing an intermediate global array. This
    # trades inexpensive arithmetic for substantially less memory traffic.
    for chunk in cutlass.range(cute.size(thread_y), unroll=1):
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            shifted = thread_x[chunk] - row_max
            numerator = cute.math.exp2(shifted * LOG2_E, fastmath=True)
            thread_y[chunk] = numerator / row_sum


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    problem_shape = (r, c)

    # Both tensors are row-major views:
    #   (row,col) -> pointer + row*c + col
    matrix_layout = cute.make_layout(problem_shape, stride=(c, 1))
    x = cute.make_tensor(xp, matrix_layout)
    y = cute.make_tensor(yp, matrix_layout)

    # One CTA spans one matrix row fragment of 256 contiguous columns. Tiling
    # in the launcher keeps the kernel focused on runtime slicing and
    # computation, matching puzzles 03 and 04.
    cta_tiler = (1, THREADS)
    tiled_x = cute.zipped_divide(x, cta_tiler)
    tiled_y = cute.zipped_divide(y, cta_tiler)

    # Coordinate values follow exactly the same tiling as data addresses and
    # provide safe predicates for the partial final column tile.
    coordinate = cute.make_identity_tensor(problem_shape)
    tiled_coordinate = cute.zipped_divide(coordinate, cta_tiler)

    kernel(tiled_x, tiled_y, tiled_coordinate, problem_shape).launch(
        grid=(r, 1, 1), block=(THREADS, 1, 1), stream=s
    )


def main():
    r, c = 23, 777
    x = random((r, c), 6, -8, 8)
    z = np.exp(x - x.max(1, keepdims=True))
    e = z / z.sum(1, keepdims=True)
    o = unfinished((r, c))
    with DeviceArray(x) as dx, DeviceArray(o) as dy:
        launch(dx.ptr, dy.ptr, r, c, stream())
        synchronize()
        check(dy.download(), e, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    main()
