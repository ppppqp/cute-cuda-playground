"""Puzzle 04: one CTA reduction per matrix row.

For X[rows, cols], CTA ``r`` computes ``sum(X[r, :])``. The reduction has two
levels:

1. Each of 256 threads owns a strided slice of one row and reduces it into one
   register value.
2. The 256 register values are written to shared memory and reduced as a tree.

The first level is expressed as CuTe layout division and slicing rather than
manual ``tid + k*256`` pointer arithmetic.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


THREADS = 256


@cute.kernel
def kernel(
    tiled_x: cute.Tensor,
    tiled_coordinate: cute.Tensor,
    y: cute.Tensor,
    problem_shape: cute.Shape,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    # The @jit launcher has already transformed the matrix into:
    #
    #   tiled_x:
    #     ((TileRow,TileCol),(RestRow,RestCol)) -> gmem address
    #     ((1,      256),    (rows,  ceil(cols/256)))
    #
    # This CTA owns RestRow=row and all RestCol chunks. This thread owns
    # TileRow=0 and TileCol=tid. Retaining RestCol with None gives:
    #
    #   thread_x:          RestCol -> gmem address
    #   thread_coordinate: RestCol -> original (row,col) coordinate
    #
    # Conceptually, thread tid receives columns:
    #   tid, tid+256, tid+512, tid+768, ...
    #
    # Notice the style now matches puzzle 03: tiling is launcher-side layout
    # algebra; the kernel only slices the pre-tiled views using runtime IDs.
    thread_slice = ((0, tid), (row, None))
    thread_x = tiled_x[thread_slice]
    thread_coordinate = tiled_coordinate[thread_slice]

    # This loop count depends on runtime `cols`, so use cutlass.range rather
    # than range_constexpr. The accumulator is explicitly Float32, matching the
    # desired device reduction type rather than Python's float semantics.
    local_sum = cutlass.Float32(0.0)
    for chunk in cutlass.range(cute.size(thread_x), unroll=1):
        # thread_coordinate[chunk] is a rank-2 (row,col) coordinate. elem_less
        # checks both components against (rows,cols). Only the column can be
        # out of range here, but retaining the full predicate is compositional.
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            local_sum += thread_x[chunk]

    # Allocate a CuTe shared-memory Tensor:
    #
    #   partial: lane -> shared_ptr + lane
    #   type: Tensor[Pointer[Float32,smem], Layout<256:1>>
    #
    # SmemAllocator determines the dynamic shared-memory requirement, so the
    # launcher does not need to manually specify a byte count.
    smem = cutlass.utils.SmemAllocator()
    partial_layout = cute.make_layout(THREADS)
    partial = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=partial_layout,
        byte_alignment=16,
        swizzle=None,
    )
    partial[tid] = local_sum

    # No lane may consume another lane's partial before every lane has stored.
    cute.arch.sync_threads()

    # Balanced shared-memory reduction tree:
    #
    #   256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    #
    # range_constexpr is expanded at JIT compile time. `offset` is therefore a
    # static integer at every stage, allowing the compiler to specialize each
    # bounds test and shared-memory address.
    for stage in cutlass.range_constexpr(8):
        offset = 128 >> stage
        if tid < offset:
            partial[tid] = partial[tid] + partial[tid + offset]

        # This barrier is required at every stage. Without it, a fast warp could
        # begin reading values that another warp has not updated yet.
        cute.arch.sync_threads()

    # After eight stages, partial[0] contains the complete row reduction. Only
    # one thread performs the global store, avoiding a write race.
    if tid == 0:
        y[row] = partial[0]


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    # Dynamic row-major matrix layout:
    #   matrix_layout(row,col) = row*c + col
    # `matrix` combines that mapping with xp's global-memory pointer.
    matrix_layout = cute.make_layout((r, c), stride=(c, 1))
    matrix = cute.make_tensor(xp, matrix_layout)

    # Define the work tile before launching, as in puzzle 03. Its logical shape
    # is one complete row fragment handled by one CTA: 1 row x 256 columns.
    cta_tiler = (1, THREADS)

    # Types after division:
    #
    #   tiled_matrix:
    #     ((1,256),(r,ceil(c/256))) -> gmem address
    #
    #   tiled_coordinate:
    #     the identical domain -> original (row,col) coordinate
    #
    # These objects are compile/JIT-time views. Passing them to the kernel does
    # not copy matrix data or materialize coordinate arrays.
    tiled_matrix = cute.zipped_divide(matrix, cta_tiler)
    coordinate = cute.make_identity_tensor((r, c))
    tiled_coordinate = cute.zipped_divide(coordinate, cta_tiler)

    # A rank-1 contiguous output tensor: row -> yp + row.
    output_layout = cute.make_layout(r)
    output = cute.make_tensor(yp, output_layout)

    # One CTA owns one row. All 256 lanes participate even when c < 256,
    # because every lane must reach every __syncthreads-equivalent barrier.
    kernel(
        tiled_matrix,
        tiled_coordinate,
        output,
        (r, c),
    ).launch(grid=(r, 1, 1), block=(THREADS, 1, 1), stream=s)


def main():
    r, c = 37, 1000
    x = random((r, c), 5)
    o = unfinished((r,))
    e = x.astype("float64").sum(1).astype("float32")
    with DeviceArray(x) as dx, DeviceArray(o) as dy:
        launch(dx.ptr, dy.ptr, r, c, stream())
        synchronize()
        check(dy.download(), e, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    main()
