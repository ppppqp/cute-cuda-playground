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
from puzzle_utils import (
    DeviceArray,
    check,
    random,
    stream,
    synchronize,
    unfinished,
)


THREADS = 256


@cute.kernel
def kernel(
    tiled_x: cute.Tensor,
    tiled_coordinate: cute.Tensor,
    y: cute.Tensor,
    problem_shape: cute.Shape,
):
    tid, _, _ = cute.arch.thread_idx()
    block_id, _, _ = cute.arch.block_idx()

    thread_x = tiled_x[
        ((0, tid), (block_id, None))
    ]  # each block has 1 row, with (c // 256) tiles
    # pick the tid th tile
    thread_x_coord: cute.Tensor = tiled_coordinate[((0, tid), (block_id, None))]

    # local reduction
    local_sum = cutlass.Float32(0.0)
    for value in cutlass.range(cute.size(thread_x)):
        if cute.elem_less(thread_x_coord[value], problem_shape):
            local_sum += thread_x[value]

    cute.printf("local_sum = {}", local_sum)
    # for step in cutlass.range_constexpr(8):
    #     offset = 128 >> step

    smem = cutlass.utils.SmemAllocator()
    s_layout = cute.make_layout(
        THREADS
    )  # allocate one for each thread, although it's probable that we don't use every thread
    shared_tensor = smem.allocate_tensor(
        element_type=cutlass.Float32, layout=s_layout, byte_alignment=16
    )
    shared_tensor[tid] = local_sum
    cute.arch.sync_threads()
    # smem reduction
    for step in cutlass.range_constexpr(8):
        offset = 128 >> step
        if tid < offset:
            shared_tensor[tid] = shared_tensor[tid] + shared_tensor[tid + offset]

        # make sure all threads finished writing
        cute.arch.sync_threads()

    if tid == 0:
        cute.printf("result {}", shared_tensor[0])
        y[block_id] = shared_tensor[0]


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    x_layout = cute.make_layout((r, c), stride=(c, 1))
    y_layout = cute.make_layout(r)
    x_tensor = cute.make_tensor(xp, x_layout)
    y_tensor = cute.make_tensor(yp, y_layout)

    cta_tiler = (1, THREADS)  # block shape
    x_tiled_tensor = cute.zipped_divide(x_tensor, cta_tiler)

    x_coord_tensor = cute.make_identity_tensor((r, c))
    x_tiled_coord_tensor = cute.zipped_divide(x_coord_tensor, cta_tiler)

    kernel(x_tiled_tensor, x_tiled_coord_tensor, y_tensor, (r, c)).launch(
        grid=(r, 1, 1), block=(THREADS, 1, 1), stream=s
    )


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
