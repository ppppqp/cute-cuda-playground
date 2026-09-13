"""Puzzle 05: numerically stable fused row softmax."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    x_tiled: cute.Tensor,
    x_coord_tiled_tensor: cute.Tensor,
    y_tiled: cute.Tensor,
    y_coord_tiled_tensor: cute.Tensor,
    rows: cutlass.Int32,
    cols: cutlass.Int32,
):
    # TODO: CTA max reduction, exp/sum reduction, then normalized stores.
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    thread_slice = ((0, tid), (bid, None))
    x_thread_tensor = x_tiled[thread_slice]
    x_thread_coord_tensor = x_coord_tiled_tensor[thread_slice]
    y_thread_tensor = y_tiled[thread_slice]

    # local max reduction
    local_max = -cutlass.Float32.inf
    for value in cutlass.range(cute.size(x_thread_tensor)):
        if cute.elem_less(x_thread_coord_tensor[value], (rows, cols)):
            local_max = cute.arch.fmax(local_max, x_thread_tensor[value])

    smem = cutlass.utils.SmemAllocator()
    s_layout = cute.make_layout(256)
    shared_tensor = smem.allocate_tensor(element_type=cutlass.Float32, layout=s_layout)

    # smem max reduction
    shared_tensor[tid] = local_max
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr(8):
        offset = 128 >> step
        if tid < offset:
            shared_tensor[tid] = cute.arch.fmax(
                shared_tensor[tid], shared_tensor[tid + offset]
            )

        cute.arch.sync_threads()

    cta_max = shared_tensor[0]

    # exp/sum local reduction
    local_sum = cutlass.Float32(0.0)
    for value in cutlass.range(cute.size(x_thread_tensor)):
        if cute.elem_less(x_thread_coord_tensor[value], (rows, cols)):
            local_sum += cute.exp(x_thread_tensor[value] - cta_max)

    shared_tensor[tid] = local_sum
    cute.arch.sync_threads()

    # smem sum reduction
    for step in cutlass.range_constexpr(8):
        offset = 128 >> step
        if tid < offset:
            shared_tensor[tid] = shared_tensor[tid] + shared_tensor[tid + offset]
        cute.arch.sync_threads()

    # sum
    cta_sum = shared_tensor[0]
    cute.arch.sync_threads()

    for value in cutlass.range(cute.size(x_thread_tensor)):
        if cute.elem_less(x_thread_coord_tensor[value], (rows, cols)):
            y_thread_tensor[value] = (
                cute.exp(x_thread_tensor[value] - cta_max) / cta_sum
            )


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    layout = cute.make_layout((r, c), stride=(c, 1))
    x_tensor = cute.make_tensor(xp, layout)
    y_tensor = cute.make_tensor(yp, layout)

    # thread layout:
    cta_tiler = (1, 256)
    x_tiled_tensor = cute.zipped_divide(x_tensor, cta_tiler)
    y_tiled_tensor = cute.zipped_divide(y_tensor, cta_tiler)

    x_coord_tensor = cute.make_identity_tensor((r, c))
    x_coord_tiled_tensor = cute.zipped_divide(x_coord_tensor, cta_tiler)

    y_coord_tensor = cute.make_identity_tensor((r, c))
    y_coord_tiled_tensor = cute.zipped_divide(y_coord_tensor, cta_tiler)

    kernel(
        x_tiled_tensor, x_coord_tiled_tensor, y_tiled_tensor, y_coord_tiled_tensor, r, c
    ).launch(grid=(r, 1, 1), block=(256, 1, 1), stream=s)


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
