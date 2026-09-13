"""Puzzle 06: fused row LayerNorm with affine epilogue."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    x_tiled_tensor: cute.Tensor,
    g_tiled_tensor: cute.Tensor,
    b_tiled_tensor: cute.Tensor,
    y_tiled_tensor: cute.Tensor,
    x_tiled_coord_tensor: cute.Tensor,
    g_tiled_coord_tensor: cute.Tensor,
    r: cutlass.Int32,
    c: cutlass.Int32,
    eps: cutlass.Float32,
):
    # TODO: paired sum/sumsq reduction, broadcast statistics, apply gamma/beta.
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    # get thread slice value
    x_thread_slice = ((0, tid), (bid, None))
    x_thread_tensor = x_tiled_tensor[x_thread_slice]
    x_thread_coord_tensor = x_tiled_coord_tensor[x_thread_slice]

    # local reduction to partial sum
    local_sum = cute.Float32(0.0)
    local_sum_sq = cute.Float32(0.0)

    for value in cutlass.range(cute.size(x_thread_tensor)):
        if cute.elem_less(x_thread_coord_tensor[value], (r, c)):
            local_sum += x_thread_tensor[value]
            local_sum_sq += x_thread_tensor[value] * x_thread_tensor[value]

    # cta reduction to sum
    smem = cutlass.utils.SmemAllocator()
    s_layout = cute.make_layout((2, 256), stride=(256, 1))
    shared_tensor = smem.allocate_tensor(element_type=cute.Float32, layout=s_layout)
    shared_tensor[0, tid] = local_sum
    shared_tensor[1, tid] = local_sum_sq
    cute.arch.sync_threads()
    for step in cutlass.range_constexpr(8):
        offset = 128 >> step
        if tid < offset:
            shared_tensor[0, tid] += shared_tensor[0, tid + offset]
            shared_tensor[1, tid] += shared_tensor[1, tid + offset]
        cute.arch.sync_threads()

    cta_sum = shared_tensor[0, 0]
    cta_mean = cta_sum / c
    cta_sum_sq = shared_tensor[1, 0]

    var_sq = cta_sum_sq / c - cta_mean * cta_mean

    # thread local layer norm
    g_thread_slice = (tid, None)
    g_thread_tensor = g_tiled_tensor[g_thread_slice]
    b_thread_tensor = b_tiled_tensor[g_thread_slice]
    y_thread_tensor = y_tiled_tensor[x_thread_slice]
    for chunk in cutlass.range(cute.size(x_thread_tensor)):
        if cute.elem_less(x_thread_coord_tensor[chunk], (r, c)):
            x_value = x_thread_tensor[chunk]
            g_value = g_thread_tensor[chunk]
            b_value = b_thread_tensor[chunk]
            y_thread_tensor[chunk] = (x_value - cta_mean) / cute.sqrt(
                var_sq
            ) * g_value + b_value


@cute.jit
def launch(
    xp: cute.Pointer,
    gp: cute.Pointer,
    bp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    eps: cutlass.Float32,
    s: cuda.CUstream,
):
    x_layout = cute.make_layout((r, c), stride=(c, 1))
    x_tensor = cute.make_tensor(xp, x_layout)
    g_layout = cute.make_layout(c)
    g_tensor = cute.make_tensor(gp, g_layout)
    b_tensor = cute.make_tensor(bp, g_layout)

    y_tensor = cute.make_tensor(yp, x_layout)

    cta_tiler = (1, 256)
    vector_tiler = (256,)
    x_tiled_tensor = cute.zipped_divide(x_tensor, cta_tiler)
    y_tiled_tensor = cute.zipped_divide(y_tensor, cta_tiler)
    g_tiled_tensor = cute.zipped_divide(g_tensor, vector_tiler)
    b_tiled_tensor = cute.zipped_divide(b_tensor, vector_tiler)

    x_coord_tensor = cute.make_identity_tensor((r, c))
    g_coord_tensor = cute.make_identity_tensor(c)
    x_tiled_coord_tensor = cute.zipped_divide(x_coord_tensor, cta_tiler)
    g_tiled_coord_tensor = cute.zipped_divide(g_coord_tensor, vector_tiler)
    kernel(
        x_tiled_tensor,
        g_tiled_tensor,
        b_tiled_tensor,
        y_tiled_tensor,
        x_tiled_coord_tensor,
        g_tiled_coord_tensor,
        r,
        c,
        eps,
    ).launch(grid=(r, 1, 1), block=(256, 1, 1), stream=s)


def main():
    r, c = 19, 768
    eps = 1e-5
    x = random((r, c), 7, -3, 3)
    g = random((c,), 8, 0.5, 1.5)
    b = random((c,), 9, -0.2, 0.2)
    m = x.mean(1, keepdims=True, dtype=np.float64)
    v = ((x - m) ** 2).mean(1, keepdims=True)
    e = ((x - m) / np.sqrt(v + eps) * g + b).astype(np.float32)
    o = unfinished((r, c))
    with (
        DeviceArray(x) as dx,
        DeviceArray(g) as dg,
        DeviceArray(b) as db,
        DeviceArray(o) as dy,
    ):
        launch(dx.ptr, dg.ptr, db.ptr, dy.ptr, r, c, eps, stream())
        synchronize()
        check(dy.download(), e, atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    main()
