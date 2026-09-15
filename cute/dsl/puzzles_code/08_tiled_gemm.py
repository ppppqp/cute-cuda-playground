"""Puzzle 08: shared-memory tiled SIMT GEMM."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished

TILE_K = 16


@cute.kernel
def kernel(
    a_tiled_tensor: cute.Tensor,
    b_tiled_tensor: cute.Tensor,
    c_tiled_tensor: cute.Tensor,
    a_tiled_coord_tensor: cute.Tensor,
    b_tiled_coord_tensor: cute.Tensor,
    c_tiled_coord_tensor: cute.Tensor,
    a_layout_tv: cute.Layout,
    b_layout_tv: cute.Layout,
    c_layout_tv: cute.Layout,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
):
    # TODO: allocate 16x16 A/B shared tiles, make tiled copies, predicate edge
    # loads, synchronize each K stage, and accumulate one C value per thread.
    tid, _, _ = cute.arch.thread_idx()
    bid_n, bid_m, _ = cute.arch.block_idx()

    c_cta_slice = ((None, None), (bid_m, bid_n))
    c_block_tensor = c_tiled_tensor[c_cta_slice]
    c_block_coord_tensor = c_tiled_coord_tensor[c_cta_slice]
    c_tv_tensor = cute.composition(c_block_tensor, c_layout_tv)
    c_tv_coord_tensor = cute.composition(c_block_coord_tensor, c_layout_tv)
    thread_slice = (tid, cute.repeat_like(None, c_tv_tensor[1]))
    c_thread_tensor = c_tv_tensor[thread_slice]
    c_thread_coord_tensor = c_tv_coord_tensor[thread_slice]

    local_m = tid // 16
    local_n = tid % 16

    smem = cutlass.utils.SmemAllocator()
    a_shared_layout = cute.make_layout((16, 16), stride=(16, 1))
    b_shared_layout = cute.make_layout((16, 16), stride=(16, 1))
    a_shared_tensor = smem.allocate_tensor(
        layout=a_shared_layout, element_type=cutlass.Float32
    )
    b_shared_tensor = smem.allocate_tensor(
        layout=b_shared_layout, element_type=cutlass.Float32
    )

    # one accumulator for each output element
    # since one thread corresponds to one output element, it is still thread-private
    accumulator = cutlass.Float32(0.0)
    for k_group in cutlass.range(cute.ceil_div(k, TILE_K)):
        # load stage
        a_cta_slice = ((None, None), (bid_m, k_group))
        b_cta_slice = ((None, None), (k_group, bid_n))

        a_block_tensor = a_tiled_tensor[a_cta_slice]
        b_block_tensor = b_tiled_tensor[b_cta_slice]
        a_block_coord_tensor = a_tiled_coord_tensor[a_cta_slice]
        b_block_coord_tensor = b_tiled_coord_tensor[b_cta_slice]

        a_tv_tensor = cute.composition(a_block_tensor, a_layout_tv)
        b_tv_tensor = cute.composition(b_block_tensor, b_layout_tv)
        a_tv_coord_tensor = cute.composition(a_block_coord_tensor, a_layout_tv)
        b_tv_coord_tensor = cute.composition(b_block_coord_tensor, b_layout_tv)

        # fuck
        a_shared_tv_tensor = cute.composition(a_shared_tensor, a_layout_tv)
        b_shared_tv_tensor = cute.composition(b_shared_tensor, b_layout_tv)

        a_thread_tensor = a_tv_tensor[thread_slice]
        b_thread_tensor = b_tv_tensor[thread_slice]
        a_thread_coord_tensor = a_tv_coord_tensor[thread_slice]
        b_thread_coord_tensor = b_tv_coord_tensor[thread_slice]
        a_shared_thread_tensor = a_shared_tv_tensor[thread_slice]
        b_shared_thread_tensor = b_shared_tv_tensor[thread_slice]

        for chunk in cutlass.range_constexpr(cute.size(a_thread_tensor)):
            if cute.elem_less(a_thread_coord_tensor[chunk], (m, k)):
                a_shared_thread_tensor[chunk] = a_thread_tensor[chunk]
            else:
                a_shared_thread_tensor[chunk] = 0.0
            if cute.elem_less(b_thread_coord_tensor[chunk], (k, n)):
                b_shared_thread_tensor[chunk] = b_thread_tensor[chunk]
            else:
                b_shared_thread_tensor[chunk] = 0.0
        cute.arch.sync_threads()

        # compute stage
        for kk in cutlass.range(TILE_K):
            accumulator += a_shared_tensor[local_m, kk] * b_shared_tensor[kk, local_n]

        cute.arch.sync_threads()

    if cute.elem_less(c_thread_coord_tensor[0], (m, n)):
        c_thread_tensor[0] = accumulator


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
    a_layout = cute.make_layout((m, k), stride=(k, 1))
    b_layout = cute.make_layout((k, n), stride=(n, 1))
    c_layout = cute.make_layout((m, n), stride=(n, 1))

    a_tensor = cute.make_tensor(ap, a_layout)
    b_tensor = cute.make_tensor(bp, b_layout)
    c_tensor = cute.make_tensor(cp, c_layout)

    a_thread_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    b_thread_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    c_thread_layout = cute.make_ordered_layout((16, 16), order=(1, 0))
    value_layout = cute.make_layout((1, 1))

    a_cta_tiler, a_layout_tv = cute.make_layout_tv(a_thread_layout, value_layout)
    b_cta_tiler, b_layout_tv = cute.make_layout_tv(b_thread_layout, value_layout)
    c_cta_tiler, c_layout_tv = cute.make_layout_tv(c_thread_layout, value_layout)

    a_tiled_tensor = cute.zipped_divide(a_tensor, a_cta_tiler)
    b_tiled_tensor = cute.zipped_divide(b_tensor, b_cta_tiler)
    c_tiled_tensor = cute.zipped_divide(c_tensor, c_cta_tiler)

    a_coord_tensor = cute.make_identity_tensor((m, k))
    b_coord_tensor = cute.make_identity_tensor((k, n))
    c_coord_tensor = cute.make_identity_tensor((m, n))

    a_tiled_coord_tensor = cute.zipped_divide(a_coord_tensor, a_cta_tiler)
    b_tiled_coord_tensor = cute.zipped_divide(b_coord_tensor, b_cta_tiler)
    c_tiled_coord_tensor = cute.zipped_divide(c_coord_tensor, c_cta_tiler)

    kernel(
        a_tiled_tensor,
        b_tiled_tensor,
        c_tiled_tensor,
        a_tiled_coord_tensor,
        b_tiled_coord_tensor,
        c_tiled_coord_tensor,
        a_layout_tv,
        b_layout_tv,
        c_layout_tv,
        m,
        n,
        k,
    ).launch(
        grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
        block=(256, 1, 1),
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
