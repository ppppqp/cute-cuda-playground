"""Puzzle 08: shared-memory tiled SIMT GEMM."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
):
    # TODO: allocate 16x16 A/B shared tiles, make tiled copies, predicate edge
    # loads, synchronize each K stage, and accumulate one C value per thread.
    pass


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
    a = cute.make_tensor(ap, cute.make_layout((m, k), stride=(k, 1)))
    b = cute.make_tensor(bp, cute.make_layout((k, n), stride=(n, 1)))
    c = cute.make_tensor(cp, cute.make_layout((m, n), stride=(n, 1)))
    kernel(a, b, c, m, n, k).launch(
        grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
        block=(16, 16, 1),
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
