"""Puzzle 11: int8 weight-only GEMM with fused per-channel scales."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    a: cute.Tensor,
    w: cute.Tensor,
    scale: cute.Tensor,
    c: cute.Tensor,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
):
    # TODO: accumulate float(a[m,k])*float(w[k,n]), then apply scale[n] once.
    # Stretch: use vectorized packed int8 loads and an integer MMA atom.
    pass


@cute.jit
def launch(
    ap: cute.Pointer,
    wp: cute.Pointer,
    sp: cute.Pointer,
    cp: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    s: cuda.CUstream,
):
    a = cute.make_tensor(ap, cute.make_layout((m, k), stride=(k, 1)))
    w = cute.make_tensor(wp, cute.make_layout((k, n), stride=(n, 1)))
    sc = cute.make_tensor(sp, cute.make_layout(n))
    c = cute.make_tensor(cp, cute.make_layout((m, n), stride=(n, 1)))
    kernel(a, w, sc, c, m, n, k).launch(
        grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
        block=(16, 16, 1),
        stream=s,
    )


def main():
    m, n, k = 31, 48, 96
    a = random((m, k), 18)
    w = np.fromfunction(
        lambda i, j: (i * n + j) * 17 % 127 - 63, (k, n), dtype=int
    ).astype(np.int8)
    sc = random((n,), 19, 0.01, 0.08)
    e = (a @ w.astype(np.float32)) * sc
    o = unfinished((m, n))
    with (
        DeviceArray(a) as da,
        DeviceArray(w) as dw,
        DeviceArray(sc) as ds,
        DeviceArray(o) as dc,
    ):
        launch(da.ptr, dw.ptr, ds.ptr, dc.ptr, m, n, k, stream())
        synchronize()
        check(dc.download(), e, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()
