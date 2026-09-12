"""Puzzle 06: fused row LayerNorm with affine epilogue."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    x: cute.Tensor,
    g: cute.Tensor,
    b: cute.Tensor,
    y: cute.Tensor,
    r: cutlass.Int32,
    c: cutlass.Int32,
    eps: cutlass.Float32,
):
    # TODO: paired sum/sumsq reduction, broadcast statistics, apply gamma/beta.
    pass


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
    lm = cute.make_layout((r, c), stride=(c, 1))
    lv = cute.make_layout(c)
    kernel(
        cute.make_tensor(xp, lm),
        cute.make_tensor(gp, lv),
        cute.make_tensor(bp, lv),
        cute.make_tensor(yp, lm),
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
