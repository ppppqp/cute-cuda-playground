"""Puzzle 05: numerically stable fused row softmax."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(x: cute.Tensor, y: cute.Tensor, rows: cutlass.Int32, cols: cutlass.Int32):
    # TODO: CTA max reduction, exp/sum reduction, then normalized stores.
    pass


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    l = cute.make_layout((r, c), stride=(c, 1))
    kernel(cute.make_tensor(xp, l), cute.make_tensor(yp, l), r, c).launch(
        grid=(r, 1, 1), block=(256, 1, 1), stream=s
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
