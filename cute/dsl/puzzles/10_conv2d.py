"""Puzzle 10: direct NHWC 3x3 convolution expressed with CuTe tensors."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    y: cute.Tensor,
    h: cutlass.Int32,
    width: cutlass.Int32,
    ci: cutlass.Int32,
    co: cutlass.Int32,
):
    # TODO: one thread per output element; loop 3x3xCi with padding predicates.
    # Stretch: transform the coordinate problem into an implicit GEMM view.
    pass


@cute.jit
def launch(
    xp: cute.Pointer,
    wp: cute.Pointer,
    yp: cute.Pointer,
    h: cutlass.Int32,
    width: cutlass.Int32,
    ci: cutlass.Int32,
    co: cutlass.Int32,
    s: cuda.CUstream,
):
    x = cute.make_tensor(
        xp, cute.make_layout((h, width, ci), stride=(width * ci, ci, 1))
    )
    w = cute.make_tensor(
        wp, cute.make_layout((3, 3, ci, co), stride=(3 * ci * co, ci * co, co, 1))
    )
    y = cute.make_tensor(
        yp, cute.make_layout((h, width, co), stride=(width * co, co, 1))
    )
    count = h * width * co
    kernel(x, w, y, h, width, ci, co).launch(
        grid=(cute.ceil_div(count, 256), 1, 1), block=(256, 1, 1), stream=s
    )


def main():
    h, width, ci, co = 13, 11, 3, 5
    x = random((h, width, ci), 16)
    w = random((3, 3, ci, co), 17)
    e = np.zeros((h, width, co), np.float32)
    for oh in range(h):
        for ow in range(width):
            for kh in range(3):
                for kw in range(3):
                    ih, iw = oh + kh - 1, ow + kw - 1
                    if 0 <= ih < h and 0 <= iw < width:
                        e[oh, ow] += x[ih, iw] @ w[kh, kw]
    with (
        DeviceArray(x) as dx,
        DeviceArray(w) as dw,
        DeviceArray(unfinished(e.shape)) as dy,
    ):
        launch(dx.ptr, dw.ptr, dy.ptr, h, width, ci, co, stream())
        synchronize()
        check(dy.download(), e, atol=7e-4, rtol=7e-4)


if __name__ == "__main__":
    main()
