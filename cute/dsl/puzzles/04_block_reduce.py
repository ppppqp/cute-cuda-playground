"""Puzzle 04: one CTA reduction per matrix row."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(x: cute.Tensor, y: cute.Tensor, rows: cutlass.Int32, cols: cutlass.Int32):
    # TODO: strip-mine one row across 256 threads, then reduce via shared memory.
    pass


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    kernel(
        cute.make_tensor(xp, cute.make_layout((r, c), stride=(c, 1))),
        cute.make_tensor(yp, cute.make_layout(r)),
        r,
        c,
    ).launch(grid=(r, 1, 1), block=(256, 1, 1), stream=s)


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
