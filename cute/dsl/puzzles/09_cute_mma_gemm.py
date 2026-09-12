"""Puzzle 09: Ampere tensor-core GEMM with a CuTe TiledMma."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    # TODO: create an SM80 warp MMA MmaF16BF16Op/TiledMma, partition A/B/C,
    # allocate register fragments, call cute.gemm, and store the accumulator.
    pass


@cute.jit
def launch(ap: cute.Pointer, bp: cute.Pointer, cp: cute.Pointer, s: cuda.CUstream):
    n = 64
    a = cute.make_tensor(ap, cute.make_layout((n, n), stride=(n, 1)))
    b = cute.make_tensor(bp, cute.make_layout((n, n), stride=(n, 1)))
    c = cute.make_tensor(cp, cute.make_layout((n, n), stride=(n, 1)))
    kernel(a, b, c).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=s)


def main():
    n = 64
    a = random((n, n), 14, -0.25, 0.25)
    b = random((n, n), 15, -0.25, 0.25)
    o = unfinished((n, n))
    with DeviceArray(a) as da, DeviceArray(b) as db, DeviceArray(o) as dc:
        launch(da.ptr, db.ptr, dc.ptr, stream())
        synchronize()
        check(dc.download(), a @ b, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    main()
