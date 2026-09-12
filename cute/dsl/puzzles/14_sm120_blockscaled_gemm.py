"""Puzzle 14 (SM120): tcgen05 block-scaled GEMM with scale-factor layouts."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import (
    DeviceArray,
    check,
    random,
    require_compute_capability,
    stream,
    synchronize,
    unfinished,
)


@cute.kernel
def kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    # TODO: choose an SM120 tcgen05 block-scaled MMA, create its scale-factor
    # tensors/layouts, stage operands, allocate TMEM accumulators, issue MMA,
    # and write float output. First implement a dense tcgen05 version.
    pass


@cute.jit
def launch(ap: cute.Pointer, bp: cute.Pointer, cp: cute.Pointer, s: cuda.CUstream):
    n = 64
    l = cute.make_layout((n, n), stride=(n, 1))
    kernel(
        cute.make_tensor(ap, l), cute.make_tensor(bp, l), cute.make_tensor(cp, l)
    ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=s)


def main():
    require_compute_capability(12, 0)
    n = 64
    a = random((n, n), 25, -0.2, 0.2)
    b = random((n, n), 26, -0.2, 0.2)
    with (
        DeviceArray(a) as da,
        DeviceArray(b) as db,
        DeviceArray(unfinished((n, n))) as dc,
    ):
        launch(da.ptr, db.ptr, dc.ptr, stream())
        synchronize()
        check(dc.download(), a @ b, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
