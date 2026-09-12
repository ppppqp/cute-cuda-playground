"""Puzzle 03: predicated, shared-memory tiled transpose."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(x: cute.Tensor, y: cute.Tensor, rows: cutlass.Int32, cols: cutlass.Int32):
    # TODO: allocate a padded 32x33 tile with SmemAllocator, cooperatively load
    # four rows per thread, synchronize, and write the transposed tile.
    pass


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    lx = cute.make_layout((r, c), stride=(c, 1))
    ly = cute.make_layout((c, r), stride=(r, 1))
    kernel(cute.make_tensor(xp, lx), cute.make_tensor(yp, ly), r, c).launch(
        grid=(cute.ceil_div(c, 32), cute.ceil_div(r, 32), 1), block=(32, 8, 1), stream=s
    )


def main():
    r, c = 93, 70
    x = random((r, c), 4)
    o = unfinished((c, r))
    with DeviceArray(x) as dx, DeviceArray(o) as dy:
        launch(dx.ptr, dy.ptr, r, c, stream())
        synchronize()
        check(dy.download(), x.T)


if __name__ == "__main__":
    main()
