"""Puzzle 02: matrix add with a dynamic rank-2 layout."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, m: cutlass.Int32, n: cutlass.Int32
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row = by * 16 + ty
    col = bx * 16 + tx
    # TODO: bounds-check (row,col) and store a[row,col] + b[row,col].


@cute.jit
def launch(
    ap: cute.Pointer,
    bp: cute.Pointer,
    cp: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    s: cuda.CUstream,
):
    layout = cute.make_layout((m, n), stride=(n, 1))
    kernel(
        cute.make_tensor(ap, layout),
        cute.make_tensor(bp, layout),
        cute.make_tensor(cp, layout),
        m,
        n,
    ).launch(
        grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
        block=(16, 16, 1),
        stream=s,
    )


def main():
    m, n = 67, 131
    a = random((m, n), 2)
    b = random((m, n), 3)
    o = unfinished((m, n))
    with DeviceArray(a) as da, DeviceArray(b) as db, DeviceArray(o) as dc:
        launch(da.ptr, db.ptr, dc.ptr, m, n, stream())
        synchronize()
        check(dc.download(), a + b)


if __name__ == "__main__":
    main()
