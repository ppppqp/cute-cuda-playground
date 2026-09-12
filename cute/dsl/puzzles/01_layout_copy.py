"""Puzzle 01: copy through a rank-1 CuTe layout."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished

THREADS = 256


@cute.kernel
def kernel(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int32):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    # TODO: map bid*THREADS+tid through the layout, bounds-check, and copy x to y.
    idx = bid * THREADS + tid
    if idx < n:
        y[idx] = x[idx]


@cute.jit
def launch(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int32, s: cuda.CUstream):
    layout = cute.make_layout(n)
    kernel(cute.make_tensor(xp, layout), cute.make_tensor(yp, layout), n).launch(
        grid=(cute.ceil_div(n, THREADS), 1, 1), block=(THREADS, 1, 1), stream=s
    )


def main():
    n = 1003
    x = random((n,), 1)
    y = unfinished((n,))
    with DeviceArray(x) as dx, DeviceArray(y) as dy:
        launch(dx.ptr, dy.ptr, n, stream())
        synchronize()
        check(dy.download(), x)


if __name__ == "__main__":
    main()
