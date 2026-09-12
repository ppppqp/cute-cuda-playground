"""Puzzle 12: fused online-softmax attention without an SxS score tensor."""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor,
    seq: cutlass.Int32,
    d: cutlass.Int32,
):
    # TODO: tile K/V through shared memory and maintain online (max,sum,O)
    # state per query row. Rescale the old accumulator whenever max increases.
    pass


@cute.jit
def launch(
    qp: cute.Pointer,
    kp: cute.Pointer,
    vp: cute.Pointer,
    op: cute.Pointer,
    seq: cutlass.Int32,
    d: cutlass.Int32,
    s: cuda.CUstream,
):
    l = cute.make_layout((seq, d), stride=(d, 1))
    kernel(
        cute.make_tensor(qp, l),
        cute.make_tensor(kp, l),
        cute.make_tensor(vp, l),
        cute.make_tensor(op, l),
        seq,
        d,
    ).launch(grid=(seq, 1, 1), block=(128, 1, 1), stream=s)


def main():
    seq, d = 32, 32
    q = random((seq, d), 20, -0.5, 0.5)
    k = random((seq, d), 21, -0.5, 0.5)
    v = random((seq, d), 22, -0.5, 0.5)
    score = q @ k.T / np.sqrt(d)
    score -= score.max(1, keepdims=True)
    p = np.exp(score)
    p /= p.sum(1, keepdims=True)
    e = p @ v
    with (
        DeviceArray(q) as dq,
        DeviceArray(k) as dk,
        DeviceArray(v) as dv,
        DeviceArray(unfinished(e.shape)) as do,
    ):
        launch(dq.ptr, dk.ptr, dv.ptr, do.ptr, seq, d, stream())
        synchronize()
        check(do.download(), e, atol=5e-4, rtol=5e-4)


if __name__ == "__main__":
    main()
