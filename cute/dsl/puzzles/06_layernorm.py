"""Puzzle 06: fused row LayerNorm with affine epilogue.

For every row, compute:

    mean = sum(x) / cols
    variance = sum(x*x) / cols - mean*mean
    y[row,col] = (x[row,col] - mean) * rsqrt(variance + eps)
                 * gamma[col] + beta[col]

One CTA owns one row. The kernel combines the sum and sum-of-squares trees so
both statistics traverse shared memory together, then fuses normalization and
the affine transform into the final global-memory pass.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


THREADS = 256


@cute.kernel
def kernel(
    tiled_x: cute.Tensor,
    tiled_gamma: cute.Tensor,
    tiled_beta: cute.Tensor,
    tiled_y: cute.Tensor,
    tiled_coordinate: cute.Tensor,
    problem_shape: cute.Shape,
    epsilon: cutlass.Float32,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    # x/y were divided by a (1,256) CTA tile:
    #   ((1,256),(rows,ceil(cols/256))) -> gmem address
    # gamma/beta were divided by a 256-element tile:
    #   (256,ceil(cols/256)) -> gmem address
    #
    # The retained RestCol mode is identical for all four tensors, so `chunk`
    # names the same original column in x, y, gamma, and beta.
    matrix_thread_slice = ((0, tid), (row, None))
    vector_thread_slice = (tid, None)
    thread_x = tiled_x[matrix_thread_slice]
    thread_y = tiled_y[matrix_thread_slice]
    thread_coordinate = tiled_coordinate[matrix_thread_slice]
    thread_gamma = tiled_gamma[vector_thread_slice]
    thread_beta = tiled_beta[vector_thread_slice]

    # Each thread independently reduces its strided subset into registers.
    # Accumulating x and x^2 in the same traversal avoids another global load.
    local_sum = cutlass.Float32(0.0)
    local_sum_square = cutlass.Float32(0.0)
    for chunk in cutlass.range(cute.size(thread_x), unroll=1):
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            value = thread_x[chunk]
            local_sum += value
            local_sum_square += value * value

    # Shared layout:
    #   (statistic,lane) -> statistic*256 + lane
    # Mode 0 selects sum or sumsq; mode 1 selects the contributing thread.
    # Neighboring lanes are contiguous, so reduction accesses are bank-friendly.
    smem = cutlass.utils.SmemAllocator()
    partial_layout = cute.make_layout((2, THREADS), stride=(THREADS, 1))
    partial = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=partial_layout,
        byte_alignment=16,
        swizzle=None,
    )
    partial[0, tid] = local_sum
    partial[1, tid] = local_sum_square
    cute.arch.sync_threads()

    # Reduce both statistics with the same balanced tree. Keeping the two
    # additions adjacent gives the compiler instruction-level parallelism.
    for stage in cutlass.range_constexpr(8):
        offset = 128 >> stage
        if tid < offset:
            partial[0, tid] = partial[0, tid] + partial[0, tid + offset]
            partial[1, tid] = partial[1, tid] + partial[1, tid + offset]
        cute.arch.sync_threads()

    count = problem_shape[1]
    mean = partial[0, 0] / count
    second_moment = partial[1, 0] / count

    # Roundoff can make E[x^2]-E[x]^2 slightly negative for nearly constant
    # rows. Clamp to zero before rsqrt. This two-moment method is educational
    # and fast, though Welford is preferable for numerically difficult inputs.
    variance = cute.arch.fmax(second_moment - mean * mean, 0.0)
    inverse_stddev = cute.math.rsqrt(variance + epsilon, fastmath=True)

    # Fused normalization + affine epilogue. Invalid coordinates are skipped,
    # so gamma/beta are never read from the padded portion of the final tile.
    for chunk in cutlass.range(cute.size(thread_y), unroll=1):
        if cute.elem_less(thread_coordinate[chunk], problem_shape):
            normalized = (thread_x[chunk] - mean) * inverse_stddev
            thread_y[chunk] = normalized * thread_gamma[chunk] + thread_beta[chunk]


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
    problem_shape = (r, c)

    # Rank-2 row-major x/y layouts:
    #   (row,col) -> pointer + row*c + col
    matrix_layout = cute.make_layout(problem_shape, stride=(c, 1))
    x = cute.make_tensor(xp, matrix_layout)
    y = cute.make_tensor(yp, matrix_layout)

    # Rank-1 gamma/beta layouts indexed only by column. These vectors are
    # shared logically by every row CTA; each CTA reads the same parameters.
    vector_layout = cute.make_layout(c)
    gamma = cute.make_tensor(gp, vector_layout)
    beta = cute.make_tensor(bp, vector_layout)

    # Perform problem-level tiling in the launcher, consistently with puzzles
    # 03–05. Matrix tiles include a singleton row mode; parameter-vector tiles
    # need only the 256-column mode.
    matrix_tiler = (1, THREADS)
    # Even though this tiler has rank one, pass it as a one-element CuTe shape
    # profile. A bare Python `int` is not an MLIR Value and the DSL's
    # zipped_divide binding cannot use it directly as operand 1.
    vector_tiler = (THREADS,)
    tiled_x = cute.zipped_divide(x, matrix_tiler)
    tiled_y = cute.zipped_divide(y, matrix_tiler)
    tiled_gamma = cute.zipped_divide(gamma, vector_tiler)
    tiled_beta = cute.zipped_divide(beta, vector_tiler)

    coordinate = cute.make_identity_tensor(problem_shape)
    tiled_coordinate = cute.zipped_divide(coordinate, matrix_tiler)

    kernel(
        tiled_x,
        tiled_gamma,
        tiled_beta,
        tiled_y,
        tiled_coordinate,
        problem_shape,
        eps,
    ).launch(grid=(r, 1, 1), block=(THREADS, 1, 1), stream=s)


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
