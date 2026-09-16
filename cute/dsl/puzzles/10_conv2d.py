"""Puzzle 10: direct NHWC 3x3 convolution expressed with CuTe tensors.

This is deliberately the simplest useful convolution kernel: one CUDA thread
computes one scalar output value. It does not use shared memory or tensor cores
yet. The learning goals are to:

* describe structured NHWC/filter tensors with CuTe layouts;
* tile a flat output domain and slice one value per thread;
* convert a linear output coordinate back to ``(oh, ow, oc)``;
* express zero padding with predicates; and
* recognize convolution as an implicit GEMM without materializing im2col.

The later optimization path is to have a CTA compute an output tile, stage
input/filter tiles in shared memory, and replace the scalar reduction with
CuTe MMA atoms. Keeping this version direct gives us a correctness baseline.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import numpy as np
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


# A 1-D CTA is natural because we first linearize the H*W*Co output domain.
THREADS = 256


@cute.kernel
def kernel(
    x: cute.Tensor,
    w: cute.Tensor,
    tiled_y: cute.Tensor,
    tiled_coordinate: cute.Tensor,
    h: cutlass.Int32,
    width: cutlass.Int32,
    ci: cutlass.Int32,
    co: cutlass.Int32,
    output_count: cutlass.Int32,
):
    """Compute a same-size, stride-1, dilation-1, 3x3 convolution.

    ``x`` has shape ``(H,W,Ci)`` and strides ``(W*Ci,Ci,1)``. Thus its
    coordinate-to-offset mapping is ``ih*(W*Ci) + iw*Ci + ic``.

    ``w`` has shape ``(3,3,Ci,Co)`` and HWIO strides
    ``(3*Ci*Co,Ci*Co,Co,1)``.

    ``tiled_y`` and ``tiled_coordinate`` have divided domain
    ``((THREADS),(ceil(H*W*Co/THREADS)))``. The first mode is position inside
    a CTA tile; the second is the tile (and therefore block) index.
    """
    tid = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]

    # This is a scalar lookup: (tid, bid) selects both modes. No ``None`` is
    # needed because we do not want to retain either mode as a tensor slice.
    linear_index = tiled_coordinate[tid, bid]

    # Predicate the partial final CTA before accessing global memory.
    if linear_index < output_count:
        # Invert linear = (oh * W + ow) * Co + oc. Output channel is fastest,
        # agreeing with the unit-stride mode of an HWC output.
        out_channel = linear_index % co
        spatial_index = linear_index // co
        out_w = spatial_index % width
        out_h = spatial_index // width

        # This private scalar normally lives in a register. Every thread has a
        # distinct accumulator and output, so the reduction has no race.
        accumulator = cutlass.Float32(0.0)

        # kh/kw have compile-time trip counts. Ci is a runtime argument, so it
        # uses cutlass.range with an explicitly non-unrolled loop.
        for kh in cutlass.range_constexpr(3):
            for kw in cutlass.range_constexpr(3):
                # Padding=1 centers filter coordinate (1,1) on (oh,ow).
                in_h = out_h + kh - 1
                in_w = out_w + kw - 1

                # Skipping out-of-range positions implements zero padding and
                # prevents illegal global-memory loads.
                if in_h >= 0:
                    if in_h < h:
                        if in_w >= 0:
                            if in_w < width:
                                for input_channel in cutlass.range(ci, unroll=1):
                                    accumulator += (
                                        x[in_h, in_w, input_channel]
                                        * w[
                                            kh,
                                            kw,
                                            input_channel,
                                            out_channel,
                                        ]
                                    )

        tiled_y[tid, bid] = accumulator


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
    # A CuTe Tensor is an engine plus a layout. Here the engine is global
    # memory, while the layouts map structured coordinates to pointer offsets.
    x = cute.make_tensor(
        xp, cute.make_layout((h, width, ci), stride=(width * ci, ci, 1))
    )
    w = cute.make_tensor(
        wp, cute.make_layout((3, 3, ci, co), stride=(3 * ci * co, ci * co, co, 1))
    )

    output_count = h * width * co

    # Contiguous HWC storage admits this zero-copy rank-1 view. Only the layout
    # changes; the output data is not copied or rearranged.
    y_flat = cute.make_tensor(yp, cute.make_layout((output_count,)))

    # A rank-1 tiler is a one-element tuple. zipped_divide composes a new
    # (WithinTile, RestTile) coordinate system with the original layout:
    #
    #   linear ~= within_tile + THREADS * rest_tile
    #
    # It changes the logical view; it does not pack or move the data.
    output_tiler = (THREADS,)
    tiled_y = cute.zipped_divide(y_flat, output_tiler)

    # An identity tensor maps a coordinate to itself. Dividing it in exactly
    # the same way lets each thread recover its original flat output index,
    # including invalid tail coordinates that must be predicated away.
    linear_coordinate = cute.make_identity_tensor((output_count,))
    tiled_coordinate = cute.zipped_divide(linear_coordinate, output_tiler)

    kernel(
        x,
        w,
        tiled_y,
        tiled_coordinate,
        h,
        width,
        ci,
        co,
        output_count,
    ).launch(
        grid=(cute.ceil_div(output_count, THREADS), 1, 1),
        block=(THREADS, 1, 1),
        stream=s,
    )


def main():
    # Non-round dimensions exercise padding and a partial final CTA.
    h, width, ci, co = 13, 11, 3, 5
    x = random((h, width, ci), 16)
    w = random((3, 3, ci, co), 17)

    # NumPy reference for cross-correlation (the deep-learning convention;
    # unlike mathematical convolution, the filter is not spatially flipped).
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
