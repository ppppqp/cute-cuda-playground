"""Puzzle 03: predicated, shared-memory tiled transpose.

The implementation uses three changes of logical coordinates without moving
the underlying global-memory allocation:

1. ``zipped_divide``: matrix -> (coordinate inside CTA tile, CTA coordinate)
2. ``composition``: (thread,value) -> coordinate inside CTA tile -> address
3. a transposed shared-memory view: output coordinate -> transposed address

The shared allocation has logical shape (32,32) but physical row stride 33.
That extra element prevents a warp reading a shared-memory column from sending
all 32 lanes to the same bank.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    tiled_x: cute.Tensor,
    tiled_y: cute.Tensor,
    tiled_x_coord: cute.Tensor,
    tiled_y_coord: cute.Tensor,
    x_shape: cute.Shape,
    y_shape: cute.Shape,
    tv_layout: cute.Layout,
):
    # CUDA has a flat 256-thread CTA. tv_layout gives those threads a logical
    # 8x32 arrangement and gives each thread four values along the row mode.
    tid, _, _ = cute.arch.thread_idx()
    block_x, block_y, _ = cute.arch.block_idx()

    # tiled_x has domain:
    #   ((32,32), (ceil(rows/32),ceil(cols/32))) -> gmem address
    # Select input CTA (block_y,block_x), retaining the complete 32x32 tile.
    x_cta_slice = ((None, None), (block_y, block_x))
    block_x_tensor = tiled_x[x_cta_slice]
    block_x_coord = tiled_x_coord[x_cta_slice]

    # y is the transposed matrix, so the same CTA occupies grid coordinate
    # (block_x,block_y) in y's coordinate system.
    y_cta_slice = ((None, None), (block_x, block_y))
    block_y_tensor = tiled_y[y_cta_slice]
    block_y_coord = tiled_y_coord[y_cta_slice]

    # Allocate 32*33 floats, but expose only a logical (32,32) tensor.
    #
    # s_layout: (row,col) -> row*33 + col
    #
    # The unused element at the end of each row rotates the bank assignment of
    # successive rows. SmemAllocator returns Tensor[Pointer[Float32,smem],Layout].
    smem = cutlass.utils.SmemAllocator()
    s_layout = cute.make_layout((32, 32), stride=(33, 1))
    shared_tile = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=s_layout,
        byte_alignment=16,
        swizzle=None,
    )

    # tv_layout maps (T,V)->(tile_row,tile_col), with T=256 and V=4.
    # Composition changes both tensors into a common ownership coordinate:
    #   thread_x:      (T,V) -> input gmem address
    #   thread_x_crd:  (T,V) -> original input (row,col)
    #   thread_shared: (T,V) -> shared-memory address
    tv_x = cute.composition(block_x_tensor, tv_layout)
    tv_x_coord = cute.composition(block_x_coord, tv_layout)
    tv_shared = cute.composition(shared_tile, tv_layout)

    # Fix T=tid and retain the full (possibly hierarchical) V mode.
    thread_slice = (tid, cute.repeat_like(None, tv_x[1]))
    thread_x = tv_x[thread_slice]
    thread_x_coord = tv_x_coord[thread_slice]
    thread_shared = tv_shared[thread_slice]

    # Global -> shared. Edge CTAs contain logical tile coordinates outside the
    # real matrix. Zero-fill them so every shared location is initialized even
    # though only in-bounds values will eventually be written to y.
    for value in cutlass.range_constexpr(cute.size(thread_x)):
        if cute.elem_less(thread_x_coord[value], x_shape):
            thread_shared[value] = thread_x[value]
        else:
            thread_shared[value] = 0.0

    # All threads must finish writing before any thread reads the tile through
    # the transposed view below.
    cute.arch.sync_threads()

    # Reuse the same shared pointer with a different layout:
    #
    #   shared_tile:       (r,c) -> 33*r + c
    #   shared_transposed: (r,c) -> r + 33*c
    #
    # Thus shared_transposed(r,c) aliases shared_tile(c,r). This is a new view;
    # no shared-memory transpose or copy happens here.
    transposed_layout = cute.make_layout((32, 32), stride=(1, 33))
    shared_transposed = cute.make_tensor(shared_tile.iterator, transposed_layout)

    # Apply the same (T,V) ownership mapping to the output tile and transposed
    # shared view. Since the output's stride-1 coordinate is its column, nearby
    # threads again issue coalesced stores.
    tv_y = cute.composition(block_y_tensor, tv_layout)
    tv_y_coord = cute.composition(block_y_coord, tv_layout)
    tv_shared_transposed = cute.composition(shared_transposed, tv_layout)

    output_thread_slice = (tid, cute.repeat_like(None, tv_y[1]))
    thread_y = tv_y[output_thread_slice]
    thread_y_coord = tv_y_coord[output_thread_slice]
    thread_shared_transposed = tv_shared_transposed[output_thread_slice]

    # Shared -> global. The output coordinate tensor supplies the correct edge
    # predicate for output shape (cols,rows).
    for value in cutlass.range_constexpr(cute.size(thread_y)):
        if cute.elem_less(thread_y_coord[value], y_shape):
            thread_y[value] = thread_shared_transposed[value]


@cute.jit
def launch(
    xp: cute.Pointer,
    yp: cute.Pointer,
    r: cutlass.Int32,
    c: cutlass.Int32,
    s: cuda.CUstream,
):
    # x: (rows,cols):(cols,1); y: (cols,rows):(rows,1).
    # These are dynamic layouts because rows/cols are runtime Int32 values.
    x_shape = (r, c)
    y_shape = (c, r)
    x_layout = cute.make_layout(x_shape, stride=(c, 1))
    y_layout = cute.make_layout(y_shape, stride=(r, 1))
    x = cute.make_tensor(xp, x_layout)
    y = cute.make_tensor(yp, y_layout)

    # 256 threads are logically arranged as 8 rows x 32 columns. Thread IDs
    # vary fastest across columns, so a warp loads/stores one contiguous row.
    thread_layout = cute.make_ordered_layout((8, 32), order=(1, 0))

    # Each thread owns four positions along the tile's row dimension and one
    # along its column dimension. Combining (8,32) threads with (4,1) values
    # gives collective CTA coverage (32,32).
    value_layout = cute.make_ordered_layout((4, 1), order=(1, 0))
    cta_tiler, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    # Convert matrices to hierarchical tiled views:
    #   ((TileRow,TileCol),(GridRow,GridCol)) -> address
    tiled_x = cute.zipped_divide(x, cta_tiler)
    tiled_y = cute.zipped_divide(y, cta_tiler)

    # Identity tensors return coordinates instead of memory values. Applying
    # the identical tiling/ownership transforms gives exact edge predicates.
    x_coord = cute.make_identity_tensor(x_shape)
    y_coord = cute.make_identity_tensor(y_shape)
    tiled_x_coord = cute.zipped_divide(x_coord, cta_tiler)
    tiled_y_coord = cute.zipped_divide(y_coord, cta_tiler)

    kernel(
        tiled_x,
        tiled_y,
        tiled_x_coord,
        tiled_y_coord,
        x_shape,
        y_shape,
        tv_layout,
    ).launch(
        # Grid follows the input: x dimension tiles columns, y tiles rows.
        grid=(cute.ceil_div(c, 32), cute.ceil_div(r, 32), 1),
        # 256, 1, 1
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
        stream=s,
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
