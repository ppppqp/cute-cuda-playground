"""Puzzle 02: matrix add via canonical CTA/thread/value partitioning.

Reading notation used below
---------------------------
CuTe layouts are functions. ``X -> Y`` means "a mapping whose domain is X and
whose result is Y". For a memory tensor, Y is an address/offset. For an
identity tensor, Y is a logical coordinate.

``None`` in a tensor subscript means "retain this mode". An integer fixes a
mode to one coordinate and removes it from the result. Thus slicing is partial
function application, not a data copy.

The ownership path in this file is:

  matrix coordinate (m,n) -> global-memory address
  ((tile_m,tile_n),(cta_m,cta_n)) -> global-memory address
  (tile_m,tile_n) -> global-memory address             [slice one CTA]
  (thread,value) -> global-memory address              [compose TV mapping]
  value -> global-memory address                       [slice one thread]

No operation in that path moves or repacks data. It only constructs views with
new coordinate systems.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from puzzle_utils import DeviceArray, check, random, stream, synchronize, unfinished


@cute.kernel
def kernel(
    tiled_a: cute.Tensor,
    tiled_b: cute.Tensor,
    tiled_c: cute.Tensor,
    tiled_coord: cute.Tensor,
    problem_shape: cute.Shape,
    tv_layout: cute.Layout,
):
    # Runtime scalar IR values (normally Int32):
    #   tid     in [0, 256)
    #   block_x in [0, ceil_div(N,16))
    #   block_y in [0, ceil_div(M,16))
    tid, _, _ = cute.arch.thread_idx()
    block_x, block_y, _ = cute.arch.block_idx()

    # Each tiled_* argument is a Tensor with hierarchical logical domain:
    #
    #   ((TileM, TileN), (RestM, RestN)) -> address
    #     (16,    16)    (ceil(M/16), ceil(N/16))
    #
    # Its engine is still the original global-memory pointer. `zipped_divide`
    # changed only the layout attached to that engine.
    #
    # cta_coord retains both within-tile modes and fixes both rest/grid modes:
    #   ((None,None), (block_y,block_x))
    cta_coord = ((None, None), (block_y, block_x))

    # Types/mappings after slicing:
    #   block_a/b/c : Tensor[gmem, Layout] with
    #                 (tile_m,tile_n) -> global-memory address
    #   block_coord : Tensor[coordinate-engine, Layout] with
    #                 (tile_m,tile_n) -> original (m,n) coordinate
    #
    # Every block tensor has logical shape (16,16), including an edge CTA.
    # Coordinates in the final edge tile may be outside the real (M,N), so
    # block_coord will later provide the predicate.
    block_a = tiled_a[cta_coord]
    block_b = tiled_b[cta_coord]
    block_c = tiled_c[cta_coord]
    block_coord = tiled_coord[cta_coord]

    # tv_layout has the conceptual type:
    #
    #   Layout: (thread_id, value_id) -> (tile_m, tile_n)
    #
    # Here T=256 and V=1. `composition(block_a, tv_layout)` is ordinary
    # function composition:
    #
    #   tv_a(t,v) = block_a(tv_layout(t,v))
    #
    # Therefore each tv_* is a Tensor whose domain is (T,V). Its engine is
    # inherited from block_*, while its layout is the composed layout:
    #
    #   tv_a/b/c : (thread,value) -> global-memory address
    #   tv_coord : (thread,value) -> original (m,n) coordinate
    tv_a = cute.composition(block_a, tv_layout)
    tv_b = cute.composition(block_b, tv_layout)
    tv_c = cute.composition(block_c, tv_layout)
    tv_coord = cute.composition(block_coord, tv_layout)

    # `tv_a` can have a hierarchical value mode in more advanced copies.
    # `cute.repeat_like(None, tv_a[1])` constructs a None-profile with exactly
    # the same hierarchy as that value mode. It does not load tv_a[1].
    #
    # thread_slice fixes T=tid and retains all of V. Partial application turns
    # (T,V)->address into V->address.
    thread_slice = (tid, cute.repeat_like(None, tv_a[1]))

    # Types/mappings owned by this CUDA thread:
    #   thread_a/b/c : Tensor[gmem, Layout], V -> global-memory address
    #   thread_coord : Tensor[coordinate-engine, Layout], V -> (m,n)
    # In this exercise each has size V=1. If value_layout later gives each
    # thread four values, this code still yields a four-element tensor.
    thread_a = tv_a[thread_slice]
    thread_b = tv_b[thread_slice]
    thread_c = tv_c[thread_slice]
    thread_coord = tv_coord[thread_slice]

    # `range_constexpr` is unrolled at JIT-compilation time because V is static.
    # `thread_coord[value]` evaluates the coordinate mapping, while
    # `thread_a[value]` and friends evaluate address mappings and then load or
    # store through their tensor engines.
    for value in cutlass.range_constexpr(cute.size(thread_a)):
        # elem_less performs component-wise coordinate comparison followed by
        # logical AND: (coord_m < M) and (coord_n < N).
        if cute.elem_less(thread_coord[value], problem_shape):
            # TODO: thread_c[value] = thread_a[value] + thread_b[value]
            pass


@cute.jit
def launch(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    cuda_stream: cuda.CUstream,
):
    # problem_shape is a CuTe Shape with two *dynamic* Int32 modes. Its Python
    # value looks like (m,n), but inside @cute.jit the elements are IR values.
    problem_shape = (m, n)

    # Type: Layout<(m,n):(n,1)>.
    # Mapping: (row,col) -> row*n + col. A Layout is shape + stride; it owns no
    # pointer and performs no memory access.
    matrix_layout = cute.make_layout(problem_shape, stride=(n, 1))

    # Type: Tensor<Pointer[Float32,gmem], Layout<(m,n):(n,1)>>.
    # Mapping: (row,col) -> a_ptr + row*n + col. All three tensors reuse the
    # same immutable layout object but have different pointer engines.
    matrix_a = cute.make_tensor(a_ptr, matrix_layout)
    matrix_b = cute.make_tensor(b_ptr, matrix_layout)
    matrix_c = cute.make_tensor(c_ptr, matrix_layout)

    # Type: static Layout<(16,16):(16,1)> (conceptually).
    # Mapping: linear thread ID -> (thread_m,thread_n), with thread_n varying
    # fastest. This gives neighboring CUDA threads neighboring matrix columns,
    # hence coalesced 32-bit memory operations.
    thread_layout = cute.make_ordered_layout((16, 16), order=(1, 0))

    # Static Layout<(1,1)>. It describes the per-thread value arrangement.
    # Both extents are one, so every thread owns exactly one logical value.
    value_layout = cute.make_layout((1, 1))

    # make_layout_tv combines ownership descriptions; it does not inspect A/B/C.
    #   cta_tiler: Shape<(16,16)> -- collective logical coverage of one CTA
    #   tv_layout: Layout with (T,V)->(tile_m,tile_n), T=256 and V=1
    cta_tiler, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    # Each result is a Tensor view with domain:
    #   ((TileM,TileN),(RestM,RestN)) -> address
    # This is a logical divide + regrouping of tile modes and rest modes. It is
    # not allocation, packing, or a memory copy.
    tiled_a = cute.zipped_divide(matrix_a, cta_tiler)
    tiled_b = cute.zipped_divide(matrix_b, cta_tiler)
    tiled_c = cute.zipped_divide(matrix_c, cta_tiler)

    # IdentityTensor<(m,n)> maps (row,col) -> (row,col), returning coordinates
    # rather than memory. Applying the identical divide/slice/compose operations
    # guarantees that each data value and its predicate coordinate have exactly
    # the same ownership.
    coordinates = cute.make_identity_tensor(problem_shape)
    tiled_coord = cute.zipped_divide(coordinates, cta_tiler)

    # mode 0 of tv_layout is its thread mode, so this evaluates to 256. A 1-D
    # CUDA block is used even though ownership is logically arranged as 16x16;
    # thread_layout performs that logical mapping.
    kernel(tiled_a, tiled_b, tiled_c, tiled_coord, problem_shape, tv_layout).launch(
        grid=(cute.ceil_div(n, 16), cute.ceil_div(m, 16), 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
        stream=cuda_stream,
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
