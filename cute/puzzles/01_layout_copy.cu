// Puzzle 01 — CuTe layout as an index function
// Goal: copy x to y using a rank-1 CuTe Layout. Do not index x/y directly
// with threadIdx.x: call the layout to produce the offset.
// Learn: Int<N>, Shape, Stride, Layout::operator(), cute::size.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void layout_copy(float const* x, float* y, int n) {
  using Layout = decltype(cute::make_layout(cute::make_shape(cute::Int<256>{}),
                                            cute::make_stride(cute::Int<1>{})));
  Layout layout;
  int block_base = int(blockIdx.x) * cute::size(layout);
  // TODO: obtain the within-block offset from `layout(threadIdx.x)`, add
  // block_base, bounds-check against n, and copy one element.
}

int main() {
  constexpr int N = 1003;
  auto x = puzzle::random_vector(N);
  auto y = puzzle::sentinel_vector(N);
  puzzle::DeviceBuffer<float> dx(N), dy(N);
  dx.upload(x); dy.upload(y);
  layout_copy<<<(N + 255) / 256, 256>>>(dx.get(), dy.get(), N);
  puzzle::after_launch();
  return puzzle::check(dy.download(), x) ? 0 : 1;
}
