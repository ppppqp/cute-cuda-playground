// Puzzle 04 — one-block reduction
// Goal: produce one sum per row. Each 256-thread block owns a row of 1000
// floats. Use a CuTe thread layout to walk the row, then reduce in shared memory.
// Learn: layout-driven strip mining, reduction trees, synchronization.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void row_sum(float const* x, float* sums, int rows, int cols) {
  using Threads = decltype(cute::make_layout(cute::make_shape(cute::Int<256>{})));
  Threads threads;
  __shared__ float partial[256];
  int lane = threads(threadIdx.x);
  // TODO: accumulate x[blockIdx.x, lane + k*size(threads)], place each local
  // sum in partial, reduce partial to lane 0, and write sums[blockIdx.x].
}

int main() {
  constexpr int R=37,C=1000;
  auto x=puzzle::random_vector(R*C,4); std::vector<float> expected(R,0), out=puzzle::sentinel_vector(R);
  for(int r=0;r<R;++r) for(int c=0;c<C;++c) expected[r]+=x[r*C+c];
  puzzle::DeviceBuffer<float> dx(R*C),dy(R); dx.upload(x);dy.upload(out);
  row_sum<<<R,256>>>(dx.get(),dy.get(),R,C); puzzle::after_launch();
  return puzzle::check(dy.download(),expected,2e-4f,2e-4f)?0:1;
}
