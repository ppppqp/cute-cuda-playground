// Puzzle 05 — numerically stable fused softmax
// Goal: one block computes each row: max reduction, exp/sum reduction, normalize.
// Never materialize intermediate arrays in global memory.
// Learn: multi-stage reductions, numerical stability, reuse of thread layouts.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void softmax(float const* x, float* y, int rows, int cols) {
  using Threads = decltype(cute::make_layout(cute::make_shape(cute::Int<256>{})));
  Threads threads;
  extern __shared__ float scratch[];
  // TODO: compute a row maximum, synchronize; compute exp(x-max) and its sum,
  // synchronize; write normalized probabilities. Support cols > blockDim.x.
}

int main(){
  constexpr int R=23,C=777; auto x=puzzle::random_vector(R*C,5,-8,8);
  std::vector<float> expected(R*C),out=puzzle::sentinel_vector(R*C);
  for(int r=0;r<R;++r){float mx=-INFINITY;for(int c=0;c<C;++c)mx=std::max(mx,x[r*C+c]);float s=0;for(int c=0;c<C;++c)s+=std::exp(x[r*C+c]-mx);for(int c=0;c<C;++c)expected[r*C+c]=std::exp(x[r*C+c]-mx)/s;}
  puzzle::DeviceBuffer<float> dx(R*C),dy(R*C);dx.upload(x);dy.upload(out);
  softmax<<<R,256,256*sizeof(float)>>>(dx.get(),dy.get(),R,C);puzzle::after_launch();
  return puzzle::check(dy.download(),expected,2e-5f,2e-4f)?0:1;
}
