// Puzzle 06 — fused LayerNorm
// Goal: one block per row computes mean/variance and applies y=(x-mean)/sqrt(var+eps)*gamma+beta.
// Accumulate in float and use E[x^2]-E[x]^2 (then consider when Welford is better).
// Learn: paired reductions, broadcast, fusion, numerical error.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void layernorm(float const* x, float const* gamma, float const* beta,
                          float* y, int rows, int cols, float epsilon) {
  using Threads = decltype(cute::make_layout(cute::make_shape(cute::Int<256>{})));
  Threads threads;
  extern __shared__ float scratch[];  // enough for sum and sum-of-squares
  // TODO: reduce sum and sumsq, derive mean and variance, then normalize every
  // element owned by this thread and apply per-column gamma/beta.
}

int main(){constexpr int R=19,C=768;constexpr float E=1e-5f;auto x=puzzle::random_vector(R*C,6,-3,3);auto g=puzzle::random_vector(C,7,.5f,1.5f);auto b=puzzle::random_vector(C,8,-.2f,.2f);std::vector<float>w(R*C),o=puzzle::sentinel_vector(R*C);for(int r=0;r<R;++r){double s=0,q=0;for(int c=0;c<C;++c){double v=x[r*C+c];s+=v;q+=v*v;}float m=s/C,v=q/C-m*m;for(int c=0;c<C;++c)w[r*C+c]=(x[r*C+c]-m)/std::sqrt(v+E)*g[c]+b[c];}puzzle::DeviceBuffer<float>dx(R*C),dg(C),db(C),dy(R*C);dx.upload(x);dg.upload(g);db.upload(b);dy.upload(o);layernorm<<<R,256,512*sizeof(float)>>>(dx.get(),dg.get(),db.get(),dy.get(),R,C,E);puzzle::after_launch();return puzzle::check(dy.download(),w,3e-4f,3e-4f)?0:1;}
