// Puzzle 08 — shared-memory tiled GEMM
// Goal: implement a 16x16x16 tiled GEMM with predicated edge loads.
// Express global/shared layouts with CuTe; double-buffering is an optional stretch.
// Learn: CTA tiling, cooperative copy, synchronization, arithmetic intensity.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void tiled_gemm(float const* a,float const* b,float* c,int m,int n,int k){
  __shared__ float as[16][16],bs[16][16];
  using Tile=decltype(cute::make_layout(cute::make_shape(cute::Int<16>{},cute::Int<16>{}),cute::LayoutRight{}));
  Tile tile;
  // TODO: each CTA computes one 16x16 C tile. For every K tile, load A/B
  // cooperatively (zero-fill OOB), synchronize, accumulate, synchronize.
  // Use `tile(cute::make_coord(row,col))` when flattening a shared tile.
}

int main(){constexpr int M=71,N=58,K=65;auto a=puzzle::random_vector(M*K,11),b=puzzle::random_vector(K*N,12);std::vector<float>w(M*N,0),o=puzzle::sentinel_vector(M*N);for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int z=0;z<K;++z)w[i*N+j]+=a[i*K+z]*b[z*N+j];puzzle::DeviceBuffer<float>da(M*K),db(K*N),dc(M*N);da.upload(a);db.upload(b);dc.upload(o);tiled_gemm<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da.get(),db.get(),dc.get(),M,N,K);puzzle::after_launch();return puzzle::check(dc.download(),w,8e-4f,8e-4f)?0:1;}
