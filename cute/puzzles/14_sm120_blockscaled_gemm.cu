// Puzzle 14 — Blackwell GeForce block-scaled GEMM (SM120, advanced)
// Goal: implement MXFP8/NVFP4 block-scaled GEMM with tcgen05 MMA and its scale
// factor layout. Configure with -DCMAKE_CUDA_ARCHITECTURES=120a and use a
// CUTLASS version/toolkit that supports the exact instruction you select.
// Learn: tcgen05 MMA, tensor memory, block scaling, scale-factor layouts.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void sm120_blockscaled_gemm(float const*a,float const*b,float*c,int n){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  // TODO(SM120): quantize/load operands and scale factors in the layout expected
  // by a block-scaled MMA atom, issue tcgen05 MMA, then write float output.
  // First milestone: implement a non-block-scaled tcgen05 GEMM with the same tile.
#endif
}

int main(){int device=0;cudaDeviceProp p{};CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaGetDeviceProperties(&p,device));if(p.major*10+p.minor<120){std::cout<<"SKIP: puzzle 14 requires SM120 (build for 120a)\n";return 0;}constexpr int N=64;auto a=puzzle::random_vector(N*N,24,-.2f,.2f),b=puzzle::random_vector(N*N,25,-.2f,.2f);std::vector<float>e(N*N,0),o=puzzle::sentinel_vector(N*N);for(int i=0;i<N;++i)for(int j=0;j<N;++j)for(int z=0;z<N;++z)e[i*N+j]+=a[i*N+z]*b[z*N+j];puzzle::DeviceBuffer<float>da(a.size()),db(b.size()),dc(o.size());da.upload(a);db.upload(b);dc.upload(o);sm120_blockscaled_gemm<<<1,128>>>(da.get(),db.get(),dc.get(),N);puzzle::after_launch();return puzzle::check(dc.download(),e,2e-2f,2e-2f)?0:1;}
