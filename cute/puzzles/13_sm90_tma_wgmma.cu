// Puzzle 13 — Hopper pipeline (SM90+, advanced)
// Goal: GEMM using TMA global->shared transfers, an mbarrier pipeline, and
// warp-group MMA. Study CUTLASS/CuTe collective builders after writing a small
// explicit pipeline. Configure with -DCMAKE_CUDA_ARCHITECTURES=90a.
// Learn: TMA tensors, GMMA descriptors, warp-group roles, producer/consumer stages.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void hopper_gemm(float const*a,float const*b,float*c,int n){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // TODO(SM90): construct tiled copies and a TiledMMA; dedicate producer warps
  // to TMA, consumer warp-groups to WGMMA, and correctly arrive/wait barriers.
#endif
}

int main(){int device=0;cudaDeviceProp p{};CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaGetDeviceProperties(&p,device));if(p.major<9){std::cout<<"SKIP: puzzle 13 requires SM90+ (build for 90a)\n";return 0;}constexpr int N=64;auto a=puzzle::random_vector(N*N,22,-.2f,.2f),b=puzzle::random_vector(N*N,23,-.2f,.2f);std::vector<float>e(N*N,0),o=puzzle::sentinel_vector(N*N);for(int i=0;i<N;++i)for(int j=0;j<N;++j)for(int z=0;z<N;++z)e[i*N+j]+=a[i*N+z]*b[z*N+j];puzzle::DeviceBuffer<float>da(a.size()),db(b.size()),dc(o.size());da.upload(a);db.upload(b);dc.upload(o);hopper_gemm<<<1,128>>>(da.get(),db.get(),dc.get(),N);puzzle::after_launch();return puzzle::check(dc.download(),e,1e-3f,1e-3f)?0:1;}
