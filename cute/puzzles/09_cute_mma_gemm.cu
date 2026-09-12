// Puzzle 09 — CuTe MMA partitioning
// Goal: replace scalar tiled GEMM with a CuTe TiledMMA and register fragments.
// Suggested path: choose an SM80 MMA_Atom, make_tiled_mma, partition A/B/C,
// clear the accumulator fragment, call cute::gemm, then store the fragment.
// Learn: MMA_Atom, TiledMMA, ThrMMA, partition_A/B/C, register fragments.
// Constraint: build for SM80+; start with multiples of 64 to avoid predication.
#include <cute/layout.hpp>
// After installing matching CUDA development headers, add:
//   #include <cute/atom/mma_atom.hpp>
//   #include <cute/atom/mma_traits_sm80.hpp>
#include "puzzle_utils.cuh"

__global__ void cute_mma_gemm(float const* a,float const* b,float* c,int m,int n,int k){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // TODO: implement an SM80 CuTe MMA kernel. You may change input storage to
  // half after first making the float interface pass. Keep float accumulators.
#endif
}

int main(){constexpr int M=64,N=64,K=64;auto a=puzzle::random_vector(M*K,13,-.25f,.25f),b=puzzle::random_vector(K*N,14,-.25f,.25f);std::vector<float>w(M*N,0),o=puzzle::sentinel_vector(M*N);for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int z=0;z<K;++z)w[i*N+j]+=a[i*K+z]*b[z*N+j];puzzle::DeviceBuffer<float>da(M*K),db(K*N),dc(M*N);da.upload(a);db.upload(b);dc.upload(o);cute_mma_gemm<<<1,128>>>(da.get(),db.get(),dc.get(),M,N,K);puzzle::after_launch();return puzzle::check(dc.download(),w,1e-3f,1e-3f)?0:1;}
