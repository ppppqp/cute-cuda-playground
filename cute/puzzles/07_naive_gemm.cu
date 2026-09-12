// Puzzle 07 — GEMM coordinate fundamentals
// Goal: C[M,N] = A[M,K] * B[K,N], with one thread per C element.
// Build row-major CuTe tensors/layouts for A, B, and C and use their coordinate
// mappings in the K loop. This version is intentionally not tiled.
// Learn: rank-2 tensor coordinates, leading dimensions, GEMM semantics.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void naive_gemm(float const* a,float const* b,float* c,int m,int n,int k){
  auto la=cute::make_layout(cute::make_shape(m,k),cute::make_stride(k,cute::Int<1>{}));
  auto lb=cute::make_layout(cute::make_shape(k,n),cute::make_stride(n,cute::Int<1>{}));
  auto lc=cute::make_layout(cute::make_shape(m,n),cute::make_stride(n,cute::Int<1>{}));
  int row=blockIdx.y*blockDim.y+threadIdx.y,col=blockIdx.x*blockDim.x+threadIdx.x;
  // TODO: if in bounds, accumulate over K using la/lb and store through lc.
}

int main(){constexpr int M=53,N=61,K=47;auto a=puzzle::random_vector(M*K,9),b=puzzle::random_vector(K*N,10);std::vector<float>w(M*N,0),o=puzzle::sentinel_vector(M*N);for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int k=0;k<K;++k)w[i*N+j]+=a[i*K+k]*b[k*N+j];puzzle::DeviceBuffer<float>da(M*K),db(K*N),dc(M*N);da.upload(a);db.upload(b);dc.upload(o);naive_gemm<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da.get(),db.get(),dc.get(),M,N,K);puzzle::after_launch();return puzzle::check(dc.download(),w,5e-4f,5e-4f)?0:1;}
