// Puzzle 11 — weight-only int8 GEMM
// Goal: C=A*dequant(W), where each output channel has scale[n] and W is int8.
// Fuse dequantization into accumulation; do not create a float weight matrix.
// Learn: mixed types, vectorized loads, per-channel scaling, epilogue fusion.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void int8_gemm(float const*a,signed char const*w,float const*scale,float*c,int m,int n,int k){
  // TODO: map (m,n) with CuTe layouts; accumulate float(a[m,k])*float(w[k,n]),
  // then multiply once by scale[n]. Stretch: load int8 weights as packed int4.
}

int main(){constexpr int M=31,N=48,K=96;auto a=puzzle::random_vector(M*K,17);auto s=puzzle::random_vector(N,18,.01f,.08f);std::vector<signed char>w(K*N);for(int i=0;i<K*N;++i)w[i]=static_cast<signed char>((i*17)%127-63);std::vector<float>e(M*N,0),o=puzzle::sentinel_vector(M*N);for(int i=0;i<M;++i)for(int j=0;j<N;++j){for(int z=0;z<K;++z)e[i*N+j]+=a[i*K+z]*float(w[z*N+j]);e[i*N+j]*=s[j];}puzzle::DeviceBuffer<float>da(a.size()),ds(s.size()),dc(o.size());puzzle::DeviceBuffer<signed char>dw(w.size());da.upload(a);dw.upload(w);ds.upload(s);dc.upload(o);int8_gemm<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da.get(),dw.get(),ds.get(),dc.get(),M,N,K);puzzle::after_launch();return puzzle::check(dc.download(),e,1e-3f,1e-3f)?0:1;}
