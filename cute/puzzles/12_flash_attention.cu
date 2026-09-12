// Puzzle 12 — fused online softmax attention
// Goal: O = softmax(Q*K^T/sqrt(D))*V for one head, without storing the SxS
// score matrix. Process keys/values in tiles and update (row_max,row_sum,O)
// with the online-softmax recurrence. No causal mask in the base puzzle.
// Learn: FlashAttention dataflow, online softmax, tiling, fusion.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void flash_attention(float const*q,float const*k,float const*v,float*o,int seq,int d){
  // TODO: one CTA owns one or more query rows. Tile K/V through shared memory.
  // For each score tile update m_new=max(m_old,tile_max), rescale the old
  // accumulator by exp(m_old-m_new), add exp(score-m_new)*V, and normalize last.
}

int main(){constexpr int S=32,D=32;auto q=puzzle::random_vector(S*D,19,-.5f,.5f),k=puzzle::random_vector(S*D,20,-.5f,.5f),v=puzzle::random_vector(S*D,21,-.5f,.5f);std::vector<float>e(S*D,0),o=puzzle::sentinel_vector(S*D),score(S);for(int i=0;i<S;++i){float mx=-INFINITY;for(int j=0;j<S;++j){float z=0;for(int d=0;d<D;++d)z+=q[i*D+d]*k[j*D+d];score[j]=z/std::sqrt(float(D));mx=std::max(mx,score[j]);}float sum=0;for(int j=0;j<S;++j)sum+=std::exp(score[j]-mx);for(int j=0;j<S;++j)for(int d=0;d<D;++d)e[i*D+d]+=std::exp(score[j]-mx)/sum*v[j*D+d];}puzzle::DeviceBuffer<float>dq(q.size()),dk(k.size()),dv(v.size()),doo(o.size());dq.upload(q);dk.upload(k);dv.upload(v);doo.upload(o);flash_attention<<<S,128>>>(dq.get(),dk.get(),dv.get(),doo.get(),S,D);puzzle::after_launch();return puzzle::check(doo.download(),e,5e-4f,5e-4f)?0:1;}
