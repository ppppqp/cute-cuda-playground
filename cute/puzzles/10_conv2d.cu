// Puzzle 10 — direct NHWC convolution
// Goal: compute a 3x3, stride-1, same-padding convolution. Start direct, then
// use CuTe layout composition to view it as implicit GEMM (stretch goal).
// Learn: rank-4 layouts, boundary predicates, convolution-to-GEMM lowering.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void conv2d(float const*x,float const*w,float*y,int h,int width,int ci,int co){
  // x:[H,W,Ci], w:[3,3,Ci,Co], y:[H,W,Co], all row-major.
  // TODO: assign one output element per thread, loop kh/kw/channel, skip
  // padding coordinates, and map every coordinate with a CuTe layout.
}

int main(){constexpr int H=13,W=11,CI=3,CO=5;auto x=puzzle::random_vector(H*W*CI,15),f=puzzle::random_vector(3*3*CI*CO,16);std::vector<float>e(H*W*CO,0),o=puzzle::sentinel_vector(H*W*CO);for(int h=0;h<H;++h)for(int w=0;w<W;++w)for(int co=0;co<CO;++co)for(int kh=0;kh<3;++kh)for(int kw=0;kw<3;++kw)for(int ci=0;ci<CI;++ci){int ih=h+kh-1,iw=w+kw-1;if(ih>=0&&ih<H&&iw>=0&&iw<W)e[(h*W+w)*CO+co]+=x[(ih*W+iw)*CI+ci]*f[((kh*3+kw)*CI+ci)*CO+co];}puzzle::DeviceBuffer<float>dx(x.size()),df(f.size()),dy(o.size());dx.upload(x);df.upload(f);dy.upload(o);int count=H*W*CO;conv2d<<<(count+255)/256,256>>>(dx.get(),df.get(),dy.get(),H,W,CI,CO);puzzle::after_launch();return puzzle::check(dy.download(),e,7e-4f,7e-4f)?0:1;}
