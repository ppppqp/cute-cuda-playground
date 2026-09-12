// Puzzle 03 — tiled transpose through shared memory
// Goal: transpose a non-square matrix, using a 32x8 block and a padded
// shared-memory tile. Use CuTe layouts for both global and shared offsets.
// Learn: composing row/column-major layouts, bank-conflict padding, __syncthreads.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void tiled_transpose(float const* x, float* y, int rows, int cols) {
  __shared__ float tile[32][33];
  auto input = cute::make_layout(cute::make_shape(rows, cols),
                                 cute::make_stride(cols, cute::Int<1>{}));
  auto output = cute::make_layout(cute::make_shape(cols, rows),
                                  cute::make_stride(rows, cute::Int<1>{}));
  // TODO: cooperatively load a 32x32 tile in four iterations, synchronize,
  // then store it transposed. Bounds-check every global access. The [32][33]
  // padding should make both shared-memory access directions conflict-free.
}

int main() {
  constexpr int R = 93, C = 70;
  auto x = puzzle::random_vector(R*C, 3);
  std::vector<float> expected(R*C), out = puzzle::sentinel_vector(R*C);
  for (int r=0;r<R;++r) for(int c=0;c<C;++c) expected[c*R+r]=x[r*C+c];
  puzzle::DeviceBuffer<float> dx(R*C), dy(R*C); dx.upload(x); dy.upload(out);
  tiled_transpose<<<dim3((C+31)/32,(R+31)/32), dim3(32,8)>>>(dx.get(),dy.get(),R,C);
  puzzle::after_launch();
  return puzzle::check(dy.download(), expected) ? 0 : 1;
}
