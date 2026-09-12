// Puzzle 02 — shape, stride, and coordinate mapping
// Goal: add two MxN row-major matrices. Derive (row,col) from the CUDA launch,
// then use a dynamic CuTe layout to map that coordinate to memory.
// Learn: make_shape, make_stride, make_coord, runtime layouts.
#include <cute/layout.hpp>
#include "puzzle_utils.cuh"

__global__ void matrix_add(float const* a, float const* b, float* c, int m, int n) {
  auto matrix = cute::make_layout(cute::make_shape(m, n),
                                  cute::make_stride(n, cute::Int<1>{}));
  int row = int(blockIdx.y * blockDim.y + threadIdx.y);
  int col = int(blockIdx.x * blockDim.x + threadIdx.x);
  // TODO: bounds-check (row,col), compute `matrix(cute::make_coord(row,col))`,
  // and store a[offset] + b[offset].
}

int main() {
  constexpr int M = 67, N = 131;
  auto a = puzzle::random_vector(M * N, 1);
  auto b = puzzle::random_vector(M * N, 2);
  std::vector<float> expected(M * N), out = puzzle::sentinel_vector(M * N);
  for (int i = 0; i < M * N; ++i) expected[i] = a[i] + b[i];
  puzzle::DeviceBuffer<float> da(M*N), db(M*N), dc(M*N);
  da.upload(a); db.upload(b); dc.upload(out);
  dim3 block(16, 16), grid((N+15)/16, (M+15)/16);
  matrix_add<<<grid, block>>>(da.get(), db.get(), dc.get(), M, N);
  puzzle::after_launch();
  return puzzle::check(dc.download(), expected) ? 0 : 1;
}
