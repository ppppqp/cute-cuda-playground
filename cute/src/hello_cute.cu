#include <cute/layout.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void check_cuda(cudaError_t result, char const* operation) {
  if (result != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
__global__ void vector_add(T const* a, T const* b, T* c, int count) {
  // A CuTe layout maps a logical coordinate to a linear offset. Both its
  // shape and stride are compile-time values here, so this abstraction costs
  // nothing in the generated kernel.
  using ThreadLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<256>{}),
      cute::make_stride(cute::Int<1>{})));

  ThreadLayout thread_layout;
  int const index = static_cast<int>(blockIdx.x) * cute::size(thread_layout) +
                    thread_layout(threadIdx.x);
  if (index < count) {
    c[index] = a[index] + b[index];
  }
}

}  // namespace

int main() {
  constexpr int kCount = 1 << 16;
  constexpr int kThreads = 256;

  std::vector<float> a(kCount);
  std::vector<float> b(kCount);
  std::vector<float> c(kCount);
  for (int i = 0; i < kCount; ++i) {
    a[i] = static_cast<float>(i) * 0.25f;
    b[i] = static_cast<float>(i % 17) - 8.0f;
  }

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  std::size_t const bytes = kCount * sizeof(float);
  check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(a)");
  check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(b)");
  check_cuda(cudaMalloc(&device_c, bytes), "cudaMalloc(c)");
  check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice),
             "copy a to device");
  check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice),
             "copy b to device");

  vector_add<<<(kCount + kThreads - 1) / kThreads, kThreads>>>(
      device_a, device_b, device_c, kCount);
  check_cuda(cudaGetLastError(), "launch vector_add");
  check_cuda(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost),
             "copy c to host");

  int errors = 0;
  for (int i = 0; i < kCount; ++i) {
    if (std::abs(c[i] - (a[i] + b[i])) > 1.0e-5f) {
      ++errors;
    }
  }

  check_cuda(cudaFree(device_a), "cudaFree(a)");
  check_cuda(cudaFree(device_b), "cudaFree(b)");
  check_cuda(cudaFree(device_c), "cudaFree(c)");

  if (errors != 0) {
    std::cerr << "FAIL: " << errors << " incorrect results\n";
    return EXIT_FAILURE;
  }
  std::cout << "PASS: CuTe vector_add produced " << kCount
            << " correct results\n";
  return EXIT_SUCCESS;
}

