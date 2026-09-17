#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define PA_CUDA_CHECK(expr) ::pa::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace pa {
inline void cuda_check(cudaError_t result, char const* expression,
                       char const* file, int line) {
  if (result != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << expression << ": "
              << cudaGetErrorString(result) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t n) : size_(n) {
    PA_CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  T* get() { return data_; }
  T const* get() const { return data_; }
  void upload(T const* source) {
    PA_CUDA_CHECK(cudaMemcpy(data_, source, size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* destination) const {
    PA_CUDA_CHECK(cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
 private:
  T* data_{};
  std::size_t size_{};
};
}  // namespace pa

