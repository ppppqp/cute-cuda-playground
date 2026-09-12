#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) ::puzzle::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace puzzle {

inline void cuda_check(cudaError_t result, char const* expression,
                       char const* file, int line) {
  if (result != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << expression
              << ": " << cudaGetErrorString(result) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  T* get() { return data_; }
  T const* get() const { return data_; }
  void upload(std::vector<T> const& host) {
    CUDA_CHECK(cudaMemcpy(data_, host.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(count_);
    CUDA_CHECK(cudaMemcpy(host.data(), data_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return host;
  }
 private:
  T* data_ = nullptr;
  std::size_t count_;
};

inline std::vector<float> random_vector(std::size_t count, int seed = 2026,
                                        float lo = -1.0f, float hi = 1.0f) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(lo, hi);
  std::vector<float> values(count);
  std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
  return values;
}

inline std::vector<float> sentinel_vector(std::size_t count) {
  return std::vector<float>(count, std::numeric_limits<float>::quiet_NaN());
}

inline bool check(std::vector<float> const& got,
                  std::vector<float> const& expected,
                  float atol = 1.0e-4f, float rtol = 1.0e-4f) {
  int errors = 0;
  float worst = 0.0f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    float error = std::abs(got[i] - expected[i]);
    float limit = atol + rtol * std::abs(expected[i]);
    if (!std::isfinite(got[i]) || error > limit) {
      worst = std::max(worst, std::isfinite(error) ? error : std::numeric_limits<float>::infinity());
      if (errors++ < 8) {
        std::cerr << "mismatch[" << i << "]: got " << got[i]
                  << ", expected " << expected[i] << '\n';
      }
    }
  }
  if (errors) {
    std::cerr << "FAIL: " << errors << '/' << got.size()
              << " values differ; worst absolute error = " << worst << '\n';
    return false;
  }
  std::cout << "PASS: all " << got.size() << " values match\n";
  return true;
}

inline void after_launch() {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace puzzle
