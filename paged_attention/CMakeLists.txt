cmake_minimum_required(VERSION 3.24)

project(paged_attention_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build")
endif()

add_library(paged_attention STATIC
  src/baseline.cu
  src/fused_v1.cu
  src/vectorized_v2.cu
  src/gqa_reuse_v3.cu
  src/split_kv_v4.cu)
target_include_directories(paged_attention PUBLIC include)
target_compile_options(paged_attention PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)

add_executable(pa_benchmark benchmarks/benchmark.cu)
target_link_libraries(pa_benchmark PRIVATE paged_attention)
target_compile_options(pa_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

enable_testing()
add_test(NAME correctness_smoke
  COMMAND pa_benchmark --case tiny --kernel baseline --check --iterations 3)

