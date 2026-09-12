cmake_minimum_required(VERSION 3.24)

project(cute_playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Override at configure time, for example: -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cute/tensor.hpp")
  message(FATAL_ERROR
    "CuTe headers were not found. Run: git clone --depth 1 --branch v3.9.2 "
    "https://github.com/NVIDIA/cutlass.git third_party/cutlass")
endif()

add_executable(hello_cute src/hello_cute.cu)
target_include_directories(hello_cute PRIVATE "${CUTLASS_ROOT}/include")
target_compile_options(hello_cute PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

