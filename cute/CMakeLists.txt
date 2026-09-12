cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build")
endif()

project(cute_playground LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Override at configure time, for example: -DCMAKE_CUDA_ARCHITECTURES=90
set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cute/tensor.hpp")
  message(FATAL_ERROR
    "CuTe headers were not found. Run: git clone --depth 1 --branch v4.6.1 "
    "https://github.com/NVIDIA/cutlass.git third_party/cutlass")
endif()

function(add_cute_exercise source)
  get_filename_component(target "${source}" NAME_WE)
  add_executable("${target}" "${source}")
  target_include_directories("${target}" PRIVATE
    "${CUTLASS_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_options("${target}" PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endfunction()

add_cute_exercise(src/hello_cute.cu)

file(GLOB PUZZLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/puzzles/[0-9][0-9]_*.cu")
foreach(source IN LISTS PUZZLE_SOURCES)
  add_cute_exercise("${source}")
endforeach()
