# CuTe kernel interview puzzles

This directory is a progressive CUDA/CuTe exercise suite. It uses the header-only CuTe
library shipped with NVIDIA CUTLASS v4.6.1 in `third_party/cutlass` and the
CUDA 12.8 toolkit selected by `activate-cuda` in `~/.bashrc`.
`env.sh` also selects GCC 13 because the machine's default GCC 15 is newer
than the host compilers supported by CUDA 12.8.

## CuTe DSL environment

The Python DSL is installed in the isolated `.venv` using NVIDIA's CUDA 12
wheel set pinned to CUTLASS 4.6.1. Activate it, JIT-compile the smoke kernel,
launch it, and check its output with:

```bash
source dsl-env.sh
python dsl/smoke_test.py
```

The smoke kernel computes `y[i] = 2*x[i] + 1` for 1003 elements. It uses a
CuTe layout, a `@cute.kernel` device function, a `@cute.jit` host launcher,
cuda-python memory management, and a NumPy reference check.

The matching CuTe DSL puzzle series lives in `dsl/puzzles`. Run a puzzle from
the project root after activating the environment:

```bash
python dsl/puzzles/01_layout_copy.py
```

Like the C++ series, unfinished DSL kernels leave a NaN-initialized output and
fail the NumPy reference check. Each file owns its kernel and JIT launcher;
`dsl/puzzles/puzzle_utils.py` only supplies CUDA allocation and checking code.

To recreate the environment later:

```bash
./setup-dsl.sh
```

The wheel contains the DSL compiler/runtime libraries, but kernel execution
still requires a compatible NVIDIA driver and GPU.

## Build and run

```bash
cd /home/qiping-pan/Documents/workspace/kernel_playground/cute
source env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/01_layout_copy
```

Every puzzle builds before it is solved. Its output buffer starts with NaNs,
so running an unfinished puzzle prints mismatches and exits nonzero. Implement
only the `TODO` inside the kernel, rebuild that target, and make its checker pass:

```bash
cmake --build build --target 01_layout_copy
./build/01_layout_copy
```

Do not change the CPU reference or loosen tolerances while solving a puzzle.
After correctness, profile the kernel and record bandwidth/FLOP throughput.

## Curriculum

| # | Puzzle | Main ideas |
|---|---|---|
| 01 | Layout copy | static shape/stride and layout-as-function |
| 02 | 2-D add | dynamic layouts and coordinates |
| 03 | Tiled transpose | shared memory, bank conflicts, predication |
| 04 | Row reduction | strip mining and reduction trees |
| 05 | Softmax | stable max/sum reductions and fusion |
| 06 | LayerNorm | paired statistics reductions and epilogues |
| 07 | Naive GEMM | matrix coordinates and leading dimensions |
| 08 | Tiled GEMM | CTA tiling and cooperative shared-memory loads |
| 09 | CuTe MMA GEMM | MMA atoms, thread partitions, fragments |
| 10 | Conv2D | rank-4 layouts and implicit-GEMM thinking |
| 11 | Quantized GEMM | int8 packing, scales, fused dequantization |
| 12 | FlashAttention | online softmax and tiled Q/K/V dataflow |
| 13 | SM90 TMA/WGMMA | asynchronous pipeline and warp-group MMA |
| 14 | SM120 block-scaled GEMM | tcgen05, tensor memory, scale layouts |

The default generated GPU architecture is SM80. Set it to match your GPU when
configuring, for example `-DCMAKE_CUDA_ARCHITECTURES=89` for an Ada GPU or
`-DCMAKE_CUDA_ARCHITECTURES=90a` for Hopper. To find the compute capability:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Local toolkit limitation

The selected CUDA 12.8 directory has only a partial development-header set, so
some C++ CuTe headers fall back to older system CUDA headers. Layout-oriented
C++ puzzles continue to build, but the full C++ MMA/TMA header graph requires a
complete, version-matched toolkit. The Python CuTe DSL wheel is self-contained
for code generation, although executing generated kernels still requires a
supported driver.

Upstream examples worth reading alongside the series are under
`third_party/cutlass/examples/cute/tutorial`.
