# CuTe playground

This directory is a minimal CUDA/CuTe project. It uses the header-only CuTe
library shipped with NVIDIA CUTLASS v3.9.2 in `third_party/cutlass` and the
CUDA 12.8 toolkit selected by `activate-cuda` in `~/.bashrc`.
`env.sh` also selects GCC 13 because the machine's default GCC 15 is newer
than the host compilers supported by CUDA 12.8.

## Build and run

```bash
cd /home/qiping-pan/Documents/workspace/kernel_playground/cute
source env.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/hello_cute
```

The default generated GPU architecture is SM80. Set it to match your GPU when
configuring, for example `-DCMAKE_CUDA_ARCHITECTURES=89` for an Ada GPU or
`-DCMAKE_CUDA_ARCHITECTURES=90` for Hopper. To find the compute capability:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

The first example uses a compile-time CuTe layout to map CUDA thread indices
to vector offsets. A good next step is to experiment with multi-dimensional
shapes and strides in `src/hello_cute.cu`, then study the upstream examples in
`third_party/cutlass/examples/cute`.
