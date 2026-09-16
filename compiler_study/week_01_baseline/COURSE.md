# Week 1 — Baseline, architecture, and measurement

## Outcomes

Explain CKL end to end, reproduce its tests, and design trustworthy compiler/runtime benchmarks.

## Reading (3 hours)

1. CKL `README.md`, then `docs/composable-kernel-language.md` sections 1, 4–7, and 9. Draw the path from Python source to device artifact.
2. CKL `docs/dsl-and-dialect-scope.md` sections “Layering,” “Verification boundary,” and “Pass pipeline.”
3. [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) through “Operations”; [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/).
4. [CUDA Best Practices Guide: Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics) and [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

In `answers.md`, answer: What information is visible at each CKL layer? What is erased by NVIDIA lowering? Where can conversion cost enter selection? Name four benchmark errors. Why are warmup and device synchronization distinct requirements?

## Build assignment (7 hours)

1. Configure CKL from a clean build directory and run all tests. Record commands, versions, duration, passed/skipped counts.
2. Run `examples/nvidia_mma.py` if the MLIR/CUDA configuration supports it. Save emitted IR at each available stage; otherwise document the first missing dependency and use checked-in MLIR tests.
3. Add `study/week-1/architecture.md` containing a diagram of frontend, dialect, passes, core planner, NVIDIA extension, codegen, and missing runtime.
4. Add `study/week-1/benchmark-design.md`: define cases, shapes, warmup, repetitions, synchronization, statistics, correctness oracle, and environment capture for future weeks.
5. Pick one test and trace every called CKL component using source links and line numbers.

## Acceptance tests

- `cmake --build build` and `ctest --test-dir build --output-on-failure` pass.
- Another person can reproduce the recorded baseline from a clean tree.
- Architecture diagram distinguishes implemented paths from proposed paths.
- Benchmark plan can detect a 10% regression without changing methodology after seeing results.

## Deliverables

Standard evidence bundle plus `architecture.md`, `benchmark-design.md`, baseline logs, and a five-minute recorded or live architecture explanation.

