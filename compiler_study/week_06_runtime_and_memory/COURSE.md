# Week 6 — Runtime interfaces, asynchronous execution, and memory

## Outcomes

Design the missing CKL host-runtime boundary and implement a small, testable vertical slice without prematurely building a full runtime.

## Reading (3 hours)

1. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/): device management, memory, streams, events, launch, error handling.
2. [MLIR Execution Engine](https://mlir.llvm.org/docs/ExecutionEngine/) and [GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) host/device boundaries.
3. CKL `README.md` “DSL and compiler,” `python/ckl/compiler.py`, `examples/nvidia_mma.py`, and `NvidiaRuntimeValidation.cpp`.
4. Review a primary allocator design: [PyTorch CUDA memory management notes](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).

In `answers.md`: specify ownership for module/context/stream/event/buffer; distinguish launch completion from execution completion; explain deferred errors; derive live ranges for a four-op graph; list hazards of reusing a buffer across streams.

## Build assignment (8 hours)

First write `runtime-rfc.md` with a C or Python-facing API, ownership, error model, synchronization semantics, artifact loading, argument ABI, and deliberate non-goals. Then implement one vertical slice:

- preferred: load CKL's NVIDIA artifact, allocate/copy/launch/synchronize/copy-back for the existing MMA example;
- fallback: a mock backend conforming to the same interface, with deterministic async event and failure tests.

Also implement standalone interval-based buffer reuse or a memory-plan analysis for a small event graph. Prove peak-byte calculation with unit tests.

## Acceptance tests

- Correct result is compared with a CPU/reference oracle.
- Launch/runtime failures propagate with operation and backend context.
- Tests cover zero-size, bad argument/shape, asynchronous completion, and one reuse hazard.
- Memory planner reports live intervals, assignment, and peak bytes; one test demonstrates safe reuse.
- RFC separates compile-time artifact production from runtime ownership.

## Deliverables

RFC, vertical-slice code/tests, memory-plan example, timeline, and standard evidence bundle. If fallback used, record exactly what blocks real launch.

