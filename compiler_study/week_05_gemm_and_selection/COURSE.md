# Week 5 — GEMM, fusion, and alternative selection

## Outcomes

Reason from GEMM shape/layout to implementation choice and connect measured kernel behavior to CKL task alternatives.

## Reading (3 hours)

1. [CUTLASS GEMM API](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api.html) and [CuTe DSL programming model](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
2. [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
3. Local CuTe puzzles 07–09 and CKL `Extensions/NVIDIA/MmaSync.*`.
4. CKL `specs/task-alternative-selection.md`, `AlternativeProvider.*`, `ResolveTaskAlternatives.cpp`, and selection tests.

In `answers.md`: derive GEMM FLOPs and bytes for two cache assumptions; explain CTA/warp/instruction tiling; list four reasons fusion loses; distinguish task alternative enumeration, scoring, selection, and materialization; identify information missing from CKL's present cost representation.

## Build assignment (8 hours)

1. Complete/re-derive CuTe puzzles 07–09 and benchmark at least six M/N/K regimes: square, skinny-M, skinny-N, small, misaligned, and large.
2. Define two competing CKL alternatives for one task or extend the existing example. Give each explicit preconditions, layout contracts, and cost provenance.
3. Add a deterministic test showing selection changes when shape/layout/cost changes. Do not encode an unexplained magic threshold.
4. Write `selection-study.md`: measured facts, model assumptions, decision boundary, and cases the policy cannot predict.

## Acceptance tests

- Kernel correctness passes for aligned and edge sizes.
- Benchmark includes warmup, synchronization, median/p95, achieved TFLOP/s, and input layout.
- CKL test proves both alternatives can win and explains tie-breaking.
- Selection report distinguishes measured cost, modeled cost, and unknown cost.

## Deliverables

Kernel results, selection patch/tests, decision table, `selection-study.md`, and standard evidence bundle.

