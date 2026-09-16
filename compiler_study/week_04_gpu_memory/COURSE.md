# Week 4 — GPU execution, layouts, and memory behavior

## Outcomes

Connect CKL layout contracts to measured coalescing, bank conflicts, predication, and reduction behavior.

## Reading (3 hours)

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): programming model, SIMT, memory hierarchy, shared memory, synchronization.
2. [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalesced access, shared memory, occupancy, timing.
3. Local `cute/README.md`, then source for puzzles 01–04 before implementing them.
4. CKL `docs/composable-kernel-language.md` sections 3–4 and `Distribution.*`, `StorageLayout.*`, `ExchangeSchedule.*`.

In `answers.md`: map logical coordinate, owner, local order, and address; derive global transactions for one warp access; predict bank conflicts for a transpose tile; distinguish occupancy limiter from performance limiter; describe a CKL conversion corresponding to a shuffle versus shared-memory exchange.

## Build assignment (8 hours)

Complete or re-derive CuTe puzzles 01–04 without changing references/tolerances. For transpose and reduction, keep a baseline and optimized variant. Profile at least three shapes each.

Then add `ckl/study/week-4/layout-case-study.md` expressing one puzzle using CKL's four semantic objects. State which facts CKL can verify today, which it can plan, and which require a target cost model.

## Acceptance tests

- All four puzzle correctness checks pass, including non-multiple edge shapes.
- Results report effective bandwidth and compare against a stated hardware bound.
- A profiler or static access analysis supports each claimed cause; occupancy alone is not accepted as causation.
- Case study includes at least one explicit conversion schedule.

## Deliverables

Patch or solution references, `results.csv`, profiler screenshots/commands, `gpu_profile.md`, layout case study, and standard evidence bundle.

