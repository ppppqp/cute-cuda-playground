# Week 8 — Event DAGs, scheduling, and overlap

## Outcomes

Represent execution separately from placement, schedule compute/conversion events under resource constraints, and quantify exposed communication.

## Reading (3 hours)

1. CKL compiler spec section 13 and `ExchangeSchedule.*`; inspect scheduling-related dialect tests.
2. [MLIR async dialect](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) and [GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) async dependencies.
3. [CUDA Programming Guide: Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).
4. Read one pipeline reference: [GPipe](https://arxiv.org/abs/1811.06965), focusing on bubbles and microbatch scheduling.

In `answers.md`: distinguish task DAG, event DAG, resource schedule, and runtime trace; derive critical path; state conditions for legal overlap; explain double buffering; describe how overlap trades latency/throughput for memory.

## Build assignment (8 hours)

Implement a small event-DAG scheduler, either in CKL if the abstraction fits or under `ckl/study/week-8/` consuming exported planner data.

- Events: compute, conversion/copy, barrier/event.
- Resources: at least two compute devices/queues and one transfer resource.
- Inputs: duration estimate, dependencies, resource eligibility.
- Outputs: start/end, resource, critical path, makespan, utilization, exposed transfer.

Implement a serial baseline and a dependency-aware list schedule. Add timeline export (CSV or Chrome Trace). Test cycle detection, deterministic tie-breaking, resource serialization, fork/join, and double-buffer capacity.

## Acceptance tests

- Every event starts after dependencies finish and never overlaps illegally on exclusive resources.
- Critical path and makespan match hand calculations on three small graphs.
- One scenario overlaps transfer/compute and improves makespan; another cannot overlap and explains why.
- Timeline can be inspected without reading scheduler code.

## Deliverables

Scheduler/tests, two timelines, hand-worked validation, limitations, and standard evidence bundle.

