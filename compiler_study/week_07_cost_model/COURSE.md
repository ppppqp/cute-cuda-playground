# Week 7 — Cost models and graph-wide planning

## Outcomes

Turn CKL alternative selection into an inspectable optimization problem whose decisions can be tested and calibrated.

## Reading (3 hours)

1. CKL `specs/task-alternative-selection.md` in full; `ResolveTaskAlternatives.cpp`, fanout, linear-pipeline, and memory-boundary tests.
2. CKL compiler spec sections 11.3–11.4, 13.2, and 19.
3. [MLIR DataFlow Framework](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/) for analysis structure.
4. [TVM auto-scheduler paper](https://www.usenix.org/conference/osdi20/presentation/zheng) sections on cost modeling and measurement; focus on separating search space from predictor.

In `answers.md`: formalize CKL selection variables, constraints, and objective; identify additive and non-additive costs; explain why unknown is not zero; compare enumeration, DP, shortest path, ILP, and heuristic search; state how calibration avoids data leakage.

## Build assignment (9 hours)

Extend or wrap CKL's current selection with an explicit cost breakdown containing at least compute, conversion/communication, fixed launch, and unknown-confidence terms. Scope may be a linear pipeline if general DAG optimization is too large.

1. Define units and provenance for every component.
2. Add deterministic synthetic tests: local-vs-global trap, fanout, forced alternative, memory boundary, tie, unavailable option, and unknown cost.
3. Calibrate at least one component from Week 4/5 measurements; keep machine-specific data outside semantic IR or justify placement.
4. Emit decision provenance: candidates rejected, component totals, constraints, winner, and confidence.
5. Compare model prediction with measured winner on held-out cases.

## Acceptance tests

- Costs with incompatible units cannot be silently combined.
- Increasing a conversion cost never makes a plan containing only that changed edge more attractive.
- Planner output is deterministic under stable inputs.
- At least one test defeats greedy per-node selection.
- Report prediction accuracy/regret on held-out cases, not only training cases.

## Deliverables

Cost-model design, implementation/tests or standalone prototype linked to CKL IR, calibration data, prediction report, and standard evidence bundle.

