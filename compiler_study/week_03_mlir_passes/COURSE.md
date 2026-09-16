# Week 3 — Rewriting, analyses, passes, and lowering

## Outcomes

Implement a nontrivial CKL transformation and explain legality, profitability, pass ordering, and analysis invalidation.

## Reading (3 hours)

1. [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) and [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/).
2. [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/) sections on operation pass restrictions, dependent dialects, analyses, and statistics.
3. CKL `ResolveTaskAlternatives.cpp`, `PlanCompositions.cpp`, `MaterializeSelectedAlternatives.cpp`, and their tests.
4. CKL `specs/task-alternative-selection.md`; write the current pipeline as pre/postconditions.

In `answers.md`: explain greedy rewrite convergence; distinguish legalization from optimization; state why mutating outside the rewriter is unsafe; give a pass-order counterexample; identify which CKL facts are analysis versus IR state.

## Build assignment (8 hours)

Implement one scoped transformation: a canonicalization for redundant conversion/composition, a pass diagnostic/statistic for selection, or a missing legalization in the existing selection pipeline. Before coding, write:

- input and output IR examples;
- legality and profitability rules;
- termination/idempotence argument;
- analyses preserved/invalidated;
- five test cases including fanout and a non-match.

Use FileCheck-style structural assertions where available. Run the pass twice and prove the second run makes no change unless the pass is deliberately iterative.

## Acceptance tests

- New focused test covers match, non-match, malformed case, fanout/multiple-use case, and idempotence.
- Existing pass-pipeline tests pass.
- `ckl-opt` output and diagnostic/statistic are captured in evidence.
- Design note explicitly separates semantic legality from estimated profitability.

## Deliverables

Implementation, tests, `pass-contract.md`, before/after IR, and standard evidence bundle.

