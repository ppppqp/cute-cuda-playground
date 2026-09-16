# Week 2 — IR, SSA, invariants, and verification

## Outcomes

Read MLIR structurally, state CKL invariants, and improve verifier behavior with negative tests.

## Reading (3 hours)

1. [MLIR LangRef](https://mlir.llvm.org/docs/LangRef/): SSA values, blocks, regions, dominance, traits, and symbol tables.
2. [MLIR Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/) and [Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/).
3. CKL `docs/dsl-and-dialect-scope.md` attributes/types/operations and verification boundary.
4. Read `CKLOps.td`, `CKLOps.cpp`, `CKLTypes.td`, and `tests/Dialect/invalid.mlir` side by side.

In `answers.md`: list invariants for `ckl.task`, `ckl.invoke`, and layout conversion; distinguish parse errors from verification errors; explain dominance in nested regions; identify three invariants currently enforced and two missing or weakly diagnosed.

## Build assignment (7 hours)

Choose one real verifier gap after inspection. Add or improve verification for it in CKL. The change must:

1. Produce a diagnostic naming the operation, violated invariant, expected value, and observed value where useful.
2. Add at least three negative cases and one positive boundary case.
3. Preserve round-trip behavior for valid IR.
4. Add a comment or documentation stating why the invariant belongs at that layer.

Preferred targets: task signature/region agreement, invocation operand/result contracts, invalid distribution/index-map combinations, or illegal memory-space transitions. Do not add a redundant check already guaranteed by an MLIR trait.

## Acceptance tests

- Relevant focused test fails before and passes after the implementation.
- Full `ctest --test-dir build --output-on-failure` passes.
- Diagnostics are asserted, not merely nonzero exit status.
- `evidence.md` includes one rejected alternative and why it was wrong-layer or redundant.

## Deliverables

Code/tests, `invariants.md` with a table of owner and enforcement stage, before/after diagnostics, and the standard evidence bundle.

