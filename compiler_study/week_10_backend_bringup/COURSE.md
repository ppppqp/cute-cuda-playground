# Week 10 — Backend and model bring-up

## Outcomes

Demonstrate a disciplined correctness-first process for adding a target capability without contaminating target-independent CKL semantics.

## Reading (3 hours)

1. CKL NVIDIA dialect, extension, transforms, tests, and `NVIDIATarget` frontend/compiler path.
2. CKL AMD `MfmaLayouts.*`; identify implemented semantics versus missing lowering/runtime.
3. [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) sections on conversion targets, legality, type conversion, and reconciliation.
4. [LLVM Testing Guide](https://llvm.org/docs/TestingGuide.html): regression tests and `lit`/FileCheck principles.

In `answers.md`: define capability discovery, legality, legalization, lowering, artifact creation, and runtime availability; explain why unsupported must not mean infinite cost silently; inventory NVIDIA assumptions leaked into generic code; design differential testing for a second target.

## Build assignment (8 hours)

Choose a bounded second-target slice, preferably AMD layout alternative registration/materialization without pretending codegen exists. Alternatively add a materially new NVIDIA capability.

1. Write `bringup-plan.md`: capability matrix, first legal IR, expected diagnostics, correctness oracle, and optimization stages.
2. Add capability declaration and one legal path or dry-run lowering boundary.
3. Add explicit, tested diagnostics for unsupported chip/feature/op/layout.
4. Add positive, negative, and fallback tests. Generic CKL tests must not require the target.
5. Produce a target-independent versus target-specific boundary review.

## Acceptance tests

- Unsupported combinations fail early with actionable diagnostics.
- Capability selection is deterministic and testable without hardware where possible.
- Target-specific attributes/ops do not leak into the semantic core without written justification.
- Full CKL test suite passes; target tests skip or fail explicitly, never silently.

## Deliverables

Bring-up plan, capability matrix, patch/tests, diagnostic transcript, boundary review, and standard evidence bundle.

