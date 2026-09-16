# Week 3 exam — 60 minutes, 20 points

1. (4) Write pseudocode for a rewrite removing `convert_layout(A->B)` followed by `convert_layout(B->A)`. List three reasons it may be illegal or unprofitable.
2. (4) Compare dialect conversion with greedy pattern rewriting. Choose one for NVIDIA lowering and justify it.
3. (4) Given `canonicalize -> select -> plan -> materialize -> lower`, swap two passes and construct a failing example.
4. (4) Explain analysis caching and invalidation. What goes wrong if selection changes the graph but preserves a stale cost analysis?
5. (4) Defend your implementation's termination, idempotence, diagnostics, and test adequacy.

Oral check: inspect an unfamiliar 20-line MLIR snippet and narrate a pass pipeline. Full credit requires identifying side effects and multiple-use hazards.

