# Week 6 exam — 75 minutes, 20 points

1. (4) Design a minimal CKL runtime API. Mark resource owners, lifetimes, and error surfaces.
2. (4) Given live intervals `A[0,3], B[1,2], C[3,5], D[4,6]` and sizes, show a legal reuse plan and explain endpoint semantics.
3. (4) Explain why freeing or reusing memory after enqueue can be safe or unsafe. Include streams and events.
4. (4) A launch succeeds but later synchronization fails. Trace error propagation without leaking resources or misattributing the fault.
5. (4) Critique your RFC: what production requirements are omitted, and which omission would fail first under concurrent serving?

Oral check: whiteboard allocation-to-result lifecycle in seven minutes. Require a clear distinction between host, driver, and device state.

