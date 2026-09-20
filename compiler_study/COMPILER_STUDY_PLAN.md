# 12-Week Study Plan: Compilers

## Goal

Prepare to discuss and demonstrate how an ML workload moves from a framework graph through intermediate representations, optimization and lowering into a runtime that schedules kernels and data movement across heterogeneous devices.

By the end of 12 weeks, produce one coherent portfolio project: a small heterogeneous ML compiler/executor with documented IR passes, cost-based partitioning, memory planning, runtime execution, correctness tests, and performance measurements. The project matters more than completing every reading.

This plan assumes **12–15 focused hours per week**. For an 8-hour week, keep the project and interview-practice blocks and reduce reading. For a 20-hour week, add implementation depth rather than more resources.

## What the job description is testing

| Role requirement | Evidence to build |
|---|---|
| IR transformations, passes, lowering, code generation | A graph IR with shape inference, constant folding, fusion, legalization, and target lowering |
| Execution planning across heterogeneous hardware | A cost-based graph partitioner with explicit transfers and a written explanation of tradeoffs |
| Runtime interfaces and kernel orchestration | An executor with device buffers, dispatch, synchronization, and trace output |
| Memory systems and hardware efficiency | A liveness-based memory planner and roofline/profile analysis |
| ML inference and serving performance | Benchmarks for a transformer-like subgraph, including batching and communication overhead |
| GPU/kernel knowledge | Correct and profiled CUDA/CuTe kernels from this repository |
| Strong C++ and/or Python | Python for rapid compiler work; C++/CUDA for systems and kernel work |
| Production engineering judgment | Tests, reproducible benchmarks, clear failure modes, design docs, and performance regression checks |

## Weekly cadence

Use roughly the same cadence each week:

- **2 hours — concepts:** Read selectively and write brief notes from memory.
- **6–8 hours — implementation:** Build the week's vertical slice.
- **2 hours — experiments:** Profile, benchmark, and explain results.
- **1–2 hours — interview practice:** One coding problem and one technical explanation.
- **1 hour — artifact:** Commit a concise README/design note and record next steps.

For every benchmark, record hardware, software versions, input shapes, warmup, repetitions, median and tail latency, and synchronization points. Never report GPU timing measured only with an unsynchronized host clock.

## Phase 1 — Foundations and measurement

### Week 1: Baseline, inference anatomy, and performance reasoning

**Learn**

- Trace a transformer inference request: graph capture, compilation, allocation, dispatch, KV cache, sampling, and response.
- Review latency versus throughput, arithmetic intensity, bandwidth, FLOPs, occupancy, launch overhead, and Amdahl's law.
- Refresh transformer operations: GEMM, normalization, attention, RoPE, KV cache, and quantization.

**Build**

- Create a small PyTorch benchmark containing `Linear -> activation -> Linear`, normalization, and attention-shaped workloads.
- Add a reliable benchmark harness with warmup, synchronization, percentiles, and correctness checks.
- Capture/probe the graph with `torch.export` or `torch.fx`; print shapes, dtypes, and users of every value.

**Deliverable**

- `docs/baseline.md`: workload breakdown, likely bottlenecks, and a roofline-style estimate for at least two operators.
- A baseline table with eager latency, operator time, memory use, and shapes.

**Interview checkpoint**

- Explain prefill versus decode, why decode is commonly bandwidth-bound, and how batching changes the bottleneck.

### Week 2: Compiler fundamentals through a tiny graph IR

**Learn**

- SSA, dominance, basic blocks, dataflow analysis, pattern rewriting, canonicalization, common subexpression elimination, dead-code elimination, and pass invariants.
- Distinguish graph-level IR, loop/tensor IR, and target/kernel IR.

**Build**

- Define a typed SSA-like graph IR in Python: operations, values, users, tensor shape/dtype/device, and verification.
- Import a small graph from Week 1 or construct one programmatically.
- Implement constant folding, DCE, algebraic simplification, and shape propagation.
- Add before/after textual IR dumps and unit tests, including malformed IR tests.

**Deliverable**

- A pass pipeline with tests and a short statement of each pass's preconditions, guarantees, and complexity.

**Interview checkpoint**

- Implement a local rewrite and explain why pass ordering can change legality, profitability, and compile time.

### Week 3: MLIR/LLVM orientation and lowering

**Learn**

- MLIR operations, regions, blocks, dialects, traits, interfaces, declarative rewriting, conversion targets, and partial/full dialect conversion.
- Walk through a representative path such as StableHLO/linalg to loops or GPU to LLVM. Focus on why multiple abstraction levels exist.

**Build**

- Complete one small MLIR tutorial exercise or standalone pass.
- Add two levels to the project IR: a high-level tensor graph and a lower-level scheduled form with explicit loops or kernel calls.
- Write a legalization/lowering pass and reject unsupported operations with useful diagnostics.

**Deliverable**

- `docs/lowering.md`: one operation shown before and after lowering, preserved semantics, lost information, and remaining optimization opportunities.

**Interview checkpoint**

- Whiteboard how an attention operation could lower through progressively more hardware-specific IRs.

## Phase 2 — Kernels, memory, and runtime

### Week 4: GPU execution and memory hierarchy

**Learn**

- Warps, CTAs, occupancy, coalescing, shared-memory bank conflicts, registers, divergence, asynchronous copies, and synchronization.
- Learn the profiler workflow: first identify the limiting resource, then change one hypothesis at a time.

**Build**

- Complete CuTe puzzles 01–04 in [`cute/`](cute/README.md): layout copy, 2-D layout, transpose, and reduction.
- Benchmark naive and optimized transpose/reduction variants.
- Use Nsight Systems for the execution timeline and Nsight Compute for kernel metrics if available.

**Deliverable**

- `docs/gpu_profile_1.md`: hypotheses, measurements, achieved bandwidth, and why each optimization helped or failed.

**Interview checkpoint**

- Diagnose an uncoalesced load, a bank conflict, low occupancy, and launch-bound execution from symptoms.

### Week 5: GEMM, fusion, and kernel selection

**Learn**

- GEMM tiling across CTA/warp/instruction levels, layouts, tensor cores, epilogues, fusion benefits and costs, and shape-dependent kernel selection.
- Understand why fewer kernels is not automatically faster: register pressure, occupancy, recomputation, and lost library specialization matter.

**Build**

- Complete CuTe puzzles 07–09: naive GEMM, tiled GEMM, and CuTe MMA GEMM.
- Add a compiler fusion pass for a small pattern such as matmul + bias + activation.
- Lower fused and unfused forms to callable implementations; allow a library fallback.

**Deliverable**

- A benchmark over several M/N/K shapes and a kernel-selection rule justified by data.

**Interview checkpoint**

- Derive GEMM arithmetic intensity and explain how tile size affects reuse, occupancy, and edge handling.

### Week 6: Runtime, dispatch, and memory planning

**Learn**

- Device APIs, streams/queues, events, asynchronous execution, allocators, buffer lifetime, memory pools, and dependency scheduling.
- Review fragmentation, peak live memory, host/device transfers, and synchronization hazards.

**Build**

- Implement a runtime interface: allocate, copy, launch/dispatch, synchronize, and free/reuse.
- Add liveness analysis and reuse buffers whose live ranges do not overlap.
- Produce a Chrome-trace-compatible event log or simple execution timeline.
- Test dependency ordering, asynchronous completion, aliasing rules, and error propagation.

**Deliverable**

- Compare naive allocation with planned reuse: allocation count, peak bytes, and latency.

**Interview checkpoint**

- Design a safe caching allocator and discuss stream ordering, fragmentation, OOM recovery, and observability.

## Phase 3 — Heterogeneous execution and inference optimization

### Week 7: Cost models and graph partitioning

**Learn**

- Graph partitioning, placement constraints, compute/communication tradeoffs, critical paths, pipeline parallelism, and dynamic programming versus heuristics.
- Separate operation support, placement legality, and placement profitability.

**Build**

- Model two devices with different supported ops, compute rates, memory capacities, and transfer bandwidth/latency.
- Implement an initial greedy partitioner, then improve it with dynamic programming for a linear graph or a local-search heuristic for a DAG.
- Insert explicit send/receive or copy operations at partition boundaries.

**Deliverable**

- Unit tests where the obvious per-op fastest placement is globally worse because of transfer cost.
- `docs/partitioning.md`: objective function, constraints, algorithm, complexity, and failure cases.

**Interview checkpoint**

- Solve a small placement problem on a whiteboard and explain how profiling data updates the cost model.

### Week 8: Scheduling and communication overlap

**Learn**

- List scheduling, resource constraints, dependency DAGs, double buffering, pipelining, collective communication basics, and overlap limits.
- Reason about throughput from pipeline stage balance and microbatch count.

**Build**

- Turn a partitioned graph into a scheduled event DAG containing compute, transfer, and synchronization events.
- Simulate or execute multiple streams/devices; compare sequential execution with overlapped transfer and compute.
- Detect cycles and resource conflicts; generate a timeline.

**Deliverable**

- A timeline and table showing critical path, device utilization, exposed communication, and throughput under multiple schedules.

**Interview checkpoint**

- Explain when communication can be hidden and why more overlap may increase memory pressure.

### Week 9: Attention, KV cache, and serving behavior

**Learn**

- Online softmax/FlashAttention, paged KV cache, continuous batching, prefix reuse, speculative decoding, and quantized KV caches.
- Read Gimlet's referenced Corsair/speculative-decoding work if publicly accessible; focus on system/compiler implications.

**Build**

- Complete CuTe puzzles 05, 06, and 12: softmax, LayerNorm, and FlashAttention. If time is limited, prioritize softmax and FlashAttention.
- Add an attention-shaped graph to the compiler and represent KV-cache state explicitly.
- Benchmark different sequence lengths, batch sizes, prefill/decode phases, and partition points.

**Deliverable**

- `docs/inference.md`: latency/throughput tradeoffs, bytes moved per token, and one proposed compiler optimization.

**Interview checkpoint**

- Explain online softmax correctness and choose placements for prefill and decode on two hypothetical accelerators.

## Phase 4 — Production quality and interview readiness

### Week 10: New backend/model bring-up

**Learn**

- Backend capability descriptions, target legalization, fallback paths, numerical tolerance, feature detection, and differential testing.

**Build**

- Add a third simulated backend or substantially different target profile.
- Bring up an unfamiliar graph end to end: import, verify, legalize, partition, lower, execute, and compare against a reference.
- Add randomized/property tests for shapes and dtypes, plus graceful unsupported-op diagnostics.

**Deliverable**

- A bring-up checklist and a debugging narrative for one real correctness or performance failure.

**Interview checkpoint**

- Answer: “A new accelerator arrived. How would you get the first model running correctly, then make it fast?”

### Week 11: Performance engineering and regression discipline

**Learn**

- Benchmark design, variance, tail latency, profiling perturbation, regression thresholds, counters, tracing, and end-to-end versus microbenchmark attribution.

**Build**

- Profile the complete project and optimize the largest measured bottleneck.
- Add a reproducible benchmark suite and machine-readable results.
- Add performance regression checks with thresholds wide enough to avoid noise but narrow enough to catch meaningful regressions.
- Measure compile time, execution time, peak memory, transfer bytes, and device utilization.

**Deliverable**

- `docs/performance_report.md`: baseline, profile evidence, change, result, remaining bottleneck, and any negative results.

**Interview checkpoint**

- Practice a 30-minute performance-debugging session: clarify metric, form hypotheses, select measurements, interpret, iterate.

### Week 12: Package the story and simulate the interview

**Build and polish**

- Make the project runnable from a clean environment with one documented command.
- Finalize architecture diagrams, tests, benchmark methodology, known limitations, and future work.
- Prepare a 10-minute project walkthrough and a 2-minute version.
- Prepare six STAR stories: hard bug, performance win, ambiguous project, tradeoff, disagreement, and failure/learning.

**Mock interview loop**

1. **Coding (45 min):** graph traversal, interval/liveness problem, or cache/scheduler implementation.
2. **Compiler design (60 min):** IR, pass pipeline, legality, lowering, diagnostics, and testing.
3. **ML systems design (60 min):** partition an LLM inference workload across heterogeneous devices.
4. **Performance (45 min):** interpret a trace/profile and propose the next experiment.
5. **Project deep dive (45 min):** defend measurements and design choices; discuss what breaks at production scale.

**Final deliverable**

- A public-quality repository or sanitized code sample with a crisp README, architecture, demo, benchmark table, test instructions, and honest limitations.

## Portfolio project specification

Keep the scope narrow enough to finish. A good minimum is:

```text
PyTorch/hand-built graph
        |
        v
 Typed tensor graph IR
        |
 shape inference -> canonicalization -> fusion -> legalization
        |
 cost-based heterogeneous partitioning + explicit transfers
        |
 kernel/library selection -> memory planning -> event scheduling
        |
 simulated devices and/or CPU + CUDA runtime
        |
 correctness comparison + trace + benchmark report
```

Required qualities:

- Correctness is checked against a trusted reference.
- Every pass verifies its input/output invariants.
- Costs include compute, communication, launch overhead, and memory capacity.
- The runtime exposes synchronization rather than hiding it accidentally.
- At least one optimization is justified with profile data.
- At least one negative result is documented; mature performance work includes ideas that did not win.

Avoid building a parser, a full framework frontend, or many operators. Those expand scope without demonstrating the role's most important skills.

## Ongoing interview drills

Rotate through these prompts during the weekly interview block:

- Topologically sort a graph; detect a cycle; compute liveness and peak memory.
- Fuse graph patterns while respecting multiple users and side effects.
- Partition a weighted DAG across two devices with transfer costs.
- Design a pass manager with analyses, invalidation, verification, and diagnostics.
- Explain SSA, dominance, canonicalization, legalization, and dialect conversion.
- Design kernel dispatch for dynamic shapes and a cache for compiled variants.
- Diagnose why an individually faster kernel makes the end-to-end graph slower.
- Compare tensor, pipeline, and expert parallelism for inference.
- Discuss allocator correctness under asynchronous streams.
- Explain how compiler decisions affect p50/p99 latency, throughput, and cost per token.

Continue ordinary data-structure practice, but bias toward graphs, intervals, queues, caches, concurrency, and clean C++/Python implementation rather than obscure puzzles.

## Read selectively

Use primary documentation and source code; do not try to read everything front to back.

- MLIR documentation: language reference, pass infrastructure, pattern rewriting, dialect conversion, GPU and LLVM lowering.
- LLVM Kaleidoscope tutorial for a compact end-to-end compiler path.
- PyTorch `torch.export`, FX, dispatcher, and custom operator documentation.
- OpenXLA/StableHLO documentation for portable ML IR and shape semantics.
- Triton tutorials for program models, fused softmax, matmul, and profiling.
- CUDA C++ Programming Guide and Best Practices Guide for execution and memory behavior.
- CUTLASS/CuTe documentation and the puzzle suite already in this repository.
- FlashAttention and PagedAttention papers for memory-aware inference design.
- TVM or XLA architecture/code for examples of scheduling, lowering, and runtime boundaries.

When reading a system, always answer: What is the IR? What invariants hold? Which choices are deferred? What is the cost model? Where are synchronization and data movement represented? How is correctness tested? How is performance measured?

## Weekly scorecard

At the end of each week, score 0–2 for each item:

- **Concept:** Can I explain it without notes?
- **Implementation:** Does the code work on unseen cases?
- **Measurement:** Do I have trustworthy numbers and a causal explanation?
- **Communication:** Can I explain the design and tradeoffs in under five minutes?
- **Artifact:** Is there a test, benchmark, commit, or document I could show?

A score below 7/10 means carry the missing deliverable into the next week. Do not repeat the entire week.

## Readiness checklist

Apply and interview when most of these are true—not only when all are perfect:

- [ ] I can trace an ML graph from framework capture to kernel dispatch.
- [ ] I can design and test an IR transformation and state its invariants.
- [ ] I can explain progressive lowering and where target-specific decisions belong.
- [ ] I can partition a graph using compute, communication, and memory costs.
- [ ] I can reason about asynchronous runtime execution and buffer lifetimes.
- [ ] I can profile GPU execution and distinguish compute-, bandwidth-, launch-, and communication-bound behavior.
- [ ] I understand attention, KV-cache behavior, batching, and inference latency/throughput tradeoffs.
- [ ] I have one measured end-to-end project and two deep technical stories.
- [ ] I can write clear, tested C++ or Python under interview time constraints.
- [ ] I can state what I do not know and describe the experiment I would run next.

## First session

1. Create a `study-log.md` with weekly scores and benchmark links.
2. Reserve recurring study blocks for all 12 weeks.
3. Run the existing CuTe smoke test and one puzzle to validate the environment.
4. Implement the Week 1 benchmark harness before collecting any baseline numbers.
5. Schedule the Week 6 and Week 12 mock interviews now.

