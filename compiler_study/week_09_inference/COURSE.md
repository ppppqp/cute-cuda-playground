# Week 9 — Inference workloads: attention, KV state, and serving

## Outcomes

Relate CKL composition to a real inference subgraph and reason separately about prefill, decode, batching, state, and communication.

## Reading (4 hours)

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762): attention definition and shapes.
2. [FlashAttention](https://arxiv.org/abs/2205.14135): IO complexity and online softmax; [PagedAttention](https://arxiv.org/abs/2309.06180): KV memory management.
3. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling and selective batching.
4. Local CuTe puzzles 05, 06, and 12; CKL composition/alternative docs. Optional: Gimlet's public Corsair/speculative-decoding material if accessible.

In `answers.md`: derive attention tensor shapes and bytes per decode token; prove online softmax recurrence; contrast prefill/decode bottlenecks; explain paged KV benefits/costs; identify compiler-visible state and scheduling decisions.

## Build assignment (9 hours)

1. Complete/re-derive CuTe softmax and FlashAttention puzzles; LayerNorm is secondary if time-boxed.
2. Represent an attention-shaped pipeline in CKL tasks: projection or tile load, score/MMA, online softmax/reduction, value accumulation, and output. If unsupported, write valid proposed IR plus a standalone executable prototype.
3. Define at least two layout/implementation alternatives and conversion boundaries.
4. Evaluate prefill and single-token decode across at least four sequence lengths and two batch sizes, measured or modeled with labels.
5. Write a heterogeneous placement proposal and calculate when transfer cost erases accelerator advantage.

## Acceptance tests

- Numerical result agrees with a stable reference across edge shapes and extreme logits.
- Stateful KV inputs/outputs are explicit; no hidden mutation in the reasoning model.
- Report separates time per request, time per output token, throughput, and bytes moved.
- Placement break-even calculation includes fixed and size-dependent transfer cost.

## Deliverables

Correctness tests, results, CKL IR/prototype, `inference-analysis.md`, placement worksheet, and standard evidence bundle.

