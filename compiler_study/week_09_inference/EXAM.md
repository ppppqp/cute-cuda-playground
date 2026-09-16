# Week 9 gate exam — 90 minutes, 20 points

1. (4) Derive stable online softmax update equations when a new score tile arrives. Explain rescaling of the accumulated output.
2. (4) Compare prefill and decode arithmetic intensity and parallelism. State how continuous batching changes both.
3. (4) Calculate KV-cache bytes for a concrete model configuration and discuss MHA, GQA, quantization, and paging.
4. (4) Partition prefill and decode across two hypothetical devices. Include transfer, cache residency, batching, and failure/fallback.
5. (4) Present the CKL attention pipeline: contracts, alternatives, conversions, missing abstractions, and measured/model evidence.

Pass condition: 15/20 with correct online-softmax and KV-cache reasoning. Oral check: adapt the design to speculative decoding and identify compiler/runtime boundaries.

