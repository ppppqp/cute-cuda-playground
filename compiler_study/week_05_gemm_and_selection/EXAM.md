# Week 5 exam — 75 minutes, 20 points

1. (4) Calculate FLOPs and minimum bytes for `M=128,N=4096,K=4096` FP16 GEMM. State assumptions and likely bottleneck.
2. (4) Explain how CTA tile size affects reuse, registers, occupancy, edge waste, and parallelism.
3. (4) Design CKL contracts for SIMT and tensor-core alternatives, including legality and cost provenance.
4. (4) A fused epilogue is slower. Give a measurement plan that distinguishes register pressure, lost specialization, launch savings, and bandwidth.
5. (4) Given three alternatives and two conversion edges, compute the globally cheapest two-task plan and show why local choice can lose.

Oral check: defend the Week 5 selection rule against an unseen dynamic shape. Full credit requires saying when to fall back or autotune.

