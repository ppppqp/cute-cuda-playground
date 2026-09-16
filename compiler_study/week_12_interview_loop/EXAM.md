# Week 12 final — 120 minutes plus oral defense, 20 points

## Written case (60 minutes, 10 points)

Gimlet has GPU A (high FLOPs, small memory), accelerator B (excellent decode, limited ops), and a slower interconnect than expected. Design compilation/execution for an LLM request.

1. Define IR levels and information retained at each.
2. Specify legality/capability discovery and fallback.
3. Formulate partitioning costs and constraints.
4. Represent transfers, state, scheduling, and synchronization.
5. Define correctness, performance, and rollout validation.

## CKL defense (45 minutes, 6 points)

Answer without slides first:

1. What exact problem does CKL solve, and what does it not solve?
2. Which invariant and design boundary are most important?
3. Show one graph-wide decision where local optimization loses.
4. What is measured versus modeled?
5. What would you build next with four weeks, and what evidence determines priority?
6. What would fail first at production scale?

## Reflection (15 minutes, 4 points)

Name three beliefs that changed, two persistent gaps, one negative result, and the first experiment you would run at Gimlet.

Rubric: 5 points technical correctness, 5 compiler structure, 4 systems/performance reasoning, 3 evidence discipline, 3 communication. Pass at 15/20. Any hidden data movement, missing correctness plan, or unsupported performance claim must be corrected in a second defense.
