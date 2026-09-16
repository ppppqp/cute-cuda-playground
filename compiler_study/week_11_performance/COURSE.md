# Week 11 — Profiling, regression control, and production evidence

## Outcomes

Find and improve one measured bottleneck in CKL or its generated workload, then prevent regression without overfitting noise.

## Reading (3 hours)

1. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) sections on the profiling model and roofline.
2. [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/) for timeline interpretation.
3. CKL compiler spec evaluation plan and current CTest structure.
4. [Google Benchmark user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md), focusing on repetitions, counters, and complexity; use concepts even if not adopting it.

In `answers.md`: distinguish microbenchmark, compiler benchmark, runtime benchmark, and end-to-end serving benchmark; list sources of variance; explain p50/p95; define practical significance; state why profiling can perturb execution.

## Build assignment (9 hours)

1. Choose one metric from compile time, selection/planning time, generated kernel time, conversion overhead, peak memory, or end-to-end latency.
2. Freeze methodology and collect at least 20 observations or justify a statistically adequate alternative.
3. Profile, form three ranked hypotheses, and run discriminating experiments.
4. Implement the smallest change supported by evidence. Record negative/neutral experiments too.
5. Add a benchmark command and machine-readable output. Propose a regression rule using historical noise; do not hard-code your best observed run.

## Acceptance tests

- Correctness suite passes before and after.
- Results include raw observations, median/p95 or suitable statistics, environment, and effect size.
- Claimed cause has counter evidence against at least one alternative hypothesis.
- Regression threshold is derived from variance and documents machine scope.
- Optimization is reverted or labeled inconclusive if evidence does not support it.

## Deliverables

Benchmark harness, raw CSV, profiles, `performance-report.md`, regression policy, patch/tests, and standard evidence bundle.

