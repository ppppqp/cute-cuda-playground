# CKL Compiler Interview Course

This is a 12-week, 12–15 hour/week course tailored to Gimlet Labs' MTS — Compilers role. The flagship is [`ckl`](../../ckl/README.md); [`cute`](../cute/README.md) supplies focused GPU labs. Each week has:

- `COURSE.md`: required reading, reading checks, build assignment, acceptance tests, and deliverables.
- `EXAM.md`: a closed-notes written/oral checkpoint with rubric.

The course intentionally does not copy external publications. “Reading materials” means exact local sections and primary-source links, with questions that prove you read them.

## Rules of engagement

1. Create a branch in CKL per week: `study/week-N-topic`.
2. Put course answers and evidence in `ckl/study/week-N/`; production-worthy code goes in its natural CKL directory.
3. Before work, save `baseline.txt` with commit, toolchain, GPU, configure command, and test result.
4. A task is complete only when its acceptance commands pass and `evidence.md` links code, tests, measurements, and one limitation.
5. Time-box readings. Build and explanation quality dominate volume.
6. Do the exam closed-notes. Mark uncertain answers; verify afterward in `corrections.md`.
7. Do not merge a study change merely to check a box. A standalone experiment is acceptable when the CKL abstraction is not ready.

## Standard evidence bundle

Each `ckl/study/week-N/` submission contains:

```text
answers.md       reading-check answers
evidence.md      acceptance checklist with links and command output summary
exam-answers.md  timed exam answers
corrections.md   errors found after grading and corrected model
results.csv      when the week includes measurement
```

Benchmark CSV columns: `git_sha,device,driver,toolchain,case,shape,warmups,runs,metric,unit,median,p95,min,max`. Store raw profiler files outside Git if large and link their location.

## Definition of done

Score each week out of 100: reading checks 15, implementation 45, tests/measurement 20, exam 20. Pass at 75 with at least half credit in implementation and exam. Weeks 4, 7, 9, and 12 are gates: do not skip their failed sections.

## Schedule

| Week | Theme | CKL outcome |
|---|---|---|
| 1 | Baseline and architecture | Reproducible map of CKL and performance harness design |
| 2 | IR, SSA, verification | Stronger verifier and negative tests |
| 3 | MLIR passes and lowering | New transformation with pass-pipeline evidence |
| 4 | GPU memory | Profiled CuTe kernels and CKL layout implications |
| 5 | GEMM and selection | Measured alternative-selection policy |
| 6 | Runtime and memory | Minimal launch/runtime design or vertical slice |
| 7 | Cost models | Calibrated graph-wide selection cost model |
| 8 | Scheduling | Explicit event DAG and overlap analysis |
| 9 | Inference | Attention-shaped composed task and serving analysis |
| 10 | Backend bring-up | Second-target capability/legalization slice |
| 11 | Performance discipline | Regression harness and evidence-backed optimization |
| 12 | Interview loop | Portfolio release and five mock interviews |

## Recommended sequencing adjustments

If CKL changes while you study, preserve the learning objective and update filenames in `evidence.md`. If no compatible GPU is available, use static compiler tests plus the scheduler simulator; label modeled numbers as estimates, never measurements.

