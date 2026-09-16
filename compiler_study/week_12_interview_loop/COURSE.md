# Week 12 — Portfolio release and interview loop

## Outcomes

Turn CKL work into a concise technical narrative and demonstrate coding, compiler design, ML systems design, performance debugging, and behavioral ownership.

## Reading (2 hours)

1. Re-read the [Gimlet role](https://jobs.ashbyhq.com/gimlet/ac6c4998-d6e7-429d-a889-930376bcc9f2). Map every requirement to evidence or an honest gap.
2. Re-read CKL README and your Weeks 1, 7, 9, and 11 reports. Remove claims unsupported by code or measurement.
3. Review your exam corrections. Build a one-page “mistake taxonomy,” not a memorized answer sheet.

In `answers.md`: write a 30-second role fit, two-minute CKL pitch, ten-minute deep-dive outline, three strongest evidence points, three limitations, and why CKL matters to heterogeneous inference.

## Build assignment (7 hours)

Create a portfolio-quality CKL release candidate:

- clean build/test instructions and tested environment;
- architecture diagram showing semantic core, dialect, selection/planning, target lowering, and runtime status;
- one end-to-end demo with expected output;
- benchmark table with methodology and limitations;
- design decision record for the hardest tradeoff;
- roadmap separating implemented, experimental, and proposed work;
- a tagged commit or immutable commit hash used in the presentation.

Do a fresh-clone rehearsal in a temporary directory. Do not rewrite unrelated CKL history or publish without reviewing secrets/licensing.

## Mock loop (6 hours)

Run on separate days or with breaks. Record scores and corrections.

1. Coding, 45 min: DAG scheduling, intervals, cache, or graph rewrite.
2. Compiler design, 60 min: IR, pass pipeline, lowering, verification, diagnostics.
3. ML systems, 60 min: heterogeneous LLM inference partitioning.
4. Performance, 45 min: trace/counter diagnosis and experiment design.
5. Project deep dive, 45 min: CKL architecture, evidence, tradeoffs, failure modes.
6. Behavioral, 30 min: ambiguity, disagreement, failure, ownership, fast learning.

## Acceptance tests

- Fresh clone reaches the documented demo without undocumented manual repair.
- Every benchmark claim links raw data and exact commit.
- Ten-minute talk finishes in 9–11 minutes and survives 15 minutes of questions.
- Each mock has written feedback, corrections, and one repeated question showing improvement.
- Role matrix has no inflated claim; gaps have a concrete learning story.

## Deliverables

Release commit, portfolio README/docs, demo transcript, role matrix, talk outline/slides, mock scorecards, six STAR stories, and standard evidence bundle.

