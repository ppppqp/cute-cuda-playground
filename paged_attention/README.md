# Paged-attention NCU lab

This is a production-shaped CUDA optimization project: one decode query attends
to a non-contiguous paged KV cache with grouped-query attention (GQA). The
starter implementation is correct by construction but intentionally repeats
the Q.K dot product for every output element. Your job is to replace measured
bottlenecks, one at a time, while keeping every stage independently runnable.

## Build and smoke test

Requirements: Linux, CMake 3.24+, a CUDA toolkit, an NVIDIA GPU, and Nsight
Compute (`ncu`). Set the architecture for your GPU; common examples are 80 for
A100, 86 for RTX 30 series, 89 for RTX 40 series, and 90 for H100.

```bash
cd paged_attention
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/pa_benchmark --case tiny --kernel baseline --check
```

Do not profile until the tiny correctness check passes. Before interpreting a
timing, lock or record GPU clocks when possible and ensure no unrelated process
is using the device.

## The contract

See `include/paged_attention/types.cuh`. Q and output use
`[batch, query_head, head_dim]`. K/V use
`[physical_page, token_in_page, kv_head, head_dim]`. A logical token maps to a
physical page through `block_tables`. Query heads are evenly grouped over KV
heads. Initially use FP32 so numerical and indexing bugs are easy to isolate;
convert storage to FP16/BF16 only after `fused_v1` is correct.

The benchmark provides five deterministic workload families. Start with
`tiny`, then use `interactive` and `ragged`. The `long` case is intended for the
split-KV stage and is deliberately impractical for the starter kernel.

## Implementation stages

### Stage 0 — understand the baseline

Read `src/baseline.cu` and predict the profile before running NCU. Calculate
how many times each Q and K element is loaded and how many Q.K FMAs are executed
per output. Confirm those predictions using instruction, memory, and source
counters. Record the profile in `EXPERIMENT_LOG.md`.

### Stage 1 — `fused_v1`

Implement one CTA per `(batch, query_head)`:

1. Map token to `(physical_page, token_in_page)`.
2. Let lanes cooperatively calculate one Q.K score using warp reductions.
3. Broadcast the score instead of recalculating it per output dimension.
4. Each lane owns one or more output dimensions.
5. Maintain online-softmax state `(maximum, denominator, output[])`.
6. Normalize after the last valid token.

For a new score `s`, use:

```text
new_m = max(old_m, s)
alpha = exp(old_m - new_m)
beta  = exp(s - new_m)
new_l = alpha * old_l + beta
new_o = alpha * old_o + beta * V[token]
```

Handle the first token explicitly or ensure `exp(-inf - -inf)` cannot occur.
Check sequence lengths 1, page boundary ±1, and a non-multiple of page size.
Never assume logical pages are physically adjacent.

```bash
for n in 1 15 16 17 31 32 33; do
  ./build/pa_benchmark --case tiny --kernel fused_v1 --context "$n" --check
done
```

### Stage 2 — `vectorized_v2`

Change only access mechanics first. Try aligned 16-byte loads, then compare a
token-major cache layout with a dimension-chunked layout. Inspect sectors per
request and L1/L2/DRAM bytes. Verify pointer and stride alignment before using a
vector type. Preserve a tail path if you expand supported head dimensions.

After the FP32 access pattern is understood, add FP16 or BF16 storage with FP32
dot-product and softmax accumulation. Treat datatype conversion as a separate
experiment from layout changes.

### Stage 3 — `gqa_reuse_v3`

Map a CTA to `(batch, kv_head)` and process 2–4 query heads that share K/V.
Load each KV tile once, then reuse it. Sweep query heads per CTA. Watch register
count, spills, shared-memory use, achieved occupancy, and eligible warps; reuse
that reduces bytes can still lose if it destroys latency hiding.

### Stage 4 — `split_kv_v4`

For long contexts, split a sequence over multiple CTAs. Each split emits
`(local_max, local_sum, local_output[])`. A second kernel combines split `i`
with global maximum `m` using weights `exp(local_max[i] - m)`. Extend the API
with an explicitly sized workspace. Sweep 1, 2, 4, 8, and 16 splits and build a
dispatch threshold rather than assuming more parallelism always wins.

## Profiling workflow

First time only:

```bash
chmod +x scripts/profile.sh
ncu --list-sets
```

Run cheap triage before expensive collections:

```bash
scripts/profile.sh tiny baseline triage
scripts/profile.sh interactive fused_v1 memory
scripts/profile.sh ragged vectorized_v2 roofline
```

The script warms up ten launches, profiles one matching launch, and writes an
`.ncu-rep` under `profiles/`. Kernel-name filtering matters because NCU replay
can otherwise collect the harness and warm-up launches. Time final results with
the benchmark outside NCU; profiler replay is not a latency measurement.

Use this diagnosis order:

1. Confirm GPU duration and launch shape.
2. Use Speed of Light / roofline to form a compute-vs-memory hypothesis.
3. Inspect total DRAM/L2 traffic and memory request efficiency.
4. Check scheduler issue rate before interpreting stall reasons.
5. Inspect dominant stalls in `WarpStateStats`.
6. Correlate hot source/SASS instructions using the included `-lineinfo`.
7. Check registers, spills, shared memory, and occupancy after every tiling
   change.

Do not optimize for occupancy alone. Low occupancy is causal only when the
schedulers lack eligible warps and more residency can hide the observed
latency. Likewise, a high percentage of peak bandwidth is useful only together
with byte counts: fusion should reduce total bytes, even if bandwidth percentage
falls.

## Definition of done

- Every stage passes `--case tiny --check`.
- Boundary-length tests cover 1, 15, 16, 17, 31, 32, and 33 tokens.
- Baseline, interactive, ragged, and long results are recorded independently.
- Each optimization has a hypothesis and before/after NCU evidence.
- Final timings include median and p90 over repeated process runs.
- At least one rejected optimization is explained with profiler evidence.
- A dispatcher selects kernels based on context length, batch shape, and GQA
  ratio.
