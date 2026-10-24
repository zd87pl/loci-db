# Benchmark Methodology

## Overview

LOCI benchmarks compare query methods against brute-force ground truth to
measure speed and recall. The checked-in script is for local iteration and
uses Qdrant's in-memory mode unless `QDRANT_URL` points to a live server.
Existing results do not establish a general speedup or customer-scale behavior.

## Comparison Methods

1. **Naive Qdrant** — Single collection, 3 independent float-range payload filters (x, y, z) + timestamp range + HNSW vector search.
2. **LOCI r4** — Historical fixed-`r4` Hilbert baseline with temporal sharding + single `MatchAny` integer pre-filter + exact payload post-filter + HNSW.
3. **LOCI r4 + overlap** — Same as above with `overlap_factor=1.2` (20% expanded spatial search region to catch boundary points) before exact post-filtering.
4. **LOCI current** — A benchmark-specific implementation of epoch-local 4D bounds, per-shard overfetch, exact payload post-filtering, temporal routing, overlap expansion, and decay-weighted re-ranking.

The `LOCI current` label does not guarantee parity with the public client: the
benchmark reimplements query logic. The public client now includes exact
coordinate/confidence ranges in Qdrant before top-k, so this copied path needs
updating and verification before its results describe the current client.
Publication/pilot runs should exercise that client directly.

For decision-grade comparisons, add a tuned indexed-range baseline and a
time-sharding-only baseline without Hilbert filtering. Tune ANN effort fairly
at equal recall and use the same ranking objective for every compared method;
decay-aware and similarity-only objectives cannot establish an equal-recall win.

## Dataset Generation

### Synthetic Dataset
- **Vectors:** D-dimensional (default 512) sampled from N(0, 1), L2-normalized.
- **Positions:** (x, y, z) sampled uniformly from [0, 1]^3.
- **Timestamps:** Uniformly distributed across a configurable time range.
- **Scene IDs:** Assigned round-robin to simulate multi-agent scenarios.
- **Sizes:** N ∈ {1,000, 10,000, 100,000} to show scaling behavior.

### Pilot Data (Required for Customer Claims)

Use representative recorded observations with verified coordinate transforms,
timestamps, embedding versions, and source provenance. Include a task set
labeled independently of LOCI's results. Synthetic coordinates and vectors do
not establish performance or usefulness on a real robot's data.

## Ground Truth Computation

Ground truth for recall@k is computed via brute-force exact search:
1. For each query vector, compute cosine similarity against ALL points in the dataset.
2. Apply exact float-range spatial and temporal filters.
3. Return top-k results sorted by similarity.
4. Recall@k = |retrieved ∩ ground_truth| / |ground_truth|, where ground truth
   contains up to k eligible records. The script scores empty ground truth as
   1.0; report empty-query frequency separately to expose its effect.

The benchmark's `LOCI current` row uses decay-aware brute-force ground truth.
Its other rows use similarity-only ground truth. For a fair head-to-head result,
align the objective and corpus cutoff across methods, report eligibility
violations separately, and verify ranking against the selected metric.

## Benchmark Scenarios

| Scenario | Description | Spatial Radius | Temporal Window | Evaluation purpose |
|---|---|---|---|---|
| A | Tight spatial query | 0.05 | full | Selective spatial filtering |
| B | Wide spatial query | 0.5 | full | Bucket expansion and fan-out cost |
| C | Combined spatial + temporal | 0.05 | tight | Combined routing and filtering |
| D | Broader spatial radius, short time window | 0.3 | 1000ms | Retrieval stress test |

This benchmark script measures the retrieval path only. Scenario D is a broader-radius, short-window query, not the `predict_and_retrieve` API.

## Synthetic World-Model Regression Harness

`benchmarks/world_model_harness.py` complements the raw retrieval benchmarks
with a deterministic demonstration. It creates a synthetic patrol episode
and a scripted predictor in `LocalLociClient`, then reports whether it can:

- retrieve a historical analog for the current state,
- retrieve the predicted future phase better than a current-state query,
- raise novelty for an out-of-distribution spatial probe,
- reconstruct the trajectory around a retrieved state.

Run it locally with:

```bash
python benchmarks/world_model_harness.py
python benchmarks/world_model_harness.py --quick
```

The JSON artifact is written to `benchmarks/results/world_model_latest.json`.
The harness is intentionally deterministic so it can be used as a regression
test. It does not evaluate a trained predictor, held-out real observations,
physical novelty detection, or customer outcomes. Evaluate those separately
with current-state retrieval, simple-predictor, and no-retrieval baselines;
keep future observations and overlapping episodes out of the retrieval corpus.

## Metrics

For each scenario and dataset size:
- **Query latency:** avg, p50, p95, p99 (ms)
- **Recall@10:** fraction of ground-truth top-10 retrieved
- **QPS estimate:** reciprocal mean serial latency; not concurrent throughput
- **Index build time:** seconds to insert all points

The current script writes latency, recall, estimated QPS, and insert-time
metrics to `benchmarks/results/latest.json`. For pilot/publication runs, also
collect concurrent throughput, resource/storage usage, collection count,
ingest/query contention, cold/warm cache behavior, and operating cost. None
of those additional measurements should be inferred from serial latency alone.

## Statistical Validity

- **Checked-in script defaults:** `n_runs=3` and `warmup_queries=5` for faster iteration.
- **Publication recommendation:** use sufficient independent repetitions and
  queries for stable tail estimates; increasing runs alone is not validation.
- **Confidence intervals:** add raw per-query/per-run output and derive intervals
  from those samples; the existing aggregate JSON does not supply them.
- **Randomization:** Queries generated with fixed seed for reproducibility.

## Hardware Specification

Benchmarks should report:
- CPU model and core count
- RAM size
- Qdrant version and configuration
- OS and kernel version
- Python version
- Dataset/model versions, index/search settings, retention, and concurrency

## Output Format

Results are saved to `benchmarks/results/latest.json` and printed as a markdown
table. Preserve an immutable artifact with code revision, backend, configuration,
raw samples, and uncertainty before citing a result; the `latest` file can be
overwritten by a later local run.

## Evidence Gate

Report workloads where LOCI wins, ties, or loses against the strongest tuned
baseline. Geometric eligibility, ANN recall, latency, operating cost, and user
task benefit are separate outcomes. Follow
[IMPLEMENTATION_READINESS.md](IMPLEMENTATION_READINESS.md) for live deployment,
real-predictor, and customer acceptance gates.
