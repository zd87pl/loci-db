# Benchmark Methodology

## Overview

LOCI benchmarks compare four query methods against a brute-force ground truth to measure both speed and recall. The checked-in script is tuned for fast local iteration; publication runs should keep the same method definitions but increase sample counts.

## Comparison Methods

1. **Naive Qdrant** — Single collection, 3 independent float-range payload filters (x, y, z) + timestamp range + HNSW vector search.
2. **LOCI r4** — Historical fixed-`r4` Hilbert baseline with temporal sharding + single `MatchAny` integer pre-filter + exact payload post-filter + HNSW.
3. **LOCI r4 + overlap** — Same as above with `overlap_factor=1.2` (20% expanded spatial search region to catch boundary points) before exact post-filtering.
4. **LOCI current** — Mirrors the shipped query path: epoch-local 4D bounds, per-shard overfetch, exact payload post-filtering, temporal shard routing, overlap-based Hilbert bucket expansion, and decay-weighted re-ranking before top-k truncation.

## Dataset Generation

### Synthetic Dataset
- **Vectors:** D-dimensional (default 512) sampled from N(0, 1), L2-normalized.
- **Positions:** (x, y, z) sampled uniformly from [0, 1]^3.
- **Timestamps:** Uniformly distributed across a configurable time range.
- **Scene IDs:** Assigned round-robin to simulate multi-agent scenarios.
- **Sizes:** N ∈ {1,000, 10,000, 100,000} to show scaling behavior.

### Real Robot Data (Optional)
- Trajectories from Orange Pi 5 robot demo.
- Pre-recorded (x, y, z) from SLAM + V-JEPA 2 embeddings.

## Ground Truth Computation

Ground truth for recall@k is computed via brute-force exact search:
1. For each query vector, compute cosine similarity against ALL points in the dataset.
2. Apply exact float-range spatial and temporal filters.
3. Return top-k results sorted by similarity.
4. Recall@k = |retrieved ∩ ground_truth| / k.

For the `LOCI current` row, the checked-in benchmark applies the same decay-weighted re-ranking as the shipped client before truncating to top-k, so its recall is measured against a decay-aware brute-force ranking.

## Benchmark Scenarios

| Scenario | Description | Spatial Radius | Temporal Window | Expected LOCI Advantage |
|---|---|---|---|---|
| A | Tight spatial query | 0.05 | full | Strongest — 10-20x faster |
| B | Wide spatial query | 0.5 | full | Moderate |
| C | Combined spatial + temporal | 0.05 | tight | Strongest combined case |
| D | Broader spatial radius, short time window | 0.3 | 1000ms | Retrieval stress test |

This benchmark script measures the retrieval path only. Scenario D is a broader-radius, short-window query, not the `predict_and_retrieve` API.

## Metrics

For each scenario and dataset size:
- **Query latency:** avg, p50, p95, p99 (ms)
- **Recall@10:** fraction of ground-truth top-10 retrieved
- **QPS:** queries per second (1 / avg_latency)
- **Index build time:** seconds to insert all points

The current script writes latency, recall, QPS, and insert-time metrics to `benchmarks/results/latest.json`. Memory usage and confidence intervals can be added for publication runs, but are not emitted by default in the checked-in script.

## Statistical Validity

- **Checked-in script defaults:** `n_runs=3` and `warmup_queries=5` for faster iteration.
- **Publication recommendation:** increase to `n_runs=10` and `warmup_queries=10`.
- **Confidence intervals:** derive 95% CIs from the saved per-run metrics in publication mode.
- **Randomization:** Queries generated with fixed seed for reproducibility.

## Hardware Specification

Benchmarks should report:
- CPU model and core count
- RAM size
- Qdrant version and configuration
- OS and kernel version
- Python version

## Output Format

Results are saved to `benchmarks/results/latest.json` and a markdown table is printed to stdout for direct copy-paste into README and paper.

## Comparison to ANN-Benchmarks Methodology

Our benchmark methodology follows the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) approach:
- Pre-computed ground truth from exact search.
- Recall@k as primary accuracy metric.
- QPS as primary throughput metric.
- Separation of build and query phases.

The key difference is that we measure spatiotemporal filtering performance, which ANN-Benchmarks does not cover — their focus is pure vector search without payload filters.
