# Benchmark Methodology

## Overview

LOCI benchmarks compare four query methods against a brute-force ground truth to measure both speed and recall. The checked-in script is tuned for fast local iteration; publication runs should keep the same method definitions but increase sample counts.

## Comparison Methods

1. **Naive Qdrant** — Single collection, 3 independent float-range payload filters (x, y, z) + timestamp range + HNSW vector search.
2. **LOCI r4** — Historical fixed-`r4` Hilbert baseline: single `MatchAny` integer pre-filter + timestamp range filter + exact payload post-filter + HNSW.
3. **LOCI r4 + overlap** — Same as above with `overlap_factor=1.2` (20% expanded spatial search region to catch boundary points) before exact post-filtering.
4. **LOCI current** — Mirrors the shipped query path: indexed timestamp-range filtering, epoch-local 4D Hilbert bounds for single-epoch windows (spatial-only cover otherwise), candidate overfetch, exact payload post-filtering, overlap-based Hilbert bucket expansion, and decay-weighted re-ranking before top-k truncation.

## Dataset Generation

### Synthetic Dataset
- **Vectors:** D-dimensional (default 512) sampled from N(0, 1), L2-normalized.
- **Positions:** (x, y, z) sampled uniformly from [0, 1]^3.
- **Timestamps:** Uniformly distributed across a configurable time range.
- **Scene IDs:** Assigned round-robin to simulate multi-agent scenarios.
- **Sizes:** N ∈ {1,000, 10,000} in the checked-in config (`benchmarks/results/latest.json` contains only these sizes; larger runs can be enabled for publication).

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

Measured results below are p50 latencies from `benchmarks/results/latest.json`
(512-dim, live Qdrant, N=1,000 and N=10,000; no larger run has been recorded):

| Scenario | Description | Spatial Radius | Temporal Window | Measured result (latest.json) |
|---|---|---|---|---|
| A | Tight spatial query | 0.05 | full | Naive Qdrant faster: LOCI 4.9x slower at N=1k (20.5ms vs 4.2ms), 3.7x slower at N=10k (162ms vs 44ms) |
| B | Wide spatial query | 0.5 | full | Naive Qdrant much faster: LOCI 23x slower at N=1k, 16x slower at N=10k |
| C | Combined spatial + temporal | 0.05 | tight | LOCI's best case: ~2x faster at N=10k (22.6ms vs 44.9ms, recall@10 0.99); still 2.1x slower at N=1k |
| D | Broader spatial radius, short time window | 0.3 | 1000ms | Naive Qdrant faster: LOCI 7.9x slower at N=1k, 1.8x slower at N=10k |

LOCI's relative performance improves with dataset size in every scenario, but
the only measured win is the combined tight spatial + temporal case (C) at
N=10,000 — where indexed temporal filtering and the Hilbert pre-filter compose.
Pure spatial filtering is not currently a speedup over naive float-range
filters; see `docs/NOVELTY.md` for the honest framing of the value proposition.

This benchmark script measures the retrieval path only. Scenario D is a broader-radius, short-window query, not the `predict_and_retrieve` API.

## World-Model Proof Harness

`benchmarks/world_model_harness.py` complements the raw retrieval benchmarks
with a deterministic product-level proof. It creates a closed-loop patrol
episode in `LocalLociClient` and reports whether LOCI can:

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
test for the concept, not only as a performance benchmark.

## World-model memory benchmark (wm_bench)

`benchmarks/wm_bench/` is the v1 implementation of RFC-0001 R3: a runnable
tasks × systems benchmark suite for "memory for world models" that anyone can
point at their own store. It grows the proof harness above into a matrix of
five tasks over five systems with a brute-force ground truth.

### Synthetic-data caveat (read this first)

**All v1 results are computed on synthetic trajectories.** The environment
this v1 was built in cannot download the embodied datasets named in the RFC
(Habitat/AI2-THOR rollouts, TartanAir, nuScenes), so v1 ships seeded
synthetic generators instead:

- **Smooth patrol** — a momentum random walk in the unit cube; embeddings
  drift along a fixed low-dimensional random manifold plus Gaussian noise.
- **Multi-scene warehouse** — scenes with fixed latent anchors, revisited in
  rotation, so re-encountered scenes yield re-encountered embeddings.
- **OOD injection** — contiguous excursion segments with positions in a novel
  spatial corner and embeddings constructed near-orthogonal to the base
  manifold, labelled `is_ood=True`.

Documented realism limitations (see the `benchmarks/wm_bench/datasets.py`
docstring for the full list): embeddings are not from a trained world model,
positions are not SLAM output, and the constructed orthogonality makes OOD
detection intentionally easy — read the novelty AUC as an upper bound on
separability, not a field result. Every artifact and report row is labelled
synthetic.

### Tasks

1. **Future-state analog recall@k** — queries are the true embedding
   `horizon` steps ahead plus noise (an imperfect "world-model prediction");
   recall@k is measured against the brute-force oracle's top-k for the
   identical query, unfiltered and time-windowed. Note: every in-memory
   engine here searches exactly, so differences appear only under
   consolidation or filtering — this task is primarily a correctness gate
   until real ANN-backed deployments run it.
2. **Novelty AUC over OOD segments** — streaming score-before-insert: each
   point is scored with `predict_novelty` against history that excludes it,
   then inserted. ROC-AUC (rank-sum, numpy, no sklearn) over `is_ood`
   labels, plus an onset AUC scoring only each excursion's first points
   (later in-segment points match their own predecessors and legitimately
   look familiar to an online detector). Systems without a novelty API
   (naive Qdrant) report `auc: null` — no proxy score is invented.
3. **Trajectory reconstruction fidelity** — fraction of a scene's true
   ordered trajectory recovered around a mid-scene anchor, plus an order
   fidelity score. Each system uses its own API: causal scene scan for
   LOCI, payload scroll for naive Qdrant, array slice for brute force.
4. **Recall vs age under consolidation** — the flight-recorder curve. A
   long stream is inserted in epoch-sized chunks with consolidation on;
   afterwards, progressively older time windows are probed. Two recalls
   per age bucket: *strict* (only surviving raw points count) and *covered*
   (a summary whose recorded source range covers the ground-truth point also
   counts), plus the resident-point compression ratio.
5. **Sustained-load latency** — interleaved single inserts and windowed
   queries with no sleeps; p50/p95 per op. Latency is the one wall-clock
   (machine-dependent) metric in the suite; everything else is
   deterministic for a given seed.

### Systems

| name | what it is |
|---|---|
| `brute_force` | Exact numpy search over everything — the recall oracle |
| `loci_local` | `LocalLociClient` (in-memory backend, decay off) |
| `loci_local_consolidated` | `LocalLociClient` + `ConsolidationPolicy` (flight-recorder config) |
| `loci_qdrant_memory` | `LociClient` over qdrant-client's `:memory:` local engine |
| `naive_qdrant` | Plain Qdrant collection, float-range payload filters, no LOCI |

Determinism notes: all streams are future-dated so LOCI's maintenance clock
(`max(wall clock, newest timestamp)`) is driven by the stream itself, and
decay is disabled — no wall-clock value enters any non-latency metric.

### How to run

```bash
python -m benchmarks.wm_bench              # full suite (~1 min, writes artifact)
python -m benchmarks.wm_bench --quick      # reduced sizes
python -m benchmarks.wm_bench --tasks novelty_auc,recall_vs_age
python -m benchmarks.wm_bench --systems loci_local,naive_qdrant --no-write
```

The artifact is `benchmarks/results/wm_bench_latest.json` (schema: config,
seed, versions, per-task per-system metrics); a markdown report prints to
stdout. Honesty rules are baked into both: the synthetic label appears in
every output, oracle rows are always shown, and losing numbers are printed
like winning ones — the checked-in run records, for example, LOCI losing
insert latency to naive Qdrant by ~29× (`loci_local`) while winning query
latency, and strict recall collapsing to 0 on fully consolidated buckets.

### Adding a dataset adapter (v2 path to real data)

Implement the `TrajectoryDataset` protocol in
`benchmarks/wm_bench/datasets.py`: expose `name`, `vector_dim`,
`synthetic = False`, and iterate `TrajectoryPoint(timestamp_ms, x, y, z,
embedding, scene_id, is_ood)` in strictly increasing timestamp order with
positions normalised to `[0, 1]^3`. Every task consumes only this protocol,
so a Habitat/TartanAir/nuScenes adapter (frames embedded with an open world
model, poses normalised per scene) plugs in without touching task code. OOD
labels are optional (`is_ood=False` everywhere disables the novelty task's
positives); timestamps should be future-dated or the consolidation task's
aging will be driven by the machine's wall clock instead of the stream.

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
