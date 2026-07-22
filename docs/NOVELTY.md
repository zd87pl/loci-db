# LOCI — Novelty Claims vs Prior Art

## Comparison Table

| Capability | LOCI | SpatCode | TANNS | WorldMem | RA-DT |
|---|:---:|:---:|:---:|:---:|:---:|
| 4D Hilbert spatial bucketing | ✓ | ✗ | ✗ | ✗ | ✗ |
| Multi-resolution overlap | ✓ | soft | ✗ | ✗ | ✗ |
| Bounded temporal aging (raw → summary) | ✓ | ✗ | partial | ✗ | ✗ |
| Predict-then-retrieve | ✓ | ✗ | ✗ | ✗ | ✗ |
| Novelty detection score | ✓ | ✗ | ✗ | ✗ | ✗ |
| World model adapters | ✓ | ✗ | ✗ | ✗ | ✗ |
| Cross-session persistence | ✓ | ✗ | ✗ | ✗ | ✗ |

## Per-Innovation Analysis

### 1. 4D Hilbert Spatial Bucketing

**What exists:** Traditional vector databases use independent float-range filters for each spatial dimension (x, y, z, t). This requires 3-4 separate filter conditions ANDed together, each traversing a separate index.

**What LOCI adds:** A single 4D Hilbert space-filling curve maps (x, y, z, t_normalized) to a single integer. Spatial queries become a single `MatchAny` filter on one indexed integer field instead of 3-4 independent range filters.

**Why it matters (measured):** The single-integer filter replaces O(n_dims) index traversals with one set-membership test — but this does **not** translate into a raw spatial speedup at the scales measured so far. On the checked-in benchmark (`benchmarks/results/latest.json`, N=1,000-10,000 against a live Qdrant, 512-dim), naive float-range filtering is 3.7-4.9x *faster* on pure tight-spatial queries (scenario A). Where the Hilbert pre-filter pays off is in combination with temporal filtering: on tight spatial + temporal queries (scenario C), LOCI is ~2x faster than naive Qdrant at N=10,000 (22.6ms vs 44.9ms p50) at 0.99 recall@10. The honest value proposition is the combined spatiotemporal query path — plus predict-then-retrieve and the zero-infrastructure in-memory mode — not spatial filtering in isolation.

### 2. Multi-Resolution Overlap

**What exists:** SpatCode (WWW 2026) uses RoPE-style soft encoding that embeds coordinates directly into the embedding space for fuzzy/semantic spatial matching. Hard-filtering approaches (including naive Hilbert bucketing) suffer from boundary recall degradation.

**What LOCI adds:** Points are encoded at three Hilbert resolutions (p=4, 8, 12). Queries use an integer Hilbert pre-filter with overlap_factor (default 1.2 = 20% expansion), then apply exact coordinate post-filtering. The default query path starts at the coarsest resolution, while `adaptive=True` can promote denser regions to finer stored resolutions.

**Why it matters:** Preserves recall@k parity with brute-force search while maintaining the speed advantage of integer-set filtering. SpatCode sacrifices exact geometric range queries for soft retrieval; LOCI preserves deterministic spatial boundaries while fixing boundary-recall degradation through overlap plus authoritative exact filtering.

### 3. Bounded Temporal Storage with Episodic-to-Semantic Aging

**What exists:** TANNS (ICDE 2025) manages timestamps within a single graph structure with internal filtering. Standard vector databases use a single collection with timestamp range filters and no memory-aging semantics.

**What LOCI adds:** All raw states live in one payload-indexed `loci_data` collection; time-windowed queries use an indexed `timestamp_ms` range filter. Epochs are a logical unit of aging: when a raw epoch leaves the configured raw window, consolidation folds it into per-scene centroid summaries in a companion `loci_summary` collection, and retention purges expired raw points with cutoff-based deletes. Queries search raw and summary data together (concurrently in the async client), so old data stays findable at reduced fidelity while storage stays bounded.

**Why it matters:** Enables a fixed storage budget over unbounded ingest (the flight-recorder property), cross-session persistence, and multi-agent memory sharing — with every operation O(1) in collection count. TANNS's single-graph approach cannot support these operational concerns.

### 4. Predict-Then-Retrieve (Strongest Claim)

**What exists:** HyDE (ACL 2023) generates hypothetical documents for retrieval. No existing system applies this concept to spatiotemporal world models.

**What LOCI adds:** An atomic pipeline: (1) call user's world model predictor to generate a predicted future-state embedding, (2) retrieve historical states matching that prediction, optionally constrained by spatial bounds or an explicit time window, (3) compute a combined similarity score, (4) return a prediction_novelty score.

**Why it matters:** This turns a vector database into a novelty detector for physical agents. A robot with LOCI can answer "Have I seen a situation like what I'm about to encounter?" before acting. No prior system provides this primitive.

### 5. Novelty Detection Score

**What exists:** No existing vector database provides a quantified novelty metric from retrieval results.

**What LOCI adds:** `prediction_novelty ∈ [0, 1]` computed from the predict-then-retrieve pipeline. 0.0 = "I've seen this before" (strong historical match). 1.0 = "This is new territory" (no historical analog).

**Why it matters:** Enables autonomous agents to modulate behavior based on situational familiarity: use cached experience for known situations, proceed cautiously in novel ones.

### 6. World Model Adapters

**What exists:** No existing spatiotemporal database provides ready-to-use integrations with specific world model architectures.

**What LOCI adds:** Production adapters for V-JEPA 2 (Meta FAIR), DreamerV3 (Hafner et al.), and generic numpy/torch models. Each adapter handles the specific output format and maps it to LOCI's WorldState schema.

**Why it matters:** Reduces integration friction from days to minutes. A researcher can plug LOCI into their existing V-JEPA 2 or DreamerV3 pipeline with ~5 lines of code.

### 7. Cross-Session Persistence

**What exists:** WorldMem and RA-DT operate within single episodes. TANNS maintains a single temporal graph.

**What LOCI adds:** A shared persistent store where data from different sessions, agents, or time periods coexists, partitioned logically by `scene_id` and indexed time rather than physically. Causal chains (prev_state_id / next_state_id) link within sessions; cross-session queries retrieve across the whole store with indexed time-range filters.

**Why it matters:** Enables multi-agent systems where agents share a common spatial memory. A fleet of robots can contribute to and query from the same LOCI instance.

## Conformal Novelty Guarantees (RFC-0001 R2)

### What the guarantee is

`loci.retrieval.novelty.ConformalNoveltyCalibrator` upgrades the novelty score from a
heuristic to a **distribution-free guaranteed false-alarm rate**. Each best-match
similarity `s` maps to a nonconformity score `a = 1 - clamp(s, 0, 1)` (low similarity =
more nonconforming). A sliding window of the most recent nonconformity scores is the
calibration set, and a new observation gets the conformal p-value

```
p = (1 + #{a_i in window : a_i >= a*}) / (n + 1)
```

with the alarm `is_novel(score)` firing when `p <= alpha` (inductive/split conformal
prediction; Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, 2005).

**Guarantee:** if the current observation is exchangeable with the window contents (e.g.
an i.i.d. in-distribution stream), the p-value is super-uniform, so
`P(false alarm) <= alpha` — finite-sample, distribution-free, valid at any window size,
with no threshold tuning. The pipeline already scores before observing
(`PredictThenRetrieve` calls `calibrated_novelty` before `observe`), which is the correct
online-conformal discipline.

### The exchangeability caveat (honest limits)

- **Drift breaks exchangeability.** A drifting score distribution weakens the exact
  finite-sample bound. The sliding window *adapts* to slow drift (old regimes are
  evicted, so alarms recover), at the cost of exact validity during the transition. On
  the checked-in evaluation, sustained downward drift over-fires by ~1.5% absolute at
  `alpha = 0.10` (0.115 measured) while staying within tolerance at 0.01 and 0.05.
- **Alarms are correlated across time** — the window is shared between nearby
  observations. The bound is on the marginal false-alarm rate, not alarm independence.
- **Observed OOD is absorbed.** The pipeline observes every sample, so a persistent
  anomaly migrates into the calibration window and alarms stop — first-encounter
  detection is high (0.96–1.0 at `alpha >= 0.05` in the eval), steady-state detection
  under contamination is much lower. This is the same mechanism as drift adaptation; if
  your deployment needs persistent alarms, gate `observe()` on the alarm decision (and
  accept that conditional observation biases the window).
- **Warm-up:** below `min_samples`, `calibrated_novelty` falls back to the raw absolute
  novelty `1 - score` (check `.warmed_up`). `is_novel`/`p_value` are valid at any
  occupancy — with a small window the p-value simply cannot reach `alpha`, so the alarm
  is conservative, never anti-conservative.

### Usage

```python
from loci import ConformalNoveltyCalibrator, LocalLociClient

client = LocalLociClient(vector_size=16)
calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=512, min_samples=30)

result = client.predict_and_retrieve(
    context_vector=embedding,
    predictor_fn=world_model.predict,
    future_horizon_ms=1000,
    current_position=(x, y, z),
    calibrator=calibrator,  # duck-typed: same slot as NoveltyCalibrator
)
# Continuous score: calibrated_novelty = 1 - p_value, so thresholding at
# 1 - alpha reproduces the guaranteed alarm exactly:
if calibrator.warmed_up and result.prediction_novelty >= 1 - calibrator.alpha:
    ...  # fires on <= ~5% of in-distribution observations
```

The legacy z-score `NoveltyCalibrator` is unchanged and remains available; at a matched
nominal threshold (`novelty >= 1 - alpha`) its implied alarm has an *uncontrolled* rate —
measured FAR between 0.000 and 0.017 across configurations regardless of the nominal
alpha, including a configuration (bimodal, alpha=0.01) with zero OOD detection.

### Evaluation

`benchmarks/conformal_eval.py` (deterministic, seeded, no network) sweeps
`alpha ∈ {0.01, 0.05, 0.1}` over 5 seeds on three stream shapes (gaussian, bimodal, slow
drift) with injected OOD segments, and writes
`benchmarks/results/conformal_latest.json`. Measured on held-out in-distribution data
(window=512, 20k eval points per cell): empirical FAR within ±1% of alpha for every
exchangeable configuration — e.g. gaussian 0.0095/0.0479/0.0995 and bimodal
0.0107/0.0493/0.0990 for alpha 0.01/0.05/0.10 — meeting the RFC-0001 R2 success metric;
the drift shape stays within tolerance except `alpha = 0.10` (0.115), as expected from
the caveat above. A coarse version of the guarantee is pinned in
`tests/test_conformal_novelty.py`.
