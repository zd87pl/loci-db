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
