# RFC-0001: LOCI as the Memory System for World Models

**Status:** Accepted (living document)
**Date:** 2026-07-17
**Scope:** Strategic direction and R&D roadmap from v0.3 to v1.0 and beyond.

---

## 1. Thesis

LOCI's own reproducible benchmarks (`benchmarks/results/latest.json`) say something
uncomfortable and clarifying: **as a pure spatial pre-filter, the Hilbert bucket layer
often loses to naive Qdrant float-range filters** (3.7–5× slower on tight spatial at
small N; ~2× faster only on combined tight spatial+temporal at 10k+). Competing with
Qdrant, Milvus, and LanceDB at the filter layer is a race we do not win by being a
middleware on top of one of them.

What no general vector database has is LOCI's **semantics layer**:

- `predict_and_retrieve` with absolute, calibrated novelty scoring
- causal trajectories and episodic context windows
- temporal decay, retention, and (new) consolidation as first-class memory aging
- a zero-infrastructure in-memory mode with microsecond-class label queries
- adapters that speak world-model (V-JEPA 2, DreamerV3) natively

**The identity is therefore: *the memory system for world models and embodied
agents* — not "a faster spatiotemporal filter."** Every bet below either strengthens
that identity or makes the system boring-and-reliable enough to trust with it.

The credibility strategy is part of the thesis: in a space full of inflated claims,
*"the memory system whose benchmarks you can reproduce"* is itself a moat. Honest
negative results (see `experimental/IDD-58*`, `docs/BENCHMARK_METHODOLOGY.md`) are
policy, not accidents.

---

## 2. R&D bets (cutting-edge)

### R1. Memory consolidation — episodic → semantic aging
**Landed as v1 in this change** (`loci/temporal/consolidation.py`, LocalLociClient).

Instead of deleting old epochs, summarize them: per-scene centroid states with pooled
embeddings, merged time ranges, and source counts, stored in coarse summary
collections that queries include transparently. Recent memory stays raw and
high-fidelity; old memory degrades to gist; storage is bounded forever while
everything remains findable.

- **Next:** wire into the Qdrant clients; configurable fidelity curves (multi-tier:
  raw → k-centroids → single scene digest); consolidation-aware novelty (a match
  against a summary is weaker evidence of familiarity than a raw match).
- **Effort:** v1 landed; Qdrant wiring ~1–2 wk; multi-tier ~3–4 wk.
- **Success metric:** flight-recorder property — N days of continuous recording holds
  a fixed storage budget with measured recall degradation curve published in the
  benchmark suite (R3).

### R2. Conformal novelty — from heuristic to guarantee
**Landed** (`ConformalNoveltyCalibrator`; empirical FAR within ±1% of alpha —
see `benchmarks/results/conformal_latest.json`).

The novelty score is now correct (absolute best-match similarity + calibration), but
it is a heuristic. Conformal prediction turns it into a guarantee:
"novelty > τ fires with ≤ α false-alarm rate," distribution-free, using a sliding
calibration window of nonconformity scores.

- **Why it matters:** the advertised robot-safety use case needs a *rate*, not a
  vibe. This is also a clean publishable result on top of an implemented system.
- **Design sketch:** replace `NoveltyCalibrator`'s z-score squash with a conformal
  quantile over the trailing window (already score-before-observe, which is the
  correct online-conformal discipline); expose `alpha` instead of a raw threshold.
- **Effort:** ~2–3 wk including evaluation.
- **Success metric:** empirical false-alarm rate within ±1% of the configured α on
  held-out trajectory data across three datasets.

### R3. The world-model memory benchmark — own the category
**Landed as v1** (`benchmarks/wm_bench/`, synthetic generators + dataset
adapter protocol; real-dataset adapters are the v2 work).

No standard benchmark exists for what LOCI does. Define it:

- **Datasets:** public embodied trajectories (Habitat/AI2-THOR rollouts, TartanAir,
  nuScenes scenes) embedded with an open world model.
- **Tasks:** future-state analog recall@k, novelty AUC for OOD segments, trajectory
  reconstruction fidelity, recall-vs-age under consolidation, insert/query latency
  under continuous write load.
- **Baselines:** naive Qdrant, pgvector+PostGIS, a flat FAISS index with brute-force
  filters. Run them honestly; publish losses as well as wins.
- `benchmarks/world_model_harness.py` is the seed; grow it into a standalone,
  pip-installable suite others can run against *their* stores.
- **Effort:** ~4–6 wk for v1 with three datasets.
- **Success metric:** at least one external system evaluated on it by someone who
  isn't us.

### R4. Adaptive query planner — make the filter layer stop losing
IDD-58's real finding: contiguous **Hilbert range runs** can replace 65k-ID
`MatchAny` filters, but only when boxes grid-align; enumeration cost dominates at
p≥8. The fix is not a better constant — it is a planner that chooses per query among:

1. Hilbert range-run conditions (few `Range` clauses) when runs are few,
2. bucket `MatchAny` when the cover is small,
3. plain float-range payload filters when neither wins (i.e., admit naive mode).

Cost model inputs already exist (`estimated_bucket_count`, adaptive density stats).
- **Effort:** ~3–4 wk including benchmark regression gates.
- **Success metric:** LOCI ≥ naive Qdrant on *every* scenario in
  `benchmarks/vs_naive_qdrant.py` (worst case: ties, because the planner picks naive).

### R5. Embedded Rust engine — "SQLite for spatiotemporal vectors"
**Stage (a) landed** (`LocalLociClient(backend="rust")`, parity-tested; 5.6x
batched insert, 85x search). Stages (b) persistence and (c) quantization
remain.

The in-memory mode is the most differentiated pragmatic asset (zero infra, µs-class
label queries) but it is Python and non-persistent. `loci-core` already has
parity-tested Hilbert + temporal math. Grow it into an embedded engine:

- Rust-native store: mmap-persisted segments per epoch, int8/binary quantization
  with exact rerank, SIMD distance kernels; Python bindings keep the current
  `LocalLociClient` API.
- Target: Jetson/Orange Pi-class on-device memory for robots — a category
  client-server vector DBs structurally cannot serve.
- **Effort:** the big one — ~2–3 months to a credible v1. Stage it: (a) Rust
  MemoryStore drop-in, (b) persistence, (c) quantization.
- **Success metric:** feature-parity with MemoryStore behind the same tests
  (`tests/test_qdrant_integration.py`-style tier), ≥5× insert throughput, working
  persistence, <200MB footprint for 1M 512-d states at int8.

---

## 3. Pragmatic engineering (usable)

### P1. Storage refactor: bounded collections *(prerequisite for production)*
**Landed in this change**: exactly two collections per tenant
(`{prefix}loci_data` + `{prefix}loci_summary`) with payload-indexed
timestamps and Hilbert buckets; epochs are purely logical and every operation
is O(1) in collection count. The old one-collection-per-5s-epoch layout and
its O(collections) cost model are gone; existing deployments migrate with the
`loci migrate-layout` CLI (dry-run, verified copy, optional `--delete-old`).
**Metric:** 30 days of continuous writes with flat p50 query latency and
bounded collection count.

### P2. MCP server — spatial memory for agents
**Landed as v1 in this change** (`loci/mcp/`, `loci-mcp` entry point,
`docs/MCP_SERVER.md`): `remember` / `recall` / `novelty` / `trajectory` /
`memory_stats` over local, Qdrant, or cloud backends. Rides the agent wave at
near-zero marginal cost; becomes genuinely compelling when R5 gives it durable
on-device storage. **Next:** text-embedding convenience layer (optional embedder
config) so agents without vectors can use it directly.

### P3. One 20-minute path to "wow"
A single runnable end-to-end example: open world-model checkpoint (V-JEPA 2) →
webcam/dataset clip → LOCI → novelty alerts + trajectory replay, in a notebook, plus
a ROS 2 node packaging the same loop for robotics users. Adapters exist; the missing
artifact is the demo that requires no assembly. **Effort:** ~2 wk. **Metric:** cold
clone to running demo in under 20 minutes, measured with a stopwatch, by someone who
didn't build it.

### P4. QueryStats observability
Queries can silently degrade (search failures log at WARNING but callers can't
see them). Return an optional stats object: data/summary searches hit/failed,
buckets enumerated, filter mode chosen (feeds R4), overfetch efficiency,
consolidation hits.
**Effort:** ~1 wk. **Metric:** a partial-outage query is distinguishable from an
empty result at the API level.

### P5. Versioned storage format + release discipline
Before v1.0: a storage format version stamp in payloads/collections, a migration
path, and the outstanding release chores (publish fixed library over PyPI 0.3.0,
register or drop `loci-core` name, apply cloud migration 005). **Effort:** days.

---

## 4. Product wedge: the robot flight recorder

The demos already gesture at it; name it and build toward it. Record
embedding+pose+time continuously (P1/R5 make it cheap), novelty flags anomalies in
real time with guaranteed false-alarm rates (R2), trajectories reconstruct incidents
after the fact, consolidation keeps a fixed storage budget forever (R1). Debugging
and compliance for robot fleets is a concrete buyer with a concrete pain, and every
capability it needs is on this roadmap. The MCP server (P2) is the same story for
software agents.

---

## 5. Sequencing

| Horizon | Items | Outcome |
|---|---|---|
| **Now** (≤1 month) | P1 storage refactor (landed), P5 releases, P4 QueryStats, R1 Qdrant wiring (landed), P3 demo path | Adoptable: a team can run LOCI in production without tripping on it |
| **Quarter** | R5 stage (a)+(b), R2 conformal novelty, R3 benchmark v1 | Defensible: measurable claims nobody else can make |
| **Two quarters** | R5 quantization, R4 planner, R1 multi-tier, multi-agent shared memory | Category-defining: "memory for world models" has a reference implementation |

## 6. Risks

- **Category risk:** "world-model memory" may consolidate inside the big robotics
  stacks (Isaac, π0 ecosystems). Mitigation: integrate with them (P3), don't compete.
- **Engine risk:** R5 is a rewrite-shaped project. Mitigation: staged behind the
  existing test tiers; MemoryStore remains the reference implementation.
- **Focus risk:** the cloud API is a distraction until the library identity wins.
  Keep it in maintenance mode; the wedge is embedded + agents, not hosted ANN.
