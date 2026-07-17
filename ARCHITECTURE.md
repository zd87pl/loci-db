# Loci Architecture

## Four-Layer Design

```
┌───────────────────────────────────────────────┐
│              Application Layer                │
│  LociClient / AsyncLociClient / LocalLociClient│
│  insert · query · predict_and_retrieve        │
├───────────────────────────────────────────────┤
│              Retrieval Layer                  │
│  predict.py — predict-then-retrieve           │
│  funnel.py  — multi-scale coarse→fine search  │
├───────────────────────────────────────────────┤
│           Indexing & Routing Layer            │
│  spatial/  — 4D Hilbert encoding + bucketing  │
│  temporal/ — logical epochs: consolidation,   │
│              retention, decay scoring         │
├───────────────────────────────────────────────┤
│              Storage Layer                    │
│  Qdrant — two bounded collections per tenant: │
│    loci_data (raw) + loci_summary (aged)      │
│  MemoryStore (in-process, no infra needed)    │
└───────────────────────────────────────────────┘
```

### Layer 1: Storage (Qdrant)

Each tenant/store uses exactly **two** Qdrant collections, created lazily on
first use and constant for the lifetime of the deployment:

- **`{prefix}loci_data`** — every raw `WorldState`. Payload indices:

  | Field          | Type      | Purpose                        |
  |----------------|-----------|--------------------------------|
  | `hilbert_r4/r8/r12` | INTEGER | Multi-resolution spatial bucket pre-filter |
  | `timestamp_ms` | INTEGER   | Indexed temporal Range filter  |
  | `scale_level`  | KEYWORD   | Multi-scale funnel search      |
  | `scene_id`     | KEYWORD   | Causal chains and funnel narrowing |

- **`{prefix}loci_summary`** — consolidated summary states produced by
  episodic-to-semantic aging. Same indices **except** the Hilbert fields:
  summaries carry no Hilbert payload, and spatial queries reach them through
  the exact geometric post-filter on `x`/`y`/`z`.

**Epochs are purely logical.** An epoch (`timestamp_ms // epoch_size_ms`) is
the unit of consolidation granularity and of Hilbert t-normalisation — it is
never a collection. Time selectivity comes from a Range condition on the
indexed `timestamp_ms` field, so every operation (insert, query, retention,
consolidation) touches at most two collections regardless of how long the
deployment has been ingesting: **O(1) in collection count**.

- **Retention** is a filter-based delete: both `RetentionPolicy` knobs
  (`max_epochs`, `max_age_ms`) reduce to a single epoch-aligned cutoff
  timestamp, and raw points older than the cutoff are deleted from
  `loci_data`. Summaries are never purged.
- **Consolidation** folds raw epochs that leave the raw window into
  `loci_summary`: the coarse group's existing summaries are selected by
  timestamp range, re-consolidated together with the stale raw states
  (lossless bookkeeping of `source_count` / `t_min_ms` / `t_max_ms`), and
  the epoch's raw points are deleted.

Distance metric is configurable: `cosine` (default), `dot`, or `euclidean`.

`LocalLociClient` swaps Qdrant for **`MemoryStore`**, an in-process backend
with the same two-collection/Hilbert layout — no external infrastructure
required.

### Layer 2: Indexing & Routing

**Spatial** — The 4D point `(x, y, z, t_normalised)` is mapped to a single
`int64` via a 4-dimensional Hilbert space-filling curve at multiple resolutions.
Queries start from the coarsest indexed resolution by default (`p=4` → 16 bins
per axis) so bounding-box expansion enumerates a manageable number of bucket IDs
for `MatchAny` filtering. When `adaptive=True`, dense regions can be promoted to
finer stored resolutions as long as the bucket fan-out stays bounded.

Key property: Hilbert curves preserve **spatial locality** — nearby points in 4D
space map to nearby indices on the 1D curve, making the integer set filter
a good proxy for a spatial bounding box.

Quantisation in `encode()` uses `round()`. Query-time expansion uses
`floor()`/`ceil()` at the boundaries to guarantee no misses at grid edges, and
an exact payload post-filter is the final source of truth for geometric bounds.

**Temporal** — `timestamp_ms // epoch_size_ms` determines the logical epoch,
which fixes how `t` is normalised into `[0, 1]` for Hilbert encoding. A query
time window becomes an indexed `timestamp_ms` Range condition; every query
issues **one search on `loci_data` plus one on `loci_summary`** and merges the
results (the async client runs the two searches **concurrently** via
`asyncio.gather`).

Hilbert cover rules: when the time window falls inside a **single epoch**, the
bucket cover uses that epoch's normalised t-bounds — full 4D selectivity. When
the window spans **multiple epochs**, or there is no window, the cover is
computed over the full t range `[0, 1]` — spatial-only selectivity — because
Hilbert t-encoding is epoch-relative, and the indexed timestamp Range carries
the t-dimension instead.

### Layer 3: Retrieval

**predict-then-retrieve** — Calls the user's predictor function to generate a
hypothetical future embedding, then retrieves stored historical analogs matching
that prediction. Callers can optionally pass an explicit absolute time window
when they are storing scheduled or future-dated states.

**funnel search** — Cascades through scale levels (sequence → frame → patch)
to progressively refine results when multi-scale embeddings are stored. Matching
epochs and scene IDs are carried forward between stages, narrowing the timestamp
Range and scene filter so finer passes do not re-scan unrelated data. Always
returns results at the finest available granularity.

**temporal decay** — Re-ranks results using exponential decay:
`score = similarity × exp(-λ × age_ms)`.  λ defaults to
`DEFAULT_DECAY_LAMBDA`, a **one-hour half-life**; derive custom rates from a
half-life with `loci.temporal.decay.lambda_from_half_life()` rather than
setting the per-millisecond `decay_lambda` directly.

### Layer 4: Application

Three client implementations share identical APIs:

- **`LociClient`** — Synchronous.  Sequential data + summary searches.
- **`AsyncLociClient`** — Asynchronous.  Runs the data and summary searches
  concurrently via `asyncio.gather`; async-safe collection creation with
  per-collection locks.
- **`LocalLociClient`** — Synchronous, backed by the in-process `MemoryStore`.
  Zero infrastructure: no Qdrant server needed.

Both support:
- `insert()` / `insert_batch()` — with automatic causal linking within scenes
- `query()` — spatiotemporal ANN search with Hilbert pre-filtering
- `predict_and_retrieve()` — the novel predict-then-search primitive
- `get_trajectory()` — causal chain traversal

## Data Flow

```
insert(WorldState)
  → ensure loci_data exists → compute logical epoch
  → normalise t within epoch → compute hilbert_id
  → find causal predecessor in same scene → link prev/next
  → upsert PointStruct to loci_data
  → maintenance pass: consolidation, then retention

insert_batch(states)
  → sort by (scene_id, timestamp_ms) → build causal chains
  → one bulk upsert to loci_data
  → patch next_state_id links
  → maintenance pass: consolidation, then retention

query(vector, bounds, time_window)
  → time window → indexed timestamp_ms Range condition
  → single-epoch window: epoch-local 4D Hilbert cover
    multi-epoch or no window: spatial-only cover (t carried by the Range)
  → search loci_data (Range + Hilbert MatchAny)
    + search loci_summary (Range only, no Hilbert payload)
    (async client runs both searches concurrently)
  → merge → exact post-filter → apply temporal decay → re-rank
  → return WorldStates with vectors

predict_and_retrieve(context_vector, predictor_fn, horizon)
  → predicted = predictor_fn(context_vector)
  → query(predicted) — searches ALL stored history for analogs by default;
    an explicit search_time_window_ms=(start, end) restricts retrieval to an
    absolute timestamp range (only useful for scheduled/future-dated states)
```

## Causal Linking

On `insert()`, Loci automatically finds the most recent state in the
same `scene_id` and links `prev_state_id` / `next_state_id`.  On
`insert_batch()`, states are sorted by `(scene_id, timestamp_ms)` and
linked within the batch.  This enables `get_trajectory()` to walk the
causal chain forward and backward from any anchor state.

## Known Limitations and Planned Refactors

One structural issue is known and deliberately deferred; it is tracked in
[ROADMAP.md](ROADMAP.md) alongside the `loci-core` distribution decision:

1. **Hand-maintained client parity.** `LociClient`, `AsyncLociClient`, and
   `LocalLociClient` implement the same API surface with roughly 77% duplicated
   logic, so every behavioral fix must be applied three times and parity drifts.
   Planned fix: extract a shared core (query planning, filter construction,
   result assembly) with thin sync/async/local transport shells.
