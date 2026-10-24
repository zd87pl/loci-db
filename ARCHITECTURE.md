# Loci Architecture

## Four-Layer Design

```
┌───────────────────────────────────────────────┐
│              Application Layer                │
│  LociClient / AsyncLociClient             │
│  insert · query · predict_and_retrieve        │
├───────────────────────────────────────────────┤
│              Retrieval Layer                  │
│  predict.py — predict-then-retrieve           │
│  funnel.py  — multi-scale coarse→fine search  │
├───────────────────────────────────────────────┤
│           Indexing & Routing Layer            │
│  spatial/  — 4D Hilbert encoding + bucketing  │
│  temporal/ — epoch sharding + decay scoring   │
├───────────────────────────────────────────────┤
│              Storage Layer                    │
│  Qdrant (one collection per temporal epoch)   │
└───────────────────────────────────────────────┘
```

### Layer 1: Storage (Qdrant)

Each temporal epoch maps to a separate Qdrant collection (`loci_{epoch_id}`).
Collections are created lazily on first insert.  Payload indices:

| Field          | Type      | Purpose                        |
|----------------|-----------|--------------------------------|
| `hilbert_r4/r8/r12` | INTEGER | Multi-resolution spatial bucket pre-filter |
| `timestamp_ms` | INTEGER   | Temporal range filter          |
| `scale_level`  | KEYWORD   | Multi-scale funnel search      |
| `scene_id`     | KEYWORD   | Causal chains and funnel narrowing |
| `x`, `y`, `z` | FLOAT | Exact geometric ranges before top-k |
| `confidence` | FLOAT | Confidence threshold before top-k |

Distance metric is configurable: `cosine` (default), `dot`, or `euclidean`.

### Layer 2: Indexing & Routing

**Spatial** — The 4D point `(x, y, z, t_normalised)` is mapped to a single
`int64` via a 4-dimensional Hilbert space-filling curve at multiple resolutions.
Queries start from the coarsest indexed resolution by default (`p=4` → 16 bins
per axis) so bounding-box expansion enumerates a manageable number of bucket IDs
for `MatchAny` filtering. When `adaptive=True`, dense regions can be promoted to
finer stored resolutions as long as the bucket fan-out stays bounded.

Hilbert curves organize grid cells along a locality-preserving traversal. A
bounding box is covered by enumerating its grid cells, not by assuming that all
nearby 4D points have nearby scalar indices.

Quantisation in `encode()` uses `round()`. Query-time expansion uses
`floor()`/`ceil()` at the boundaries to cover intersecting grid cells. Exact
coordinate and confidence ranges are included in the backend filter before
top-k, with a final geometric post-check. This does not guarantee ANN top-k
recall; candidate limits, per-shard overfetch, and ranking need evaluation.

**Temporal** — `timestamp_ms // epoch_size_ms` determines the epoch.  Queries
compute which epochs overlap the requested time window and fan out searches.
The async client searches all matching shards **concurrently** via `asyncio.gather`.

### Layer 3: Retrieval

**predict-then-retrieve** — Calls the user's predictor function to generate a
hypothetical future embedding, then searches stored observations for analogs.
The default adds no timestamp restriction; it does not automatically search
`[now, now + horizon]` or exclude future-dated observations. Callers can pass
`search_time_window_ms=(start, end)` to restrict the absolute range, including
a historical cutoff for evaluation without future-data leakage.

**funnel search** — Cascades through scale levels (sequence → frame → patch)
to progressively refine results when multi-scale embeddings are stored. Matching
epochs and scene IDs are carried forward between stages so finer passes do not
re-scan unrelated shards. Always returns results at the finest available granularity.

**temporal decay** — Re-ranks results using exponential decay:
`decayed_score = similarity × exp(-λ × age_ms)` for cosine/dot scores.
Euclidean raw `score` is `-distance` (higher is better); its `decayed_score`
is `1 / (1 + distance) × exp(-λ × age_ms)`, so aging cannot improve a negative
distance score. Decay is configurable via `decay_lambda`.

### Layer 4: Application

Two client implementations share identical APIs:

- **`LociClient`** — Synchronous.  Sequential shard iteration.
- **`AsyncLociClient`** — Asynchronous.  Parallel shard fan-out via
  `asyncio.gather`.  Async-safe collection creation with per-collection locks.

Both support:
- `insert()` / `insert_batch()` — with automatic causal linking within scenes
- `query()` — spatiotemporal ANN search with Hilbert pre-filtering
- `predict_and_retrieve()` — predictor callback followed by analog retrieval
- `get_trajectory()` — chronological scene-neighborhood retrieval

## Data Flow

```
insert(WorldState)
  → compute epoch_id → ensure collection exists
  → normalise t within epoch → compute hilbert_id
  → find causal predecessor in same scene → link prev/next
  → upsert PointStruct to qdrant

insert_batch(states)
  → sort by (scene_id, timestamp_ms) → build causal chains
  → group by epoch → one upsert per collection
  → patch next_state_id links

query(vector, bounds, time_window)
  → determine epoch range → build epoch-local 4D bounds
  → choose Hilbert resolution → expand bounds to bucket IDs
  → fan-out with MatchAny + exact coordinate/time/confidence ranges before top-k
  → exact post-filter → apply temporal decay → re-rank
  → return WorldStates with vectors

predict_and_retrieve(context_vector, predictor_fn, horizon)
  → predicted = predictor_fn(context_vector)
  → query(predicted, time_window=search_time_window_ms)  # default None
  → re-rank candidates → compute retrieval-based novelty score
```

`future_horizon_ms` does not pass a horizon argument to `predictor_fn`; the
caller configures the predictor for its intended horizon. Without an explicit
search time window, retrieval ranks analogs by vector similarity. This operation
composes calls and does not provide a cross-call transaction.

## Readiness Boundaries

Per-epoch Qdrant persistence is implemented; automatic storage-tier migration
is planned. The optional Rust extension exposes compute primitives, while
client multi-resolution encoding currently uses the Python `HilbertIndex` path.
Primitive benchmarks should not be presented as end-to-end client speedups.
See [IMPLEMENTATION_READINESS.md](docs/IMPLEMENTATION_READINESS.md) for pilot
data contracts, reliability checks, performance comparisons, and acceptance gates.

## Causal Linking

On `insert()`, Loci automatically finds the most recent state in the
same `scene_id` and links `prev_state_id` / `next_state_id`.  On
`insert_batch()`, states are sorted by `(scene_id, timestamp_ms)` and
linked within the batch; separate batches are not automatically stitched
together. `get_trajectory()` retrieves and sorts records from the anchor's
scene, rather than following the stored links. Out-of-order and concurrent
writes need further linking guarantees before applications rely on those
links as a consistent chain.
