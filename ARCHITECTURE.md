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
| `hilbert_id`   | INTEGER   | Spatial bucket pre-filter      |
| `timestamp_ms` | INTEGER   | Temporal range filter          |
| `scale_level`  | KEYWORD   | Multi-scale funnel search      |

Distance metric is configurable: `cosine` (default), `dot`, or `euclidean`.

### Layer 2: Indexing & Routing

**Spatial** — The 4D point `(x, y, z, t_normalised)` is mapped to a single
`int64` via a 4-dimensional Hilbert space-filling curve.  The resolution order
(default `p=4` → 16 bins per axis) is deliberately low so that bounding-box
expansion enumerates a manageable number of bucket IDs for `MatchAny` filtering.

Key property: Hilbert curves preserve **spatial locality** — nearby points in 4D
space map to nearby indices on the 1D curve, making the integer set filter
a good proxy for a spatial bounding box.

Quantisation in `encode()` uses `round()`.  The `expand_bounding_box()` function
uses `floor()`/`ceil()` at the boundaries to guarantee no misses at grid edges.

**Temporal** — `timestamp_ms // epoch_size_ms` determines the epoch.  Queries
compute which epochs overlap the requested time window and fan out searches.
The async client searches all matching shards **concurrently** via `asyncio.gather`.

### Layer 3: Retrieval

**predict-then-retrieve** — Calls the user's predictor function to generate a
hypothetical future embedding, then runs a standard query filtered to
`[now, now + future_horizon_ms]`.

**funnel search** — Cascades through scale levels (sequence → frame → patch)
to progressively refine results when multi-scale embeddings are stored.
Always returns results at the finest available granularity.

**temporal decay** — Re-ranks results using exponential decay:
`score = similarity × exp(-λ × age_ms)`.  Configurable via `decay_lambda`.

### Layer 4: Application

Two client implementations share identical APIs:

- **`LociClient`** — Synchronous.  Sequential shard iteration.
- **`AsyncLociClient`** — Asynchronous.  Parallel shard fan-out via
  `asyncio.gather`.  Async-safe collection creation with per-collection locks.

Both support:
- `insert()` / `insert_batch()` — with automatic causal linking within scenes
- `query()` — spatiotemporal ANN search with Hilbert pre-filtering
- `predict_and_retrieve()` — the novel predict-then-search primitive
- `get_trajectory()` — causal chain traversal

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
  → determine epoch range → expand bounding box to hilbert IDs
  → fan-out search across collections with MatchAny + Range filters
  → apply temporal decay → re-rank → return WorldStates with vectors

predict_and_retrieve(context_vector, predictor_fn, horizon)
  → predicted = predictor_fn(context_vector)
  → query(predicted, time_window=[now, now+horizon])
```

## Causal Linking

On `insert()`, Loci automatically finds the most recent state in the
same `scene_id` and links `prev_state_id` / `next_state_id`.  On
`insert_batch()`, states are sorted by `(scene_id, timestamp_ms)` and
linked within the batch.  This enables `get_trajectory()` to walk the
causal chain forward and backward from any anchor state.
