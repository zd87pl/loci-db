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
│  temporal/ — epoch sharding + decay scoring   │
├───────────────────────────────────────────────┤
│              Storage Layer                    │
│  Qdrant (one collection per temporal epoch)   │
│  MemoryStore (in-process, no infra needed)    │
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

Distance metric is configurable: `cosine` (default), `dot`, or `euclidean`.

`LocalLociClient` swaps Qdrant for **`MemoryStore`**, an in-process backend
with the same epoch/Hilbert layout — no external infrastructure required.

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

**Temporal** — `timestamp_ms // epoch_size_ms` determines the epoch.  Queries
compute which epochs overlap the requested time window and fan out searches.
The async client searches all matching shards **concurrently** via `asyncio.gather`.

### Layer 3: Retrieval

**predict-then-retrieve** — Calls the user's predictor function to generate a
hypothetical future embedding, then retrieves stored historical analogs matching
that prediction. Callers can optionally pass an explicit absolute time window
when they are storing scheduled or future-dated states.

**funnel search** — Cascades through scale levels (sequence → frame → patch)
to progressively refine results when multi-scale embeddings are stored. Matching
epochs and scene IDs are carried forward between stages so finer passes do not
re-scan unrelated shards. Always returns results at the finest available granularity.

**temporal decay** — Re-ranks results using exponential decay:
`score = similarity × exp(-λ × age_ms)`.  λ defaults to
`DEFAULT_DECAY_LAMBDA`, a **one-hour half-life**; derive custom rates from a
half-life with `loci.temporal.decay.lambda_from_half_life()` rather than
setting the per-millisecond `decay_lambda` directly.

### Layer 4: Application

Three client implementations share identical APIs:

- **`LociClient`** — Synchronous.  Sequential shard iteration.
- **`AsyncLociClient`** — Asynchronous.  Parallel shard fan-out via
  `asyncio.gather`.  Async-safe collection creation with per-collection locks.
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
  → fan-out search across collections with MatchAny + Range filters
  → exact post-filter → apply temporal decay → re-rank
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

Two structural issues are known and deliberately deferred; both are tracked
in [ROADMAP.md](ROADMAP.md):

1. **Unbounded collection growth.** One Qdrant collection per temporal epoch
   means collection count grows linearly with wall-clock time — at the default
   5-second epoch, roughly 17,000 collections per day of continuous ingest.
   Operations that enumerate collections (shard routing, compaction, health
   checks) are O(collections). Planned fix: migrate to payload-indexed epoch
   IDs within a bounded set of collections, keeping the same epoch-pruned
   query semantics.

2. **Hand-maintained client parity.** `LociClient`, `AsyncLociClient`, and
   `LocalLociClient` implement the same API surface with roughly 77% duplicated
   logic, so every behavioral fix must be applied three times and parity drifts.
   Planned fix: extract a shared core (query planning, filter construction,
   result assembly) with thin sync/async/local transport shells.
