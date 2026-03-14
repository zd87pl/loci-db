# Engram

**A 4D spatiotemporal vector database middleware for AI world models.**

[![CI](https://github.com/zd87pl/engram-db/actions/workflows/ci.yml/badge.svg)](https://github.com/zd87pl/engram-db/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## The Problem

Modern world models — V-JEPA 2, DreamerV3, GAIA-1, UniSim — produce embeddings
where every vector has an implicit **4D spatiotemporal address** `(x, y, z, t)`.
Existing vector databases (Qdrant, Milvus, Weaviate) treat all embedding dimensions
equally: a spatial query requires 3+ float-range payload filters evaluated
independently, time-based retrieval has no native sharding, and there is no
concept of "predict the future then find what's nearby."

## The Solution

Engram is a middleware layer on top of [Qdrant](https://qdrant.tech) that makes
spatiotemporal structure **first-class** through three novel primitives:

### 1. Hilbert Curve Spatial Bucketing

Encode `(x, y, z, t)` as a single `int64` via a 4D Hilbert space-filling curve.
Spatial bounding-box queries decompose into a `MatchAny` filter on Hilbert bucket
IDs — replacing 3 independent float-range filters with a single integer set lookup.

```
         Naive Qdrant               Engram
    ┌──────────────────┐     ┌──────────────────┐
    │ x_min ≤ x ≤ x_max│     │                  │
    │ y_min ≤ y ≤ y_max│ →   │ hilbert_id ∈ {…} │
    │ z_min ≤ z ≤ z_max│     │   (single filter) │
    └──────────────────┘     └──────────────────┘
```

### 2. Temporal Sharding

Automatic routing of vectors to **time-partitioned Qdrant collections**
(`engram_{epoch_id}`). Configurable epoch size. Queries fan out only to
epochs that overlap the requested time window — with the async client,
all shards are searched **concurrently** via `asyncio.gather`.

### 3. Predict-then-Retrieve

An **atomic API call** that composes a user-supplied world model with
vector search:

```python
results = client.predict_and_retrieve(
    context_vector=current_embedding,
    predictor_fn=my_world_model,       # you supply this
    future_horizon_ms=2000,
)
```

1. Call `predictor_fn(context_vector)` → predicted future embedding
2. Query the store for nearest neighbours to that prediction, filtered
   to `[now, now + future_horizon_ms]`

This enables **anticipatory retrieval** — finding stored states that
are similar to what the world model *predicts will happen next*.

## Quick Start

```bash
pip install engram-db          # or: pip install -e ".[dev]"
docker run -p 6333:6333 qdrant/qdrant
```

### Sync API

```python
from engram import EngramClient, WorldState

client = EngramClient(
    "http://localhost:6333",
    vector_size=512,
    epoch_size_ms=5000,
    distance="cosine",           # or "dot", "euclidean"
)

# Insert world states
state = WorldState(
    x=0.5, y=0.3, z=0.8,
    timestamp_ms=1700000000000,
    vector=[0.1] * 512,
    scene_id="warehouse_sim",
    scale_level="patch",
)
state_id = client.insert(state)

# Batch insert (truly batched — one Qdrant call per epoch)
ids = client.insert_batch(states)

# Spatiotemporal query
results = client.query(
    vector=query_embedding,
    spatial_bounds={"x_min": 0.2, "x_max": 0.8,
                    "y_min": 0.0, "y_max": 1.0,
                    "z_min": 0.0, "z_max": 1.0},
    time_window_ms=(start_ms, end_ms),
    limit=10,
)

# Predict-then-retrieve
results = client.predict_and_retrieve(
    context_vector=current_embedding,
    predictor_fn=my_world_model,
    future_horizon_ms=2000,
)

# Trajectory reconstruction via causal links
trajectory = client.get_trajectory(state_id, steps_back=20, steps_forward=20)
```

### Async API (parallel shard fan-out)

```python
from engram import AsyncEngramClient

async with AsyncEngramClient(
    "http://localhost:6333",
    vector_size=512,
    distance="cosine",
) as client:
    await client.insert(state)
    results = await client.query(vector=query_embedding, limit=10)
    predicted = await client.predict_and_retrieve(
        context_vector=emb,
        predictor_fn=model,
        future_horizon_ms=1000,
    )
```

## Core Data Model

```python
@dataclass
class WorldState:
    x: float                    # normalised [0, 1]
    y: float                    # normalised [0, 1]
    z: float                    # normalised [0, 1]
    timestamp_ms: int           # unix milliseconds

    vector: list[float]         # arbitrary dimension (512, 1024, 1408, …)

    scene_id: str = ""
    scale_level: str = "patch"  # "patch" | "frame" | "sequence"
    confidence: float = 1.0

    # Causal links (auto-populated on insert)
    prev_state_id: str | None = None
    next_state_id: str | None = None
```

## Architecture

```
┌───────────────────────────────────────────────┐
│              Application Layer                │
│  EngramClient / AsyncEngramClient             │
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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

## Development

```bash
git clone https://github.com/zd87pl/engram-db.git
cd engram-db
pip install -e ".[dev]"
pytest tests/ -v
```

## Performance

**10,000 points · 512-dim vectors · 100 random spatiotemporal queries · 21 temporal shards**

| Method | Avg latency (ms) | P95 latency (ms) | Recall@10 |
|:--|--:|--:|--:|
| Naive Qdrant (3 float-range filters) | 166.36 | 197.23 | 1.000 |
| Engram (Hilbert bucketing + temporal sharding) | 120.04 | 216.85 | 1.000 |
| **Speedup** | **1.39×** | — | — |

Engram's temporal sharding reduces the search space per query by routing
only to epochs that overlap the time window.  On a production Qdrant server
with payload indexes, the Hilbert `MatchAny` integer-set pre-filter provides
additional speedup over float-range evaluation — the numbers above use
qdrant-client's in-memory mode where payload indexes are not active.

Reproduce with:
```bash
python benchmarks/vs_naive_qdrant.py
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the v0.1 → v1.0 plan.

## License

Apache 2.0
