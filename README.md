# LOCI

**A 4D spatiotemporal vector database for AI world models.**

[![CI](https://github.com/zd87pl/loci-db/actions/workflows/ci.yml/badge.svg)](https://github.com/zd87pl/loci-db/actions)
[![PyPI version](https://img.shields.io/pypi/v/loci-stdb.svg)](https://pypi.org/project/loci-stdb/)
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

LOCI is a middleware layer on top of [Qdrant](https://qdrant.tech) that makes
spatiotemporal structure **first-class** through three novel primitives:

### 1. Multi-Resolution Hilbert Bucketing

Encode `(x, y, z, t)` at multiple Hilbert resolutions (p=4, 8, 12).
Spatial bounding-box queries use a Hilbert integer pre-filter with overlap, then
apply an exact payload post-filter as the authoritative geometric check. By
default queries start at the coarsest indexed resolution; with `adaptive=True`,
dense regions can be promoted to finer Hilbert resolutions at query time.

```
         Naive Qdrant               LOCI
    ┌──────────────────┐     ┌──────────────────┐
    │ x_min ≤ x ≤ x_max│     │                  │
    │ y_min ≤ y ≤ y_max│ →   │ hilbert_r4 ∈ {…} │
    │ z_min ≤ z ≤ z_max│     │  (single filter)  │
    └──────────────────┘     └──────────────────┘
```

### 2. Temporal Sharding

Automatic routing of vectors to **time-partitioned Qdrant collections**
(`loci_{epoch_id}`). Configurable epoch size. Queries fan out only to
epochs that overlap the requested time window — with the async client,
all shards are searched **concurrently** via `asyncio.gather`.

### 3. Predict-then-Retrieve with Novelty Detection

An **atomic API call** that composes a user-supplied world model with
vector search, returning both results and a **novelty score**:

```python
result = client.predict_and_retrieve(
    context_vector=current_embedding,
    predictor_fn=my_world_model,
    future_horizon_ms=2000,
    current_position=(0.5, 0.3, 0.8),
)
print(f"Novelty: {result.prediction_novelty:.2f}")
# 0.0 = "I've seen this before"
# 1.0 = "This is new territory"
```

By default this searches stored history for analogs to the predicted future
embedding. Pass `search_time_window_ms=(start, end)` only when you explicitly
want to restrict retrieval to an absolute timestamp range.

## Quick Start

### Quick Start with Docker

The fastest way to run LOCI with a persistent Qdrant backend:

```bash
docker compose up
```

This starts two services:
- **loci** — the LOCI REST API on `http://localhost:8000`
- **qdrant** — the Qdrant vector store on `http://localhost:6333`

Qdrant data is persisted in a named volume so it survives restarts.

Once running, insert and query world states via the HTTP API:

```bash
# Health check
curl http://localhost:8000/health

# Insert a world state. The vector length must match the server's
# LOCI_VECTOR_SIZE (512 in docker-compose.yml) — wrong-length vectors are
# rejected with HTTP 422, so the payload is generated rather than hand-typed:
python3 -c 'import json; print(json.dumps({
    "x": 0.5, "y": 0.3, "z": 0.8,
    "timestamp_ms": 1700000000000,
    "vector": [0.1] * 512,
    "scene_id": "s1"}))' \
  | curl -X POST http://localhost:8000/insert \
      -H 'Content-Type: application/json' -d @-

# Query by vector similarity. Spatial bounds and the time window are optional —
# omit them to search everything:
python3 -c 'import json; print(json.dumps({
    "vector": [0.1] * 512,
    "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
    "z_min": 0.0, "z_max": 1.0,
    "limit": 10}))' \
  | curl -X POST http://localhost:8000/query \
      -H 'Content-Type: application/json' -d @-
```

Interactive API docs: `http://localhost:8000/docs`

---

### No Docker? No problem — in-memory mode

Try LOCI instantly with zero infrastructure using `LocalLociClient`:

```bash
pip install loci-stdb          # or: pip install -e ".[dev]"
```

```python
from loci import LocalLociClient, WorldState

client = LocalLociClient(vector_size=512)

# Insert a world state
state = WorldState(
    x=0.5, y=0.3, z=0.8,
    timestamp_ms=1000,
    vector=[0.1] * 512,
    scene_id="my_scene",
)
state_id = client.insert(state)

# Query by vector similarity + spatial bounds + time window
results = client.query(
    vector=[0.1] * 512,
    spatial_bounds={"x_min": 0.0, "x_max": 1.0,
                    "y_min": 0.0, "y_max": 1.0,
                    "z_min": 0.0, "z_max": 1.0},
    time_window_ms=(0, 5000),
    limit=10,
)
```

### With Qdrant (production)

```bash
pip install loci-stdb
docker run -p 6333:6333 qdrant/qdrant
```

```python
from loci import LociClient, WorldState

client = LociClient(
    "http://localhost:6333",
    vector_size=512,
    epoch_size_ms=5000,
    distance="cosine",
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

# Spatiotemporal query with overlap factor
results = client.query(
    vector=query_embedding,
    spatial_bounds={"x_min": 0.2, "x_max": 0.8,
                    "y_min": 0.0, "y_max": 1.0,
                    "z_min": 0.0, "z_max": 1.0},
    time_window_ms=(start_ms, end_ms),
    limit=10,
    overlap_factor=1.2,  # 20% expanded search for boundary recall
)

# Predict-then-retrieve with novelty scoring
result = client.predict_and_retrieve(
    context_vector=current_embedding,
    predictor_fn=my_world_model,
    future_horizon_ms=2000,
    current_position=(0.5, 0.3, 0.8),
)

# Trajectory reconstruction via scroll API
trajectory = client.get_trajectory(state_id, steps_back=20, steps_forward=20)

# Episodic context window
context = client.get_causal_context(state_id, window_ms=5000)
```

### Async API (parallel shard fan-out)

```python
from loci import AsyncLociClient

async with AsyncLociClient(
    "http://localhost:6333",
    vector_size=512,
    distance="cosine",
) as client:
    await client.insert(state)
    results = await client.query(vector=query_embedding, limit=10)
```

### World Model Adapters

```python
from loci.adapters.vjepa2 import VJEPA2Adapter
from loci.adapters.dreamer import DreamerV3Adapter
from loci.adapters.generic import GenericAdapter

# V-JEPA 2
adapter = VJEPA2Adapter()
states = adapter.batch_clip_to_states(clip_output, ts, scene_id)

# DreamerV3
adapter = DreamerV3Adapter()
ws = adapter.rssm_to_world_state(h_t, z_t, position, ts, scene_id)

# Generic numpy/torch
adapter = GenericAdapter(expected_dim=512)
ws = adapter.from_numpy(embedding, position, ts, scene_id)
```

## Performance

All numbers below come straight from `benchmarks/results/retrieval_latest.json`
(in-memory `LocalLociClient` backend, 128-dim vectors, 500 queries per scenario,
seed 42, Apple Silicon / arm64, Python 3.14). Latency depends heavily on query
type: **label-filtered retrieval (a `scene_id` keyword filter with no spatial or
temporal bounds — the demo path) runs at ~78µs p50 at N=100**, while spatial
bounding-box queries are dominated by the exact geometric post-filter and take
tens to hundreds of milliseconds at these dataset sizes.

| N objects | Query type | P50 | P99 |
|--:|:--|--:|--:|
| 100 | Label-filtered (`scene_id` keyword, no spatial/temporal bounds) | 78µs | 101µs |
| 100 | Vector-only ANN | 195µs | 252µs |
| 100 | Spatial + temporal window | 40.0ms | 43.6ms |
| 100 | Spatial bounding box | 97.5ms | 108.0ms |
| 1,000 | Label-filtered | 479µs | 524µs |
| 1,000 | Vector-only ANN | 1.67ms | 1.92ms |
| 1,000 | Spatial + temporal window | 297ms | 309ms |
| 1,000 | Spatial bounding box | 580ms | 636ms |

Adding a temporal window to a spatial query roughly halves its cost through
epoch shard pruning (40.0ms vs 97.5ms p50 at N=100), but the exact spatial
post-filter — the authoritative geometric check — dominates spatial query time
in the pure-Python in-memory backend. Accelerating that path is the motivation
for the optional native Rust primitives in `loci-core/`.

Insert throughput: **~60,000-67,000 states/s** (in-memory backend, 128-dim vectors).

Run the retrieval benchmark on your hardware:

```bash
python benchmarks/benchmark_retrieval.py
```

Run the world-model proof harness:

```bash
python benchmarks/world_model_harness.py --quick
```

For a LOCI-vs-naive-Qdrant comparison benchmark:

```bash
# In-memory (no Qdrant server needed):
python benchmarks/vs_naive_qdrant.py

# Against a live Qdrant server:
QDRANT_URL=http://localhost:6333 python benchmarks/vs_naive_qdrant.py
```

Results are written to `benchmarks/results/` and printed as markdown tables.

## Why not SpatCode?

SpatCode (WWW 2026, arXiv 2601.09530) encodes coordinates into the embedding
space for soft/fuzzy retrieval via RoPE-style positional encoding. LOCI uses
Hilbert bucketing for **exact geometric range queries** with deterministic behavior.

**Use SpatCode** when semantic proximity matters (e.g., "find images taken
near this location").

**Use LOCI** when physical boundaries matter (e.g., "find all observations
within this 3D bounding box in the last 5 seconds").

## Why not TANNS?

TANNS (ICDE 2025) builds a single graph managing all timestamps internally
with a Timestamp Graph structure. LOCI uses collection-level sharding with
storage tiering.

**Use TANNS** for single-session temporal ANN where all data fits in one graph.

**Use LOCI** when you need cross-session persistence, multi-agent memory sharing,
hot/warm/cold storage tiering, or predict-then-retrieve.

## Architecture

```
┌───────────────────────────────────────────────┐
│              Application Layer                │
│  LociClient / AsyncLociClient / LocalLociClient│
│  insert · query · predict_and_retrieve        │
├───────────────────────────────────────────────┤
│              Retrieval Layer                  │
│  predict.py — predict-then-retrieve + novelty │
│  funnel.py  — multi-scale coarse→fine search  │
├───────────────────────────────────────────────┤
│           Indexing & Routing Layer            │
│  spatial/  — multi-res Hilbert + overlap      │
│  temporal/ — epoch sharding + decay scoring   │
├───────────────────────────────────────────────┤
│              Adapters Layer                   │
│  V-JEPA 2 · DreamerV3 · Generic numpy/torch  │
├───────────────────────────────────────────────┤
│              Storage Layer                    │
│  Qdrant (one collection per temporal epoch)   │
│  MemoryStore (in-process, no infra needed)    │
└───────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design
- [docs/NOVELTY.md](docs/NOVELTY.md) — Novelty claims vs prior art
- [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) — Benchmark replication guide
- [docs/WORLD_MODEL_INTEGRATION.md](docs/WORLD_MODEL_INTEGRATION.md) — Integration guides

## Development

```bash
git clone https://github.com/zd87pl/loci-db.git
cd loci-db
pip install -e ".[dev]"
pytest tests/ -v

# Linting & formatting (must pass in CI)
ruff check loci/ tests/
ruff format --check loci/ tests/
mypy loci/
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the v0.1 → v1.0 plan.

## Citation

```bibtex
@misc{loci2026,
  title={LOCI: A 4D Spatiotemporal Vector Database for AI World Models},
  author={Dyras, Zygmunt},
  year={2026},
  url={https://github.com/zd87pl/loci-db}
}
```

## License

Apache 2.0
