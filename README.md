# LOCI

**A 4D spatiotemporal vector database for AI world models.**

[![CI](https://github.com/zd87pl/loci-db/actions/workflows/ci.yml/badge.svg)](https://github.com/zd87pl/loci-db/actions)
[![PyPI version](https://img.shields.io/pypi/v/loci-stdb.svg)](https://pypi.org/project/loci-stdb/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## The Problem

Applications that store world-model embeddings often need to attach a position,
timestamp, and source observation to each vector. They also need to retrieve
relevant past observations within explicit spatial and temporal bounds. This
requires an application-level contract for coordinates, embedding compatibility,
and retrieval behavior alongside the vector store.

## The Solution

LOCI is a middleware layer on top of [Qdrant](https://qdrant.tech) that makes
spatiotemporal structure explicit through three capabilities:

### 1. Multi-Resolution Hilbert Bucketing

Encode `(x, y, z, t)` at multiple Hilbert resolutions (p=4, 8, 12).
Spatial bounding-box queries combine a Hilbert integer pre-filter with exact
coordinate ranges in Qdrant, followed by a geometric post-check. By
default queries start at the coarsest indexed resolution; with `adaptive=True`,
dense regions can be promoted to finer Hilbert resolutions at query time.

```
         Range filters             LOCI pre-filter
    ┌──────────────────┐     ┌──────────────────┐
    │ x_min ≤ x ≤ x_max│     │                  │
    │ y_min ≤ y ≤ y_max│ →   │ hilbert_r4 ∈ {…} │
    │ z_min ≤ z ≤ z_max│     │ + exact ranges   │
    └──────────────────┘     └──────────────────┘
```

### 2. Temporal Sharding

Automatic routing of vectors to **time-partitioned Qdrant collections**
(`loci_{epoch_id}`). Configurable epoch size. Queries fan out only to
epochs that overlap the requested time window — with the async client,
all shards are searched **concurrently** via `asyncio.gather`.

### 3. Predict-then-Retrieve with Novelty Detection

A single API call composes a user-supplied world model with vector search,
returning results and a **retrieval-based novelty score**. It is not a storage
transaction, and the score needs task-specific evaluation before use:

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

By default this searches stored observations without a timestamp restriction
for analogs to the predicted future embedding. Pass
`search_time_window_ms=(start, end)` to restrict retrieval to an absolute range,
including a historical cutoff when evaluating predictions.

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

# Insert a world state (512-dim vector)
python3 -c 'import json; print(json.dumps({"x":0.5,"y":0.3,"z":0.8,"timestamp_ms":1700000000000,"vector":[0.1]*512,"scene_id":"s1"}))' | \
curl -X POST http://localhost:8000/insert \
  -H 'Content-Type: application/json' \
  --data-binary @-

# Query (spatial + time window)
python3 -c 'import json; print(json.dumps({"vector":[0.1]*512,"x_min":0.0,"x_max":1.0,"time_start_ms":1700000000000,"time_end_ms":1700000005000,"limit":10}))' | \
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  --data-binary @-
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

### With a persistent Qdrant backend

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

These helpers convert supplied model outputs to `WorldState` objects; they do
not load or validate a production model pipeline. The V-JEPA 2 helper maps
tubelet grid indices to normalized coordinates, not calibrated 3D positions.
Applications must define and validate that mapping.

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

The following reported local measurements use the **in-memory backend** with
128-dimensional vectors on Apple Silicon. They do not establish live-Qdrant
latency, a general speedup, or customer-scale performance.

| N objects | Query type | P50 | P99 |
|--:|:--|--:|--:|
| 100 | Label-filtered (demo path) | 75µs | 124µs |
| 100 | Vector-only ANN | 212µs | 217µs |
| 100 | Temporal shard pruning | 156µs | 188µs |
| 500 | Label-filtered (demo path) | 259µs | 281µs |
| 1,000 | Label-filtered (demo path) | 469µs | 514µs |
| 1,000 | Vector-only ANN | 1.86ms | 2.08ms |

Insert throughput: **~59,000 states/s** (in-memory backend, 128-dim vectors).

Run the retrieval benchmark on your hardware:

```bash
python benchmarks/benchmark_retrieval.py
```

Run the synthetic world-model regression harness:

```bash
python benchmarks/world_model_harness.py --quick
```

For the existing LOCI-vs-naive-Qdrant comparison benchmark:

```bash
# In-memory (no Qdrant server needed):
python benchmarks/vs_naive_qdrant.py

# Against a live Qdrant server:
QDRANT_URL=http://localhost:6333 python benchmarks/vs_naive_qdrant.py
```

Results are written to `benchmarks/results/` and printed as markdown tables.
Before making performance claims, compare the current client on live Qdrant
against tuned range filters and a time-sharding-only baseline at equal recall.
See the [implementation readiness plan](docs/IMPLEMENTATION_READINESS.md).

## Research and Alternative Approaches

LOCI combines explicit geometric bounds, per-epoch routing, and retrieval using
predicted embeddings. Related systems such as SpatCode and TANNS deserve
task-matched evaluation; this repository does not establish superiority over
them or a comprehensive claim of research priority.

Collection-level temporal sharding is implemented. Automatic hot/warm/cold
storage migration is planned. Exact geometric filtering controls which results
are eligible; ANN candidate retrieval still requires recall measurement.
See [implementation and research claims](docs/NOVELTY.md).

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
- [docs/NOVELTY.md](docs/NOVELTY.md) — Implemented capabilities and research hypotheses
- [docs/IMPLEMENTATION_READINESS.md](docs/IMPLEMENTATION_READINESS.md) — Pilot scope, evidence, and delivery gates
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
