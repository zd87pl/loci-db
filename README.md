# LOCI

**Spatiotemporal memory for AI agents.** LOCI stores world-model embeddings with a 4D address — position (x, y, z) + timestamp — so your robot can answer *"what did I see near here, and when?"* instead of just *"what is this similar to?"*

[![CI](https://github.com/zd87pl/loci-db/actions/workflows/ci.yml/badge.svg)](https://github.com/zd87pl/loci-db/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI: planned v1.0](https://img.shields.io/badge/PyPI-planned%20v1.0-lightgray.svg)](ROADMAP.md)

```python
from loci import LocalLociClient, WorldState
import time

client = LocalLociClient(vector_size=512)  # zero infrastructure; install: pip install -e .

# Every observation stores WHERE and WHEN, not just WHAT
client.insert(WorldState(x=0.5, y=0.3, z=0.0,
                         timestamp_ms=int(time.time() * 1000),
                         vector=my_world_model_embedding,
                         scene_id="warehouse"))

# Query: what did the robot see near (0.5, 0.3) in the last 5 seconds?
results = client.query(vector=query_vec,
                       spatial_bounds={"x_min": 0.3, "x_max": 0.7,
                                       "y_min": 0.1, "y_max": 0.5,
                                       "z_min": 0.0, "z_max": 1.0},
                       time_window_ms=(now_ms - 5000, now_ms))

# Predict then retrieve: what does my world model predict, and have I seen it before?
result = client.predict_and_retrieve(context_vector=obs,
                                     predictor_fn=my_world_model,
                                     current_position=(0.5, 0.3, 0.0))
print(f"Novelty: {result.prediction_novelty:.2f}")  # 0.0 familiar → 1.0 new
```

---

## Start here

| Resource | Description |
|----------|-------------|
| [**Getting Started notebook**](notebooks/getting_started.ipynb) | 10-minute walkthrough — warehouse robot scenario, no Qdrant needed |
| [**Docker Compose demo**](demo/README.md) | Interactive warehouse robot demo: `docker compose up` |
| [**Benchmark results**](#performance) | LOCI vs naive Qdrant — when does spatiotemporal indexing help? |
| [**Architecture deep-dive**](ARCHITECTURE.md) | Hilbert bucketing, temporal sharding, novelty detection internals |
| [**World model integration guide**](docs/WORLD_MODEL_INTEGRATION.md) | V-JEPA 2, DreamerV3, and generic adapter examples |

---

## Why LOCI?

Modern world models — V-JEPA 2, DreamerV3, GAIA-1, UniSim — produce embeddings where every vector has an **implicit 4D spatiotemporal address** `(x, y, z, t)`. Existing vector databases treat all embedding dimensions equally: spatial queries require 3+ independent float-range payload filters, there is no native time sharding, and there is no primitive for *"predict the future then find what's nearby."*

LOCI makes spatiotemporal structure **first-class** through three primitives:

### 1. Multi-Resolution Hilbert Bucketing

Encode `(x, y, z)` at multiple Hilbert resolutions (p=4, 8, 12). A bounding-box spatial query becomes a **single integer set-membership filter** instead of three independent float comparisons.

```
      Naive Qdrant                     LOCI
┌──────────────────────┐     ┌──────────────────────┐
│ x_min ≤ x ≤ x_max    │     │                      │
│ y_min ≤ y ≤ y_max    │ →   │  hilbert_r4 ∈ {…}    │
│ z_min ≤ z ≤ z_max    │     │  (single set filter) │
└──────────────────────┘     └──────────────────────┘
```

With `adaptive=True`, dense spatial regions are automatically promoted to finer Hilbert resolutions at query time.

### 2. Temporal Epoch Sharding

Vectors are automatically routed to **time-partitioned collections** (`loci_{epoch_id}`). Queries fan out only to epochs that overlap the requested time window — with the async client, all shards are searched **concurrently** via `asyncio.gather`. Temporal decay scoring penalises older results gently.

### 3. Predict-then-Retrieve with Novelty Detection

```python
result = client.predict_and_retrieve(
    context_vector=current_embedding,
    predictor_fn=my_world_model,        # any callable: vector → vector
    future_horizon_ms=2000,
    current_position=(0.5, 0.3, 0.8),
)
# result.results        — stored states matching the prediction
# result.prediction_novelty  — 0.0 familiar, 1.0 new territory
```

This is LOCI's analogue to HyDE (ACL 2023) for spatiotemporal world models — the robot's hippocampus, not just a filing cabinet.

---

## Quick Start

### In-memory (zero infrastructure)

> **Note:** loci-db is not yet published to PyPI (planned for v1.0). Install from source:

```bash
git clone https://github.com/zd87pl/loci-db.git
cd loci-db
pip install -e .
```

```python
from loci import LocalLociClient, WorldState
import time

client = LocalLociClient(vector_size=512)

state = WorldState(
    x=0.5, y=0.3, z=0.8,
    timestamp_ms=int(time.time() * 1000),
    vector=[0.1] * 512,
    scene_id="my_scene",
    scale_level="patch",   # "patch" | "frame" | "sequence"
    confidence=0.95,
)
state_id = client.insert(state)

# Spatiotemporal query
results = client.query(
    vector=[0.1] * 512,
    spatial_bounds={"x_min": 0.0, "x_max": 1.0,
                    "y_min": 0.0, "y_max": 1.0,
                    "z_min": 0.0, "z_max": 1.0},
    time_window_ms=(int(time.time() * 1000) - 5000,
                    int(time.time() * 1000)),
    limit=10,
)

# Walk causal chain
trajectory = client.get_trajectory(state_id, steps_back=10, steps_forward=10)
```

> See [notebooks/getting_started.ipynb](notebooks/getting_started.ipynb) for a full walkthrough.

### With Qdrant (production)

```bash
# Install from source (see above)
docker run -p 6333:6333 qdrant/qdrant
```

```python
from loci import LociClient

client = LociClient("http://localhost:6333", vector_size=512, epoch_size_ms=5000)
# Same API as LocalLociClient — swap without changing application code
```

### One-command demo (Docker Compose)

```bash
git clone https://github.com/zd87pl/loci-db.git
cd loci-db
docker compose up
```

Open **http://localhost:8000** — interactive warehouse robot demo with live novelty detection.

### Async API (parallel shard fan-out)

```python
from loci import AsyncLociClient

async with AsyncLociClient("http://localhost:6333", vector_size=512) as client:
    await client.insert(state)
    results = await client.query(vector=query_vec, limit=10)
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

# Generic numpy / torch
adapter = GenericAdapter(expected_dim=512)
ws = adapter.from_numpy(embedding, position=(0.5, 0.3, 0.8), timestamp_ms=ts, scene_id="env")
```

---

## Performance

### Benchmark Results

Benchmark: LOCI vs naive Qdrant (single collection, 3 independent float-range payload filters).  
Backend: Qdrant in-memory client. Results from `benchmarks/results/latest.json`.  
Config: 512-dim vectors, 50 queries (5 warmup, 3 runs), `epoch_size_ms=5000`. Seed: 42.  
To reproduce on your hardware: `python benchmarks/vs_naive_qdrant.py` (see [instructions below](#reproduce-on-your-hardware)).

**Query scenarios:**

| ID | Description | Spatial radius | Time window |
|----|-------------|---------------|-------------|
| A  | Tight spatial query | 0.05 (narrow box) | Full dataset |
| B  | Wide spatial query | 0.50 (half the space) | Full dataset |
| C  | Tight spatial + temporal | 0.05 | 10% of dataset span |
| D  | Broader spatial, short window | 0.30 | 10% of dataset span |

**N = 1,000 vectors** (in-memory Qdrant):

| Scenario | Method | p50 (ms) | p99 (ms) | QPS | Recall@10 |
|----------|--------|----------|----------|-----|-----------|
| A (tight spatial) | Naive Qdrant | 4.2 | 5.0 | 232 | 1.000 |
| A | LOCI r4 | 20.5 | 27.0 | 50 | 0.990 |
| B (wide spatial) | Naive Qdrant | 8.0 | 27.2 | 118 | 1.000 |
| B | LOCI r4 | 187.5 | 279.3 | 5 | 1.000 |
| C (spatial+temporal) | Naive Qdrant | 4.3 | 5.2 | 227 | 1.000 |
| C | LOCI r4 | 9.2 | 14.0 | 109 | 1.000 |
| D (stress test) | Naive Qdrant | 6.3 | 7.5 | 156 | 1.000 |
| D | LOCI r4 | 49.4 | 85.4 | 19 | 0.982 |

**N = 10,000 vectors** (in-memory Qdrant):

| Scenario | Method | p50 (ms) | p99 (ms) | QPS | Recall@10 |
|----------|--------|----------|----------|-----|-----------|
| A (tight spatial) | Naive Qdrant | 43.6 | 112.0 | 22 | 1.000 |
| A | LOCI r4 | 162.3 | 270.3 | 6 | 1.000 |
| B (wide spatial) | Naive Qdrant | 75.1 | 99.7 | 13 | 1.000 |
| B | LOCI r4 | 1,182.0 | 1,558.1 | 0.9 | 1.000 |
| **C (spatial+temporal)** | **Naive Qdrant** | **44.9** | **75.2** | **22** | **1.000** |
| **C** | **LOCI r4** | **22.6** | **30.3** | **44** | **0.990** |
| D (stress test) | Naive Qdrant | 63.2 | 115.5 | 14 | 1.000 |
| D | LOCI r4 | 116.1 | 217.4 | 8 | 1.000 |

**When LOCI wins:** Scenario C at N=10,000 — tight spatial bounds combined with a narrow time window. LOCI returns results **2× faster** because epoch sharding eliminates 90% of data before the Hilbert filter runs, and the Hilbert pre-filter then reduces candidates to a single integer comparison. The advantage grows with N.

**When naive Qdrant wins:** Wide spatial queries (Scenario B) and small datasets (N ≤ 1,000) — the Hilbert encoding overhead is not amortised when filtering removes few candidates.

> Full per-run data: `benchmarks/results/latest.json`.  
> Methodology: [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md).

### Reproduce on your hardware

```bash
# In-memory (no Qdrant server needed):
python benchmarks/vs_naive_qdrant.py

# Against a live Qdrant server:
QDRANT_URL=http://localhost:6333 python benchmarks/vs_naive_qdrant.py

# Local in-memory backend (insert throughput + query latency):
python benchmarks/local_benchmark.py
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      Application Layer                    │
│  LociClient / AsyncLociClient / LocalLociClient           │
│  insert · insert_batch · query · predict_and_retrieve     │
│  get_trajectory · get_causal_context · funnel_query       │
├───────────────────────────────────────────────────────────┤
│                      Retrieval Layer                      │
│  predict.py  — predict-then-retrieve + novelty scoring    │
│  funnel.py   — multi-scale coarse→fine search             │
├───────────────────────────────────────────────────────────┤
│                  Indexing & Routing Layer                 │
│  spatial/    — multi-res Hilbert bucketing + overlap      │
│  temporal/   — epoch sharding + decay re-ranking          │
├───────────────────────────────────────────────────────────┤
│                      Adapters Layer                       │
│  V-JEPA 2 · DreamerV3 · Generic numpy/torch              │
├───────────────────────────────────────────────────────────┤
│                      Storage Layer                        │
│  Qdrant (one collection per temporal epoch)               │
│  MemoryStore (in-process, no infrastructure required)     │
└───────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document with internals.

---

## Comparison

### vs SpatCode (WWW 2026)

SpatCode encodes coordinates into embedding space for soft/fuzzy retrieval via RoPE-style positional encoding. LOCI uses Hilbert bucketing for **exact geometric range queries**.

- **Use SpatCode** when semantic proximity matters (*"find images taken near this location"*).
- **Use LOCI** when physical boundaries matter (*"find all observations within this 3D bounding box in the last 5 seconds"*).

### vs TANNS (ICDE 2025)

TANNS builds a single graph managing timestamps with a Timestamp Graph structure. LOCI uses collection-level sharding with storage tiering.

- **Use TANNS** for single-session temporal ANN where all data fits in one graph.
- **Use LOCI** when you need cross-session persistence, multi-agent memory sharing, or predict-then-retrieve.

---

## Documentation

| Document | Audience |
|----------|----------|
| [notebooks/getting_started.ipynb](notebooks/getting_started.ipynb) | Researchers — first contact |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Contributors — system internals |
| [docs/NOVELTY.md](docs/NOVELTY.md) | Reviewers — novelty claims vs prior art |
| [docs/BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) | Researchers — benchmark replication |
| [docs/WORLD_MODEL_INTEGRATION.md](docs/WORLD_MODEL_INTEGRATION.md) | Practitioners — integration guides |

---

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

Current version: **v0.3** (performance). Full detail in [ROADMAP.md](ROADMAP.md).

| Version | Status | Focus |
|---------|--------|-------|
| v0.1 | ✅ Done | Core primitives: WorldState, Hilbert encoding, temporal sharding, predict-then-retrieve |
| v0.2 | ✅ Done | Async client, causal chain linking, configurable distances, 70+ tests |
| v0.3 | 🚧 Current | Adaptive Hilbert resolution, funnel search, in-progress: result caching, cross-system benchmarks |
| v0.4 | Planned | Cross-scale causal linking, scale-aware temporal decay |
| v1.0 | Planned | PyPI release, gRPC transport, authentication, Kubernetes Helm chart, OpenTelemetry |

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
