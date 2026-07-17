# Loci — Next Steps Plan

Status: v0.3 is on PyPI (`loci-stdb`). Connection retry logic, funnel search
exposure (`funnel_query()` on all three clients), and adaptive Hilbert
resolution are shipped. This document prioritizes the remaining work.

---

## Priority 1: Deferred architecture refactors

The remaining known structural debt (see "Known Limitations and Planned
Refactors" in [ARCHITECTURE.md](ARCHITECTURE.md)):

### 1a. Bounded epoch storage — DONE
Shipped: each tenant/store now uses exactly two collections
(`{prefix}loci_data` for raw states, `{prefix}loci_summary` for consolidated
summaries) with payload-indexed timestamps and Hilbert buckets. Epochs are
purely logical (consolidation granularity + Hilbert t-normalisation), time
selectivity comes from the indexed `timestamp_ms` range filter, and every
operation is O(1) in collection count. Existing per-epoch deployments migrate
with `loci migrate-layout` (dry-run, verified copy, optional `--delete-old`).

### 1b. Shared client core (sync/async/local parity)
**Why:** `LociClient`, `AsyncLociClient`, and `LocalLociClient` duplicate
roughly 77% of their logic; every fix must be applied three times and parity
drifts.

- Extract query planning, filter construction, and result assembly into a
  shared core module
- Reduce the three clients to thin transport shells (sync Qdrant, async
  Qdrant, in-process MemoryStore)
- Add a parity test suite that runs the same scenarios against all three

---

## Priority 2: Packaging follow-up

### 2a. Decide the loci-core distribution story
The native Rust crate is currently a local-only uv dependency group
(`uv sync --group native`; pip users: `pip install -e ./loci-core`) because
`loci-core` is not registered on PyPI — publishing an extra that depends on an
unregistered name would be a dependency-confusion vector. Either:

- Register `loci-core` on PyPI (even as a placeholder) and publish wheels via
  maturin, then reintroduce a real `[native]` extra, **or**
- Keep it group-only and document that as the supported path

### 2b. Adaptive resolution persistence
- Persist `density_stats` across client restarts (Qdrant metadata collection
  or local file)
- Decide whether to flip the constructor default to `adaptive=True` once
  persistence lands

---

## Priority 3: Performance

### 3a. Result caching for repeated spatial queries
- LRU cache keyed on `(hilbert_ids_frozenset, time_window, top_k, distance_metric)`
- TTL-based expiry (configurable, default 5s)
- Cache invalidated on insert to overlapping region
- Bounded memory (max 1000 entries default)

### 3b. Batch predict-then-retrieve
- Accept `list[vector]` + `predictor_fn` → fan out predictions in parallel
- For async client, run all predictions concurrently via `asyncio.gather`
- Deduplicate overlapping time windows across predictions before querying
- Return results grouped by input context vector

### 3c. Competitive benchmarks (Milvus, Weaviate)
- Extend `benchmarks/` with comparable setups against Milvus and Weaviate
  spatial filter queries
- Measure: query latency (p50/p95/p99), insert throughput, memory usage
- Publish results in a `BENCHMARKS.md` doc

---

## Priority 4: v0.4 — Multi-Scale

### 4a. Cross-scale causal linking
- Extend causal chain to link across scale levels (e.g., a `sequence` state
  links to its constituent `frame` states)
- Add `parent_state_id` / `child_state_ids` fields to `WorldState`
- Update `insert_batch` to detect and link hierarchical relationships

### 4b. Scale-aware temporal decay
- Different decay rates per scale level (sequences decay slower than patches)
- Configurable `decay_lambda_map: dict[str, float]`
- Integrate into existing decay scoring in `temporal/decay.py`

---

## Priority 5: Production hardening (v1.0 prep)

### 5a. Observability
- Add OpenTelemetry spans to insert/query/predict_and_retrieve
- Prometheus counters: queries_total, inserts_total, errors_total
- Histogram: query_latency_seconds, insert_latency_seconds
- Structured logging (replace bare `logging` with structlog)

### 5b. Error handling audit
- Audit all `except` blocks — several silently swallow errors (especially in
  causal linking predecessor lookup)
- Surface errors as warnings or raise custom `LociError` hierarchy
- Add `LociConnectionError`, `LociValidationError`, `LociQueryError`

### 5c. CI hardening (remaining)
- Add coverage reporting (pytest-cov) with a minimum threshold
- Extend the Docker smoke test into a fuller integration suite against real
  Qdrant (the CI `docker-smoke` job currently covers health/insert/query)

---

## Suggested execution order

| Step | Item | Effort | Impact |
|------|------|--------|--------|
| 1 | 2a — loci-core distribution decision | Small | High (unblocks native path) |
| 2 | 1a — Bounded epoch storage | Done | Shipped (two-collection layout + `loci migrate-layout`) |
| 3 | 1b — Shared client core | Large | High (correctness/parity) |
| 4 | 5b — Error handling audit | Small | Medium (correctness) |
| 5 | 3a — Result caching | Medium | Medium (performance) |
| 6 | 3b — Batch predict-then-retrieve | Medium | Medium (API) |
| 7 | 4a — Cross-scale causal linking | Large | High (differentiation) |
| 8 | 4b — Scale-aware decay | Small | Medium |
| 9 | 3c — Competitive benchmarks | Medium | High (positioning) |
| 10 | 5a — Observability | Medium | High (production) |
