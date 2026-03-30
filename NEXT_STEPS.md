# Loci — Next Steps Plan

Status: v0.2 core is complete. This document prioritizes work to finish v0.2,
then move through v0.3 and v0.4 based on impact and existing code assets.

---

## Priority 1: Finish v0.2 — Robustness (remaining items)

### 1a. Connection pooling and retry logic
**Why now:** Production users will hit transient Qdrant failures immediately.
Without retries, a single network blip drops writes/reads silently.

- Add configurable retry decorator (exponential backoff, max 3 retries) to all
  Qdrant RPC calls in `client.py` and `async_client.py`
- Expose `max_retries` and `retry_backoff_base` in client constructors
- Consider connection health checks on client init
- Add tests using mock Qdrant that raises transient errors

### 1b. Shard lifecycle — warm/cold migration policy
**Why now:** Unbounded collection proliferation as epochs accumulate. Old shards
waste resources.

- Add `ShardPolicy` config: `warm_retention_ms`, `cold_action` (archive/delete)
- Implement `client.compact()` method that migrates or removes cold epochs
- For "archive" mode, merge cold epoch points into a single `loci_archive`
  collection (lower-resolution Hilbert IDs acceptable)
- For "delete" mode, drop the collection entirely
- Add background task variant for `AsyncLociClient`

---

## Priority 2: Integrate existing v0.3 code (high leverage — code already exists)

### 2a. Adaptive Hilbert resolution follow-up
**Status:** `AdaptiveResolution` is already wired into the clients behind the
optional `adaptive=True/False` flag, and `density_stats` is exposed for
inspection.

- Persist density stats across client restarts (serialize to Qdrant metadata
  collection or local file)
- Decide whether to flip the constructor default to `True` after persistence
  is in place
- Update README/examples to show `adaptive=True` when users want the feature

### 2b. Expose funnel search in client API
**Code exists:** `loci/retrieval/funnel.py` — `funnel_search()` is complete.
Cascades sequence -> frame -> patch. Not exposed via clients.

- Add `funnel_query()` method to `LociClient` and `AsyncLociClient`
- Wire through to `funnel_search()` with proper shard fan-out
- Add to `LocalLociClient` using memory backend
- Add integration tests and an example in `examples/`

---

## Priority 3: New v0.3 features — Performance

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
- Automate with a benchmark CI job (nightly or manual trigger)
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

### 5c. CI hardening
- Make mypy non-optional (remove `|| true` from CI)
- Add ruff/black for formatting
- Add coverage reporting (pytest-cov) with minimum threshold
- Add integration test job that runs against real Qdrant (Docker service)

---

## Suggested execution order

| Step | Item | Effort | Impact |
|------|------|--------|--------|
| 1 | 2a — Wire adaptive resolution | Small | High (code exists) |
| 2 | 2b — Expose funnel search | Small | High (code exists) |
| 3 | 1a — Connection retry logic | Medium | High (reliability) |
| 4 | 5b — Error handling audit | Small | Medium (correctness) |
| 5 | 5c — CI hardening | Small | Medium (quality) |
| 6 | 3a — Result caching | Medium | Medium (performance) |
| 7 | 1b — Shard lifecycle | Medium | Medium (operations) |
| 8 | 3b — Batch predict-then-retrieve | Medium | Medium (API) |
| 9 | 4a — Cross-scale causal linking | Large | High (differentiation) |
| 10 | 4b — Scale-aware decay | Small | Medium |
| 11 | 3c — Competitive benchmarks | Medium | High (marketing) |
| 12 | 5a — Observability | Medium | High (production) |

The first two items are highest-leverage: the code is already written and tested
in isolation — they just need to be wired into the client API. This gives
immediate feature uplift with minimal risk.
