# Engram Roadmap

## v0.1 — Foundation

- [x] WorldState data model with validation
- [x] Hilbert curve spatial encoding (4D)
- [x] Temporal sharding with epoch-based collections
- [x] EngramClient: insert, insert_batch, query
- [x] Predict-then-retrieve primitive
- [x] Temporal decay scoring
- [x] Basic test suite

## v0.2 — Robustness

- [x] AsyncEngramClient with parallel shard fan-out
- [x] Causal chain linking in insert and insert_batch
- [x] Configurable distance metrics (cosine, dot, euclidean)
- [x] Input validation (confidence, timestamps, spatial bounds)
- [x] py.typed marker for downstream type checking
- [x] CI pipeline (GitHub Actions, Python 3.11 + 3.12)
- [x] Comprehensive test suite (70+ tests)
- [x] Connection retry logic with exponential backoff
- [ ] Shard lifecycle: warm → cold migration policy

## v0.3 — Performance (current)

- [x] Integrate adaptive Hilbert resolution into clients (density tracking + stats)
- [x] Integrate funnel search into client API (`funnel_query()` on all clients)
- [ ] Result caching for repeated spatial queries
- [ ] Benchmarks against Milvus and Weaviate spatial filters
- [ ] Batch predict-then-retrieve (multiple context vectors)

## v0.4 — Multi-Scale

- [ ] Cross-scale causal linking
- [ ] Scale-aware temporal decay

## v1.0 — Production Ready

- [ ] gRPC transport option
- [ ] Authentication and multi-tenancy
- [ ] Observability (OpenTelemetry traces, Prometheus metrics)
- [ ] Helm chart for Kubernetes deployment
- [ ] Published to PyPI
