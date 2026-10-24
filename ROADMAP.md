# Loci Roadmap

## Immediate Priority — One Customer Pilot

Version milestones below track features, not production readiness or validated
customer value. Follow [IMPLEMENTATION_READINESS.md](docs/IMPLEMENTATION_READINESS.md)
before expanding scope. Robot incident retrieval and regression curation is a
provisional first workflow; confirm it with an existing user or partner.

- [ ] Agree one workflow, dataset, buyer, acceptance thresholds, and paid continuation decision
- [ ] Define coordinate frames, embedding versions, timestamps, and source provenance
- [ ] Verify failure handling, persistence/recovery, concurrent writers, and expected scale
- [ ] Benchmark the current client on live Qdrant against tuned filters and time-sharding-only at equal recall
- [ ] Evaluate a real predictor on held-out episodes against retrieval without prediction
- [ ] Measure incident investigation time and useful regression cases with pilot users
- [ ] Proceed only when agreed reliability, quality, cost, and commercial gates pass

## v0.1 — Foundation

- [x] WorldState data model with validation
- [x] Hilbert curve spatial encoding (4D)
- [x] Temporal sharding with epoch-based collections
- [x] LociClient: insert, insert_batch, query
- [x] Predict-then-retrieve primitive
- [x] Temporal decay scoring
- [x] Basic test suite

## v0.2 — Robustness

- [x] AsyncLociClient with parallel shard fan-out
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
- [x] Authentication and multi-tenancy (cloud API: API keys, per-tenant namespaces, rate limits)
- [ ] Observability (OpenTelemetry traces, Prometheus metrics)
- [ ] Helm chart for Kubernetes deployment
- [x] Published to PyPI (as `loci-stdb`)
