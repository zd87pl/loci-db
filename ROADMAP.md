# Loci Roadmap

## v0.1 — Foundation

- [x] WorldState data model with validation
- [x] Hilbert curve spatial encoding (4D)
- [x] Temporal epoch model (since superseded: epochs are now purely logical
      over the bounded two-collection layout)
- [x] LociClient: insert, insert_batch, query
- [x] Predict-then-retrieve primitive
- [x] Temporal decay scoring
- [x] Basic test suite

## v0.2 — Robustness

- [x] AsyncLociClient with concurrent search execution
- [x] Causal chain linking in insert and insert_batch
- [x] Configurable distance metrics (cosine, dot, euclidean)
- [x] Input validation (confidence, timestamps, spatial bounds)
- [x] py.typed marker for downstream type checking
- [x] CI pipeline (GitHub Actions, Python 3.11 + 3.12)
- [x] Comprehensive test suite (70+ tests)
- [x] Connection retry logic with exponential backoff
- [x] Warm → cold data aging (delivered as memory consolidation: stale raw
      epochs fold into the summary collection; see v0.4)

## v0.3 — Performance (current)

- [x] Integrate adaptive Hilbert resolution into clients (density tracking + stats)
- [x] Integrate funnel search into client API (`funnel_query()` on all clients)
- [ ] Result caching for repeated spatial queries
- [ ] Benchmarks against Milvus and Weaviate spatial filters
- [ ] Batch predict-then-retrieve (multiple context vectors)

## v0.4 — Multi-Scale & Memory Semantics

Direction set by [RFC-0001](docs/RFC-0001-memory-for-world-models.md)
("the memory system for world models"): invest above the filter layer.

- [x] Memory consolidation v1 — episodic→semantic aging on all three clients
      (`ConsolidationPolicy`; RFC-0001 R1)
- [x] MCP server — spatial memory for agents (`loci-stdb[mcp]`, `loci-mcp`;
      RFC-0001 P2)
- [x] Consolidation for the Qdrant clients (sync + async)
- [ ] Multi-tier consolidation fidelity curves (raw → k-centroids → scene digest)
- [ ] Conformal novelty — false-alarm-rate guarantees (RFC-0001 R2)
- [ ] World-model memory benchmark suite (RFC-0001 R3)
- [ ] Adaptive query planner: Hilbert range-runs vs buckets vs naive filters
      chosen per query (RFC-0001 R4)
- [ ] Cross-scale causal linking
- [ ] Scale-aware temporal decay

## Deferred architecture refactors

Known structural debts, documented in ARCHITECTURE.md ("Known Limitations and
Planned Refactors"):

- [x] Bounded epoch storage — **done**: exactly two collections per tenant
      (`{prefix}loci_data` + `{prefix}loci_summary`) with payload-indexed
      timestamps and Hilbert buckets; epochs are now purely logical and every
      operation is O(1) in collection count. Existing per-epoch deployments
      migrate with `loci migrate-layout`
- [ ] Shared client core — extract common query planning, filter construction,
      and result assembly to eliminate the ~77% duplicated logic across
      `LociClient` / `AsyncLociClient` / `LocalLociClient`
- [ ] loci-core distribution — register `loci-core` on PyPI and publish maturin
      wheels (restoring a safe `[native]` extra), or commit to the local-only
      dependency-group install path

## v1.0 — Production Ready

- [ ] gRPC transport option
- [x] Authentication and multi-tenancy (cloud API: API keys, per-tenant namespaces, rate limits)
- [ ] Observability (OpenTelemetry traces, Prometheus metrics)
- [ ] Helm chart for Kubernetes deployment
- [x] Published to PyPI (as `loci-stdb`)
