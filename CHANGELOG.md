# Changelog

All notable changes to loci-db are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
loci-db uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `SECURITY.md` for open-source readiness.
- `CORS_ORIGINS` env var for `demo_spatial` — replaces hardcoded wildcard.
- Bounded `limit` (1–1000) and `overlap_factor` (0–10) fields in `QueryRequest` via Pydantic `Field` constraints.
- Vector size validation on `/query` endpoint (mirrors existing `/insert` check).
- LRU eviction (`max_cache_size`, default 4096) for `AdaptiveResolution._resolution_cache`.
- `.gitignore` entries for secrets, IDE files, and coverage artifacts.
- Integration tests for `server.py` REST endpoints (`/health`, `/insert`, `/query`).

### Fixed
- Consistent nearest-first Euclidean ranking across sync, async, and local clients.
- Retention invalidates deleted collections; readers discover shards added or removed by other writers.
- Cloud insertion/query preserves metadata, scale, confidence, and causal links; both SDKs forward confidence filters.
- Exact spatial/confidence predicates apply before backend top-k selection, preventing avoidably empty results.
- Query backend failures propagate instead of silently producing incomplete results; unsupported cloud operations raise explicit errors.
- Dependency-audit CI installs the application and cloud runtime dependencies before scanning.
- `InsertRequest.metadata` mutable default replaced with `Field(default_factory=dict)` to prevent cross-request data leakage.

### Upgrade notes
- Euclidean raw scores now consistently use negative distance; decayed scores use reciprocal distance similarity multiplied by recency weight.
- Existing Qdrant collections need FLOAT indexes on `x`, `y`, `z`, and `confidence`. Deploy the cloud API before or with the updated SDK. See [implementation readiness](docs/IMPLEMENTATION_READINESS.md) for the scoped backfill and pilot validation plan.

---

## [0.3.0] — 2026-03-01

### Added
- ADR-3: Spatial deduplication cross-frame NMS pipeline.
- ADR-2: Cross-frame temporal consensus buffer.
- ADR-1: Confidence filtering pipeline.
- Precomputed numpy Hilbert LUT replacing itertools enumeration.
- Retrieval benchmark and vectorised `MemoryStore`.

---

## [0.2.0] — 2025-12-01

### Added
- Phase A demo: `demo_spatial` assistive AI backend (voice + camera + WebSocket).
- `LociClient.insert` / `LociClient.query` high-level API.
- FastAPI REST server (`server.py`).

---

## [0.1.0] — 2025-10-01

### Added
- Initial release: `WorldState`, `MemoryStore`, Qdrant integration, Hilbert curve spatial indexing.
