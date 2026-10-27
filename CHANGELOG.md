# Changelog

All notable changes to loci-db are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
loci-db uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] — Unreleased

### Changed — BREAKING
- **Storage layout: bounded two-collection design** (RFC-0001 P1). Each
  tenant/store now uses exactly two collections — `{prefix}loci_data` for raw
  states (payload-indexed `hilbert_r4/r8/r12`, `timestamp_ms`, `scene_id`,
  `scale_level`) and `{prefix}loci_summary` for consolidated summaries —
  replacing the old one-collection-per-epoch layout (`loci_{epoch}` /
  `loci_sum_{coarse}`, which grew by ~17k collections per day of continuous
  ingest at the default 5s epoch). Epochs are now purely logical (the unit of
  consolidation granularity and Hilbert t-normalization); time selectivity
  comes from an indexed `timestamp_ms` range filter, and every operation is
  O(1) in collection count. **Existing deployments must run
  `loci migrate-layout` once** to copy legacy per-epoch collections into the
  new layout before upgrading their clients.
- **`RetentionPolicy.archive_callback` removed.** Retention is now a
  filter-based delete of raw points older than an epoch-aligned cutoff
  (`max_epochs` / `max_age_ms` both reduce to a cutoff timestamp); there is no
  archive hook. Use a `ConsolidationPolicy` to age old epochs into summaries
  instead of losing them.
- **`RetentionManager` and consolidation helper APIs changed** for direct
  callers: `RetentionManager.maybe_purge(now_ms, delete_before)` takes an
  injected cutoff-deleter instead of enumerating epoch collections, and the
  per-epoch/per-coarse collection-name helpers are replaced by
  `data_collection_name()` / `summary_collection_name()` /
  `coarse_time_range()` / `fold_cutoff_ms()` in `loci.temporal.consolidation`.

### Added
- **`loci migrate-layout` CLI command**: one-shot, idempotent migration of a
  legacy per-epoch Qdrant deployment to the bounded layout. Discovers legacy
  collections under a `--prefix` (other tenants untouched), copies every point
  verbatim (same ids, vectors, payloads) into `{prefix}loci_data` /
  `{prefix}loci_summary` (created with the same payload indexes the clients
  create), supports `--dry-run` (plan only), and only drops
  legacy collections with `--delete-old` after a per-collection point-count
  verification. Safe to re-run: already-copied points are skipped.
- **Memory consolidation** (RFC-0001 R1, all three clients): a
  `ConsolidationPolicy` that summarizes epochs older than a raw window into
  per-scene centroid states stored in the single summary collection instead of
  deleting them. Queries include summaries transparently (marked with
  `metadata["consolidated"]`); trajectories and causal context ignore them;
  storage stays bounded while old data remains findable.
- **MCP server** (RFC-0001 P2): `pip install "loci-stdb[mcp]"` + `loci-mcp`
  exposes LOCI as a Model Context Protocol server — `remember` / `recall` /
  `novelty` / `trajectory` / `memory_stats` over local, Qdrant, or cloud
  backends. See `docs/MCP_SERVER.md`.
- **RFC-0001** (`docs/RFC-0001-memory-for-world-models.md`): strategic
  direction and R&D roadmap to v1.0.

### Correctness fixes
- Temporal decay now defaults to a **one-hour half-life** (`DEFAULT_DECAY_LAMBDA`, derived via `lambda_from_half_life()`); the decay exponent is clamped so very old results degrade to similarity-order instead of collapsing to a 0.0 tie. Derive custom rates from a half-life rather than setting the per-millisecond `decay_lambda` directly.
- `predict_and_retrieve` novelty is now computed on an absolute scale: 0 = strong historical analog, 1 = no analog, independent of the rest of the result batch.
- Consistent nearest-first Euclidean ranking across the sync, async and local clients: the raw `score` is `-distance` (higher is better) and `decayed_score` is `1 / (1 + distance) × exp(-λ × age)`, so temporal decay can no longer promote a farther, older match (previously the negative raw score was multiplied by the recency weight, which inverted the order), and re-ranking and novelty behave consistently across distance metrics.
- Exact spatial, temporal and confidence predicates (`x`/`y`/`z`, `timestamp_ms` and `confidence` Range conditions) are part of the backend filter on both `loci_data` and `loci_summary` **before** top-k selection, so filtered queries are no longer avoidably empty when ineligible decoys share a Hilbert bucket; the geometric post-filter remains the final check. New collections get FLOAT payload indexes on `x`, `y`, `z` and `confidence`.
- Query backend failures now propagate (once the retry policy gives up) instead of being logged and folded into a partial or empty result set. The one tolerated case is a collection deleted out from under a client: the query returns an empty result and the client's readiness cache is reset so the collection is re-probed on the next call.
- Scroll pagination advances correctly across pages in multi-page scans.
- `min_confidence` filtering overfetches candidates so filtered queries can still fill the requested `limit`.
- LRU eviction (`max_cache_size`, default 4096) for `AdaptiveResolution._resolution_cache`.

### Removed
- The Fly.io managed stack. Removed: `cloud/api` (FastAPI service with Supabase API keys and Qdrant Cloud), the `cloud/edge` proxy Worker, the Supabase migrations, the AWS Terraform scaffold, the `cloud-api.yml` Fly deploy workflow and the `cloud-tests` CI job. LOCI Cloud runs on Cloudflare only (`cloud/workers`).
- The `loci cloud keys` CLI commands, which managed keys on that service. Cloudflare projects and keys come from `cloud/workers/scripts/bootstrap.py`.

### Cloud API
- State metadata now survives the insert → query round-trip.
- `scale_level`, `confidence` and the causal links (`prev_state_id` / `next_state_id`) round-trip through `/insert` and `/query`.
- `/query` accepts `min_confidence`, and both SDKs (`LociClient` in cloud mode and `AsyncLociClient`) forward it instead of raising.
- Retention cache correctness fix.
- Stricter namespace validation.

### Servers
- Root REST server (`server.py`): validation errors return **422** (parity with FastAPI request validation); `/query` spatial bounds are now optional — omit them to search everywhere.
- Request body size is capped.
- Vector size validation on `/query` (mirrors the `/insert` check); bounded `limit` (1–1000) and `overlap_factor` (0–10) via Pydantic `Field` constraints.
- `InsertRequest.metadata` mutable default replaced with `Field(default_factory=dict)` to prevent cross-request data leakage.
- `CORS_ORIGINS` env var for `demo_spatial` — replaces hardcoded wildcard; fixed an XSS vector in the `demo_spatial` frontend.
- Research `CodeRunner` now executes generated code sandboxed.
- Integration tests for `server.py` REST endpoints (`/health`, `/insert`, `/query`).

### Packaging & CI
- **Removed the published `[native]` extra** — it depended on `loci-core`, an unregistered PyPI name (dependency-confusion vector). The native bindings are now a local uv dependency group: `uv sync --group native`; pip users should `pip install -e ./loci-core`.
- Root `Dockerfile` now builds `loci` from the working tree instead of installing the last PyPI release under HEAD's `server.py` (which broke `/insert` in the compose quick start).
- `demo/Dockerfile` copies `README.md` so the hatchling build no longer aborts.
- Removed stale root `fly.toml` (duplicate of `cloud/fly.toml` without health checks; a root `flyctl deploy` would have clobbered production config).
- `benchmarks/docker-compose.yml` binds Qdrant to 127.0.0.1 and pins the image tag.
- CI: dependency audit now installs the project (library + dev + cloud API deps) before `pip-audit`; new Rust workflow runs `cargo test` for `loci-core`; the Fly deploy gates on the full library suite + ruff + mypy, not only cloud tests; new `docker-smoke` job builds the root image and exercises `/health`, `/insert`, `/query` against Qdrant; workflows declare least-privilege `permissions: contents: read`; publishing verifies the release tag matches the package version; mypy runs without a global `--ignore-missing-imports` (per-module overrides in `pyproject.toml`).
- `.gitignore` entries for secrets, IDE files, and coverage artifacts.

### Docs
- README repositioned around **spatial memory for Physical AI**: banner, demo GIF, a verified 60-second in-process quick start, use cases, and a one-line MCP setup. Adds `CITATION.cff`, issue/PR templates, and a social-preview image (`docs/assets/`); the internal project review and next-steps plan moved under `docs/`.
- README performance section rewritten from the actual benchmark artifact (`benchmarks/results/retrieval_latest.json`), including the spatial and spatial+temporal numbers; Docker quick-start examples now send correctly-sized vectors.
- `docs/NOVELTY.md` and `docs/BENCHMARK_METHODOLOGY.md` corrected to the measured results in `benchmarks/results/latest.json` (naive Qdrant is faster on pure spatial filtering; LOCI's measured win is combined tight spatial+temporal at N=10k, ~2x).
- `ARCHITECTURE.md`: documents all three clients (incl. `LocalLociClient`/`MemoryStore`), the actual `predict_and_retrieve` default (searches all stored history; explicit window optional), and a new "Known Limitations and Planned Refactors" section.
- `SECURITY.md` points to GitHub private vulnerability reporting; `docs/WORLD_MODEL_INTEGRATION.md` fixes the install name (`loci-stdb`); `NEXT_STEPS.md` (now `docs/NEXT_STEPS.md`) and `ROADMAP.md` refreshed with the deferred refactors.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `SECURITY.md` added for open-source readiness.
- `docs/IMPLEMENTATION_READINESS.md` (new, linked from `ROADMAP.md`): pilot scope, ingestion data contracts, reliability checks, live-Qdrant benchmark protocol and acceptance gates.

### Upgrade notes
- **Euclidean scores change scale.** Before 0.4.0 the Qdrant clients returned Qdrant's raw Euclidean distance as `score` (lower was better, and decay multiplied that raw value) while the local client returned `-distance`. All three clients now return `score = -distance` (higher is better) and `decayed_score = 1 / (1 + distance) × recency` in `(0, 1]`. Re-tune any thresholds on `score` or `decayed_score` for `distance="euclidean"`.
- **FLOAT payload indexes are recommended, not required.** New `{prefix}loci_data` / `{prefix}loci_summary` collections are created with FLOAT indexes on `x`, `y`, `z` and `confidence`. `loci migrate-layout` creates them on the target collections it creates (an existing target is left as is). Collections created before this change keep returning correct results because Qdrant evaluates Range conditions on unindexed payload fields by scanning; add the indexes with `create_payload_index(collection, field, PayloadSchemaType.FLOAT)` for both collections before benchmarking (snippet in `docs/IMPLEMENTATION_READINESS.md`).
- **Deploy the cloud API before or together with the updated SDK.** Older servers silently ignore `min_confidence`, `prev_state_id` and `next_state_id` in requests and omit them from responses.

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
