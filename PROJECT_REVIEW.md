# LOCI Project Review

**Date:** 2026-07-17
**Scope:** Full repository at commit `2e9ae47` — core library (`loci/`), REST server (`server.py`), cloud API (`cloud/`), Rust core (`loci-core/`), demos, research pipeline, benchmarks, tests, docs, CI/CD.
**Method:** 13 parallel dimension reviews (spatial, temporal, retrieval, clients, server, cloud security, Rust, demos, docs accuracy, CI/packaging, research, test quality, architecture) plus a build/test health check. Every medium-or-higher finding was independently adversarially verified (high/critical findings by a 3-lens panel: technical correctness, impact, and concrete reproduction). 81 candidate findings were raised; 79 were confirmed, 2 refuted. Many findings below were verified by **empirical reproduction**, not just code reading.

---

## Executive Summary

LOCI is a well-engineered research prototype with a genuinely clean core: the Hilbert geometry is provably correct, layering matches the architecture doc, the cloud API's tenant isolation is sound on its data paths, and the suite (345 + 36 tests) passes clean with zero lint violations. The project's biggest problems are not in the math — they are at the **seams**:

1. **The advertised Docker quick start is completely broken** (critical): the image pairs HEAD's `server.py` with the year-old PyPI release, so every `/insert` returns 500.
2. **Ranking correctness collapses in common configurations**: decay underflows to 0.0 for data older than ~2 hours (making ranking arbitrary), `distance="euclidean"` inverts rankings in all three clients, and the headline `prediction_novelty` score is mathematically capped at ~0.5 — the documented 0.8 alert threshold can never fire.
3. **Cloud mode is materially out of contract**: methods documented to raise `CloudModeUnsupportedError` instead crash with `AttributeError` or silently return empty; metadata is silently dropped.
4. **The documentation oversells measured performance**: the README's "~75µs raw spatiotemporal query latency" headline describes a label-filter-only benchmark; the repo's own checked-in results put real spatial queries at ~97.5 ms p50, and the claimed "5–20× speedup vs naive Qdrant" is contradicted by the checked-in `latest.json` (LOCI is 3.7–5× *slower* on the cited scenario).
5. **The architecture's one-collection-per-5-second-epoch decision will not survive production**: ~17,280 collections/day/tenant, O(collections) everywhere, and an all-time query window materializes a 350-million-element list.

None of these are hard to fix individually, and the codebase's hygiene (tests, lint, layering, docs infrastructure) makes fixing them tractable. But the pattern is consistent: **the parts that are exercised by the test suite are solid; the parts that only run against real infrastructure (Qdrant, Docker, PyPI, decay-at-real-timescales) are broken**, because the suite tests the Qdrant clients exclusively against `MagicMock`.

### Scorecard

| Area | Grade | One-line assessment |
|---|---|---|
| Spatial indexing (Hilbert) | **A−** | Geometry provably correct, Rust/Python parity exact; defects are in query orchestration, not math |
| Temporal sharding | **B** | Boundary math correct and tested; retention/cache integration and decay numerics broken |
| Retrieval / novelty | **D** | Headline novelty score degenerate; metric-direction bugs; predictor output unvalidated |
| Client layer | **C+** | Clean mirrored API, good no-mutation discipline; silent-wrong-results bugs on real Qdrant paths |
| REST server (`server.py`) | **C** | Sane wrapper, but its deployment artifact is broken and 422 parity doesn't hold |
| Cloud API security | **B+** | Tenant isolation, key hashing, SQLi, admin gating all sound; three medium hardening gaps |
| Rust core | **B** | Bit-for-bit correct vs Python, zero unsafe; barely wired in, PyO3 layer unhardened, zero CI |
| Demos | **B−** | Functional and mostly security-aware; one broken deploy, one index-invariant bug, one XSS |
| Documentation accuracy | **C−** | API references excellent; performance claims contradicted by the repo's own artifacts |
| CI/CD & packaging | **B−** | Trusted publishing, SHA-pinned deploys; three broken gates incl. a live dependency-confusion vector |
| Research pipeline | **C+** | Clean single-pass design; unsandboxed LLM code execution is disqualifying as shipped |
| Test suite | **B−** | Excellent behavioral coverage of pure-Python paths; Qdrant backend never tested against anything real |
| Architecture | **B−** | Clean layering and API surface; two load-bearing design decisions (epoch-per-collection, parity-by-hand) are high-risk |

---

## Health Check (run in this environment)

- **Install:** `uv sync --all-extras` — OK (Python 3.11).
- **pytest:** `345 passed, 1 skipped` (skip is an opt-in perf test) in 2m16s. Zero failures.
- **ruff:** `check` — 0 violations; `format --check` — 61 files, all formatted.
- **mypy:** 2 errors, both `[import-untyped]` for the `loci_core` extension (wheel lacks a `py.typed` marker); plus a note about unused overrides for `edge_tts`/`faster_whisper`/`segno`.
- **cargo test (loci-core):** 30/30 passed (lib unit tests took 265s in debug mode — consider `--release` locally).
- **cloud tests:** fail out of the box with `ModuleNotFoundError: asyncpg` (cloud deps aren't in the root dev extras); after installing `asyncpg` + `python-json-logger`: `36 passed` in 1.5s.

---

## What's Genuinely Good

- **Hilbert correctness:** encoding uses `round(x*side)` while covers use `floor(lo*side)..ceil(hi*side)` plus a mandatory one-cell pad, so **no false negatives at box boundaries or at exactly 0.0/1.0**; the LUT fast path is test-pinned identical to brute force; Rust and Python agree bit-for-bit on 1,800 randomized cross-checks plus exhaustive order-4 sampling.
- **The "exact post-filter is authoritative" contract holds structurally** in all three clients, and `server.py`, the cloud API, funnel, and predict-then-retrieve all delegate rather than reimplement.
- **Cloud API security fundamentals:** namespace comes exclusively from the authenticated key row (never client input), keys are stored as SHA-256 of 256-bit random tokens, admin routes are DB-flag-gated, admin SQL is fully parameterized, RLS is enabled on all four tables, and the async-safety fixes from commit `8b7a26c` genuinely hold.
- **Layering matches ARCHITECTURE.md:** `spatial/`/`temporal/`/`retrieval/` never import the clients; funnel inverts its dependency through Protocols; no Qdrant types leak into the public API.
- **Write idempotency by construction:** client-side UUID point IDs make retried upserts safe; insert paths don't mutate caller `WorldState`s (test-pinned in all three clients).
- **Publishing hygiene:** PyPI OIDC trusted publishing with protected environments, pre-publish test gate, SHA-pinned deploy action, `py.typed` shipped, loopback-bound root compose with a written threat rationale.
- **Honest experimental culture:** IDD-58 docs report negative results (hypotheses C and D rejected), and every claimed artifact exists.

---

## Critical

### C1. The README Docker quick start cannot insert or query (Dockerfile / docker-compose)
`Dockerfile:5` installs the library from PyPI (`pip install "loci-stdb>=0.3.0"`) and copies only HEAD's `server.py`. PyPI has exactly one release (0.3.0, 2026-03-01). `WorldState.metadata` was added in June 2026 (commit `3bd2315`) and never published, but `server.py:100` passes `metadata=req.metadata` and `server.py:141` reads `r.state.metadata`. **Verified empirically by running HEAD `server.py` against the extracted 0.3.0 wheel: every valid `/insert` returns 500 (`TypeError: unexpected keyword argument 'metadata'`), and `/query` 500s on any non-empty result.** Since `docker-compose.yml` builds this image (`build: .`), the README's `docker compose up` quick start is broken for its core write/read path — and local changes to `loci/` never reach the container at all.
**Fix:** build the image from the working tree (`COPY loci/ pyproject.toml README.md` + `pip install .`), and add a CI smoke test that builds the image and exercises `/insert` + `/query`.

---

## High Severity

### Ranking & retrieval correctness

- **H1. Decay underflow makes ranking of older data arbitrary** (`loci/temporal/decay.py:28`). Default `decay_lambda=1e-4`/ms is a ~6.9 s half-life; `math.exp` underflows to exactly 0.0 for anything older than ~2 hours. All old results tie at 0.0 and the stable sort preserves epoch scan order — **reproduced: a 0.10-similarity match outranked a 0.99 one.** The default λ is wildly aggressive for real timestamps; either rescale (half-life in minutes/hours), or rank on `log(sim) − λ·age` to avoid underflow.
- **H2. `distance="euclidean"` inverts rankings in all three clients** (`decay.py` + `predict.py`). Qdrant returns euclid scores where smaller-is-better; MemoryStore returns negative distances. `apply_decay` and the novelty pipeline both assume higher-is-better positive scores: sync/async clients sort worst-first and truncation keeps the farthest points; in-memory novelty is pinned at 0.5. Both constructors advertise euclidean. Normalize score direction per metric at the backend boundary.
- **H3. `prediction_novelty` is degenerate — the advertised 0.8 alert threshold can never fire** (`loci/retrieval/predict.py:92`). The rank-relative min-max term always maps the best candidate to 1.0, so novelty ≤ 0.5 whenever ≥2 results exist. **Verified: a database containing only orthogonal junk still reports low novelty; an exact cosine match reads 0.13 instead of ~0.** Novelty must come from an absolute, metric-aware similarity of the best match, not rank-relative normalization. (Related mediums: NaN predictor output silently yields novelty 0.0 — the most dangerous failure for the advertised robot-safety use case; the no-`query_scored` fallback is binary; the calibrator saturates at ±1σ.)
- **H4. `min_confidence` silently drops valid results** (`loci/client.py:422`, all three clients). The confidence filter is post-fetch only, with no over-fetch and no `Range` pre-filter — **reproduced: 10 valid high-confidence matches exist, query returns 0.** Extend the 3× overfetch trigger to `min_confidence` or push it down as a payload `Range` condition.
- **H5. Ordered scrolls never paginate — trajectories truncate at 256** (`loci/client.py:871`). Qdrant returns `next_page_offset=None` whenever `order_by` is set, and rejects offset+order_by. `_scroll_all` passes `order_by="timestamp_ms"` and pages via offset, so it stops after one page: `get_trajectory`/`get_causal_context` cap at 256 points and predecessor selection is wrong on busy scenes. Paginate via `order_by.start_from`, or scroll unordered and sort client-side.
- **H6. Insert-before-query hides all pre-existing data** (`loci/client.py:156`). `_discover_collections` early-returns when `_known_collections` is non-empty, but `_ensure_collection` (every insert) also populates that set — so a client that inserts first never runs discovery and **all data from previous sessions is invisible**. Same root cause: discovery runs once per process, so long-lived readers and scaled cloud instances permanently miss epochs created by other writers (a new collection is minted every 5 s). Track discovery with a dedicated flag and give the cache a TTL / re-discover on cache miss.
- **H7. Retention purge leaves `_known_collections` stale** (`loci/client.py:731`, all three clients). `maybe_purge`'s return value (dropped names) is discarded; nothing ever removes them. Late inserts into a purged epoch **crash (verified `KeyError` on the memory backend; 404-after-retries on Qdrant)**, the epoch can never be recreated, and archive callbacks re-fire for already-dropped epochs on every insert.

### Cloud mode

- **H8. The documented cloud-mode contract is not implemented** (`loci/cloud_transport.py:9`). The docstring promises unsupported methods raise `CloudModeUnsupportedError`; none do. **Verified:** `insert_batch` crashes with `AttributeError` (`_qdrant` is `cast(QdrantClient, None)`); `query_scored`/`get_trajectory`/`get_causal_context` **silently return `[]`**. Add explicit guards per method.
- **H9. Cloud transport silently violates the WorldState contract** (`cloud_transport.py:35`). `metadata` (documented "stored and returned verbatim") and causal links are dropped on insert; query results lose `scale_level`/`confidence`/`metadata`. Also: async cloud `query` silently ignores `min_confidence` where sync raises (parity drift that has already shipped).

### Server & deployment

- **H10. "422 parity" (commit `2e9ae47`) does not hold on the root server** (`server.py:92`). The fix was applied only to `cloud/api/server.py`. Root `/insert` **returns 500 (verified)** for out-of-range coordinates, negative timestamps, and invalid `scale_level`; `/query` accepts NaN and inverted bounds and silently drops half-open time windows. Mirror the cloud validators (ideally extract a shared request model).
- **H11. `demo/Dockerfile` cannot build at all** (`demo/Dockerfile:12`). `pyproject.toml` declares `readme = "README.md"` but the Dockerfile never copies it; hatchling aborts (`Readme file does not exist`) — **reproduced**. This is the advertised Railway deploy path. One-line fix (`COPY README.md`), plus a docker-build smoke job in CI.

### Documentation & benchmark honesty

- **H12. The README performance headline misrepresents what was measured** (`README.md:233`). "~75µs raw spatiotemporal query latency" is the `label_filter` scenario — a plain `scene_id` keyword filter with **no spatial bounds and no time window**. The same checked-in results file shows real spatial queries at ~97.5 ms p50 (N=100) — about 1300× slower — and those rows are omitted from the README table. The table's "Temporal shard pruning" rows cite a scenario that doesn't exist in `retrieval_latest.json`, and every other row disagrees with the artifact's numbers.
- **H13. The "5–20× speedup vs naive Qdrant" claim is contradicted by the repo's own results** (`docs/NOVELTY.md:23`, `docs/BENCHMARK_METHODOLOGY.md`). `benchmarks/results/latest.json` shows LOCI **3.7–5× slower** than naive Qdrant on scenario A (the exact scenario the claim cites as "strongest"), and contains no N=100k data at all. LOCI wins only on combined tight spatial+temporal at larger N (~2×). Either fix the claims or produce results that substantiate them.
- **H14. README curl examples fail against the stack the README starts** (`README.md:96`) — 1-dim vectors against an enforced 512-dim server (400 rejection, and the insert is even labeled "512-dim vector").
- **H15. IDD-58's headline compression claim is a grid-alignment artifact** (`experimental/.../RESULTS.md:48`). The 16.7M-IDs→14-ranges result occurs because the tested p=8 bounds quantize to power-of-2-aligned cells; non-aligned bounds do not reproduce it, and the report's own judge scoring calls the enumeration step intractable at p≥8. Re-run with offset boxes before recommending Hypothesis B.

### CI, packaging, supply chain

- **H16. The CI CVE audit audits nothing** (`.github/workflows/ci.yml:104`). The security job installs only the scanners and runs `pip-audit` against `pip freeze` of that environment — the project's dependencies are never installed. A known-vulnerable `qdrant-client`/`numpy` would pass silently.
- **H17. Live dependency-confusion vector: the published `[native]` extra depends on `loci-core`, an unregistered PyPI name** (`pyproject.toml:43`). Verified: `loci-stdb` metadata on PyPI carries `Requires-Dist: loci-core; extra == "native"` and `pypi.org/pypi/loci-core` is 404. Anyone can register that name today and every `pip install "loci-stdb[native]"` would execute their code. Register the name (even as a placeholder) or remove the extra from published metadata.
- **H18. The Rust extension isn't used on the path it advertises accelerating** (`loci-core/src/lib.rs:60`). All three clients' insert paths call the pure-Python `HilbertIndex.encode`; the "~14,000× speedup" never engages for per-point encoding, and most of the extension's exported API (batch prep, novelty, 3D decode, bucket enumeration) has zero Python callers — while the genuinely useful 4D bucket enumeration in `hilbert_experiments.rs` is not exported at all.

### Research pipeline & demos

- **H19. `CodeRunner` executes LLM-generated code on the host with no sandbox** (`research/runners/code.py:84`) — in-place in the user's real source tree, via `shell=True` pytest, with full user privileges and network, contradicting its own "temporary file" docstring. The only backup of the original file is held in memory; a hard kill during the (up to 120 s × 5 variants) test window **permanently destroys the user's file**. Evaluate variants in a temp copy of the project at minimum.
- **H20. Optimizer's `max_tokens=4096` cannot hold its own required output** (`research/agents/optimizer.py:84`) — 5 full-replacement variants of a real source file reliably truncate; the JSON parse then crashes the run *after* the paid API calls. Check `stop_reason`, scale the budget, or generate variants in separate requests.
- **H21. `demo_spatial` merge breaks both index invariants** (`demo_spatial/app/spatial_memory.py:273`). `_try_merge` updates x/y/z/timestamp via `set_payload` but never re-encodes `hilbert_r*` or re-shards across epochs — **verified: a moved object disappears from spatial region queries and time-window queries**, a broken core feature of the assistive demo. Delete + re-insert on merge.

### Test-suite blind spots (the root cause behind most of the above)

- **H22. The Qdrant-backed clients are tested exclusively against `MagicMock`** (`tests/test_client.py:21`). No test uses `QdrantClient(location=':memory:')` — even though the benchmarks prove it works — so real filter semantics, ID round-tripping (see M-class finding on `uuid4().hex` vs hyphenated IDs), scroll pagination, tenant `collection_prefix` isolation (the exact mechanism the cloud API relies on), funnel, trajectory, causal context, and retention purge have **zero tests against real logic**. This is precisely why H5, H6, H7, and the ID-format bug shipped. Adding a `:memory:`-backed integration tier is the single highest-leverage testing investment available.
- **H23. The ADR-5 VLM confidence-calibration tests are tautological** (`tests/test_vlm_confidence_calibration.py:52`) — they reimplement the 0.6× penalty formula inside the test body; the production penalty line could be deleted without any failure.

---

## Medium Severity (selected, deduplicated)

**Cloud hardening (no isolation bypass found, but three real gaps):**
- `LOCI_MAX_BODY_BYTES` is documented but never enforced — oversized vectors are fully parsed before rejection; authenticated OOM DoS against the 512 MB VM (`cloud/api/server.py:72`).
- `generate_key.py` bypasses the `^[a-z0-9]{3,64}$` namespace regex the server enforces and *defaults* to an underscore namespace — reintroducing the exact cross-tenant prefix collision commit `8b7a26c` fixed (`cloud/api/generate_key.py:40`). Add a DB CHECK constraint.
- `--forwarded-allow-ips '*'` + the edge worker forwarding client headers verbatim lets a client spoof the IP the brute-force auth throttle keys on (`cloud/api/Dockerfile:34`).

**Performance / scalability:**
- `epochs_in_range` materializes `list(range(first, last+1))` — an all-time window is ~350 million ints (~10+ GB) before any existence filtering; the cloud `/query` accepts unbounded `time_end_ms` (`loci/temporal/sharding.py:73`). Intersect with the known-epoch set instead.
- Neither REST server can express "no spatial filter": defaults cover the full box, which enumerates all 65,536 r4 bucket IDs into a `MatchAny` filter per epoch, plus 3× overfetch (`server.py:118`).
- `expand_bounding_box` rebuilds the 65,536-entry LUT on every call — 355 ms measured per invocation (`loci/spatial/buckets.py:67`).
- One-collection-per-5s-epoch: ~17,280 collections/day/tenant, all client operations O(collections) (`loci/client.py:72`). Consider payload-indexed epoch fields in a small fixed set of collections, or much coarser epochs.
- Cross-epoch top-k merge is not decay-aware: per-shard truncation happens on raw score, so the advertised decay-weighted top-k can be wrong even without underflow (`loci/client.py:422`).

**Correctness / robustness:**
- No vector-dimension validation on the local path — one bad insert makes the whole collection unqueryable (verified numpy ragged-array crash) (`loci/backends/memory.py:74`).
- MemoryStore stores and returns live mutable references — caller mutations corrupt the store (verified both directions) (`memory.py:76`).
- Point IDs are `uuid4().hex` (hyphenless) but real Qdrant returns canonical hyphenated UUIDs — `get_trajectory` anchor matching likely never matches against a real server (`loci/client.py:231`).
- Retry logic never retries qdrant-client connection errors/timeouts — they arrive wrapped in `ResponseHandlingException`, which matches neither the status-code path nor the name-keyword path (`loci/retry.py:27`).
- Rust/Python behavioral divergence: negative time values raise `OverflowError` only when the extension is installed (u64 params); NaN coordinates silently encode to corner cells in Rust but raise in Python; no `order` validation (silent truncation at 4D order ≥ 17); the only exported bucket-enumeration function is 3D and incompatible with the 4D `hilbert_r*` payloads.
- `spatial_resolution` constructor parameter is documented, stored, and never read (`loci/client.py:106`).
- Adaptive density cells key on epoch-relative time, spreading a hot location's counts across ~16 cells so escalation triggers ~16× late (`loci/spatial/adaptive.py:124`).
- Sync `_ensure_collection` has a check-then-create race, and the cloud API drives it from a threadpool with a shared client (`loci/client.py:168`).
- Per-shard query failures are swallowed with a bare `except Exception: continue` and no logging — outages degrade to quietly smaller answers (`loci/client.py:471`).
- `/query` silently drops half-open time windows (only builds a window when *both* endpoints are present) — verified (`server.py:114`).
- `predict_and_retrieve` with an explicit historical window measures temporal proximity against `now + horizon/2` instead of the window midpoint, distorting scores (`loci/retrieval/predict.py:61`).
- DOM XSS in `demo_spatial` frontends: untrusted VLM/API labels into `innerHTML` and an inline `onclick` (the sibling warehouse demo escapes correctly) (`demo_spatial/static/index.html:771`).
- Unbounded memory growth in the publicly deployed warehouse simulation — no retention, unauthenticated `/api/simulation/start` (`demo/app/simulation.py:404`).
- Research pipeline: unvalidated `int()`/`float()` over LLM JSON can kill the run at the judge/scoring stages despite the explicit crash-tolerance intent; a hallucinated `winner_id` silently maps to no content.
- `docs/WORLD_MODEL_INTEGRATION.md` says `pip install loci-db` — wrong package name (`loci-stdb`).
- Production Fly deploy gates only on `cloud/tests`, not the library suite/lint/typecheck; a stale root `fly.toml` targets the same production app without health checks; `benchmarks/docker-compose.yml` exposes unauthenticated Qdrant on 0.0.0.0 (the root compose documents and avoids exactly this); the Rust crate has zero CI.
- Cloud auth logic (`require_api_key`: hash lookup, revocation, throttle) is never executed by any test — the dependency is always overridden (`cloud/tests/conftest.py:67`).

---

## Low Severity (selected)

Banker's-rounding divergence between Python `round()` and Rust `f64::round()` at half-cell boundaries · `maybe_purge_async` uses `iscoroutine` instead of `isawaitable` · `insert_batch` doesn't link to predecessors already in the store (batch vs sequential chains differ) · sync client has no `close()`/context manager · root `/health` echoes the internal Qdrant URL · per-key rate limit is checked only after auth has already cost a SELECT+UPDATE · Terraform S3 state has no encryption/locking · `Cargo.toml` 0.1.0 vs `loci-core/pyproject.toml` 0.3.0 version skew · ARCHITECTURE.md documents two clients (ships three) and a stale predict-then-retrieve data-flow line · SECURITY.md's "email the maintainers" channel has no email anywhere in the repo · `uv.lock` is committed but consumed by nothing · publish workflow never verifies the version matches the release tag · no top-level `permissions:` restriction on workflows · CLI `--runner=metric` silently substitutes the paid LLM runner · rate-limit test is flaky on minute rollover · advertised `LOCI_INTEGRATION=true` test mode doesn't exist · decay rerank integration test is conditionally vacuous (`if len(results) == 2` guard).

---

## Contested / Refuted During Verification

- **"funnel_search drops the true top-k"** — the mechanism is real (finer passes are gated by epochs/scenes from coarse hits, so fine-scale data in an epoch without a coarse entry is unreachable), but this is *intended, documented, and test-pinned* behavior (`tests/test_funnel_query.py:96-164`). Kept out of the defect list; worth a docs note since the surprise factor is high.
- **"Causal linking makes every insert O(scene history)"** — refuted: the feared full-scene paged scroll cannot occur, because `order_by` disables offset pagination entirely… which is exactly finding H5 (the scroll truncates instead).

---

## Priority Recommendations

1. **Fix the shipped quick start** (C1 + H11 + H14): build Docker images from the working tree, add a compose smoke test to CI, fix the curl examples. These are first-contact experiences and all three are broken.
2. **Add a real-Qdrant integration test tier** using `QdrantClient(':memory:')` (H22). It would have caught H5, H6, H7, the ID-format bug, and the `collection_prefix` isolation gap — the highest-leverage single change in the repo.
3. **Fix ranking correctness as a unit** (H1–H4): metric-aware score normalization at the backend boundary, a sane decay default (half-life parameter), absolute novelty, overfetch for `min_confidence`.
4. **Make cloud mode honest** (H8, H9): explicit `CloudModeUnsupportedError` guards, metadata support or loud rejection, async/sync parity.
5. **Close the supply-chain/CI gaps** (H16, H17): register or drop `loci-core` on PyPI; make pip-audit audit the actual dependencies; gate the Fly deploy on the full suite.
6. **Re-ground the performance narrative** (H12, H13): the honest story ("~2× on combined tight spatial+temporal at larger N; label-filtered lookups in µs; in-memory mode great for robotics prototyping") is still a good story — the current claims are falsified by the repo's own artifacts and will not survive a skeptical reader.
7. **Sandbox the research CodeRunner** (H19) before anyone runs it on a machine they care about, and write the backup to disk.
8. **Decide the epoch-per-collection question deliberately** (architecture): either coarsen epochs dramatically, or move the epoch into a payload-indexed field within a bounded collection set. Everything else in the scalability column follows from this.

---

*Review conducted with a multi-agent workflow: 13 specialized reviewers + build/test health check, 81 raised findings, each medium+ finding adversarially verified by independent agents (3-lens panels for high/critical), 79 confirmed / 2 refuted. Reproduction commands and evidence excerpts are preserved in the review transcripts.*
