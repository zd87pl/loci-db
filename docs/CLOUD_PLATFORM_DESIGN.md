# LOCI Cloud on Cloudflare — Platform and Business Design

Status: adopted. Proposed 2026-10-26; Cloudflare only was confirmed on 2026-10-27, and the earlier Fly.io + Supabase + Qdrant Cloud stack (`cloud/api`, `cloud/edge`, Supabase migrations, AWS Terraform) was removed from the repo. The open-source library (`loci/`, `loci-core/`) is unchanged and stays the offline/edge runtime.

## 1. Thesis

LOCI is spatial memory for machines that move through the physical world: *what was where, when, and what changed*. The same primitive serves several markets:

| Vertical | Who writes | Typical question | Specific need |
|---|---|---|---|
| Assistive devices (first) | ESP32 pendant, handheld Pi, phone | "Where did I leave my keys?" | Low-power devices, offline cache, voice answers |
| Elder care | Room cameras, wearables, care robots | "Has she taken her pills today? Where is her walker?" | Health-data compliance, EU residency, audit, retention |
| Robotics | Robot fleets via ROS 2 | "Where did any robot last see a pallet of SKU X?" | High write rate, map frames, fleet-wide search |
| Drones | Drones, ground stations | "What changed on this roof since last month's flight?" | Geodetic frames, large extents, mission replay |

So the answer is yes: build a horizontal "spatial memory database as a service", with vertical SDKs and templates on top. The assistive ESP32 product is the first customer of our own platform. That is how Supabase grew: one general database, many app types.

What must be true for that to work, and what this design enforces:

- **Frames are explicit.** Every observation says which coordinate frame it is in (room, map, camera, or WGS84/ENU geo). The unit cube is an implementation detail, not something customers see.
- **Devices are first-class.** Device identity, per-device revocation, offline sync and small payloads are part of the API, not an afterthought.
- **Tenants are physically isolated.** Each space lives in its own storage unit, which also gives data residency per project.

## 2. Product shape (Supabase-style)

```
Organization ── billing, members, SSO (later)
 └─ Project ── one endpoint https://<ref>.loci.cloud, one region/jurisdiction, keys, usage
     ├─ Spaces ── a home, a warehouse map, a drone survey area; has a frame and extents
     │    └─ Observations ── label, position, time, confidence, embedding, metadata, media ref
     ├─ Devices ── ESP32 / robot / drone; scoped to spaces; own credentials; last-seen; revocable
     └─ Keys ── secret (server), publishable (browser/app, read-scoped), device tokens
```

Key types mirror Supabase's anon and service-role split:

| Credential | Holder | Scope | Lifetime |
|---|---|---|---|
| Secret key `loci_sk_…` | Customer's backend | Full project | Until rotated |
| Publishable key `loci_pk_…` | App or web page | Read and ask on allowed spaces, rate-limited | Until rotated |
| Device token (JWT) | ESP32, robot | Write and read on its assigned spaces | 1 hour, refreshed with a device secret |

Developer surface: dashboard (projects, spaces, live map, devices, usage, keys), REST API with OpenAPI, SDKs (Python wraps the existing `LociClient` cloud mode, TypeScript, a C/Arduino client for ESP32, a ROS 2 bridge), an MCP server for agents (already in `loci/mcp/`), and the open-source library for self-hosting.

## 3. Architecture on Cloudflare

```
                 devices / apps / backends
                          │ HTTPS, WebSocket
                ┌─────────▼──────────┐
                │ Router Worker      │ auth (key hash in KV cache / device JWT), rate limit
                │ <ref>.loci.cloud   │ binding, request validation, routing, usage events
                └──┬───────┬───────┬─┘
     small writes  │       │ batch  │ queries / ask / sync
                   │  ┌────▼────┐   │
                   │  │ Queues  │   │
                   │  └────┬────┘   │
                ┌──▼───────▼────────▼──┐      ┌──────────────┐
                │ Space Durable Object │◄────►│ Vectorize    │ cross-space semantic search
                │ (SQLite, one/space)  │      │ one index per │
                │ observations, index, │      │ project       │
                │ revision log, live   │      └──────────────┘
                │ WebSockets, alarms   │      ┌──────────────┐
                └──────────┬───────────┘      │ R2           │ keyframes, audio, exports
                           │                  └──────────────┘
   ┌────────────┐   ┌──────▼──────┐   ┌────────────────────┐
   │ D1 control │   │ Analytics   │   │ Workers AI         │ embeddings, optional
   │ plane      │   │ Engine      │──►│ (priced add-on)    │ detection / TTS for thin clients
   │ orgs, keys │   │ usage       │   └────────────────────┘
   └────────────┘   └──────┬──────┘
                    hourly rollup → D1 → Stripe metered billing
```

### 3.1 Data plane: one Durable Object per space

A SQLite-backed Durable Object is a small, strongly consistent database with compute next to it. One per space gives physical tenant isolation, per-space locality, and a natural place for live subscriptions. LOCI's query model maps directly onto it:

```sql
CREATE TABLE obs (
  id TEXT PRIMARY KEY,           -- ULID
  device_id TEXT, seq INTEGER,   -- idempotency: UNIQUE(device_id, seq)
  label TEXT NOT NULL,
  x REAL, y REAL, z REAL,        -- metres in the space frame
  h8 INTEGER, h12 INTEGER,       -- Hilbert cells of the normalised position (loci-core)
  t INTEGER NOT NULL,            -- event time, ms
  confidence REAL,
  emb BLOB,                      -- int8-quantised embedding (256 B for 256-d)
  emb_model TEXT,                -- model id + version
  meta TEXT,                     -- JSON, capped at 4 KB
  media TEXT,                    -- R2 key
  rev INTEGER NOT NULL           -- space revision when written, for delta sync
);
CREATE INDEX obs_label_t ON obs(label, t DESC);
CREATE INDEX obs_h8_t    ON obs(h8, t);
CREATE INDEX obs_rev     ON obs(rev);
```

- **Where is X:** index seek on `(label, t)`, then the same merge, phrasing and hedging rules the demo already has.
- **Region and time:** Hilbert cell ranges plus exact float ranges, the same pre-filter-then-exact logic as `LocalLociClient`; top-k by cosine on the candidates.
- **Changes since:** compare latest versus earlier positions per label inside a window (`SpatialMemory.changes()` semantics).
- **Sync:** `rev` gives devices a cheap "everything since revision N" pull.
- **Live:** hibernatable WebSockets on the same object push new observations to phones and dashboards.
- **Maintenance:** alarms run retention, consolidation (merging repeated sightings into one record with a count) and Vectorize upserts.

The Hilbert encoding and cosine maths come from `loci-core` compiled to WebAssembly. That needs one change: make `rayon` an optional feature next to the existing optional `python` feature, since Workers have no threads.

Limits that shape the design (Cloudflare docs, checked 2026-10-26): 10 GB of storage per object, a soft limit of about 1,000 requests per second per object, rows up to 2 MB. A home or a single robot's map fits in one object for years. Warehouses, fleets and drone surveys shard a space into tiles (one object per Hilbert tile at a coarse order) with a small coordinator object that fans out region queries.

### 3.2 Cross-space search: Vectorize

Vectorize holds one index per project with namespaces per space, up to 20 million vectors, 1,536 dimensions and 10 metadata indexes. It supports numeric range filters (`$gte`/`$lt`) and returns at most 50 results with metadata. It answers "where have any of my robots seen something like this?" and feeds candidates back to the space objects for exact filtering. It is not the source of truth; the Durable Objects are.

### 3.3 Ingest: synchronous for small writes, queued for bulk

- Up to 50 observations per call: the router calls the space object directly and returns ids within one round trip. This is the ESP32 path.
- Bulk uploads (robot logs, drone missions, backfills): accepted into a Queue, written by a consumer in batches, so a burst never overloads a single object and clients get `202 Accepted` with a job id.
- Every write carries `(device_id, seq)` or an `Idempotency-Key`, so flaky Wi-Fi retries never duplicate data.

### 3.4 Control plane, metering and billing

D1 holds organisations, projects, spaces, devices, key hashes, plans and usage rollups. Workers KV caches key and device lookups at the edge. Every request writes one Analytics Engine event (project, endpoint, rows read and written, bytes, AI units); an hourly cron rolls them into D1 and reports metered usage to Stripe. The dashboard is a Pages app on the same APIs.

### 3.5 Residency and compliance

A project's jurisdiction is chosen at creation: Durable Objects can be pinned to `eu` (or `fedramp`), and R2 buckets and D1 have matching location options. Caveat from the docs: the restriction covers where data is stored and processed by the object, while object ids still appear in Cloudflare's billing logs and requests can arrive from anywhere.

Elder care in the EU means special-category health data under GDPR: EU jurisdiction, a data processing agreement, retention policies per space, export and delete endpoints, and an audit log are required before selling there. US healthcare needs a HIPAA business associate agreement; we must confirm Cloudflare's BAA terms for every product in the path before promising that.

### 3.6 Why not the other Cloudflare options

- **Python API in Cloudflare Containers:** Containers are generally available (April 2026) and bill only active CPU, but they have no persistent disk, so the vector store would still live outside Cloudflare (Qdrant Cloud). That is the current stack moved, not a Cloudflare-native one. Containers stay useful for heavy jobs: exports, re-embedding, model inference not offered by Workers AI.
- **Vectorize alone:** fine for "similar things", but LOCI's core questions are label-plus-time and region-plus-time with exact semantics and per-space consistency, which Vectorize's 50-result, 10-index filter model cannot answer reliably.
- **D1 for observations:** D1 is single-region with database-wide write serialisation; the per-space object model scales writes horizontally and adds live push.

## 4. API v1 (excerpt)

```
POST   /v1/devices/claim                 claim code from the device's QR → device id + secret (once)
POST   /v1/devices/token                 device secret → 1 h JWT
POST   /v1/spaces/{space}/observations   batch ≤ 50, idempotent
POST   /v1/spaces/{space}/observations:bulk   queued, returns job id
GET    /v1/spaces/{space}/where?label=cup&limit=5
POST   /v1/spaces/{space}/query          region / time / vector / label filters
GET    /v1/spaces/{space}/changes?since=… or window_s=…
GET    /v1/spaces/{space}/digest?since_rev=N  compact per-label latest state for device caches
POST   /v1/spaces/{space}/ask            {"text": "where is my cup"} → answer text (+ audio, add-on)
GET    /v1/spaces/{space}/live           WebSocket
PUT    /v1/media/{key}                   presigned R2 upload for keyframes
POST   /v1/search                        cross-space semantic search (Vectorize)
```

The current `/insert` and `/query` stay as a compatibility layer for the Python SDK's cloud mode, mapped onto a default space.

## 5. First use case: the ESP32 assistant

The ESP32 boards already speak HTTP to a host (`docs/ESP32_THIN_CLIENT_DEMO.md`). The cloud becomes their memory and, optionally, their brain:

1. **Provision.** The board shows a QR or claim code on first boot; the user claims it in the app, assigning it to "Home". The board stores a device secret in NVS and fetches a 1-hour token.
2. **Push.** Two modes, chosen per device:
   - *Edge inference* (P4 board, Pi): the device or the home host runs detection and sends observations in batches of up to 50 every few seconds, with `(device_id, seq)`.
   - *Cloud inference* (cheapest boards): the board uploads a keyframe to R2 and posts a reference; a Workers AI job runs detection and embedding and writes observations. This is a priced add-on.
3. **Pull.** On boot and every minute the board calls `digest?since_rev=N` and keeps a small per-label cache on its SD card, so "where is my cup?" still works offline from cached phrases, exactly as the current firmware does.
4. **Ask.** With connectivity, the board posts the recognised question (on-device ESP-SR) or audio to `/ask`; the space object answers with the existing phrasing rules. Text-to-speech audio for the board's speaker is an add-on.
5. **Share.** A family member's phone subscribes to `/live` for the same space.

The handheld and the Mac or Pi demo host gain a sync adapter: the persistence journal already records every upsert and delete, so mirroring it to `observations` with idempotency keys is a small, contained change.

## 6. Business model

**Open core.** The library stays Apache 2.0 and runs anywhere (Pi, robot, laptop). The managed service, dashboard, device management and add-ons are the paid product, and self-hosters are the funnel, as with Supabase.

**Plans (hypotheses to validate in customer interviews):**

| Plan | Price | Includes |
|---|---|---|
| Free | $0 | 1 project, 3 spaces, 5 devices, 100k stored observations, 1M operations/month, community support; paused after 7 days idle |
| Pro | $25/month + usage | 10 spaces, 50 devices, 5M observations, 20M operations, daily backups, email support |
| Team | $599/month + usage | SSO, audit log, EU jurisdiction, 99.9% SLA, retention policies |
| Enterprise / OEM | contract | dedicated jurisdiction, BAA where available, per-device licensing for hardware makers |

**Metered dimensions:** stored observations (GB), writes, reads, media GB, Vectorize queries, AI units (detection, embeddings, speech), active devices.

**Device licensing for hardware makers** (assistive pendants, care robots): a per-active-device monthly fee that bundles a write and storage allowance, so an OEM can price its own subscription. This is likely the main revenue line for the assistive and elder-care verticals.

**Unit economics check** (Durable Objects list prices, Workers Paid): writes cost $1.00 per million rows and each indexed insert writes several rows, storage costs $0.20 per GB-month. An assistant that stores one raw sighting per second while active, around 10,000 a day, would cost on the order of $1 a month in writes alone. Two design choices keep the margin healthy:

- Consolidate on the device side or in the object (the demo already merges repeated sightings within 2 s and 0.35 m), which cuts stored rows by an order of magnitude.
- Store embeddings int8-quantised (256 bytes instead of 1 KB) and keep raw frames out of the database (R2, with retention).

With those, a device costs cents per month to serve, and a $2–5 per-device price leaves room for support and AI add-ons.

## 7. Security model

- Secret and publishable keys stored as SHA-256 hashes in D1, cached in KV for 60 s; revocation propagates within a minute.
- Device JWTs signed per project; refresh requires the device secret; revoking a device kills refresh immediately and the token within the hour.
- Per-project and per-device rate limits through the Workers rate-limiting binding, plus Cloudflare WAF and bot protection on the dashboard.
- Tenant isolation by construction: a request can only reach space objects whose ids are derived from its own project id.
- Audit log of key, device and data-export events per project (Team plan and above).

## 8. Reuse and risks

| Part | Plan |
|---|---|
| `loci/` Python library | Stays the edge and offline runtime; `LociClient` cloud mode targets v1 |
| `loci-core` Rust | Compiled to Wasm for Hilbert and cosine inside Workers (feature-gate `rayon`) |
| `cloud/api` FastAPI service | Removed 2026-10-27; its validation and namespace rules (last in commit `917cc68`) are the reference for the v1 contract |
| `cloud/edge` Worker | Removed; replaced by the router Worker in `cloud/workers` |
| Supabase migrations | Removed; the control plane lives in D1 (`cloud/workers/migrations`) |

Risks:

- **Two implementations of query semantics** (Python and TypeScript/Wasm). Mitigation: a shared conformance suite of golden queries run against both in CI. It has been in place since #80 for `query`, and §11 extends it to every README promise.
- **Durable Object hot spots** for busy spaces. Mitigation: tile sharding from day one in the schema, enabled per space when needed.
- **Vendor concentration** on Cloudflare. Mitigation: the open-source library and a documented export format keep a self-host exit.
- **Compliance claims** for health data must be verified product by product before sales.

## 9. Phased plan

| Phase | Weeks | Delivers | Exit test |
|---|---|---|---|
| 0 | 2 | Router Worker, D1 control plane, device claim and tokens, space object with batch insert, where, changes, digest; ESP32 and demo-host sync | The ESP32 assistant answers "where is my cup?" from the cloud, and from its SD cache when offline |
| 1 | 3 | Queues bulk ingest, R2 media, Vectorize search, live WebSocket, Analytics Engine metering, minimal dashboard | A robot log of 1M observations ingests without errors; usage visible per project |
| 2 | 3 | Stripe billing, Free and Pro plans, EU jurisdiction, TypeScript and C SDKs, ROS 2 bridge, conformance suite | First paid design partner on Pro |
| 3 | ongoing | Workers AI add-ons, Containers jobs, Team plan, audit, retention, OEM device licensing | First hardware partner on device licensing |

Section 11 turns parity with the library and the robotics work into milestones that run alongside phases 1 and 2.

## 10. Decisions needed

1. Confirm Cloudflare-native (Durable Objects + Vectorize) over the Fly/Qdrant stack. This retires Qdrant Cloud and Supabase from the managed path. **Confirmed 2026-10-27.**
2. Domain for project endpoints (`loci.cloud` or similar).
3. Default jurisdiction for the beta: US, EU, or per project from day one.
4. Whether cloud inference for cheap boards is in the beta or deferred.
5. Plan prices, to validate in the design-partner interviews already planned in the GTM track.

### Defaults adopted for Phase 0 (2026-10-26)

| Decision | Default | Revisit when |
|---|---|---|
| Stack | Cloudflare-native (Workers, Durable Objects, D1, Vectorize later). Confirmed 2026-10-27: Fly.io, Qdrant Cloud and Supabase are off the managed path, and `cloud/api` was removed | Never for the beta |
| Domain | No custom domain in Phase 0: `*.workers.dev`, with the project ref as a path segment (`/p/<ref>/v1/...`); a `PROJECT_HOST_SUFFIX` setting switches to `<ref>.<domain>` once a domain is bought | Before the first external design partner |
| Jurisdiction | Per project from day one; default `us` (no restriction), `eu` selectable at project creation and pinned into the space object ids | First EU customer, or any elder-care pilot |
| Cloud inference | Deferred; Phase 0 accepts observations only (edge inference on the P4 board, Pi or home host) | Phase 3 add-ons |
| Prices | As in section 6, unpublished; Phase 0 meters usage but bills nothing | After 5–10 design-partner interviews |

## 11. Keeping the README's promise, and robots

The README promises what the open-source library does: a 4D Hilbert address, one query that combines similarity, place and time, novelty, trajectories and memory that ages. LOCI Cloud has to keep the same promise, or it is not LOCI. This section is the plan for that, extended to robots, which are the most demanding users of spatial memory. It refines phases 1 and 2.

### 11.1 Where the cloud stands (main, 2026-10-28)

| README promise | Library | Cloud today |
|---|---|---|
| Similarity, where and when in one `query(...)` | Yes | Yes. The conformance suite (`cloud/workers/test/conformance-golden.json`) runs 12 golden queries through both, with the same ids in the same order. |
| Hilbert-bucketed 4D address | `(x, y, z, t)` at orders 4, 8 and 12, time normalised per epoch | No. A 3D key at order 8 only; time is a plain column, and nothing indexes it on its own. |
| "Is this new?" | `predict_and_retrieve` with a novelty score and an optional calibrator | Partly. The basic form runs as a plain query; the scored and calibrated forms raise `CloudModeUnsupportedError`. |
| Trajectories and causal links | `get_trajectory`, `get_causal_context` | No. The links are stored, but nothing queries them. |
| Memory that ages, bounded storage | `ConsolidationPolicy`, `RetentionPolicy` | No. |
| `insert_batch`, `query_scored`, `funnel_query` | Yes | No. |
| Scale | Exact in-process search, or Qdrant (approximate, HNSW) | Exact search over at most the newest 20,000 matching rows per vector query; 10 GB and about 1,000 requests per second per space |

### 11.2 Decisions

1. **One engine, not a light and a full product.** The device API (`observations`, `where`, `ask`, `digest`, board routes) and the SDK surface read the same table in the same space object. Two engines would need two conformance suites, and would leave customers asking which one keeps the promise. Size becomes a plan setting (exact search for small spaces; a vector index and tiling for large ones), not a second product.
2. **There is no SQLite vector search to use.** SQLite in Durable Objects supports a subset of extensions (FTS5, JSON, math functions) and cannot load others, so sqlite-vec and libSQL's vector index are not available; D1 is the same. The space object implements vector search itself, as `LocalLociClient` does.
3. **Exact search stays the reference.** The library's in-process client keeps every promise with exact scoring behind the Hilbert and time pre-filter, and the space object does the same. An approximate index is added only for large spaces. Exact mode stays available per query, and the conformance suite always runs in it.
4. **Vectorize is for search across spaces only.** Its limits (at most 50 results with metadata, 10 metadata indexes, writes queryable in under 30 s at the median and under 2 min at p99) suit "where has any robot seen this", not the strongly consistent per-space path.
5. **The definition of done is a conformance row per promise.** Each row of the README's comparison table gets golden cases that run against the library and the Worker in CI. Once a row's cases pass, the README states that LOCI Cloud supports that row.

### 11.3 Library parity in the space object

- **4D address.**
  - Store `hilbert_r4`, `hilbert_r8` and `hilbert_r12` exactly as `HilbertIndex.encode` computes them: 4D, with time normalised within the observation's epoch. Each space sets `epoch_ms` (default 5,000, as in the library) when it is created.
  - Index `(epoch, hilbert_r12)`. A query whose window lies within one epoch becomes `hilbert_r12` prefix ranges, the same construction as today's `hilbertRanges3` in four dimensions.
  - Windows spanning several epochs, or queries without a window, use the existing 3D `h8` cover plus a new index on `t`. This is the library's own rule: multi-epoch windows use a spatial-only cover and a timestamp range.
  - The keys are at most 48 bits, so they are exact in JavaScript.
  - The Hilbert golden file gains 4D cases, so the Worker must produce the library's address for the same state.
- **Scoring.**
  - Cosine scoring and Hilbert encoding move into `loci-core` compiled to WebAssembly, with `rayon` behind a feature flag (§3.1), and use SIMD for int8 and float32 rows.
  - The 20,000-row scan cap stays until the vector index lands.
- **Vector index for large spaces (IVF).**
  - When a space passes a size threshold (about 50,000 embedded rows), an alarm trains k-means centroids on a sample, with k of about √N.
  - Each row gets a `list` column, assigned at insert, so writes stay synchronous and consistent; it is indexed with `t`.
  - A query probes the nearest lists, applies the Hilbert, time and label filters in SQL, then re-ranks exactly with decay, as `query` does today.
  - The alarm retrains when list sizes drift.
  - Target: recall@10 of at least 0.95 against exact search on a benchmark, with the number of probed lists chosen to meet it. Qdrant's HNSW is approximate too, so this is the same promise class as the library on Qdrant.
- **The full SDK surface.** `insert_batch`, `query_scored` (scores are already computed), `get_trajectory` (by scene and window), `get_causal_context` (along the stored links) and `funnel_query` (coarse to fine through r4, r8 and r12).
  - `predict_and_retrieve` and its calibrator run in the SDK on top of `query_scored`, reusing the library's code, so novelty is computed one way.
  - Each method leaves `CloudModeUnsupportedError` as it lands.
- **Memory that ages.**
  - Per-space `ConsolidationPolicy` and `RetentionPolicy`, with the library's fields, run from Durable Object alarms.
  - Summaries go to a `summary` table that queries search alongside raw rows, as `loci_summary` is searched; deleted rows leave tombstones for `digest`.
  - Storage per space stays bounded however long a robot runs.

### 11.4 Robots

The reference platform is Menlo Research's Asimov v1, an open-source humanoid with a Raspberry Pi 5 for media and networking and a Radxa CM5 for motion control. LOCI runs on the Pi 5 as it already does in the handheld demo (Pi 5 and OAK-D Lite), and nothing touches the motion controller. Until an Asimov is available, the handheld rig stands in for it. We found no confirmed ROS 2 support for Asimov, so integration goes through its own software stack or the bridge below.

- **Two tiers.**
  - The library on the robot is its working memory: in-process, available offline, with the whole promise.
  - The cloud space is the shared memory of a site or fleet, fed with consolidated sightings.
  - Neither tier belongs in a control or safety loop.
- **Client and bridge.**
  - A Python client for the v1 API: spaces, coordinates in metres in the space's frame, device tokens, upsert.
  - A ROS 2 node that turns detection and pose topics into observations and serves `where`, `query` and `changes`.
  - The `WorldState` compatibility layer stays for normalised SDK data.
- **Write path.** Robots see far more than they should upload.
  - Sightings are merged on the robot with the demo's rule (the same label within 2 s and 0.35 m).
  - They are sent with stable ids as upserts (#81, #82), so retries and restarts never duplicate them.
  - Bursts and log backfills go through Queues (§3.3) and get a job id.
  - The Phase 0 limit (120 writes per minute of up to 50 sightings, about 100 per second per robot) becomes a plan setting.
- **Maps that change.** SLAM loop closures move past poses, and maps get rebuilt. A sighting stored only in map coordinates goes stale when that happens. Neither the library nor the cloud handles this today, and it is the most robot-specific gap.
  - Observations gain optional `map_id`, `map_version` and `anchor` (a pose-graph node or keyframe id), with the offset from that anchor.
  - `POST /v1/spaces/{space}/maps/{map}/corrections` takes updated anchor poses. An alarm re-projects the affected rows, recomputing coordinates and Hilbert keys, and bumps the revision.
  - Queries can ask for a map version.
  - A rebuilt map is a new version, with an optional transform from the old one.
  - The library gains the same fields, so both tiers agree.
- **Fleet scale.**
  - A site is a space.
  - Busy or large sites shard into tiles (§3.1) once they near 10 GB or about 1,000 requests per second.
  - Questions across sites use Vectorize (decision 4).
  - Vector queries take an `emb_model` and compare only rows with the same model, so a fleet mixing embedding models never compares unlike vectors.
- **Time.** Robots stamp event time and must keep their clocks synchronised (NTP or PTP); the router already rejects times more than a day ahead.
- **Safety.** The memory is advisory and can be stale. Every answer carries the sighting's age and confidence, as the spoken answers already do, and the API documentation says the memory is not an obstacle map.

### 11.5 Milestones

| Milestone | Delivers | Exit test |
|---|---|---|
| M1 Parity core | 4D address, `t` index and per-space `epoch_ms`; `insert_batch`, `query_scored`, `get_trajectory`, `get_causal_context` | Golden cases for the 4D address and for trajectories and causal links pass on the library and the Worker |
| M2 Novelty | `predict_and_retrieve` with novelty and calibrator, and `funnel_query`, in cloud mode | A familiar and an unseen view score the same novelty on both |
| M3 Ageing | Consolidation and retention alarms, the summary table | A long-run golden case ends with the library's summaries and row counts |
| M4 Robot client | v1 Python client, ROS 2 bridge, merging on the robot, Queues bulk ingest | The handheld rig streams an hour-long session into a site space without loss; a 1M-observation robot log ingests (the Phase 1 exit test) |
| M5 Scale | WebAssembly scoring, the IVF index, exact mode per query | On a 1M-row space, recall@10 of at least 0.95 against exact search; latency measured and published |
| M6 Fleets and maps | Map versions and corrections, tiling, cross-space search | A replayed loop closure moves stored sightings to the corrected poses; two robots share one site space |

The README's comparison table gains a LOCI Cloud column as M1 to M3 land. M1 to M3 run alongside Phase 1 (§9); M4 to M6 span phases 1 and 2.

**Open questions.**
- How map frames line up with ROS conventions (`map`, `odom`, REP 105).
- Which embedding model a fleet standardises on.
- How robot write volumes are priced.

Sources: [Durable Objects limits](https://developers.cloudflare.com/durable-objects/platform/limits/), [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/), [Durable Objects data location](https://developers.cloudflare.com/durable-objects/reference/data-location/), [Vectorize limits](https://developers.cloudflare.com/vectorize/platform/limits/), [Vectorize metadata filtering](https://developers.cloudflare.com/vectorize/reference/metadata-filtering/), [Containers GA changelog](https://developers.cloudflare.com/changelog/post/2026-04-13-containers-sandbox-ga/), [Containers pricing](https://developers.cloudflare.com/containers/pricing/), [SQLite storage API (supported extensions)](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/), [Vectorize write latency changelog, 2026-06-30](https://developers.cloudflare.com/changelog/post/2026-06-30-improved-wal-throughput/), [Menlo Research: Asimov](https://menlo.ai/products/asimov), [Asimov v1 release](https://www.opensourceforu.com/2026/05/asimov-v1-open-sources-humanoid-robotics-development/).
