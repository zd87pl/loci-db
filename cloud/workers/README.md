# LOCI Cloud on Cloudflare — Phase 0

The Cloudflare-native managed service described in [docs/CLOUD_PLATFORM_DESIGN.md](../../docs/CLOUD_PLATFORM_DESIGN.md):

- **Router Worker** (`src/index.ts`): keys, device tokens, permissions, validation, rate limiting, usage.
- **Control plane** (D1, `migrations/`, `src/control.ts`): projects, secret and publishable keys, spaces, devices.
- **Data plane** (`src/space.ts`): one SQLite-backed Durable Object per space with observations, a revision log for device sync, and where / changes / digest / ask / query.
- **SDK compatibility** (`src/compat.ts`): the old `/insert` and `/query`, so `LociClient(base_url=..., api_key=...)` works against a project with no client changes beyond the URL.

LOCI Cloud runs on Cloudflare only. The earlier Fly.io + Supabase + Qdrant Cloud service (`cloud/api`) was removed on 2026-10-27 (see [`cloud/README.md`](../README.md)). The SDK compatibility contract comes from its `cloud/api/server.py`; to read it, run `git show 917cc68:cloud/api/server.py`.

## Quickstart: insert and query from Python

**1. Get a project.** A project is created with the operator's admin token. `scripts/bootstrap.py --sdk` creates one and prints the two values the SDK needs.

Locally, run the Worker in the local Workers runtime (no Cloudflare account needed):

```bash
cd cloud/workers
npm ci
printf 'ADMIN_TOKEN=%s\nDEV_ALLOW_UNPINNED=1\n' "$(openssl rand -hex 16)" > .dev.vars
npm run db:migrate:local
npx wrangler dev --port 8787
```

In a second shell, from the repository root:

```bash
python3 cloud/workers/scripts/bootstrap.py --url http://127.0.0.1:8787 \
  --admin-token "$(sed -n 's/^ADMIN_TOKEN=//p' cloud/workers/.dev.vars)" --sdk > .loci-sdk.env
chmod 600 .loci-sdk.env
set -a; . ./.loci-sdk.env; set +a
```

On the hosted beta, the operator runs the same command with `--url https://loci-cloud.zd-66b.workers.dev` and the production admin token, then hands you the output.

The output holds two lines:

- `LOCI_BASE_URL=<worker>/p/<ref>`: the project URL.
- `LOCI_API_KEY=loci_sk_…`: the secret key. It can read and write; keep it on servers.

A commented line also carries the publishable key (`loci_pk_…`), which can query but not insert. Never commit either key.

**2. Insert and query.**

```bash
pip install -e .   # from the repository root: the SDK must send its own User-Agent (see Production)
```

```python
import os
import time

from loci import LociClient, WorldState

client = LociClient(
    base_url=os.environ["LOCI_BASE_URL"], api_key=os.environ["LOCI_API_KEY"], vector_size=4
)
now = int(time.time() * 1000)
door = client.insert(
    WorldState(
        x=0.42,
        y=0.17,
        z=0.88,  # normalised to [0, 1]
        timestamp_ms=now - 60_000,
        vector=[1.0, 0.0, 0.0, 0.0],
        scene_id="hallway",
        metadata={"label": "door", "source": "cam0"},
    )
)
chair = client.insert(
    WorldState(
        x=0.10,
        y=0.50,
        z=0.20,
        timestamp_ms=now,
        vector=[0.6, 0.8, 0.0, 0.0],
        scene_id="hallway",
        confidence=0.7,
    )
)

hits = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)  # chair.prev_state_id == door
near_door = client.query(
    vector=[1.0, 0.0, 0.0, 0.0],
    spatial_bounds={"x_min": 0.3, "x_max": 0.5},  # missing keys default to 0 and 1
    time_window_ms=(now - 120_000, now),
    min_confidence=0.8,
    limit=5,
)
```

Each result is a full `WorldState`: metadata, scale level, confidence, causal links, and the vector. Vectors are stored as float32, so `0.6` comes back as `0.6000000238418579`.

`AsyncLociClient` takes the same arguments (`async with AsyncLociClient(...) as client`) and needs `httpx`. Only `insert` and `query` work in cloud mode; the other methods raise `CloudModeUnsupportedError`.

**3. The same calls with curl** (curl's own User-Agent is accepted):

```bash
curl -X POST "$LOCI_BASE_URL/insert" -H "Authorization: Bearer $LOCI_API_KEY" -H "Content-Type: application/json" \
  -d '{"x": 0.42, "y": 0.17, "z": 0.88, "timestamp_ms": 1793140000000, "vector": [1, 0, 0, 0],
       "scene_id": "hallway", "metadata": {"label": "door"}}'
# → {"id": "0bab12a5-..."}
curl -X POST "$LOCI_BASE_URL/query" -H "Authorization: Bearer $LOCI_API_KEY" -H "Content-Type: application/json" \
  -d '{"vector": [1, 0, 0, 0], "x_min": 0.3, "x_max": 0.5, "min_confidence": 0.8, "limit": 5, "include_vectors": false}'
# → {"results": [{"id": "...", "x": 0.42, ..., "scene_id": "hallway", "scale_level": "patch", "confidence": 1,
#                 "metadata": {"label": "door"}, "prev_state_id": null, "next_state_id": "...", "vector": []}]}
```

**4. Beyond the SDK.** The `default` space is an ordinary space, so the v1 endpoints below read the same data, for example:

```bash
curl -X POST "$LOCI_BASE_URL/v1/spaces/default/query" -H "Authorization: Bearer $LOCI_API_KEY" \
  -H "Content-Type: application/json" -d '{"label": "door", "limit": 5}'
curl "$LOCI_BASE_URL/v1/spaces/default/where?label=door" -H "Authorization: Bearer $LOCI_API_KEY"
```

## API (v1, Phase 0)

Path routing on `workers.dev`: `https://<worker>/p/<ref>/v1/...`. Setting `PROJECT_HOST_SUFFIX` (for example `loci.cloud`) switches to `https://<ref>.loci.cloud/v1/...`.

| Method and path | Credential | Purpose |
|---|---|---|
| `POST /admin/projects` | `ADMIN_TOKEN` | Create a project; returns its ref, secret key and publishable key |
| `GET /spaces` · `POST /spaces` | read: any · create: secret | List or create spaces (`{"space":"home","frame":"room","extents":{"lo":[..],"hi":[..]}}`) |
| `POST /devices` · `GET /devices` · `DELETE /devices/<id>` | secret | Provision (returns a one-time claim code, valid 30 min), list, revoke |
| `POST /devices/claim` | none (claim code) | Exchange the claim code for a device id and device secret, once |
| `POST /devices/token` | none (device secret) | 1-hour device token (HS256), scoped to the device's spaces |
| `POST /spaces/<s>/observations` | secret, or device on its spaces | Up to 50 observations and 200 deletes per call; idempotent by `id` or `(device, seq)` |
| `GET /spaces/<s>/where?label=&limit=&room=` | any reader | Newest sightings of a label |
| `GET /spaces/<s>/changes?window_s=&room=` | any reader | New / moved / seen per room and label |
| `GET /spaces/<s>/digest?since_rev=` | any reader | Latest state per room and label changed since a revision, plus removals, for device caches |
| `POST /spaces/<s>/query` | any reader | Region, time, label, room, confidence and vector filters (below) |
| `POST /spaces/<s>/ask` | any reader | `{"text":"where is my mug"}` → spoken-style answer |
| `GET /spaces/<s>/stats` | any reader | Counts and revision |
| `POST /p/<ref>/insert` · `POST /p/<ref>/query` | insert: secret · query: any reader | Python SDK cloud mode on the project's `default` space (below) |

Publishable keys can read and ask but not write. Revoking a device bumps its token epoch, so its tokens stop working within a minute (immediately on the same isolate).

### Query

```json
POST /p/<ref>/v1/spaces/home/query
{
  "region": {"lo": [-2, 0, 1], "hi": [1, 2, 3]},
  "time": {"start": 1793140000000, "end": 1793150000000},
  "labels": ["cup", "mug"],
  "room": "kitchen",
  "min_confidence": 0.5,
  "vector": [0.1, -0.4, 0.9],
  "half_life_s": 3600,
  "limit": 10,
  "include_embedding": false
}
```

Every field is optional. `region` is a box in the space's frame (metres for `room`), and `region` and `time` bounds are inclusive. `label` (one string) can replace `labels` (up to 16). `limit` is 1–1000 (default 10).

- **Without `vector`**, results are newest first.
- **With `vector`**, only observations with an embedding of the same dimension are candidates. Ranking follows `LociClient.query` in the library: the top `3 × limit` by cosine similarity are re-ranked by `similarity × exp(-ln 2 × age / half_life)`. The default half-life is one hour; `half_life_s: 0` ranks by similarity alone. Each result adds `similarity` and `score`. A vector query scores at most the 20,000 newest matching rows; `truncated: true` says it stopped there.

Results have the `where` shape (`id`, `label`, `x`, `y`, `z`, `t`, `confidence`, `room`, `metadata`, ...), plus `embedding` when `include_embedding` is set. Observation embeddings are stored int8-quantised, so they come back approximately.

A region becomes Hilbert cell ranges (`hilbertRanges3` in `src/hilbert.ts`) over the stored order-8 cell, an indexed pre-filter; the exact bounds then decide membership.

### Python SDK (compatibility layer)

`LociClient(base_url="https://<worker>/p/<ref>", api_key="loci_sk_...")` posts to `<base_url>/insert` and `<base_url>/query`. With `PROJECT_HOST_SUFFIX` set, the base URL is `https://<ref>.<suffix>`.

- **Default space.** Everything goes to the project's `default` space, created by the first insert from a secret key with unit extents (frame `room`). WorldState coordinates are its coordinates.
- **Contract.** Validation, payloads and responses follow the retired `cloud/api/server.py`:
  - coordinates, confidence and `min_confidence` in [0, 1];
  - `scale_level` is `patch`, `frame` or `sequence`;
  - `scene_id` is 1–256 characters;
  - metadata is up to 16 KB;
  - the time window applies only when both ends are given;
  - `limit` is 1–1000.
- **Vector dimension.** The first insert fixes the space's vector dimension; later inserts and queries with another dimension get `422`.
- **Vectors** are stored as float32, as Qdrant stores them, and returned unless `include_vectors` is false.
- **Causal links.** An insert links to the latest earlier state of the same `scene_id`, and that state's `next_state_id` becomes the new id, as `LociClient.insert` does.
- **Labels.** A state is filed under its `metadata.label` (normalised), so `where`, `ask`, `digest` and v1 `query` label filters see SDK data. States without a label are stored with an empty label and left out of `where`, `ask`, `changes` and `digest`.

Clients must send their own `User-Agent` (see Production below); the SDK sends `loci-python/<version>`.

## Develop and test locally

```bash
cd cloud/workers
npm ci
npm test                 # Vitest in the local Workers runtime (Miniflare): Hilbert parity, API, conformance
npm run typecheck
```

Run it and connect the demo host:

```bash
printf 'ADMIN_TOKEN=%s\nDEV_ALLOW_UNPINNED=1\n' "$(openssl rand -hex 16)" > .dev.vars
npm run db:migrate:local
npx wrangler dev --port 8787
python3 scripts/bootstrap.py --url http://127.0.0.1:8787 --admin-token <ADMIN_TOKEN> --space home > ../../.cloud.env
```

Then start the demo host (`demo_spatial`) with the `LOCI_CLOUD_*` lines from `.cloud.env` in its environment; it mirrors its memory into the space (`GET /api/cloud/status` on the host shows progress). Deletes go through a breaker: a pass that would delete more than 200 sightings or more than half of them, or anything while the local memory is empty, holds its deletes (`delete_breaker: "open"`, `deletes_held`) until you check and release them with `POST /api/cloud/deletes/accept` (`{"confirm": true, "expect": <deletes_held>}` plus the host's `X-Device-Token`). See `demo_spatial/app/cloud_sync.py`.

For the Python SDK instead, `bootstrap.py --sdk` prints `LOCI_BASE_URL` and `LOCI_API_KEY` (see the quickstart above).

`DEV_ALLOW_UNPINNED=1` exists only because the local runtime does not implement Durable Object jurisdictions; without it an EU project fails closed. Never set it in production.

## Production (Phase 0)

Deployed 2026-10-27 to `https://loci-cloud.zd-66b.workers.dev` (D1 `loci-control`, id in `wrangler.jsonc`). The admin token lives outside the repo in `~/.loci/cloud-admin-token` on the operator's machine; project credentials from `bootstrap.py` go to `~/.loci/cloud.env` (mode 600).

Clients must send their own `User-Agent`: Cloudflare's edge rejects the default `Python-urllib/x.y` agent with error 1010 (`cloud_sync.py` and `bootstrap.py` do).

## Deploy (needs a Cloudflare account)

```bash
npx wrangler login
npx wrangler d1 create loci-control          # put the printed database_id into wrangler.jsonc
npm run db:migrate:remote
npx wrangler secret put ADMIN_TOKEN
npm run deploy
```

## Limits and errors

| What | Limit |
|---|---|
| Writes (`/observations`, `/insert`) | 120 per minute per key or device |
| Queries (`/query`, `/v1/.../query`) | 600 per minute per key or device |
| Embedding or vector dimension | 1–1536 |
| Query `limit` | 1–1000 |
| Request body | 512 KB |

Errors have the shape `{"detail": "<message>", "code": "<code>"}`:

| Status | Code | Typical cause |
|---|---|---|
| 400 | `bad_json` | The body is not valid JSON |
| 401 | `unauthenticated` | Missing, malformed or revoked key or token |
| 403 | `forbidden` | Key from another project, or a write with a publishable key |
| 404 | `not_found` | Unknown project path or space |
| 405 | `method_not_allowed` | `/insert` and `/query` take `POST` only |
| 413 | `too_large` | Body or batch too large |
| 422 | `invalid` | Validation failed, e.g. `vector must have 4 dimensions, got 3` |
| 429 | `rate_limited` | Rate limit exceeded |

## Hilbert parity

`src/hilbert.ts` ports the `hilbertcurve` package that `loci/spatial/hilbert.py` uses. `test/hilbert-golden.json` holds 203 values generated by the Python package (3-D and 4-D, orders 4, 8 and 12, plus LOCI's round-half-even unit quantisation); the test fails if the port drifts.

## Conformance with the library

`test/conformance-golden.json` holds states, golden queries and the expected result keys in order, covering ranking with decay and over-fetch, region, time and confidence filters with inclusive bounds, labels, and causal links. Two tests use the same file:

- `test/conformance.test.ts` sends it through the Worker's SDK `/query` and v1 `/query`.
- `tests/test_cloud_conformance.py` runs it through `LocalLociClient`, and through `LociClient` and `AsyncLociClient` in cloud mode against a fake server.

Regenerate the expectations from the library after an intended ranking change: `LOCI_REGEN_GOLDEN=1 pytest tests/test_cloud_conformance.py`.

## Schema

The space object migrates itself on construction (`SCHEMA_VERSION` in `src/space.ts`). Version 2 adds the WorldState columns (`scene`, `scale`, `prev_id`, `next_id`) and the `obs_h8` and `obs_scene_t` indexes to Phase 0 objects in place. `emb_scale IS NULL` marks a float32 embedding; otherwise `emb` is int8.
