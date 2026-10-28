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

`AsyncLociClient` takes the same arguments (`async with AsyncLociClient(...) as client`) and needs `httpx`.

Cloud mode supports `insert`, `insert_batch`, `query`, `query_scored`, `get_trajectory` and `get_causal_context`, each checked against the library by the conformance suite. `funnel_query` and the scored or calibrated forms of `predict_and_retrieve` still raise `CloudModeUnsupportedError`; they are M2 in [docs/CLOUD_PLATFORM_DESIGN.md](../../docs/CLOUD_PLATFORM_DESIGN.md) §11.

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
| `GET /spaces` · `POST /spaces` | read: any · create: secret | List or create spaces (`{"space":"home","frame":"room","extents":{"lo":[..],"hi":[..]},"epoch_ms":5000}`) |
| `POST /devices` · `GET /devices` · `DELETE /devices/<id>` | secret | Provision (returns a one-time claim code, valid 30 min), list, revoke |
| `POST /devices/claim` | none (claim code) | Exchange the claim code for a device id and device secret, once |
| `POST /devices/token` | none (device secret) | 1-hour device token (HS256), scoped to the device's spaces; `expires_at` (Unix s) and `expires_in` (s) |
| `POST /spaces/<s>/observations` | secret, or device on its spaces | Up to 50 observations and 200 deletes per call; idempotent by `id` or `(device, seq)`; `"upsert": true` rewrites known ids (below) |
| `GET /spaces/<s>/where?label=&limit=&room=` | any reader | Newest sightings of a label |
| `GET /spaces/<s>/changes?window_s=&room=` | any reader | New / moved / seen per room and label |
| `GET /spaces/<s>/digest?since_rev=` | any reader | Latest state per room and label changed since a revision, plus removals, for device caches |
| `POST /spaces/<s>/query` | any reader | Region, time, label, room, confidence and vector filters (below) |
| `POST /spaces/<s>/ask` | any reader | `{"text":"where is my mug"}` → spoken-style answer |
| `GET /spaces/<s>/stats` | any reader | Counts and revision |
| `GET /spaces/<s>/board/memory` · `memory.pcm` · `phrase.pcm?text=` · `POST .../board/ask` · `.../board/voice` | any reader | ESP32 boards: cache JSON and spoken answers (below) |
| `POST /p/<ref>/insert` · `POST /p/<ref>/query` | insert: secret · query: any reader | Python SDK cloud mode on the project's `default` space (below) |

Publishable keys can read and ask but not write. Revoking a device bumps its token epoch, so its tokens stop working within a minute (immediately on the same isolate).

### Upsert

Without the flag, an observation whose `id` is already stored is counted as a duplicate and left alone. With `"upsert": true` the stored row is rewritten in place (position, time, confidence, metadata; the embedding, media and causal links are kept unless the upsert brings new ones), so a sighting that a writer merged or saw again keeps one row and one id. An upsert only applies when:

- the same writer stored the row (the same device, or a secret key for rows written with one);
- `label` and `room` are unchanged;
- `t` does not go back.

Ids that fail these checks come back in `conflicts` and are not touched. The reply counts `accepted` (new), `updated`, `duplicates` (unchanged) and `deleted`; an update bumps the revision, so `digest?since_rev=` reports it. The demo host's mirror (`demo_spatial/app/cloud_sync.py`) sends merged sightings this way, under `h_<sighting_id>`.

### Board routes (ESP32)

The ESP32 boards work locally first (on-device detection, MultiNet speech commands, the microSD cache) and then use these routes, in the formats the demo host's `/api/esp32/*` routes use, so the firmware's cache and playback code works against either. A device token (or the secret key) for the space is the credential. Audio is `audio/L16; rate=16000; channels=1`, synthesised by Workers AI (Aura-1, voice `asteria`; transcription by Whisper large-v3-turbo).

| Route | Request | Reply |
|---|---|---|
| `GET board/memory` | none | `{"rev","count","objects":[{"label","where","room","last_seen_ms","age_s"}],"summary","now_ms"}`. One object per label (its newest sighting), people left out, at most 12, newest first. `rev` hashes what a board would say, so it changes when a place or a label changes, not on every write. Strings carry no quotes, backslashes or control characters, and the JSON has no spaces: the firmware reads it with `strstr()` |
| `GET board/memory.pcm` | none | The `summary` spoken (`X-Memory-Rev`) |
| `GET board/phrase.pcm?text=` | 1–160 characters | That sentence spoken |
| `POST board/ask` | `{"text":"where is my cup"}` | The answer spoken; `X-Answer` (percent-encoded), `X-Intent`, `X-Label` |
| `POST board/voice` | raw PCM16, 16 kHz mono, 0.1–10 s | Transcribed, then answered; `X-Transcription` too. `422 not_heard` when nothing was said (Whisper's stock phrases such as "Thank you." count as nothing) |

Spoken sentences are cached per space (200, least recently used dropped), so the same phrase is synthesised once. The Cache API would do nothing on `workers.dev`, so the cache is a table in the space's Durable Object.

Provision a board with `scripts/devices.py`. It prints a one-time claim code and the console line to type on the board:

```bash
python3 scripts/devices.py --env-file ~/.loci/cloud.env create --name "P4 desk board"
# device     dev_0123456789abcdef  (P4 desk board, space home)
# claim code ABCD-EFGH-JKLM  (single use, expires in 30 min)
# on the board's serial console:
#   cloud claim https://loci-cloud.<acct>.workers.dev/p/<ref>/v1 home ABCD-EFGH-JKLM
python3 scripts/devices.py --env-file ~/.loci/cloud.env list
python3 scripts/devices.py --env-file ~/.loci/cloud.env revoke dev_0123456789abcdef
```

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

Results have the `where` shape (`id`, `label`, `x`, `y`, `z`, `t`, `confidence`, `room`, `metadata`, ...), plus `address` and, when `include_embedding` is set, `embedding`. Observation embeddings are stored int8-quantised, so they come back approximately.

### The 4D address

Every observation stores the library's 4D address: `{"epoch", "r4", "r8", "r12"}`. These are the `hilbert_r4`, `hilbert_r8` and `hilbert_r12` values that `loci`'s `HilbertIndex.encode` gives the position, normalised to the space's extents, and the time within its epoch. A space's `epoch_ms` is set when it is created: 100 ms to one day, default 5,000 ms as in the library. The conformance suite checks the values against the library's.

Queries use it as the library does:

- **A window inside one epoch** becomes `epoch = e` plus `r12` ranges covering the box and the window (`hilbertRanges` in `src/hilbert.ts`), through the `(epoch, r12)` index.
- **A window spanning epochs, or no window,** uses the 3D order-8 cell for the box and the time index for the window.

Either way the exact bounds decide membership, so the index only narrows the scan.

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
- **`insert_batch`** takes up to 100 states and links them as `LocalLociClient.insert_batch` does:
  - in (scene, time) order, each state after the previous one of its scene;
  - the first of a scene after the latest earlier state in the store, keeping its own `prev_state_id` when there is none;
  - every state with a predecessor becomes that state's `next_state_id`.

  The reply has the ids in input order and each scene's last id (`chains`). The SDK sends bigger batches in sorted chunks and passes `chains` on, so a split batch links exactly like one call.
- **`query` with `"include_scores": true`** adds `score` (cosine similarity) and `decayed_score` to each result, for `query_scored`.
- **`trajectory`** (`state_id`, `steps_back`, `steps_forward`, each 0–1000, default 10) and **`causal_context`** (`state_id`, `window_ms`, default 5,000) return the anchor's scene in time order, as `get_trajectory` and `get_causal_context` do. An unknown id returns no results. `causal_context` stops at 10,000 states and then adds `"truncated": true`.
- **Labels.** A state is filed under its `metadata.label` (normalised), so `where`, `ask`, `digest` and v1 `query` label filters see SDK data. States without a label are stored with an empty label and left out of `where`, `ask`, `changes` and `digest`.

Clients must send their own `User-Agent` (see Production below); the SDK sends `loci-python/<version>`.

## Develop and test locally

```bash
cd cloud/workers
npm ci
npm test                 # Vitest in the local Workers runtime (Miniflare): Hilbert parity, API, conformance, boards
npm run typecheck
```

Tests never call Cloudflare: `vitest.config.ts` turns remote bindings off and `test/board.test.ts` installs a Workers AI stand-in (`setAiForTests`). `wrangler dev` does call Workers AI for the board routes, which bills the account.

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

Deployed 2026-10-27 to `https://loci-cloud.zd-66b.workers.dev` (D1 `loci-control`, id in `wrangler.jsonc`). `GET /health` reports the deployed commit (`sha`, `dev` for a deploy without one). The admin token lives outside the repo in `~/.loci/cloud-admin-token` on the operator's machine; project credentials from `bootstrap.py` go to `~/.loci/cloud.env` (mode 600).

Clients must send their own `User-Agent`: Cloudflare's edge rejects the default `Python-urllib/x.y` agent with error 1010 (`cloud_sync.py` and `bootstrap.py` do).

## Deploy (needs a Cloudflare account)

The top level of `wrangler.jsonc` is production (`loci-cloud`); `env.staging` is a second Worker (`loci-cloud-staging`) with its own D1 database, Durable Objects and rate-limit namespaces, for trying board and firmware changes first.

First time, by hand:

```bash
npx wrangler login
npx wrangler d1 create loci-control          # put the printed database_id into wrangler.jsonc
npm run db:migrate:remote
npx wrangler secret put ADMIN_TOKEN
npm run deploy

# staging
npx wrangler d1 create loci-control-staging  # replace env.staging's placeholder database_id
npm run db:migrate:staging
npx wrangler secret put ADMIN_TOKEN --env staging    # a different token from production's
npm run deploy:staging
python3 scripts/bootstrap.py --url https://loci-cloud-staging.<acct>.workers.dev \
    --admin-token <STAGING_ADMIN_TOKEN> --space home > ~/.loci/cloud-staging.env
python3 scripts/seed_staging.py --from-env ~/.loci/cloud.env --to-env ~/.loci/cloud-staging.env
```

After that, `.github/workflows/cloud-deploy.yml` deploys:

- **staging** on every push to `main` that touches `cloud/workers`, or by hand;
- **production** only by hand (Actions → Cloud deploy → Run workflow → `production`). It runs in the GitHub environment `production`, so give that environment a required reviewer.

Both run the tests, apply D1 migrations, deploy with `GIT_SHA` set and check `/health`. They need repository secrets `CLOUDFLARE_API_TOKEN` (a token with Workers Scripts: Edit and D1: Edit on the account) and `CLOUDFLARE_ACCOUNT_ID`, and skip with a notice while those are missing. Set the repository variable `CLOUDFLARE_WORKERS_SUBDOMAIN` if the `workers.dev` subdomain is not `zd-66b`.

## Limits and errors

| What | Limit |
|---|---|
| Writes (`/observations`, `/insert`) | 120 per minute per key or device |
| Queries (`/query`, `/v1/.../query`) | 600 per minute per key or device |
| Speech (board TTS and transcription; cache hits are free) | 30 per minute per key or device |
| Board voice clip | 0.1–10 s of 16 kHz PCM16 (320,000 bytes) |
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
| 422 | `not_heard` | `board/voice`: the clip held no words |
| 429 | `rate_limited` | Rate limit exceeded; `Retry-After: 60` |
| 502 | `upstream` | Workers AI returned no audio |
| 503 | `unavailable` | Speech is not configured on the deployment |

## Hilbert parity

`src/hilbert.ts` ports the `hilbertcurve` package that `loci/spatial/hilbert.py` uses. `test/hilbert-golden.json` holds 203 values generated by the Python package (3-D and 4-D, orders 4, 8 and 12, plus LOCI's round-half-even unit quantisation); the test fails if the port drifts.

## Conformance with the library

`test/conformance-golden.json` holds states, golden queries and the expected results. It covers:

- ranking with decay and over-fetch;
- region, time and confidence filters with inclusive bounds;
- labels;
- causal links from single inserts and from batches, including a batch split into chained requests;
- scores, trajectories and causal context;
- the 4D address at epoch boundaries.

Two tests use the same file:

- `test/conformance.test.ts` sends it through the Worker's SDK endpoints and v1 `/query`.
- `tests/test_cloud_conformance.py` runs it through `LocalLociClient`, and through `LociClient` and `AsyncLociClient` in cloud mode against a fake server.

Regenerate the expectations from the library after an intended ranking change: `LOCI_REGEN_GOLDEN=1 pytest tests/test_cloud_conformance.py`.

## Schema

The space object migrates itself on construction (`SCHEMA_VERSION` in `src/space.ts`):

- **Version 2** adds the WorldState columns (`scene`, `scale`, `prev_id`, `next_id`) and the `obs_h8` and `obs_scene_t` indexes to Phase 0 objects in place.
- **Version 3** adds the 4D address (`epoch`, `r4`, `r8`, `r12`, with an `(epoch, r12)` index) and an index on `t`.
  - Existing rows are backfilled 5,000 per pass: the first pass runs on construction, the rest from alarms.
  - Until the backfill finishes, queries use the 3D cell and the time index.

`emb_scale IS NULL` marks a float32 embedding; otherwise `emb` is int8.
