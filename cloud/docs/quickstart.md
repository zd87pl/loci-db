# LOCI Cloud — Quickstart

From zero to inserting and querying 4D world states from Python on LOCI Cloud, the Cloudflare Workers service in [`cloud/workers`](../workers/README.md).

The Python SDK's cloud mode (`LociClient(base_url=..., api_key=...)`) talks to a project's compatibility endpoints, `/insert` and `/query`. Everything lands in the project's `default` space. The v1 API (`/v1/spaces/...`) reads the same data.

---

## 1. Get a project

A project is created with the operator's admin token. `scripts/bootstrap.py --sdk` creates one and prints the two values the SDK needs.

**Locally**, run the Worker in the local Workers runtime (no Cloudflare account needed):

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
```

**On the hosted beta** (`https://loci-cloud.zd-66b.workers.dev`), the operator runs the same command with `--url https://loci-cloud.zd-66b.workers.dev` and the production admin token, then hands you the output.

`.loci-sdk.env` looks like this:

```
LOCI_BASE_URL=http://127.0.0.1:8787/p/k3v9x2m1q8ab
LOCI_API_KEY=loci_sk_k3v9x2m1q8ab_<48 hex>
# publishable key (read-only queries): loci_pk_k3v9x2m1q8ab_<48 hex>
```

`LOCI_BASE_URL` is the project URL: `<worker>/p/<ref>`.

The secret key (`loci_sk_…`) can read and write; keep it on servers. The publishable key (`loci_pk_…`) can only read, so it can query but not insert. Never commit either key.

Load the values into your shell:

```bash
set -a; . ./.loci-sdk.env; set +a
```

---

## 2. Insert from Python

```bash
pip install -e .        # from the repository root
```

The SDK must send its own `User-Agent`, because Cloudflare rejects Python's default one with error 1010. This repository's `loci` does; releases before this change do not.

```python
import os
import time

from loci import LociClient, WorldState

client = LociClient(
    base_url=os.environ["LOCI_BASE_URL"],
    api_key=os.environ["LOCI_API_KEY"],
    vector_size=4,
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
```

`insert` returns the new state's id (a UUID).

Rules, the same as the old managed API:

- **Coordinates** `x`, `y`, `z` and `confidence` are in [0, 1].
- **`scale_level`** is `patch`, `frame` or `sequence`.
- **`scene_id`** is 1–256 characters.
- **`metadata`** is any JSON object up to 16 KB.
- **Vector dimension.** The first insert into a project fixes it; later vectors of another length get `422`.
- **Causal links.** Each insert is linked to the latest earlier state in the same `scene_id`, so above `chair.prev_state_id == door`, and `door.next_state_id == chair`.
- **Labels.** `metadata["label"]` (normalised) becomes the observation's label, which `where`, `ask` and v1 label filters use.

---

## 3. Query from Python

```python
hits = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
for s in hits:
    print(s.id, s.x, s.y, s.z, s.timestamp_ms, s.scene_id, s.metadata, s.prev_state_id)

near_door = client.query(
    vector=[1.0, 0.0, 0.0, 0.0],
    spatial_bounds={"x_min": 0.3, "x_max": 0.5},  # missing keys default to 0 and 1
    time_window_ms=(now - 120_000, now),
    min_confidence=0.8,
    limit=5,
)
```

Results are ranked as the library ranks them:

1. Cosine similarity picks the top `3 × limit` candidates.
2. Those are re-ranked by `similarity × exp(-ln 2 × age / 1 h)`.

The shared conformance suite checks that the Worker and `LocalLociClient` return the same ids in the same order.

Bounds and the time window are inclusive. The time window applies only when both ends are given.

Each result is a full `WorldState`: `metadata`, `scale_level`, `confidence`, causal links, and the vector (stored as float32, so `0.6` comes back as `0.6000000238418579`).

Only `insert` and `query` work in cloud mode. Other methods raise `CloudModeUnsupportedError`.

### Async

```python
import asyncio
import os

from loci import AsyncLociClient


async def main():
    async with AsyncLociClient(
        base_url=os.environ["LOCI_BASE_URL"],
        api_key=os.environ["LOCI_API_KEY"],
        vector_size=4,
    ) as client:
        print(await client.query(vector=[0.0, 1.0, 0.0, 0.0], limit=5))


asyncio.run(main())
```

The async client uses `httpx` (`pip install httpx`).

---

## 4. The same calls with curl

curl's own `User-Agent` is accepted.

```bash
curl -X POST "$LOCI_BASE_URL/insert" \
  -H "Authorization: Bearer $LOCI_API_KEY" -H "Content-Type: application/json" \
  -d '{"x": 0.42, "y": 0.17, "z": 0.88, "timestamp_ms": 1793140000000,
       "vector": [1, 0, 0, 0], "scene_id": "hallway", "metadata": {"label": "door"}}'
# → {"id": "0bab12a5-..."}

curl -X POST "$LOCI_BASE_URL/query" \
  -H "Authorization: Bearer $LOCI_API_KEY" -H "Content-Type: application/json" \
  -d '{"vector": [1, 0, 0, 0], "x_min": 0.3, "x_max": 0.5, "min_confidence": 0.8,
       "limit": 5, "include_vectors": false}'
# → {"results": [{"id": "...", "x": 0.42, ..., "scene_id": "hallway", "scale_level": "patch",
#                 "confidence": 1, "metadata": {"label": "door"}, "prev_state_id": null,
#                 "next_state_id": "...", "vector": []}]}
```

---

## 5. Beyond the SDK: the v1 API

The `default` space is an ordinary space, so the v1 endpoints in [cloud/workers/README.md](../workers/README.md#api-v1-phase-0) work on it. For example:

**Label filters, newest first:**

```bash
curl -X POST "$LOCI_BASE_URL/v1/spaces/default/query" \
  -H "Authorization: Bearer $LOCI_API_KEY" -H "Content-Type: application/json" \
  -d '{"label": "door", "limit": 5}'
```

**Where something was last seen:**

```bash
curl "$LOCI_BASE_URL/v1/spaces/default/where?label=door" -H "Authorization: Bearer $LOCI_API_KEY"
```

Other spaces, with frames in metres, devices and device tokens, are created through `/v1/spaces` and `/v1/devices` (see `bootstrap.py` without `--sdk`).

---

## Limits

| What | Limit |
|---|---|
| Writes | 120 per minute per key or device |
| Queries (`/query`, `/v1/.../query`) | 600 per minute per key or device |
| Vector dimension | 1–1536 |
| `limit` | 1–1000 |
| Request body | 512 KB |

## Errors

Errors have the shape `{"detail": "<message>", "code": "<code>"}`.

| Status | Code | Typical cause |
|-------:|------|---------------|
| 400 | `bad_json` | Body is not valid JSON |
| 401 | `unauthenticated` | Missing, malformed or revoked key |
| 403 | `forbidden` | Key from another project, or a publishable key calling `/insert` |
| 404 | `not_found` | Unknown project path or space |
| 405 | `method_not_allowed` | `/insert` and `/query` take `POST` only |
| 413 | `too_large` | Body over 512 KB |
| 422 | `invalid` | Validation failed: see `detail`, for example `vector must have 4 dimensions, got 3` |
| 429 | `rate_limited` | Rate limit exceeded |
