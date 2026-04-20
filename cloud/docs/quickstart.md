# LOCI Cloud API — Quickstart

Five minutes from zero to your first 4D spatiotemporal vector query.

---

## 1. Get an API key

Beta partners receive a bootstrap admin key out-of-band. To create a regular
per-project key for your application, use either the CLI or the admin HTTP
endpoint directly.

### Using the `loci` CLI

```bash
pip install loci-stdb

export LOCI_API_KEY=loci_<your-admin-key>
export LOCI_BASE_URL=https://api.loci.ai

loci cloud keys create \
  --email app@example.com \
  --namespace myapp_prod \
  --label "production"
```

Output (the raw key is shown **once** — save it immediately):

```
=== API key created ===
Key ID    : 5bf1e7d9-...
Tenant ID : 9a4de8f0-...
Namespace : myapp_prod
Admin     : False
Prefix    : loci_8fa3c21

RAW KEY (shown only once — store securely):
  loci_<64 hex chars>
```

### Listing and revoking keys

```bash
loci cloud keys list
loci cloud keys revoke <KEY_ID>
```

Pass `--json` on any subcommand to get machine-readable output.

### Using `curl` directly

```bash
curl -X POST https://api.loci.ai/admin/keys \
  -H "Authorization: Bearer $LOCI_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_email": "app@example.com",
    "namespace": "myapp_prod",
    "label": "production"
  }'
```

---

## 2. Insert a vector

### curl

```bash
curl -X POST https://api.loci.ai/insert \
  -H "Authorization: Bearer $LOCI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "x": 0.42, "y": 0.17, "z": 0.88,
    "timestamp_ms": 1713628800000,
    "vector": [0.1, 0.2, ... 512 dims ...],
    "scene_id": "scene_001",
    "scale_level": "patch",
    "confidence": 0.95
  }'
# → {"id": "7f3c8a4b..."}
```

### Python SDK — cloud mode

```python
from loci import LociClient, WorldState

client = LociClient(
    base_url="https://api.loci.ai",
    api_key="loci_...",          # your API key
    vector_size=512,
)

state = WorldState(
    x=0.42, y=0.17, z=0.88,
    timestamp_ms=1713628800000,
    vector=[0.1] * 512,
    scene_id="scene_001",
)
point_id = client.insert(state)
```

### Python SDK — local mode (unchanged)

```python
from loci import LociClient, WorldState

# Existing local-only usage keeps working exactly as before.
client = LociClient("http://localhost:6333", vector_size=512)
client.insert(state)
```

The same `LociClient` class transparently routes to the cloud API when
`base_url` is set. Omit it, and the client talks to Qdrant directly.

---

## 3. Query vectors

### curl

```bash
curl -X POST https://api.loci.ai/query \
  -H "Authorization: Bearer $LOCI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "x_min": 0.0, "x_max": 1.0,
    "y_min": 0.0, "y_max": 1.0,
    "z_min": 0.0, "z_max": 1.0,
    "time_start_ms": 1713628000000,
    "time_end_ms":   1713629000000,
    "limit": 10
  }'
```

### Python SDK

```python
hits = client.query(
    vector=[0.1] * 512,
    spatial_bounds={
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    },
    time_window_ms=(1713628000000, 1713629000000),
    limit=10,
)
for s in hits:
    print(s.id, s.x, s.y, s.z, s.timestamp_ms, s.scene_id)
```

---

## 4. Async usage

```python
import asyncio
from loci import AsyncLociClient, WorldState

async def main():
    async with AsyncLociClient(
        base_url="https://api.loci.ai",
        api_key="loci_...",
        vector_size=512,
    ) as client:
        await client.insert(WorldState(
            x=0.5, y=0.5, z=0.5,
            timestamp_ms=1713628800000,
            vector=[0.0] * 512,
            scene_id="s",
        ))
        hits = await client.query(vector=[0.0] * 512, limit=5)
        print(hits)

asyncio.run(main())
```

The async client uses `httpx`; install with `pip install httpx` if you don't
already depend on it.

---

## Rate limits

Each API key has a configurable per-minute request limit (default: 600 rpm)
stored in the `rate_limit_rpm` column. Exceeding the limit returns **429 Too
Many Requests** with a ``Retry-After`` header in seconds.

Rate limits apply per **namespace** — one misbehaving integration does not
affect other keys or tenants.

---

## Error reference

| Status | Meaning | Typical cause |
|-------:|---------|---------------|
| 200 | OK | Request succeeded |
| 201 | Created | New API key created |
| 401 | Unauthorized | Missing or invalid `Authorization` header |
| 403 | Forbidden | Non-admin key calling `/admin/*` |
| 404 | Not Found | Key id does not exist (revoke) |
| 409 | Conflict | Namespace already taken (`POST /admin/keys`) |
| 422 | Unprocessable Entity | Validation error — see `detail` for field paths |
| 429 | Too Many Requests | Rate limit exceeded |
| 503 | Service Unavailable | `/ready` returned degraded — check upstream health |

All error responses have the shape:

```json
{ "detail": "<human-readable message or field list>" }
```

Every response carries an ``X-Request-Id`` header — include it when filing
support tickets.

---

## Reference

- [cloud/README.md](../README.md) — architecture overview, env vars, CI/CD
- [cloud/docs/runbooks/secrets-rotation.md](./runbooks/secrets-rotation.md) — rotating cloud secrets
- [OpenAPI schema](https://api.loci.ai/openapi.json) — always available, machine-readable
