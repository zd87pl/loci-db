# LOCI Cloud API

This directory contains the infrastructure and service code for running LOCI as a managed cloud API — similar to Pinecone or Weaviate Cloud.

## Architecture Overview

```
                         ┌──────────────────────┐
  Developer              │  Cloudflare Workers   │
  curl/SDK  ──HTTPS──►   │  (edge auth + rate    │
                         │   limiting)           │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Fly.io (us-east)    │
                         │  loci-api            │
                         │  cloud/api/          │
                         └──┬───────────┬───────┘
                            │           │
              ┌─────────────▼──┐  ┌─────▼──────────────┐
              │ Qdrant Cloud    │  │ Supabase (Postgres) │
              │ (vector store)  │  │ api_keys, tenants,  │
              └────────────────┘  │ usage_events        │
                                  └─────────────────────┘
```

## Directory Structure

```
cloud/
├── README.md                     # This file
├── fly.toml                      # Fly.io app config (health checks, metrics, scaling)
├── api/                          # Hardened LOCI API service
│   ├── server.py                 # FastAPI app with auth, namespacing, rate limiting, metrics
│   ├── auth.py                   # API key middleware (Supabase lookup)
│   ├── generate_key.py           # Key generation utility
│   ├── Dockerfile
│   └── requirements.txt
├── edge/                         # Cloudflare Workers edge layer
│   ├── worker.js                 # Bearer token validation + /health passthrough
│   └── wrangler.toml             # Wrangler deploy config
├── migrations/                   # Supabase SQL migrations
│   ├── 001_api_keys.sql          # api_keys + tenants schema
│   ├── 002_rate_limits.sql       # Per-tenant rate_limit_rpm column
│   └── 003_rls.sql               # Row-Level Security (service-role only)
├── terraform/                    # Terraform scaffold (placeholder for future IaC)
│   ├── main.tf
│   └── variables.tf
├── tests/                        # Integration tests (mocked, CI-friendly)
│   ├── conftest.py
│   ├── test_api.py
│   └── pytest.ini
└── docs/
    └── runbooks/
        └── secrets-rotation.md   # Secret rotation procedures for all services
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | PoC — Fly.io + Supabase + Qdrant Cloud end-to-end | ✅ Done |
| 1 | API hardening (auth, namespacing, rate limiting, input validation, tests) | ✅ Done |
| 2 | Infrastructure maturity (CI/CD deploy, Cloudflare edge, Supabase RLS, monitoring, secrets runbook) | ✅ Done |
| 3 | Developer experience (SDK update, dashboard, CLI) | Planned |
| 4 | Billing & metering (Stripe, usage tracking, tier enforcement) | Planned |
| 5 | Production hardening (multi-region, load testing, security audit) | Planned |

## Quickstart (local)

```bash
# Run the API locally against a local Qdrant instance
cd cloud/api
QDRANT_URL=http://localhost:6333 \
QDRANT_API_KEY=local \
DATABASE_URL=postgresql://... \
LOCI_DEV_MODE=true \
uvicorn server:app --reload
```

## Environment Variables

| Variable | Description | Where set |
|----------|-------------|-----------|
| `QDRANT_URL` | Qdrant Cloud endpoint | Fly.io secret |
| `QDRANT_API_KEY` | Qdrant Cloud API key | Fly.io secret |
| `DATABASE_URL` | Supabase Postgres connection string | Fly.io secret |
| `LOCI_CORS_ORIGINS` | Comma-separated allowed origins | Fly.io secret |
| `LOCI_DEV_MODE` | Enable Swagger/ReDoc UI (`true`/`false`) | Fly.io secret |

## CI/CD

GitHub Actions (`.github/workflows/cloud-api.yml`):

1. **test** — runs on every push/PR touching `cloud/**` or `loci/client.py`
2. **deploy** — runs `flyctl deploy --remote-only` on push to `main`, gated on tests passing

**Required GitHub secret:** `FLY_API_TOKEN` — generate with `flyctl auth token` and add in repo Settings → Secrets → Actions.

## Edge Layer (Cloudflare Workers)

The `cloud/edge/` Worker sits in front of Fly.io and:
- Validates `Authorization: Bearer <key>` format at the edge (401 fast-path before Fly.io cold start)
- Routes `/health` and `/ready` without auth
- Forwards all other requests to the Fly.io upstream

Deploy: `cd cloud/edge && wrangler deploy` (set `UPSTREAM_URL` secret in Cloudflare dashboard).

## Supabase Migrations

Apply in order via Supabase SQL editor or `psql`:

```bash
psql $DATABASE_URL < cloud/migrations/001_api_keys.sql
psql $DATABASE_URL < cloud/migrations/002_rate_limits.sql
psql $DATABASE_URL < cloud/migrations/003_rls.sql
```

RLS is enabled on `api_keys`, `tenants`, `usage_events`, `usage_monthly`. The service role bypasses RLS; anonymous/public access is denied.

## Secrets Rotation

See `cloud/docs/runbooks/secrets-rotation.md` for rotation procedures covering `QDRANT_API_KEY`, `DATABASE_URL`, `FLY_API_TOKEN`, `LOGTAIL_TOKEN`, and `UPSTREAM_URL`.

## Pricing Model (planned)

| Tier | Price | Vectors | Requests/mo |
|------|-------|---------|-------------|
| Free | $0 | 10k | 100k |
| Developer | $29/mo | 1M | 5M |
| Pro | $149/mo | 10M | 50M |
| Enterprise | Custom | Unlimited | Unlimited |

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines.
Cloud-specific contributions should target the Fly.io + Supabase + Qdrant Cloud stack.
