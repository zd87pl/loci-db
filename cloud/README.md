# LOCI Cloud API

This directory contains the infrastructure and service code for running LOCI as a managed cloud API — similar to Pinecone or Weaviate Cloud.

## Architecture Overview

```
                         ┌─────────────────┐
  Developer              │  AWS API Gateway │
  curl/SDK  ──HTTPS──►   │  + CloudFront    │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  ECS Fargate     │
                         │  (loci-api)      │
                         │  cloud/api/      │
                         └──┬──────┬───────┘
                            │      │
              ┌─────────────▼─┐  ┌─▼──────────────┐
              │ Qdrant Cloud   │  │ Aurora Serverless│
              │ (vector store) │  │ (API keys, usage)│
              └───────────────┘  └────────────────┬─┘
                                                   │
                                          ┌────────▼───────┐
                                          │ ElastiCache     │
                                          │ (rate limiting) │
                                          └────────────────┘
```

## Directory Structure

```
cloud/
├── README.md            # This file
├── api/                 # Hardened LOCI API service
│   ├── server.py        # FastAPI app with auth, namespacing, metrics
│   ├── auth.py          # API key middleware
│   ├── namespacing.py   # Per-tenant Qdrant collection isolation
│   ├── metrics.py       # Prometheus /metrics endpoint
│   ├── Dockerfile
│   └── requirements.txt
├── terraform/           # Infrastructure-as-Code (AWS)
│   ├── main.tf
│   ├── vpc.tf
│   ├── ecs.tf
│   ├── rds.tf
│   ├── redis.tf
│   ├── api_gateway.tf
│   ├── iam.tf
│   └── variables.tf
└── migrations/          # Database schema migrations
    └── 001_api_keys.sql
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | API hardening (auth, namespacing, rate limiting, input validation) | Planned |
| 2 | Infrastructure-as-Code (Terraform, AWS Fargate + Qdrant Cloud) | Planned |
| 3 | Developer experience (SDK update, dashboard, CLI) | Planned |
| 4 | Billing & metering (Stripe, usage tracking, tier enforcement) | Planned |
| 5 | Production hardening (multi-AZ, WAF, load testing, security audit) | Planned |

## Quickstart (local simulation)

```bash
# Run the hardened API locally against a local Qdrant instance
docker compose -f cloud/docker-compose.yml up
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant endpoint | `http://qdrant:6333` |
| `LOCI_API_DB_URL` | Postgres connection string for API key DB | required |
| `LOCI_REDIS_URL` | Redis URL for rate limiting | required |
| `LOCI_CORS_ORIGINS` | Comma-separated allowed origins | `*` (dev only) |
| `LOCI_RATE_LIMIT_FREE` | Requests/minute for free tier | `60` |
| `LOCI_RATE_LIMIT_PRO` | Requests/minute for pro tier | `1000` |

## Pricing Model

| Tier | Price | Vectors | Requests/mo |
|------|-------|---------|-------------|
| Free | $0 | 10k | 100k |
| Developer | $29/mo | 1M | 5M |
| Pro | $149/mo | 10M | 50M |
| Enterprise | Custom | Unlimited | Unlimited |

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines.
Cloud-specific contributions require Terraform knowledge and AWS account access.
