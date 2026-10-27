# LOCI Cloud

LOCI Cloud runs on Cloudflare only: Workers, D1 and Durable Objects.

- **Service code, API reference, local development and deploy:** [`workers/README.md`](./workers/README.md).
- **Platform and business design, phased plan:** [`docs/CLOUD_PLATFORM_DESIGN.md`](../docs/CLOUD_PLATFORM_DESIGN.md).
- **Demo host mirroring its memory into a space:** `demo_spatial/app/cloud_sync.py`.

```
cloud/
└── workers/      # router Worker, D1 control plane, one Durable Object per space
```

The open-source library (`loci/`, `loci-core/`) is separate from the managed service. It still runs anywhere, with its own Qdrant or in-memory backends.

## History

Until 2026-10-27 this directory also held an earlier managed stack:
- a FastAPI service (`cloud/api`) on Fly.io, with Supabase for API keys and Qdrant Cloud for vectors;
- a Cloudflare proxy Worker in front of it (`cloud/edge`);
- Supabase migrations;
- an AWS Terraform scaffold;
- the `loci cloud keys` CLI.

All of that was removed when the Cloudflare design was confirmed. Commit `917cc68` is the last one that has it, if its validation or namespace rules are ever needed for reference.
