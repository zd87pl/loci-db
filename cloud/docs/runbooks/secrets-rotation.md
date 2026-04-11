# Secrets Rotation Runbook — LOCI Cloud API

This runbook covers rotation of all secrets for the LOCI Cloud API stack.
Rotate proactively every 90 days and immediately on any suspected compromise.

## Secret inventory

| Secret name        | Where stored              | Used by                        | Rotation period |
|--------------------|---------------------------|--------------------------------|-----------------|
| `QDRANT_API_KEY`   | Fly.io secrets            | `cloud/api/server.py`          | 90 days         |
| `DATABASE_URL`     | Fly.io secrets            | `cloud/api/auth.py`            | 90 days         |
| `FLY_API_TOKEN`    | GitHub Actions secrets    | CI/CD deploy workflow          | 90 days         |
| `LOGTAIL_TOKEN`    | Fly.io secrets            | `flyctl log-drains`            | 90 days         |
| `UPSTREAM_URL`     | Cloudflare Worker secrets | `cloud/edge/worker.js`         | On Fly URL change |

---

## 1. Rotate `QDRANT_API_KEY`

1. Log in to [Qdrant Cloud console](https://cloud.qdrant.io).
2. Navigate to **Cluster → API Keys → Create new key**.
3. Copy the new key.
4. Update Fly.io secret — **zero downtime** because Fly re-deploys rolling:
   ```bash
   flyctl secrets set QDRANT_API_KEY=<new-key> --app loci-api
   ```
5. Verify the API is healthy after the rolling restart:
   ```bash
   curl https://loci-api.fly.dev/ready
   ```
6. Delete the old Qdrant API key from the console.

---

## 2. Rotate `DATABASE_URL` (Supabase)

The `DATABASE_URL` includes the Supabase Postgres password. Rotate via:

1. Log in to [Supabase dashboard](https://supabase.com/dashboard).
2. Go to **Project Settings → Database → Reset database password**.
3. Copy the new connection string (use the **session mode** URI, port 5432).
4. Update Fly.io:
   ```bash
   flyctl secrets set DATABASE_URL="postgresql://postgres:<new-password>@<host>:5432/postgres" --app loci-api
   ```
5. Verify `/ready` returns HTTP 200.

> **Note:** If using Supabase connection pooling (Supavisor), use port 6543 and
> append `?pgbouncer=true` to disable prepared statements.

---

## 3. Rotate `FLY_API_TOKEN` (GitHub Actions deploy)

1. Log in to [Fly.io dashboard](https://fly.io/dashboard).
2. Go to **Account → Access tokens → Create token** (name it `github-actions-loci`).
3. Copy the token.
4. In GitHub: **Settings → Secrets → Actions → Update `FLY_API_TOKEN`**.
5. Trigger a manual CI run to verify deploy still works:
   ```bash
   gh workflow run cloud-api.yml --ref main
   ```
6. Delete the old Fly.io token.

---

## 4. Rotate `LOGTAIL_TOKEN` (Fly.io log drain)

1. Log in to [Better Stack (Logtail)](https://logtail.com).
2. Navigate to **Sources → loci-api → Settings → Regenerate token**.
3. Update Fly.io:
   ```bash
   flyctl secrets set LOGTAIL_TOKEN=<new-token> --app loci-api
   ```
4. Re-create the log drain (Fly deletes and recreates it):
   ```bash
   flyctl log-drains destroy --app loci-api  # destroys existing drain
   flyctl log-drains create --type logtail --token $LOGTAIL_TOKEN --app loci-api
   ```
5. Confirm logs are flowing in the Logtail UI within 5 minutes.

---

## 5. Update `UPSTREAM_URL` in Cloudflare Worker

Only needed if the Fly.io app URL changes (e.g. renamed app).

```bash
wrangler secret put UPSTREAM_URL
# paste: https://loci-api.fly.dev
```

---

## Verification checklist after any rotation

- [ ] `curl https://loci-api.fly.dev/health` → 200 OK
- [ ] `curl https://loci-api.fly.dev/ready` → 200 OK
- [ ] Test auth: `curl -H "Authorization: Bearer loci_<valid-key>" https://loci-api.fly.dev/query -d '{...}'`
- [ ] Check Logtail for recent log lines (within 2 min)
- [ ] Fly.io metrics dashboard shows no error spike

## Emergency: full revocation

If a secret is confirmed compromised, revoke immediately without waiting for
a replacement to be tested:

```bash
# Fly.io: unset the secret entirely (will crash the app — acceptable in an emergency)
flyctl secrets unset QDRANT_API_KEY --app loci-api

# Then rotate and set the new value as above.
```

For `FLY_API_TOKEN`: delete the token in Fly.io dashboard immediately —
this disables all GitHub Actions deploys but does not affect the running app.
