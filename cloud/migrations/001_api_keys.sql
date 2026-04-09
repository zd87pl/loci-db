-- LOCI Cloud API — initial schema
-- API keys, tenants, and usage tracking

CREATE TABLE tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    tier        TEXT NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'developer', 'pro', 'enterprise')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash    TEXT NOT NULL UNIQUE,   -- SHA-256 of the raw key
    prefix      TEXT NOT NULL,          -- first 8 chars for display (e.g. "loci_abc")
    namespace   TEXT NOT NULL UNIQUE,   -- Qdrant collection prefix for this key
    label       TEXT,
    revoked     BOOLEAN NOT NULL DEFAULT false,
    last_used_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_hash   ON api_keys(key_hash);

CREATE TABLE usage_events (
    id              BIGSERIAL PRIMARY KEY,
    api_key_id      UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint        TEXT NOT NULL,
    vectors_count   INTEGER NOT NULL DEFAULT 0,
    latency_ms      INTEGER,
    status_code     SMALLINT,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_usage_key_time ON usage_events(api_key_id, recorded_at DESC);

-- Monthly aggregated usage (materialised for billing)
CREATE TABLE usage_monthly (
    api_key_id   UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    year_month   TEXT NOT NULL,  -- e.g. "2026-04"
    total_requests BIGINT NOT NULL DEFAULT 0,
    total_vectors  BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key_id, year_month)
);
