-- LOCI Cloud API — Phase 3: admin API keys
-- Adds an is_admin flag to api_keys. Admin keys can call /admin/* endpoints
-- (create, list, revoke tenant API keys). Regular keys can only call /insert
-- and /query against their own namespace.

ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN api_keys.is_admin IS
    'When true, this key can call /admin/* endpoints to manage tenant keys.';

CREATE INDEX IF NOT EXISTS idx_api_keys_is_admin
    ON api_keys(is_admin) WHERE is_admin = true;
