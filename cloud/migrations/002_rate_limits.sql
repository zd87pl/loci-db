-- LOCI Cloud API — Phase 1: per-key rate limiting
-- Adds configurable requests-per-minute limit to each API key.
-- NULL means "use the server default" (60 rpm).

ALTER TABLE api_keys
    ADD COLUMN IF NOT EXISTS rate_limit_rpm INTEGER
        CHECK (rate_limit_rpm IS NULL OR rate_limit_rpm > 0);

COMMENT ON COLUMN api_keys.rate_limit_rpm IS
    'Per-minute request limit for this key. NULL = server default (60 rpm).';
