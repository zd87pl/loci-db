-- LOCI Cloud API — Phase 2: Row-Level Security
-- Enable RLS on api_keys and usage_events so anon/public Supabase clients
-- cannot read or write these tables. Service-role (server-side) bypasses RLS.

-- ── api_keys ──────────────────────────────────────────────────────────────
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- No anon policy: public/anon role gets nothing.
-- Service role (used by Fly.io API server via DATABASE_URL) bypasses RLS.
-- If you later need tenant self-service (e.g. Supabase auth users managing
-- their own keys), add a policy here:
--
--   CREATE POLICY tenant_own_keys ON api_keys
--     FOR ALL
--     USING (tenant_id = (SELECT id FROM tenants WHERE email = auth.email()))
--     WITH CHECK (tenant_id = (SELECT id FROM tenants WHERE email = auth.email()));

-- ── tenants ───────────────────────────────────────────────────────────────
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
-- No anon policy — service-role only.

-- ── usage_events ──────────────────────────────────────────────────────────
ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
-- No anon policy — service-role only.

-- ── usage_monthly ─────────────────────────────────────────────────────────
ALTER TABLE usage_monthly ENABLE ROW LEVEL SECURITY;
-- No anon policy — service-role only.

-- Verify RLS is on (run in psql to confirm):
-- SELECT tablename, rowsecurity FROM pg_tables
--   WHERE schemaname = 'public'
--   AND tablename IN ('tenants','api_keys','usage_events','usage_monthly');
