-- 005: enforce the namespace-isolation pattern at the database layer.
--
-- Namespaces must be 3-64 lowercase alphanumerics with NO underscores. Qdrant
-- collections are named {namespace}_loci_{epoch} and collection discovery
-- matches on the {namespace}_loci_ prefix; if underscores were allowed,
-- namespace "foo" would match collections belonging to namespace "foo_loci"
-- (prefix "foo_loci_") — a cross-tenant read. The API (server.py) and the CLI
-- (generate_key.py) validate this before insert; this CHECK is the backstop
-- for any writer that bypasses them.
--
-- DEFENSIVE NOTE — validate existing rows BEFORE running this migration:
--
--     SELECT id, namespace FROM api_keys
--     WHERE namespace !~ '^[a-z0-9]{3,64}$';
--
-- Any rows returned were created through the unvalidated CLI path (e.g. the
-- old default namespace 'loci_admin'). Migrate their tenants to a compliant
-- namespace (and rename/copy the matching Qdrant collections) or revoke the
-- keys first. The VALIDATE step below will fail — safely, without corrupting
-- anything — if violating rows remain.

-- Step 1: add the constraint without validating existing rows. NOT VALID keeps
-- the ALTER to a brief metadata-only lock; new/updated rows are checked
-- immediately.
ALTER TABLE api_keys
    ADD CONSTRAINT api_keys_namespace_format
    CHECK (namespace ~ '^[a-z0-9]{3,64}$')
    NOT VALID;

-- Step 2: validate existing rows. Takes only SHARE UPDATE EXCLUSIVE (does not
-- block reads/writes). Fails — leaving the constraint in place as NOT VALID —
-- if any pre-existing row violates the pattern.
ALTER TABLE api_keys VALIDATE CONSTRAINT api_keys_namespace_format;
