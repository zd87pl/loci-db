-- LOCI Cloud control plane (D1). Phase 0: projects, keys, spaces, devices, daily usage.

CREATE TABLE projects (
  ref           TEXT PRIMARY KEY,                       -- 12 lowercase alphanumerics
  name          TEXT NOT NULL,
  owner_email   TEXT,
  jurisdiction  TEXT NOT NULL DEFAULT 'us' CHECK (jurisdiction IN ('us', 'eu')),
  plan          TEXT NOT NULL DEFAULT 'free',
  token_secret  TEXT NOT NULL,                          -- HMAC key for device JWTs (hex)
  created_at    INTEGER NOT NULL
);

CREATE TABLE api_keys (
  id            TEXT PRIMARY KEY,
  project_ref   TEXT NOT NULL REFERENCES projects(ref) ON DELETE CASCADE,
  kind          TEXT NOT NULL CHECK (kind IN ('secret', 'publishable')),
  key_hash      TEXT NOT NULL UNIQUE,                   -- SHA-256 of the raw key
  prefix        TEXT NOT NULL,                          -- display prefix, e.g. loci_sk_ab12cd34ef56
  label         TEXT,
  revoked       INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL
);
CREATE INDEX api_keys_project ON api_keys(project_ref);

CREATE TABLE spaces (
  project_ref   TEXT NOT NULL REFERENCES projects(ref) ON DELETE CASCADE,
  space         TEXT NOT NULL,                          -- slug, e.g. home
  name          TEXT,
  frame         TEXT NOT NULL DEFAULT 'room',
  extents       TEXT NOT NULL,                          -- JSON {"lo":[x,y,z],"hi":[x,y,z]} in metres
  created_at    INTEGER NOT NULL,
  PRIMARY KEY (project_ref, space)
);

CREATE TABLE devices (
  id               TEXT PRIMARY KEY,                    -- dev_<hex>
  project_ref      TEXT NOT NULL REFERENCES projects(ref) ON DELETE CASCADE,
  name             TEXT NOT NULL,
  spaces           TEXT NOT NULL,                       -- JSON array of space slugs
  secret_hash      TEXT,                                -- set when claimed
  claim_hash       TEXT UNIQUE,                         -- SHA-256 of the one-time claim code
  claim_expires_at INTEGER,
  claimed_at       INTEGER,
  revoked          INTEGER NOT NULL DEFAULT 0,
  token_epoch      INTEGER NOT NULL DEFAULT 0,          -- bump to invalidate issued tokens
  last_seen_at     INTEGER,
  created_at       INTEGER NOT NULL
);
CREATE INDEX devices_project ON devices(project_ref);

CREATE TABLE usage_daily (
  project_ref   TEXT NOT NULL,
  day           TEXT NOT NULL,                          -- YYYY-MM-DD (UTC)
  requests      INTEGER NOT NULL DEFAULT 0,
  writes        INTEGER NOT NULL DEFAULT 0,             -- observations written
  PRIMARY KEY (project_ref, day)
);
