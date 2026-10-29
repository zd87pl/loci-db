-- Per-space logical epoch length (ms) for the 4D address; the library's default is 5 s.
ALTER TABLE spaces ADD COLUMN epoch_ms INTEGER NOT NULL DEFAULT 5000;
