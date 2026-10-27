/**
 * SpaceObject: one SQLite-backed Durable Object per space (a home, a map, a survey area).
 * Holds the observations, a revision log for device delta sync, and answers
 * where / changes / digest / ask with strong per-space consistency.
 */

import { DurableObject } from "cloudflare:workers";
import { age, changesAnswer, listAnswer, parseIntent, resolveLabel, wherePhrase, whereAnswer } from "./ask";
import { hilbert3 } from "./hilbert";
import type { Change, Digest, Env, Frame, InsertResult, Observation, ObservationIn, SpaceMeta } from "./types";

const HILBERT_ORDER = 8;
const MOVED_THRESHOLD_M = 0.5;

type Row = Record<string, SqlStorageValue>;

function quantiseEmbedding(v: number[]): { blob: ArrayBuffer; scale: number } {
  const maxAbs = v.reduce((m, x) => Math.max(m, Math.abs(x)), 0) || 1;
  const scale = maxAbs / 127;
  const q = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round((v[i] ?? 0) / scale)));
  return { blob: q.buffer, scale };
}

function rowToObs(r: Row): Observation {
  return {
    id: String(r.id),
    label: String(r.label),
    frame: String(r.frame) as Frame,
    room: (r.room as string | null) ?? null,
    x: Number(r.x),
    y: Number(r.y),
    z: Number(r.z),
    t: Number(r.t),
    confidence: Number(r.confidence),
    device_id: (r.device_id as string | null) ?? null,
    metadata: r.meta ? (JSON.parse(String(r.meta)) as Record<string, unknown>) : {},
    media: (r.media as string | null) ?? null,
    rev: Number(r.rev),
  };
}

function dist(a: Observation, b: Observation): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export class SpaceObject extends DurableObject<Env> {
  private sql: SqlStorage;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS obs (
        id TEXT PRIMARY KEY,
        device_id TEXT,
        label TEXT NOT NULL,
        frame TEXT NOT NULL,
        room TEXT,
        x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
        h8 INTEGER NOT NULL,
        t INTEGER NOT NULL,
        confidence REAL NOT NULL,
        emb BLOB, emb_scale REAL, emb_model TEXT,
        meta TEXT, media TEXT,
        rev INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS obs_label_t ON obs(label, t DESC);
      CREATE INDEX IF NOT EXISTS obs_rev ON obs(rev);
      CREATE TABLE IF NOT EXISTS tombstones (id TEXT PRIMARY KEY, label TEXT NOT NULL, room TEXT, rev INTEGER NOT NULL);
      CREATE INDEX IF NOT EXISTS tombstones_rev ON tombstones(rev);
    `);
  }

  private getMeta(): SpaceMeta | null {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'space'").toArray()[0];
    return r ? (JSON.parse(String(r.v)) as SpaceMeta) : null;
  }

  private rev(): number {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'rev'").toArray()[0];
    return r ? Number(r.v) : 0;
  }

  /** Store (or update) the space's frame and extents. Idempotent. */
  configure(meta: SpaceMeta): void {
    this.sql.exec("INSERT INTO meta (k, v) VALUES ('space', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", JSON.stringify(meta));
  }

  insertBatch(meta: SpaceMeta, observations: ObservationIn[], deletes: string[], deviceId: string | null): InsertResult {
    if (!this.getMeta()) this.configure(meta);
    const rev = this.rev() + 1;
    const ids: string[] = [];
    let accepted = 0;
    let duplicates = 0;
    let deleted = 0;
    const { lo, hi } = meta.extents;
    this.ctx.storage.transactionSync(() => {
      for (const id of deletes) {
        const row = this.sql.exec("SELECT label, room FROM obs WHERE id = ?", id).toArray()[0];
        if (!row) continue;
        this.sql.exec("DELETE FROM obs WHERE id = ?", id);
        this.sql.exec("INSERT OR REPLACE INTO tombstones (id, label, room, rev) VALUES (?, ?, ?, ?)", id, row.label, row.room, rev);
        deleted++;
      }
      for (const o of observations) {
        const id = o.id ?? (deviceId && o.seq !== undefined ? `d_${deviceId}_${o.seq}` : crypto.randomUUID());
        ids.push(id);
        const frame = o.frame ?? meta.frame;
        const unit = [o.x, o.y, o.z].map((v, k) => (v - lo[k]!) / (hi[k]! - lo[k]!)) as [number, number, number];
        const emb = o.embedding ? quantiseEmbedding(o.embedding) : null;
        const cur = this.sql.exec(
          `INSERT OR IGNORE INTO obs (id, device_id, label, frame, room, x, y, z, h8, t, confidence, emb, emb_scale, emb_model, meta, media, rev)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          id, deviceId, o.label, frame, o.room ?? null, o.x, o.y, o.z, hilbert3(unit, HILBERT_ORDER), o.t,
          o.confidence ?? 1, emb?.blob ?? null, emb?.scale ?? null, o.embedding_model ?? null,
          o.metadata ? JSON.stringify(o.metadata) : null, o.media ?? null, rev,
        );
        if (cur.rowsWritten > 0) accepted++;
        else duplicates++;
      }
      if (accepted > 0 || deleted > 0) {
        this.sql.exec("INSERT INTO meta (k, v) VALUES ('rev', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", String(rev));
      }
    });
    return { accepted, duplicates, deleted, ids, rev: accepted > 0 || deleted > 0 ? rev : rev - 1 };
  }

  where(label: string, limit: number, room: string | null = null): Observation[] {
    const rows = room
      ? this.sql.exec("SELECT * FROM obs WHERE label = ? AND room = ? ORDER BY t DESC LIMIT ?", label, room, limit)
      : this.sql.exec("SELECT * FROM obs WHERE label = ? ORDER BY t DESC LIMIT ?", label, limit);
    return rows.toArray().map(rowToObs);
  }

  labels(): string[] {
    return this.sql.exec("SELECT DISTINCT label FROM obs").toArray().map((r) => String(r.label));
  }

  /** Latest observation per (room, label), newest first. */
  latest(room: string | null = null, limit = 200): Observation[] {
    const q = `SELECT o.* FROM obs o JOIN (SELECT label, room, MAX(t) AS mt FROM obs GROUP BY label, room) g
               ON o.label = g.label AND o.room IS g.room AND o.t = g.mt
               ${room ? "WHERE o.room = ?" : ""} ORDER BY o.t DESC LIMIT ?`;
    const rows = room ? this.sql.exec(q, room, limit) : this.sql.exec(q, limit);
    const seen = new Set<string>();
    return rows.toArray().map(rowToObs).filter((o) => {
      const k = `${o.room}|${o.label}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }

  changes(windowS: number, room: string | null, nowMs: number = Date.now()): Change[] {
    const since = nowMs - windowS * 1000;
    const out: Change[] = [];
    for (const latest of this.latest(room, 500)) {
      if (latest.t < since) continue;
      const history = this.sql
        .exec("SELECT * FROM obs WHERE label = ? AND room IS ? ORDER BY t DESC LIMIT 50", latest.label, latest.room)
        .toArray()
        .map(rowToObs);
      const first = this.sql.exec("SELECT MIN(t) AS f FROM obs WHERE label = ? AND room IS ?", latest.label, latest.room).toArray()[0];
      const firstSeen = Number(first?.f ?? latest.t);
      const prev = history.slice(1).find((h) => h.t < latest.t && dist(h, latest) > MOVED_THRESHOLD_M);
      const kind: Change["kind"] = prev ? "moved" : firstSeen >= since ? "new" : "seen";
      out.push({
        kind,
        label: latest.label,
        room: latest.room,
        t: latest.t,
        first_seen: firstSeen,
        position: [latest.x, latest.y, latest.z],
        previous_position: prev ? [prev.x, prev.y, prev.z] : null,
        distance_m: prev ? Math.round(dist(prev, latest) * 100) / 100 : null,
      });
    }
    const order = { moved: 0, new: 1, seen: 2 } as const;
    return out.sort((a, b) => order[a.kind] - order[b.kind] || b.t - a.t);
  }

  /** Per-(room, label) latest state changed since a revision, for device caches. */
  digest(sinceRev: number): Digest {
    const rev = this.rev();
    const full = sinceRev <= 0;
    let objects = this.latest(null, 500);
    if (!full) {
      const changed = new Set(
        this.sql
          .exec("SELECT DISTINCT label, room FROM obs WHERE rev > ? UNION SELECT DISTINCT label, room FROM tombstones WHERE rev > ?", sinceRev, sinceRev)
          .toArray()
          .map((r) => `${r.room}|${r.label}`),
      );
      objects = objects.filter((o) => changed.has(`${o.room}|${o.label}`));
    }
    const present = new Set(this.latest(null, 500).map((o) => `${o.room}|${o.label}`));
    const removed = full
      ? []
      : this.sql
          .exec("SELECT DISTINCT label, room FROM tombstones WHERE rev > ?", sinceRev)
          .toArray()
          .map((r) => ({ label: String(r.label), room: (r.room as string | null) ?? null }))
          .filter((r) => !present.has(`${r.room}|${r.label}`));
    return {
      rev,
      full,
      objects: objects.map((o) => ({
        label: o.label, room: o.room, frame: o.frame, x: o.x, y: o.y, z: o.z, t: o.t, confidence: o.confidence, where: wherePhrase(o),
      })),
      removed,
    };
  }

  ask(text: string, nowMs: number = Date.now()): { answer: string; intent: string; label: string | null; results: Observation[] } {
    const intent = parseIntent(text);
    if (intent.kind === "where") {
      const { label, spoken } = resolveLabel(intent.noun, this.labels());
      const results = label ? this.where(label, 5, intent.room) : [];
      const answer = label ? whereAnswer(spoken, label, results, nowMs) : `I haven't seen your ${spoken} yet.`;
      return { answer, intent: "where", label, results };
    }
    if (intent.kind === "changes") {
      return { answer: changesAnswer(this.changes(600, intent.room, nowMs), intent.room), intent: "changes", label: null, results: [] };
    }
    if (intent.kind === "list") {
      const results = this.latest(intent.room, 20);
      return { answer: listAnswer(results, intent.room, nowMs), intent: "list", label: null, results };
    }
    return {
      answer: "You can ask where something is, what changed, or what I have seen.",
      intent: "unknown",
      label: null,
      results: [],
    };
  }

  stats(): { observations: number; labels: number; rev: number; newest: number | null; newest_age: string | null } {
    const r = this.sql.exec("SELECT COUNT(*) AS n, COUNT(DISTINCT label) AS l, MAX(t) AS m FROM obs").toArray()[0];
    const newest = r?.m === null || r?.m === undefined ? null : Number(r.m);
    return {
      observations: Number(r?.n ?? 0),
      labels: Number(r?.l ?? 0),
      rev: this.rev(),
      newest,
      newest_age: newest === null ? null : age(Date.now(), newest),
    };
  }
}
