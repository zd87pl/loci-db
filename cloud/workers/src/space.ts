/**
 * SpaceObject: one SQLite-backed Durable Object per space (a home, a map, a survey area).
 * Holds the observations, a revision log for device delta sync, and answers
 * where / changes / digest / ask / query with strong per-space consistency.
 */

import { DurableObject } from "cloudflare:workers";
import { age, changesAnswer, listAnswer, parseIntent, resolveLabel, wherePhrase, whereAnswer } from "./ask";
import { hilbert3, hilbert4, hilbertRanges, hilbertRanges3 } from "./hilbert";
import type {
  Address, BatchResult, Change, Digest, Env, Extents, Frame, InsertResult, Observation, ObservationIn, QueryHit, QueryResult,
  QuerySpec, SdkFields, SdkStateIn, SpaceMeta,
} from "./types";
import { DEFAULT_EPOCH_MS } from "./types";

const HILBERT_ORDER = 8;
const MOVED_THRESHOLD_M = 0.5;
const SCHEMA_VERSION = 3;
/** Causal-context replies stop at this many states. */
const MAX_CONTEXT = 10_000;
/**
 * Delay before the next backfill pass. The clock is frozen within an event, so an alarm set to
 * Date.now() from alarm() would equal the running alarm's time and be cleared with it.
 */
const BACKFILL_DELAY_MS = 100;
/** Vector queries score at most this many matching rows, newest first. */
const MAX_VECTOR_SCAN = 20_000;
/** Same over-fetch and decay clamp as loci's query_scored and temporal.decay. */
const OVERFETCH = 3;
const MIN_DECAY_EXPONENT = -700;
/** Spoken sentences kept per space, least recently used dropped first. */
const TTS_CACHE_ENTRIES = 200;

type Row = Record<string, SqlStorageValue>;

function quantiseEmbedding(v: number[]): { blob: ArrayBuffer; scale: number } {
  const maxAbs = v.reduce((m, x) => Math.max(m, Math.abs(x)), 0) || 1;
  const scale = maxAbs / 127;
  const q = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round((v[i] ?? 0) / scale)));
  return { blob: q.buffer, scale };
}

/** `emb` holds int8 values times `emb_scale`, or float32 values when `emb_scale` is NULL (SDK vectors, kept exact). */
function decodeEmbedding(blob: ArrayBuffer, scale: number | null): number[] {
  return scale === null ? Array.from(new Float32Array(blob)) : Array.from(new Int8Array(blob), (v) => v * scale);
}

function norm(v: number[]): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

/** Cosine similarity as loci's memory backend computes it: 0 for a zero query, zero-norm rows count as norm 1. */
function cosine(q: number[], qNorm: number, v: number[]): number {
  if (qNorm === 0) return 0;
  let dot = 0;
  for (let i = 0; i < q.length; i++) dot += (q[i] as number) * (v[i] as number);
  return dot / ((norm(v) || 1) * qNorm);
}

function vec3InUnit(v: [number, number, number], lo: [number, number, number], hi: [number, number, number]) {
  return v.map((x, k) => (x - lo[k]!) / (hi[k]! - lo[k]!)) as [number, number, number];
}

function epochOf(meta: SpaceMeta): number {
  return meta.epoch_ms ?? DEFAULT_EPOCH_MS; // Phase 0 objects stored their meta without one
}

/** The stored keys of a position: the 3D cell for region queries, and the library's 4D address. */
function addressOf(extents: Extents, epochMs: number, x: number, y: number, z: number, t: number): Address & { h8: number } {
  const u = vec3InUnit([x, y, z], extents.lo, extents.hi);
  // As loci's epoch_id and LociClient._normalise_time.
  const epoch = Math.floor(t / epochMs);
  const p: [number, number, number, number] = [u[0], u[1], u[2], Math.min(1, Math.max(0, (t - epoch * epochMs) / epochMs))];
  return { h8: hilbert3(u, HILBERT_ORDER), epoch, r4: hilbert4(p, 4), r8: hilbert4(p, 8), r12: hilbert4(p, 12) };
}

function rowToAddress(r: Row): Address | null {
  if (r.r12 === null || r.r12 === undefined) return null;
  return { epoch: Number(r.epoch), r4: Number(r.r4), r8: Number(r.r8), r12: Number(r.r12) };
}

function rowToHit(r: Row, includeEmbedding: boolean, similarity: number | null = null, score: number | null = null): QueryHit {
  return {
    obs: rowToObs(r),
    sdk: rowToSdk(r),
    address: rowToAddress(r),
    similarity,
    score,
    embedding: includeEmbedding && r.emb ? decodeEmbedding(r.emb as ArrayBuffer, r.emb_scale as number | null) : null,
  };
}

function rowToSdk(r: Row): SdkFields {
  return {
    scene_id: (r.scene as string | null) ?? null,
    scale_level: (r.scale as string | null) ?? null,
    prev_state_id: (r.prev_id as string | null) ?? null,
    next_state_id: (r.next_id as string | null) ?? null,
  };
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
      CREATE TABLE IF NOT EXISTS tts_cache (k TEXT PRIMARY KEY, pcm BLOB NOT NULL, used INTEGER NOT NULL);
    `);
    this.migrate();
    if (this.backfillAddresses()) void ctx.storage.setAlarm(Date.now() + BACKFILL_DELAY_MS);
  }

  /** Rows the address backfill handles per pass; the rest continue from an alarm. */
  static backfillPerPass = 5000;

  /** Additive schema changes for objects created by earlier versions. */
  private migrate(): void {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'schema'").toArray()[0];
    const from = r ? Number(r.v) : 1;
    if (from >= SCHEMA_VERSION) return;
    this.ctx.storage.transactionSync(() => {
      if (from < 2) {
        // v2: WorldState fields for the SDK compatibility layer, and a Hilbert index for region queries.
        // Observations without a label (SDK states whose metadata has none) are stored with label ''.
        this.sql.exec(`
          ALTER TABLE obs ADD COLUMN scene TEXT;
          ALTER TABLE obs ADD COLUMN scale TEXT;
          ALTER TABLE obs ADD COLUMN prev_id TEXT;
          ALTER TABLE obs ADD COLUMN next_id TEXT;
          CREATE INDEX IF NOT EXISTS obs_scene_t ON obs(scene, t);
          CREATE INDEX IF NOT EXISTS obs_h8 ON obs(h8);
        `);
      }
      if (from < 3) {
        // v3: the library's 4D address (epoch and hilbert_r4/r8/r12) and a time index. Existing rows are
        // backfilled; until that finishes, queries use the 3D cell and the time index only.
        this.sql.exec(`
          ALTER TABLE obs ADD COLUMN epoch INTEGER;
          ALTER TABLE obs ADD COLUMN r4 INTEGER;
          ALTER TABLE obs ADD COLUMN r8 INTEGER;
          ALTER TABLE obs ADD COLUMN r12 INTEGER;
          CREATE INDEX IF NOT EXISTS obs_epoch_r12 ON obs(epoch, r12);
          CREATE INDEX IF NOT EXISTS obs_t ON obs(t);
        `);
        this.sql.exec("INSERT INTO meta (k, v) VALUES ('address', 'pending') ON CONFLICT(k) DO UPDATE SET v = excluded.v");
      }
      this.sql.exec("INSERT INTO meta (k, v) VALUES ('schema', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", String(SCHEMA_VERSION));
    });
  }

  private addressReady(): boolean {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'address'").toArray()[0];
    return !r || r.v === "ready";
  }

  /** One pass of the v3 backfill. Returns true while rows remain, so the caller schedules an alarm. */
  private backfillAddresses(): boolean {
    if (this.addressReady()) return false;
    const meta = this.getMeta();
    const rows = this.sql.exec("SELECT id, x, y, z, t FROM obs WHERE r12 IS NULL LIMIT ?", SpaceObject.backfillPerPass).toArray();
    if (!meta && rows.length) return false; // cannot happen: every write stores the meta first
    this.ctx.storage.transactionSync(() => {
      for (const r of rows) {
        const a = addressOf(meta!.extents, epochOf(meta!), Number(r.x), Number(r.y), Number(r.z), Number(r.t));
        this.sql.exec("UPDATE obs SET epoch = ?, r4 = ?, r8 = ?, r12 = ? WHERE id = ?", a.epoch, a.r4, a.r8, a.r12, r.id);
      }
      if (rows.length < SpaceObject.backfillPerPass) this.sql.exec("UPDATE meta SET v = 'ready' WHERE k = 'address'");
    });
    return rows.length === SpaceObject.backfillPerPass;
  }

  async alarm(): Promise<void> {
    if (this.backfillAddresses()) await this.ctx.storage.setAlarm(Date.now() + BACKFILL_DELAY_MS);
  }

  private getMeta(): SpaceMeta | null {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'space'").toArray()[0];
    return r ? (JSON.parse(String(r.v)) as SpaceMeta) : null;
  }

  private rev(): number {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'rev'").toArray()[0];
    return r ? Number(r.v) : 0;
  }

  private setRev(rev: number): void {
    this.sql.exec("INSERT INTO meta (k, v) VALUES ('rev', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", String(rev));
  }

  /** Vector dimension fixed by the first SDK insert into this space, or null. */
  private sdkDim(): number | null {
    const r = this.sql.exec("SELECT v FROM meta WHERE k = 'sdk_dim'").toArray()[0];
    return r ? Number(r.v) : null;
  }

  /** Store (or update) the space's frame and extents. Idempotent. */
  configure(meta: SpaceMeta): void {
    this.sql.exec("INSERT INTO meta (k, v) VALUES ('space', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", JSON.stringify(meta));
  }

  /**
   * Deletes first, then inserts that ignore known ids. With `upsert`, a known id is rewritten in
   * place (a merged or moved sighting keeps its id) when the same writer owns it, label and room
   * match and `t` does not go back; anything else is reported in `conflicts` and left alone.
   */
  insertBatch(
    meta: SpaceMeta, observations: ObservationIn[], deletes: string[], deviceId: string | null, upsert = false,
  ): InsertResult {
    if (!this.getMeta()) this.configure(meta);
    const rev = this.rev() + 1;
    const ids: string[] = [];
    const conflicts: string[] = [];
    let accepted = 0;
    let duplicates = 0;
    let updated = 0;
    let deleted = 0;
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
        const emb = o.embedding ? quantiseEmbedding(o.embedding) : null;
        const metaJson = o.metadata ? JSON.stringify(o.metadata) : null;
        const old = upsert ? this.sql.exec("SELECT * FROM obs WHERE id = ?", id).toArray()[0] : undefined;
        if (old) {
          const room = o.room ?? null;
          if ((old.device_id ?? null) !== deviceId || old.label !== o.label || (old.room ?? null) !== room || o.t < Number(old.t)) {
            conflicts.push(id);
            continue;
          }
          const confidence = o.confidence ?? 1;
          const same = old.frame === frame && Number(old.x) === o.x && Number(old.y) === o.y && Number(old.z) === o.z
            && Number(old.t) === o.t && Number(old.confidence) === confidence && (old.meta ?? null) === (metaJson ?? old.meta ?? null)
            && !emb && (o.media === undefined || old.media === o.media);
          if (same) {
            duplicates++;
            continue;
          }
          // emb / media / scene / causal links are kept unless the upsert brings new ones.
          this.sql.exec(
            `UPDATE obs SET frame = ?, x = ?, y = ?, z = ?, h8 = ?, t = ?, confidence = ?,
               emb = COALESCE(?, emb), emb_scale = CASE WHEN ? IS NULL THEN emb_scale ELSE ? END,
               emb_model = COALESCE(?, emb_model), meta = COALESCE(?, meta), media = COALESCE(?, media), rev = ?
             WHERE id = ?`,
            frame, o.x, o.y, o.z, hilbert3(vec3InUnit([o.x, o.y, o.z], lo, hi), HILBERT_ORDER), o.t, confidence,
            emb?.blob ?? null, emb?.blob ?? null, emb?.scale ?? null, emb ? (o.embedding_model ?? null) : null,
            metaJson, o.media ?? null, rev, id,
          );
          updated++;
          continue;
        }
        const cur = this.sql.exec(
          `INSERT OR IGNORE INTO obs (id, device_id, label, frame, room, x, y, z, h8, t, confidence, emb, emb_scale, emb_model, meta, media, rev,
                                      epoch, r4, r8, r12)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          id, deviceId, o.label, frame, o.room ?? null, o.x, o.y, o.z, a.h8, o.t,
          o.confidence ?? 1, emb?.blob ?? null, emb?.scale ?? null, o.embedding_model ?? null,
          metaJson, o.media ?? null, rev,
        );
        if (cur.rowsWritten > 0) accepted++;
        else duplicates++;
      }
      if (accepted > 0 || deleted > 0 || updated > 0) this.setRev(rev);
    });
    const wrote = accepted > 0 || deleted > 0 || updated > 0;
    return { accepted, duplicates, updated, conflicts, deleted, ids, rev: wrote ? rev : rev - 1 };
  }

  where(label: string, limit: number, room: string | null = null): Observation[] {
    const rows = room
      ? this.sql.exec("SELECT * FROM obs WHERE label = ? AND room = ? ORDER BY t DESC LIMIT ?", label, room, limit)
      : this.sql.exec("SELECT * FROM obs WHERE label = ? ORDER BY t DESC LIMIT ?", label, limit);
    return rows.toArray().map(rowToObs);
  }

  labels(): string[] {
    return this.sql.exec("SELECT DISTINCT label FROM obs WHERE label <> ''").toArray().map((r) => String(r.label));
  }

  /** Newest sighting per label (any room), newest first, without people: what a board caches. */
  newestPerLabel(limit: number): Observation[] {
    const rows = this.sql.exec(
      `SELECT o.* FROM obs o JOIN (SELECT label, MAX(t) AS mt FROM obs WHERE label NOT IN ('', 'person') GROUP BY label) g
         ON o.label = g.label AND o.t = g.mt
       ORDER BY o.t DESC, o.id LIMIT ?`,
      limit,
    );
    const seen = new Set<string>();
    return rows.toArray().map(rowToObs).filter((o) => !seen.has(o.label) && Boolean(seen.add(o.label)));
  }

  /** Spoken sentences already synthesised for this space (TTS costs per character). */
  ttsGet(key: string): Uint8Array | null {
    const r = this.sql.exec("SELECT pcm FROM tts_cache WHERE k = ?", key).toArray()[0];
    if (!r) return null;
    this.sql.exec("UPDATE tts_cache SET used = ? WHERE k = ?", Date.now(), key);
    return new Uint8Array(r.pcm as ArrayBuffer);
  }

  ttsPut(key: string, pcm: Uint8Array, keep = TTS_CACHE_ENTRIES): void {
    this.sql.exec(
      "INSERT INTO tts_cache (k, pcm, used) VALUES (?, ?, ?) ON CONFLICT(k) DO UPDATE SET pcm = excluded.pcm, used = excluded.used",
      key, pcm, Date.now(),
    );
    this.sql.exec("DELETE FROM tts_cache WHERE k NOT IN (SELECT k FROM tts_cache ORDER BY used DESC LIMIT ?)", keep);
  }

  /** Latest observation per (room, label), newest first. */
  latest(room: string | null = null, limit = 200): Observation[] {
    const q = `SELECT o.* FROM obs o JOIN (SELECT label, room, MAX(t) AS mt FROM obs WHERE label <> '' GROUP BY label, room) g
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
          .exec("SELECT DISTINCT label, room FROM tombstones WHERE rev > ? AND label <> ''", sinceRev)
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

  /**
   * Region / time / label / room / confidence filters, then either newest first or, with a vector,
   * the library's ranking: top limit x OVERFETCH by cosine, re-ranked by similarity x exp(-ln2 age / half-life).
   */
  query(meta: SpaceMeta, spec: QuerySpec, nowMs: number = Date.now()): QueryResult {
    const where: string[] = [];
    const args: SqlStorageValue[] = [];
    const cond = (sql: string, ...values: SqlStorageValue[]) => {
      where.push(sql);
      args.push(...values);
    };
    // Index pre-filters, as in the library; the exact bounds below decide membership.
    const { lo, hi } = meta.extents;
    const epochMs = epochOf(meta);
    const { timeStart: t0, timeEnd: t1 } = spec;
    if (t0 !== null && t1 !== null && Math.floor(t0 / epochMs) === Math.floor(t1 / epochMs) && this.addressReady()) {
      // A window inside one epoch: the 4D address, through obs_epoch_r12.
      const epoch = Math.floor(t0 / epochMs);
      cond("epoch = ?", epoch);
      if (spec.region) {
        const tn = (t: number) => (t - epoch * epochMs) / epochMs;
        const ranges = hilbertRanges(
          [...vec3InUnit(spec.region.lo, lo, hi), tn(t0)], [...vec3InUnit(spec.region.hi, lo, hi), tn(t1)], 12,
        );
        cond(`(${ranges.map(() => "r12 BETWEEN ? AND ?").join(" OR ")})`, ...ranges.flat());
      }
    } else if (spec.region) {
      // Spanning epochs, or no window: the 3D cell through obs_h8, and time through obs_t.
      const ranges = hilbertRanges3(vec3InUnit(spec.region.lo, lo, hi), vec3InUnit(spec.region.hi, lo, hi), HILBERT_ORDER);
      cond(`(${ranges.map(() => "h8 BETWEEN ? AND ?").join(" OR ")})`, ...ranges.flat());
    }
    if (spec.region) {
      const [rlo, rhi] = [spec.region.lo, spec.region.hi];
      cond("x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ?", rlo[0], rhi[0], rlo[1], rhi[1], rlo[2], rhi[2]);
    }
    if (spec.timeStart !== null) cond("t >= ?", spec.timeStart);
    if (spec.timeEnd !== null) cond("t <= ?", spec.timeEnd);
    if (spec.labels) cond(`label IN (${spec.labels.map(() => "?").join(", ")})`, ...spec.labels);
    if (spec.room) cond("room = ?", spec.room);
    if (spec.minConfidence !== null) cond("confidence >= ?", spec.minConfidence);
    if (spec.vector) cond("emb IS NOT NULL");
    const filter = where.length ? `WHERE ${where.join(" AND ")}` : "";
    const hit = (r: Row, similarity: number | null, score: number | null) => rowToHit(r, spec.includeEmbedding, similarity, score);

    if (!spec.vector) {
      const rows = this.sql.exec(`SELECT * FROM obs ${filter} ORDER BY t DESC, id LIMIT ?`, ...args, spec.limit).toArray();
      return { hits: rows.map((r) => hit(r, null, null)), truncated: false };
    }

    const q = spec.vector;
    const qNorm = norm(q);
    const scored: { id: string; t: number; similarity: number; score: number }[] = [];
    let seen = 0;
    const cursor = this.sql.exec(`SELECT id, t, emb, emb_scale FROM obs ${filter} ORDER BY t DESC, id LIMIT ?`, ...args, MAX_VECTOR_SCAN + 1);
    for (const r of cursor) {
      if (++seen > MAX_VECTOR_SCAN) break;
      const v = decodeEmbedding(r.emb as ArrayBuffer, r.emb_scale as number | null);
      if (v.length === q.length) scored.push({ id: String(r.id), t: Number(r.t), similarity: cosine(q, qNorm, v), score: 0 });
    }
    const bySimilarity = (a: (typeof scored)[number], b: (typeof scored)[number]) =>
      b.similarity - a.similarity || b.t - a.t || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
    scored.sort(bySimilarity);
    const lambda = spec.halfLifeMs > 0 ? Math.LN2 / spec.halfLifeMs : 0;
    const top = scored.slice(0, spec.limit * OVERFETCH);
    for (const c of top) c.score = c.similarity * Math.exp(Math.max(-lambda * Math.max(0, nowMs - c.t), MIN_DECAY_EXPONENT));
    top.sort((a, b) => b.score - a.score || bySimilarity(a, b));
    return {
      hits: top.slice(0, spec.limit).map((c) => hit(this.sql.exec("SELECT * FROM obs WHERE id = ?", c.id).one(), c.similarity, c.score)),
      truncated: seen > MAX_VECTOR_SCAN,
    };
  }

  /** SDK /query: the space's SDK vector dimension, once set, is enforced like the old server's vector_size. */
  sdkQuery(meta: SpaceMeta, spec: QuerySpec): QueryResult | { error: string } {
    const dim = this.sdkDim();
    const n = spec.vector?.length ?? 0;
    if (dim !== null && n !== dim) return { error: `vector must have ${dim} dimensions, got ${n}` };
    return this.query(meta, spec);
  }

  /** Latest state of `scene` strictly before `t` (ties: the last written), as loci's _find_latest_predecessor. */
  private predecessor(scene: string, t: number): string | null {
    const r = this.sql.exec("SELECT id FROM obs WHERE scene = ? AND t < ? ORDER BY t DESC, rowid DESC LIMIT 1", scene, t).toArray()[0];
    return r ? String(r.id) : null;
  }

  /** One SDK state, with an exact float32 vector and both Hilbert keys. */
  private writeState(meta: SpaceMeta, s: SdkStateIn, label: string, deviceId: string | null, id: string, rev: number, prev: string | null) {
    const a = addressOf(meta.extents, epochOf(meta), s.x, s.y, s.z, s.timestamp_ms);
    this.sql.exec(
      `INSERT INTO obs (id, device_id, label, frame, room, x, y, z, h8, t, confidence, emb, emb_scale, emb_model, meta, media, rev,
                        scene, scale, prev_id, next_id, epoch, r4, r8, r12)
       VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      id, deviceId, label, meta.frame, s.x, s.y, s.z, a.h8, s.timestamp_ms,
      s.confidence, new Float32Array(s.vector).buffer, Object.keys(s.metadata).length ? JSON.stringify(s.metadata) : null, rev,
      s.scene_id, s.scale_level, prev, s.next_state_id, a.epoch, a.r4, a.r8, a.r12,
    );
  }

  /**
   * SDK /insert: one WorldState, stored with an exact float32 vector. Links it after the latest earlier
   * state of the same scene (that state's next_state_id becomes this id), as LociClient.insert does.
   */
  insertState(meta: SpaceMeta, s: SdkStateIn, label: string, deviceId: string | null): { id: string; rev: number } | { error: string } {
    if (!this.getMeta()) this.configure(meta);
    const dim = this.sdkDim();
    if (dim !== null && s.vector.length !== dim) return { error: `vector must have ${dim} dimensions, got ${s.vector.length}` };
    const id = crypto.randomUUID();
    const rev = this.rev() + 1;
    this.ctx.storage.transactionSync(() => {
      const pred = this.predecessor(s.scene_id, s.timestamp_ms);
      if (pred) this.sql.exec("UPDATE obs SET next_id = ? WHERE id = ?", id, pred);
      this.writeState(meta, s, label, deviceId, id, rev, pred ?? s.prev_state_id);
      if (dim === null) this.sql.exec("INSERT INTO meta (k, v) VALUES ('sdk_dim', ?)", String(s.vector.length));
      this.setRev(rev);
    });
    return { id, rev };
  }

  /**
   * SDK /insert_batch, as LocalLociClient.insert_batch: states are linked per scene in (scene, time) order.
   * The first state of a scene follows `chains[scene]` (the last id of the previous chunk) or else the latest
   * earlier state in the store, and keeps its own prev_state_id when there is neither. Every state that ends
   * up with a predecessor becomes that state's next_state_id. Returns ids in input order and, per scene, the
   * last id, which the SDK passes on with the next chunk.
   */
  insertStates(
    meta: SpaceMeta, states: SdkStateIn[], labels: string[], chains: Record<string, string>, deviceId: string | null,
  ): BatchResult | { error: string } {
    if (!this.getMeta()) this.configure(meta);
    const known = this.sdkDim();
    const dim = known ?? states[0]?.vector.length ?? 0;
    for (const [i, st] of states.entries()) {
      if (st.vector.length !== dim) return { error: `states[${i}].vector must have ${dim} dimensions, got ${st.vector.length}` };
    }
    const order = states.map((_, i) => i).sort((a, b) => {
      const [sa, sb] = [states[a]!, states[b]!];
      return (sa.scene_id < sb.scene_id ? -1 : sa.scene_id > sb.scene_id ? 1 : 0) || sa.timestamp_ms - sb.timestamp_ms;
    });
    const ids = states.map(() => crypto.randomUUID());
    const chain = new Map(Object.entries(chains));
    const rev = this.rev() + 1;
    this.ctx.storage.transactionSync(() => {
      const links: [string, string][] = [];
      for (const i of order) {
        const st = states[i]!;
        const prev = chain.get(st.scene_id) ?? this.predecessor(st.scene_id, st.timestamp_ms) ?? st.prev_state_id;
        this.writeState(meta, st, labels[i] ?? "", deviceId, ids[i]!, rev, prev);
        chain.set(st.scene_id, ids[i]!);
        if (prev) links.push([prev, ids[i]!]);
      }
      for (const [prev, id] of links) this.sql.exec("UPDATE obs SET next_id = ? WHERE id = ?", id, prev);
      if (known === null && states.length) this.sql.exec("INSERT INTO meta (k, v) VALUES ('sdk_dim', ?)", String(dim));
      if (states.length) this.setRev(rev);
    });
    const scenes = new Set(states.map((st) => st.scene_id));
    return { ids, chains: Object.fromEntries([...chain].filter(([scene]) => scenes.has(scene))), rev: states.length ? rev : rev - 1 };
  }

  /** As LociClient.get_trajectory: the anchor's scene in time order, `back` states before it and `forward` after. */
  trajectory(id: string, back: number, forward: number, includeEmbedding: boolean): QueryHit[] {
    const a = this.sql.exec("SELECT *, rowid AS rid FROM obs WHERE id = ?", id).toArray()[0];
    if (!a) return [];
    if (!a.scene) return [rowToHit(a, includeEmbedding)];
    const before = this.sql
      .exec(
        "SELECT * FROM obs WHERE scene = ? AND (t < ? OR (t = ? AND rowid < ?)) ORDER BY t DESC, rowid DESC LIMIT ?",
        a.scene, a.t, a.t, a.rid, back,
      )
      .toArray()
      .reverse();
    const after = this.sql
      .exec("SELECT * FROM obs WHERE scene = ? AND (t > ? OR (t = ? AND rowid > ?)) ORDER BY t, rowid LIMIT ?", a.scene, a.t, a.t, a.rid, forward)
      .toArray();
    return [...before, a, ...after].map((r) => rowToHit(r, includeEmbedding));
  }

  /** As LociClient.get_causal_context: the anchor's scene within ±windowMs, in time order (anchor included). */
  causalContext(id: string, windowMs: number, includeEmbedding: boolean): QueryResult {
    const a = this.sql.exec("SELECT scene, t FROM obs WHERE id = ?", id).toArray()[0];
    if (!a || !a.scene) return { hits: [], truncated: false };
    const at = Number(a.t);
    const rows = this.sql
      .exec("SELECT * FROM obs WHERE scene = ? AND t BETWEEN ? AND ? ORDER BY t, rowid LIMIT ?", a.scene, at - windowMs, at + windowMs, MAX_CONTEXT + 1)
      .toArray();
    return { hits: rows.slice(0, MAX_CONTEXT).map((r) => rowToHit(r, includeEmbedding)), truncated: rows.length > MAX_CONTEXT };
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
    const r = this.sql.exec("SELECT COUNT(*) AS n, COUNT(DISTINCT NULLIF(label, '')) AS l, MAX(t) AS m FROM obs").toArray()[0];
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
