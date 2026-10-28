import { FRAMES, type Extents, type Frame, type ObservationIn, type QuerySpec } from "./types";
import { HttpError } from "./util";

export const MAX_BATCH = 50;
export const MAX_DELETES = 200;
const MAX_METADATA_BYTES = 4096;
export const MAX_EMBED_DIM = 1536;
export const MAX_QUERY_LIMIT = 1000;
const MAX_QUERY_LABELS = 16;
const MAX_HALF_LIFE_S = 365 * 86_400;
const ID_RE = /^[A-Za-z0-9_.:-]{1,96}$/;
const DAY_MS = 86_400_000;

export function bad(msg: string): never {
  throw new HttpError(422, msg, "invalid");
}

export function finite(v: unknown, name: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) bad(`${name} must be a finite number`);
  return v;
}

export function normLabel(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, " ");
}

export function normRoom(s: unknown): string | null {
  if (s === undefined || s === null || s === "") return null;
  if (typeof s !== "string" || s.length > 64) bad("room must be a string of at most 64 characters");
  return s.trim().toLowerCase().replace(/\s+/g, "_");
}

export function validateObservation(o: unknown, i: number, nowMs: number): ObservationIn {
  if (typeof o !== "object" || o === null) bad(`observations[${i}] must be an object`);
  const r = o as Record<string, unknown>;
  const at = `observations[${i}]`;
  if (typeof r.label !== "string" || !normLabel(r.label) || r.label.length > 64) bad(`${at}.label must be 1-64 characters`);
  if (r.id !== undefined && (typeof r.id !== "string" || !ID_RE.test(r.id))) bad(`${at}.id must match ${ID_RE}`);
  if (r.seq !== undefined && (!Number.isInteger(r.seq) || (r.seq as number) < 0)) bad(`${at}.seq must be a non-negative integer`);
  if (r.frame !== undefined && !FRAMES.includes(r.frame as Frame)) bad(`${at}.frame must be one of ${FRAMES.join(", ")}`);
  const t = finite(r.t, `${at}.t`);
  if (!Number.isInteger(t) || t < 0 || t > nowMs + DAY_MS) bad(`${at}.t must be epoch milliseconds, not in the future`);
  const confidence = r.confidence === undefined ? 1 : finite(r.confidence, `${at}.confidence`);
  if (confidence < 0 || confidence > 1) bad(`${at}.confidence must be in [0, 1]`);
  const embedding = r.embedding === undefined ? undefined : validateVector(r.embedding, `${at}.embedding`);
  if (r.metadata !== undefined) {
    if (typeof r.metadata !== "object" || r.metadata === null || Array.isArray(r.metadata)) bad(`${at}.metadata must be an object`);
    if (JSON.stringify(r.metadata).length > MAX_METADATA_BYTES) bad(`${at}.metadata exceeds ${MAX_METADATA_BYTES} bytes`);
  }
  if (r.media !== undefined && (typeof r.media !== "string" || r.media.length > 256)) bad(`${at}.media must be a string of at most 256 characters`);
  if (r.embedding_model !== undefined && (typeof r.embedding_model !== "string" || r.embedding_model.length > 64)) {
    bad(`${at}.embedding_model must be at most 64 characters`);
  }
  return {
    id: r.id as string | undefined,
    seq: r.seq as number | undefined,
    label: normLabel(r.label),
    frame: r.frame as Frame | undefined,
    room: normRoom(r.room),
    x: finite(r.x, `${at}.x`),
    y: finite(r.y, `${at}.y`),
    z: finite(r.z, `${at}.z`),
    t,
    confidence,
    embedding,
    embedding_model: r.embedding_model as string | undefined,
    metadata: r.metadata as Record<string, unknown> | undefined,
    media: r.media as string | undefined,
  };
}

export function validateBatch(
  body: unknown,
  nowMs: number,
): { observations: ObservationIn[]; deletes: string[]; upsert: boolean } {
  if (typeof body !== "object" || body === null) bad("body must be an object");
  const b = body as Record<string, unknown>;
  const obs = b.observations ?? [];
  const dels = b.deletes ?? [];
  const upsert = b.upsert ?? false;
  if (!Array.isArray(obs)) bad("observations must be an array");
  if (!Array.isArray(dels)) bad("deletes must be an array");
  if (typeof upsert !== "boolean") bad("upsert must be true or false");
  if (obs.length > MAX_BATCH) throw new HttpError(413, `at most ${MAX_BATCH} observations per request`, "too_large");
  if (dels.length > MAX_DELETES) throw new HttpError(413, `at most ${MAX_DELETES} deletes per request`, "too_large");
  if (obs.length === 0 && dels.length === 0) bad("nothing to write");
  for (const d of dels) if (typeof d !== "string" || !ID_RE.test(d)) bad("deletes must be observation ids");
  return { observations: obs.map((o, i) => validateObservation(o, i, nowMs)), deletes: dels as string[], upsert };
}

export function validateExtents(v: unknown, frame: Frame): Extents {
  if (v === undefined) {
    if (frame === "image") return { lo: [0, 0, 0], hi: [1, 1, 1] };
    if (frame === "camera") return { lo: [-3, -1.5, 0], hi: [3, 1.5, 6] };
    return { lo: [-10, -1, -10], hi: [10, 4, 10] };
  }
  const e = v as { lo?: unknown; hi?: unknown };
  const lo = triple(e.lo, "extents.lo");
  const hi = triple(e.hi, "extents.hi");
  if (lo.some((l, k) => l >= (hi[k] as number))) bad("extents.lo must be below extents.hi on every axis");
  return { lo, hi };
}

function triple(a: unknown, name: string): [number, number, number] {
  if (!Array.isArray(a) || a.length !== 3) bad(`${name} must be [x, y, z]`);
  return a.map((x, k) => finite(x, `${name}[${k}]`)) as [number, number, number];
}

export function validateVector(v: unknown, name: string): number[] {
  if (!Array.isArray(v) || v.length === 0 || v.length > MAX_EMBED_DIM) bad(`${name} must have 1-${MAX_EMBED_DIM} numbers`);
  return v.map((x, k) => finite(x, `${name}[${k}]`));
}

function epochMs(v: unknown, name: string): number {
  if (!Number.isSafeInteger(v) || (v as number) < 0) bad(`${name} must be epoch milliseconds`);
  return v as number;
}

/** POST /spaces/:space/query. Region and time bounds are inclusive, as in the library. */
export function validateQuery(body: unknown): QuerySpec {
  if (typeof body !== "object" || body === null || Array.isArray(body)) bad("body must be an object");
  const b = body as Record<string, unknown>;

  let region: Extents | null = null;
  if (b.region !== undefined && b.region !== null) {
    const r = b.region as { lo?: unknown; hi?: unknown };
    const lo = triple(r.lo, "region.lo");
    const hi = triple(r.hi, "region.hi");
    if (lo.some((l, k) => l > (hi[k] as number))) bad("region.lo must not exceed region.hi on any axis");
    region = { lo, hi };
  }

  let timeStart: number | null = null;
  let timeEnd: number | null = null;
  if (b.time !== undefined && b.time !== null) {
    if (typeof b.time !== "object" || Array.isArray(b.time)) bad("time must be {start, end}");
    const t = b.time as { start?: unknown; end?: unknown };
    timeStart = t.start === undefined || t.start === null ? null : epochMs(t.start, "time.start");
    timeEnd = t.end === undefined || t.end === null ? null : epochMs(t.end, "time.end");
    if (timeStart !== null && timeEnd !== null && timeStart > timeEnd) bad("time.start must be <= time.end");
  }

  if (b.label !== undefined && b.labels !== undefined) bad("send label or labels, not both");
  const rawLabels = b.labels ?? (b.label === undefined ? undefined : [b.label]);
  let labels: string[] | null = null;
  if (rawLabels !== undefined) {
    if (!Array.isArray(rawLabels) || rawLabels.length === 0 || rawLabels.length > MAX_QUERY_LABELS) {
      bad(`labels must have 1-${MAX_QUERY_LABELS} entries`);
    }
    labels = rawLabels.map((l, i) => {
      if (typeof l !== "string" || !normLabel(l) || l.length > 64) bad(`labels[${i}] must be 1-64 characters`);
      return normLabel(l);
    });
  }

  let minConfidence: number | null = null;
  if (b.min_confidence !== undefined && b.min_confidence !== null) {
    minConfidence = finite(b.min_confidence, "min_confidence");
    if (minConfidence < 0 || minConfidence > 1) bad("min_confidence must be in [0, 1]");
  }

  const halfLifeS = b.half_life_s === undefined ? 3600 : finite(b.half_life_s, "half_life_s");
  if (halfLifeS < 0 || halfLifeS > MAX_HALF_LIFE_S) bad(`half_life_s must be in [0, ${MAX_HALF_LIFE_S}]`);
  const limit = b.limit === undefined ? 10 : b.limit;
  if (!Number.isInteger(limit) || (limit as number) < 1 || (limit as number) > MAX_QUERY_LIMIT) {
    bad(`limit must be an integer in [1, ${MAX_QUERY_LIMIT}]`);
  }
  if (b.include_embedding !== undefined && typeof b.include_embedding !== "boolean") bad("include_embedding must be a boolean");

  return {
    region,
    timeStart,
    timeEnd,
    labels,
    room: normRoom(b.room),
    minConfidence,
    vector: b.vector === undefined || b.vector === null ? null : validateVector(b.vector, "vector"),
    halfLifeMs: halfLifeS * 1000,
    limit: limit as number,
    includeEmbedding: b.include_embedding === true,
  };
}
