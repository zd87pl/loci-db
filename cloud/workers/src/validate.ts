import { FRAMES, type Extents, type Frame, type ObservationIn } from "./types";
import { HttpError } from "./util";

export const MAX_BATCH = 50;
export const MAX_DELETES = 200;
const MAX_METADATA_BYTES = 4096;
const MAX_EMBED_DIM = 1536;
const ID_RE = /^[A-Za-z0-9_.:-]{1,96}$/;
const DAY_MS = 86_400_000;

function bad(msg: string): never {
  throw new HttpError(422, msg, "invalid");
}

function finite(v: unknown, name: string): number {
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
  let embedding: number[] | undefined;
  if (r.embedding !== undefined) {
    if (!Array.isArray(r.embedding) || r.embedding.length === 0 || r.embedding.length > MAX_EMBED_DIM) {
      bad(`${at}.embedding must have 1-${MAX_EMBED_DIM} numbers`);
    }
    embedding = r.embedding.map((v, k) => finite(v, `${at}.embedding[${k}]`));
  }
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

export function validateBatch(body: unknown, nowMs: number): { observations: ObservationIn[]; deletes: string[] } {
  if (typeof body !== "object" || body === null) bad("body must be an object");
  const b = body as Record<string, unknown>;
  const obs = b.observations ?? [];
  const dels = b.deletes ?? [];
  if (!Array.isArray(obs)) bad("observations must be an array");
  if (!Array.isArray(dels)) bad("deletes must be an array");
  if (obs.length > MAX_BATCH) throw new HttpError(413, `at most ${MAX_BATCH} observations per request`, "too_large");
  if (dels.length > MAX_DELETES) throw new HttpError(413, `at most ${MAX_DELETES} deletes per request`, "too_large");
  if (obs.length === 0 && dels.length === 0) bad("nothing to write");
  for (const d of dels) if (typeof d !== "string" || !ID_RE.test(d)) bad("deletes must be observation ids");
  return { observations: obs.map((o, i) => validateObservation(o, i, nowMs)), deletes: dels as string[] };
}

export function validateExtents(v: unknown, frame: Frame): Extents {
  if (v === undefined) {
    if (frame === "image") return { lo: [0, 0, 0], hi: [1, 1, 1] };
    if (frame === "camera") return { lo: [-3, -1.5, 0], hi: [3, 1.5, 6] };
    return { lo: [-10, -1, -10], hi: [10, 4, 10] };
  }
  const e = v as { lo?: unknown; hi?: unknown };
  const triple = (a: unknown, n: string): [number, number, number] => {
    if (!Array.isArray(a) || a.length !== 3) bad(`extents.${n} must be [x, y, z]`);
    return a.map((x, k) => finite(x, `extents.${n}[${k}]`)) as [number, number, number];
  };
  const lo = triple(e.lo, "lo");
  const hi = triple(e.hi, "hi");
  if (lo.some((l, k) => l >= (hi[k] as number))) bad("extents.lo must be below extents.hi on every axis");
  return { lo, hi };
}
