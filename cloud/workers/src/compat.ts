/**
 * Compatibility layer for the Python SDK's cloud mode (`LociClient(base_url=..., api_key=...)`):
 *
 *   POST <project>/insert   one WorldState → {"id"}
 *   POST <project>/query    vector + box + time window → {"results": [WorldState...]}
 *
 * <project> is /p/<ref> (or <ref>.<PROJECT_HOST_SUFFIX>). Payloads, validation and responses follow
 * the retired FastAPI service (cloud/api/server.py). The project comes only from the route and the
 * credential, never the body; everything lands in the project's `default` space, which the first
 * insert creates with unit extents, so WorldState's normalised coordinates are its coordinates.
 */

import { canRead, canWrite, principalKey } from "./auth";
import { createSpace, getSpace, recordUsage, spaceStub } from "./control";
import type { Env, Principal, QueryResult, QuerySpec, SdkStateIn, SpaceMeta } from "./types";
import { bad, finite, MAX_QUERY_LIMIT, normLabel, validateVector } from "./validate";
import { HttpError, json, readJson } from "./util";

export const DEFAULT_SPACE = "default";
const MAX_METADATA_BYTES = 16 * 1024;
const SCALE_LEVELS = ["patch", "frame", "sequence"];
/** LociClient's default decay: one-hour half-life. */
const HALF_LIFE_MS = 3_600_000;
const enc = new TextEncoder();

function unit(v: unknown, name: string): number {
  const n = finite(v, name);
  if (n < 0 || n > 1) bad(`${name} must be in [0, 1]`);
  return n;
}

function bounded(v: unknown, name: string, def: number): number {
  if (v === undefined) return def;
  const n = finite(v, name);
  if (n < -1e9 || n > 1e9) bad(`${name} must be in [-1e9, 1e9]`);
  return n;
}

function optId(v: unknown, name: string): string | null {
  if (v === undefined || v === null) return null;
  if (typeof v !== "string" || v.length > 256) bad(`${name} must be a string of at most 256 characters`);
  return v;
}

function epochMs(v: unknown, name: string): number {
  if (!Number.isSafeInteger(v) || (v as number) < 0) bad(`${name} must be a non-negative integer`);
  return v as number;
}

function object(body: unknown): Record<string, unknown> {
  if (typeof body !== "object" || body === null || Array.isArray(body)) bad("body must be an object");
  return body as Record<string, unknown>;
}

export function validateSdkInsert(body: unknown): SdkStateIn {
  const b = object(body);
  if (typeof b.scene_id !== "string" || b.scene_id.length < 1 || b.scene_id.length > 256) bad("scene_id must be 1-256 characters");
  const scale = b.scale_level ?? "patch";
  if (typeof scale !== "string" || !SCALE_LEVELS.includes(scale)) bad(`scale_level must be one of ${SCALE_LEVELS.join(", ")}`);
  const confidence = b.confidence === undefined ? 1 : unit(b.confidence, "confidence");
  const metadata = b.metadata ?? {};
  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) bad("metadata must be an object");
  if (enc.encode(JSON.stringify(metadata)).length > MAX_METADATA_BYTES) bad(`metadata exceeds ${MAX_METADATA_BYTES} bytes JSON-serialized`);
  return {
    x: unit(b.x, "x"),
    y: unit(b.y, "y"),
    z: unit(b.z, "z"),
    timestamp_ms: epochMs(b.timestamp_ms, "timestamp_ms"),
    vector: validateVector(b.vector, "vector"),
    scene_id: b.scene_id,
    scale_level: scale,
    confidence,
    metadata: metadata as Record<string, unknown>,
    prev_state_id: optId(b.prev_state_id, "prev_state_id"),
    next_state_id: optId(b.next_state_id, "next_state_id"),
  };
}

/** The old /query body as a QuerySpec on the default space. Returns null when the box misses the unit cube. */
export function validateSdkQuery(body: unknown): { empty: boolean; spec: QuerySpec } {
  const b = object(body);
  const vector = validateVector(b.vector, "vector");
  const lo = (["x", "y", "z"] as const).map((a) => bounded(b[`${a}_min`], `${a}_min`, 0)) as [number, number, number];
  const hi = (["x", "y", "z"] as const).map((a) => bounded(b[`${a}_max`], `${a}_max`, 1)) as [number, number, number];
  ["x", "y", "z"].forEach((a, k) => {
    if (lo[k]! > hi[k]!) bad(`${a}_min must be <= ${a}_max`);
  });
  const start = b.time_start_ms === undefined || b.time_start_ms === null ? null : epochMs(b.time_start_ms, "time_start_ms");
  const end = b.time_end_ms === undefined || b.time_end_ms === null ? null : epochMs(b.time_end_ms, "time_end_ms");
  if (start !== null && end !== null && start > end) bad("time_start_ms must be <= time_end_ms");
  const limit = b.limit ?? 10;
  if (!Number.isInteger(limit) || (limit as number) < 1 || (limit as number) > MAX_QUERY_LIMIT) {
    bad(`limit must be an integer in [1, ${MAX_QUERY_LIMIT}]`);
  }
  // Accepted for compatibility; the box is applied exactly, so widening it changes nothing.
  const overlap = b.overlap_factor === undefined ? 1 : finite(b.overlap_factor, "overlap_factor");
  if (overlap < 0.1 || overlap > 10) bad("overlap_factor must be in [0.1, 10]");
  let minConfidence: number | null = null;
  if (b.min_confidence !== undefined && b.min_confidence !== null) minConfidence = unit(b.min_confidence, "min_confidence");
  if (b.include_vectors !== undefined && typeof b.include_vectors !== "boolean") bad("include_vectors must be a boolean");

  // WorldState coordinates live in [0, 1]; clip the box so every result parses on the client.
  const clipLo = lo.map((v) => Math.max(0, v)) as [number, number, number];
  const clipHi = hi.map((v) => Math.min(1, v)) as [number, number, number];
  const empty = clipLo.some((v, k) => v > clipHi[k]!);
  return {
    empty,
    spec: {
      region: { lo: clipLo, hi: clipHi },
      // As before: the window applies only when both ends are given.
      timeStart: start !== null && end !== null ? start : null,
      timeEnd: start !== null && end !== null ? end : null,
      labels: null,
      room: null,
      minConfidence,
      vector,
      halfLifeMs: HALF_LIFE_MS,
      limit: limit as number,
      includeEmbedding: b.include_vectors !== false,
    },
  };
}

/** The label a WorldState is filed under: its metadata.label, normalised, or '' (unlabelled). */
export function sdkLabel(metadata: Record<string, unknown>): string {
  const l = metadata.label;
  return typeof l === "string" && l.length <= 64 ? normLabel(l) : "";
}

function toWorldStates(r: QueryResult) {
  return r.hits.map((h) => ({
    id: h.obs.id,
    x: h.obs.x,
    y: h.obs.y,
    z: h.obs.z,
    timestamp_ms: h.obs.t,
    scene_id: h.sdk.scene_id ?? "",
    scale_level: h.sdk.scale_level ?? "patch",
    confidence: h.obs.confidence,
    metadata: h.obs.metadata,
    prev_state_id: h.sdk.prev_state_id,
    next_state_id: h.sdk.next_state_id,
    vector: h.embedding ?? [],
  }));
}

async function defaultSpace(env: Env, ref: string, create: boolean): Promise<SpaceMeta | null> {
  let row = await getSpace(env, ref, DEFAULT_SPACE);
  if (!row && create) {
    try {
      row = await createSpace(env, ref, { space: DEFAULT_SPACE, name: "Default (SDK)", frame: "room", extents: { lo: [0, 0, 0], hi: [1, 1, 1] } });
    } catch (e) {
      if (!(e instanceof HttpError && e.status === 409)) throw e;
      row = await getSpace(env, ref, DEFAULT_SPACE); // created by a concurrent insert
    }
  }
  return row ? { ref, space: DEFAULT_SPACE, frame: row.frame, extents: row.extents } : null;
}

export async function handleCompat(
  req: Request, env: Env, ctx: ExecutionContext, ref: string, p: Principal, action: "insert" | "query",
): Promise<Response> {
  if (req.method !== "POST") throw new HttpError(405, "method not allowed", "method_not_allowed");

  if (action === "insert") {
    if (!canWrite(p, DEFAULT_SPACE)) throw new HttpError(403, "not allowed to write to this project", "forbidden");
    if (env.WRITE_LIMITER) {
      const { success } = await env.WRITE_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
      if (!success) throw new HttpError(429, "write rate limit exceeded", "rate_limited");
    }
    const state = validateSdkInsert(await readJson(req));
    // Only a secret key may create the space; a device must already be assigned to it.
    const meta = await defaultSpace(env, ref, p.kind === "secret");
    if (!meta) throw new HttpError(404, `space ${DEFAULT_SPACE} not found`, "not_found");
    const stub = await spaceStub(env, ref, DEFAULT_SPACE);
    const res = await stub.insertState(meta, state, sdkLabel(state.metadata), p.kind === "device" ? p.deviceId : null);
    if ("error" in res) throw new HttpError(422, res.error, "invalid");
    ctx.waitUntil(recordUsage(env, ref, 1));
    env.USAGE?.writeDataPoint({ indexes: [ref], blobs: ["write", DEFAULT_SPACE], doubles: [1, 0] });
    return json({ id: res.id });
  }

  if (!canRead(p, DEFAULT_SPACE)) throw new HttpError(403, "not allowed to read this project", "forbidden");
  if (env.QUERY_LIMITER) {
    const { success } = await env.QUERY_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
    if (!success) throw new HttpError(429, "query rate limit exceeded", "rate_limited");
  }
  const { empty, spec } = validateSdkQuery(await readJson(req));
  ctx.waitUntil(recordUsage(env, ref, 0));
  const meta = await defaultSpace(env, ref, false);
  if (!meta || empty) return json({ results: [] });
  const res = await (await spaceStub(env, ref, DEFAULT_SPACE)).sdkQuery(meta, spec);
  if ("error" in res) throw new HttpError(422, res.error, "invalid");
  return json({ results: toWorldStates(res) });
}
