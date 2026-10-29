/**
 * Compatibility layer for the Python SDK's cloud mode (`LociClient(base_url=..., api_key=...)`):
 *
 *   POST <project>/insert          one WorldState → {"id"}
 *   POST <project>/insert_batch    {"states": [...], "chains": {scene: id}} → {"ids", "chains"}
 *   POST <project>/query           vector + box + time window → {"results": [WorldState...]}
 *                                  ("include_scores": true adds "score" and "decayed_score")
 *   POST <project>/trajectory      {"state_id", "steps_back", "steps_forward"} → {"results"}
 *   POST <project>/causal_context  {"state_id", "window_ms"} → {"results"}
 *
 * <project> is /p/<ref> (or <ref>.<PROJECT_HOST_SUFFIX>). Payloads, validation and responses follow
 * the retired FastAPI service (`git show 917cc68:cloud/api/server.py`). The project comes only from
 * the route and the credential, never the body; everything lands in the project's `default` space,
 * which the first insert creates with unit extents, so WorldState's normalised coordinates are its
 * coordinates.
 */

import { canRead, canWrite, principalKey } from "./auth";
import { createSpace, getSpace, recordUsage, spaceStub } from "./control";
import type { BatchResult, Env, Principal, QueryHit, QueryResult, QuerySpec, SdkStateIn, SpaceMeta } from "./types";
import { bad, finite, MAX_QUERY_LIMIT, normLabel, validateVector } from "./validate";
import { HttpError, json, readJson } from "./util";

export const DEFAULT_SPACE = "default";
export const COMPAT_ACTIONS = ["insert", "insert_batch", "query", "trajectory", "causal_context"] as const;
export type CompatAction = (typeof COMPAT_ACTIONS)[number];
/** States per insert_batch request; the SDK splits larger batches and chains them. */
export const MAX_BATCH_STATES = 100;
const MAX_BATCH_BYTES = 2 * 1024 * 1024;
const MAX_CHAINS = 1000;
const MAX_STEPS = 1000;
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

function intIn(v: unknown, name: string, def: number, min: number, max: number): number {
  if (v === undefined) return def;
  if (!Number.isSafeInteger(v) || (v as number) < min || (v as number) > max) bad(`${name} must be an integer in [${min}, ${max}]`);
  return v as number;
}

function flag(v: unknown, name: string, def: boolean): boolean {
  if (v === undefined) return def;
  if (typeof v !== "boolean") bad(`${name} must be a boolean`);
  return v;
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
  const includeVectors = flag(b.include_vectors, "include_vectors", true);
  flag(b.include_scores, "include_scores", false);

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
      includeEmbedding: includeVectors,
    },
  };
}

/** insert_batch: up to MAX_BATCH_STATES states, plus each scene's last id from the previous chunk. */
export function validateSdkBatch(body: unknown): { states: SdkStateIn[]; chains: Record<string, string> } {
  const b = object(body);
  if (!Array.isArray(b.states) || b.states.length === 0) bad("states must be a non-empty array");
  if (b.states.length > MAX_BATCH_STATES) throw new HttpError(413, `at most ${MAX_BATCH_STATES} states per request`, "too_large");
  const states = b.states.map((st, i) => {
    try {
      return validateSdkInsert(st);
    } catch (e) {
      if (e instanceof HttpError) bad(`states[${i}]: ${e.detail}`);
      throw e;
    }
  });
  const chains = b.chains ?? {};
  if (typeof chains !== "object" || chains === null || Array.isArray(chains)) bad("chains must be an object of scene ids to state ids");
  const entries = Object.entries(chains);
  if (entries.length > MAX_CHAINS) bad(`chains must have at most ${MAX_CHAINS} entries`);
  for (const [scene, id] of entries) {
    if (scene.length > 256 || typeof id !== "string" || id.length < 1 || id.length > 256) bad("chains must map scene ids to state ids");
  }
  return { states, chains: chains as Record<string, string> };
}

function anchorId(b: Record<string, unknown>): string {
  if (typeof b.state_id !== "string" || b.state_id.length < 1 || b.state_id.length > 256) bad("state_id must be 1-256 characters");
  return b.state_id;
}

/** The label a WorldState is filed under: its metadata.label, normalised, or '' (unlabelled). */
export function sdkLabel(metadata: Record<string, unknown>): string {
  const l = metadata.label;
  return typeof l === "string" && l.length <= 64 ? normLabel(l) : "";
}

function toWorldStates(hits: QueryHit[], scores = false) {
  return hits.map((h) => ({
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
    ...(scores ? { score: h.similarity, decayed_score: h.score } : {}),
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
  return row ? { ref, space: DEFAULT_SPACE, frame: row.frame, extents: row.extents, epoch_ms: row.epoch_ms } : null;
}

export async function handleCompat(
  req: Request, env: Env, ctx: ExecutionContext, ref: string, p: Principal, action: CompatAction,
): Promise<Response> {
  if (req.method !== "POST") throw new HttpError(405, "method not allowed", "method_not_allowed");
  const deviceId = p.kind === "device" ? p.deviceId : null;

  if (action === "insert" || action === "insert_batch") {
    if (!canWrite(p, DEFAULT_SPACE)) throw new HttpError(403, "not allowed to write to this project", "forbidden");
    if (env.WRITE_LIMITER) {
      const { success } = await env.WRITE_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
      if (!success) throw new HttpError(429, "write rate limit exceeded", "rate_limited");
    }
    const body = await readJson(req, action === "insert_batch" ? MAX_BATCH_BYTES : undefined);
    const batch = action === "insert_batch" ? validateSdkBatch(body) : { states: [validateSdkInsert(body)], chains: {} };
    // Only a secret key may create the space; a device must already be assigned to it.
    const meta = await defaultSpace(env, ref, p.kind === "secret");
    if (!meta) throw new HttpError(404, `space ${DEFAULT_SPACE} not found`, "not_found");
    const stub = await spaceStub(env, ref, DEFAULT_SPACE);
    const labels = batch.states.map((st) => sdkLabel(st.metadata));
    const res: { id: string } | BatchResult | { error: string } = action === "insert"
      ? await stub.insertState(meta, batch.states[0]!, labels[0]!, deviceId)
      : await stub.insertStates(meta, batch.states, labels, batch.chains, deviceId);
    if ("error" in res) throw new HttpError(422, res.error, "invalid");
    const n = batch.states.length;
    ctx.waitUntil(recordUsage(env, ref, n));
    env.USAGE?.writeDataPoint({ indexes: [ref], blobs: ["write", DEFAULT_SPACE], doubles: [n, 0] });
    return json("id" in res ? { id: res.id } : { ids: res.ids, chains: res.chains });
  }

  if (!canRead(p, DEFAULT_SPACE)) throw new HttpError(403, "not allowed to read this project", "forbidden");
  if (env.QUERY_LIMITER) {
    const { success } = await env.QUERY_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
    if (!success) throw new HttpError(429, "query rate limit exceeded", "rate_limited");
  }
  const body = object(await readJson(req));
  ctx.waitUntil(recordUsage(env, ref, 0));
  const meta = await defaultSpace(env, ref, false);
  const stub = () => spaceStub(env, ref, DEFAULT_SPACE);
  // The RPC stub types these replies as never (metadata is Record<string, unknown>), so name them here.

  if (action === "query") {
    const { empty, spec } = validateSdkQuery(body);
    if (!meta || empty) return json({ results: [] });
    const res = (await (await stub()).sdkQuery(meta, spec)) as QueryResult | { error: string };
    if ("error" in res) throw new HttpError(422, res.error, "invalid");
    return json({ results: toWorldStates(res.hits, body.include_scores === true) });
  }
  const id = anchorId(body);
  const includeVectors = flag(body.include_vectors, "include_vectors", true);
  if (action === "trajectory") {
    const back = intIn(body.steps_back, "steps_back", 10, 0, MAX_STEPS);
    const forward = intIn(body.steps_forward, "steps_forward", 10, 0, MAX_STEPS);
    const hits = meta ? ((await (await stub()).trajectory(id, back, forward, includeVectors)) as QueryHit[]) : [];
    return json({ results: toWorldStates(hits) });
  }
  const windowMs = intIn(body.window_ms, "window_ms", 5000, 0, Number.MAX_SAFE_INTEGER);
  if (!meta) return json({ results: [] });
  const res = (await (await stub()).causalContext(id, windowMs, includeVectors)) as QueryResult;
  return json({ results: toWorldStates(res.hits), ...(res.truncated ? { truncated: true } : {}) });
}
