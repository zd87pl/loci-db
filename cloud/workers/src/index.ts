/**
 * LOCI Cloud router Worker (Phase 0).
 *
 *   GET  /health
 *   POST /admin/projects                         Bearer ADMIN_TOKEN
 *   /p/<ref>/v1/...   (or <ref>.<PROJECT_HOST_SUFFIX>/v1/... when the suffix is set)
 *     GET    spaces                              secret | publishable
 *     POST   spaces                              secret
 *     GET    devices                             secret
 *     POST   devices                             secret → claim code
 *     DELETE devices/<id>                        secret
 *     POST   devices/claim                       claim code → device secret (no bearer)
 *     POST   devices/token                       device secret → 1 h token (no bearer)
 *     POST   spaces/<space>/observations         secret | device(space)
 *     GET    spaces/<space>/where?label=&limit=&room=
 *     GET    spaces/<space>/changes?window_s=&room=
 *     GET    spaces/<space>/digest?since_rev=
 *     POST   spaces/<space>/query                region / time / label / room / confidence / vector filters
 *     POST   spaces/<space>/ask                  {"text": "..."}
 *     GET    spaces/<space>/stats
 *     *      spaces/<space>/board/...            ESP32 boards: cache JSON, spoken answers (src/board.ts)
 *   /p/<ref>/{insert,insert_batch,query,trajectory,causal_context}
 *                                                  Python SDK cloud mode, on the default space (src/compat.ts)
 */

import { authenticate, canRead, canWrite, principalKey, requireSecret } from "./auth";
import { handleBoard } from "./board";
import {
  claimDevice, createDevice, createProject, createSpace, getSpace, issueDeviceToken, listDevices, listSpaces,
  recordUsage, revokeDevice, spaceStub,
} from "./control";
import { COMPAT_ACTIONS, type CompatAction, handleCompat } from "./compat";
import type { Env, Principal, QueryResult, SpaceMeta } from "./types";
import { normLabel, normRoom, validateBatch, validateQuery } from "./validate";
import { HttpError, errorResponse, json, readJson, timingSafeEqualStr } from "./util";

export { SpaceObject } from "./space";

const VERSION = "0.1.0";

/** Splits a project URL into its ref and the rest: "v1/<path>" or the SDK's "insert" / "query". */
function routeProject(url: URL, env: Env): { ref: string; rest: string } | null {
  const suffix = env.PROJECT_HOST_SUFFIX;
  let ref: string;
  let path: string;
  if (suffix && url.hostname.endsWith(`.${suffix}`)) {
    ref = url.hostname.slice(0, -(suffix.length + 1));
    path = url.pathname;
  } else {
    const m = url.pathname.match(/^\/p\/([a-z0-9]{12})(\/.*)$/);
    if (!m) return null;
    ref = m[1] as string;
    path = m[2] as string;
  }
  const m = path.match(/^\/(v1\/.*|insert|insert_batch|query|trajectory|causal_context)$/);
  return m ? { ref, rest: m[1] as string } : null;
}

function queryResponse(r: QueryResult) {
  return {
    results: r.hits.map((h) => ({
      ...h.obs,
      address: h.address,
      ...(h.score === null ? {} : { similarity: h.similarity, score: h.score }),
      ...(h.embedding === null ? {} : { embedding: h.embedding }),
    })),
    truncated: r.truncated,
  };
}

function intParam(url: URL, name: string, def: number, min: number, max: number): number {
  const v = url.searchParams.get(name);
  if (v === null || v === "") return def;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new HttpError(422, `${name} must be a number`, "invalid");
  return Math.min(max, Math.max(min, Math.floor(n)));
}

async function handleSpace(
  req: Request, env: Env, ctx: ExecutionContext, url: URL, ref: string, p: Principal, space: string, action: string,
): Promise<Response> {
  const row = await getSpace(env, ref, space);
  if (!row) throw new HttpError(404, `space ${space} not found`, "not_found");
  const stub = await spaceStub(env, ref, space);

  if (action === "observations" && req.method === "POST") {
    if (!canWrite(p, space)) throw new HttpError(403, "not allowed to write to this space", "forbidden");
    if (env.WRITE_LIMITER) {
      const { success } = await env.WRITE_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
      if (!success) throw new HttpError(429, "write rate limit exceeded", "rate_limited");
    }
    const { observations, deletes, upsert } = validateBatch(await readJson(req), Date.now());
    const meta: SpaceMeta = { ref, space, frame: row.frame, extents: row.extents, epoch_ms: row.epoch_ms };
    const result = await stub.insertBatch(meta, observations, deletes, p.kind === "device" ? p.deviceId : null, upsert);
    ctx.waitUntil(recordUsage(env, ref, result.accepted + result.updated));
    env.USAGE?.writeDataPoint({
      indexes: [ref], blobs: ["write", space], doubles: [result.accepted, result.deleted, result.updated],
    });
    return json(result, result.accepted > 0 || result.deleted > 0 || result.updated > 0 ? 201 : 200);
  }

  if (!canRead(p, space)) throw new HttpError(403, "not allowed to read this space", "forbidden");
  ctx.waitUntil(recordUsage(env, ref, 0));

  if (action === "where" && req.method === "GET") {
    const label = normLabel(url.searchParams.get("label") ?? "");
    if (!label) throw new HttpError(422, "label is required", "invalid");
    const results = await stub.where(label, intParam(url, "limit", 5, 1, 100), normRoom(url.searchParams.get("room")));
    return json({ label, results });
  }
  if (action === "query" && req.method === "POST") {
    if (env.QUERY_LIMITER) {
      const { success } = await env.QUERY_LIMITER.limit({ key: `${ref}:${principalKey(p)}` });
      if (!success) throw new HttpError(429, "query rate limit exceeded", "rate_limited");
    }
    const spec = validateQuery(await readJson(req));
    const meta: SpaceMeta = { ref, space, frame: row.frame, extents: row.extents, epoch_ms: row.epoch_ms };
    return json(queryResponse(await stub.query(meta, spec)));
  }
  if (action === "changes" && req.method === "GET") {
    const changes = await stub.changes(intParam(url, "window_s", 600, 1, 30 * 86_400), normRoom(url.searchParams.get("room")));
    return json({ changes });
  }
  if (action === "digest" && req.method === "GET") {
    return json(await stub.digest(intParam(url, "since_rev", 0, 0, Number.MAX_SAFE_INTEGER)));
  }
  if (action === "ask" && req.method === "POST") {
    const body = await readJson<{ text?: unknown }>(req, 8 * 1024);
    if (typeof body.text !== "string" || !body.text.trim() || body.text.length > 500) {
      throw new HttpError(422, "text must be 1-500 characters", "invalid");
    }
    return json(await stub.ask(body.text));
  }
  if (action === "stats" && req.method === "GET") return json({ space, ...(await stub.stats()) });
  throw new HttpError(404, "not found", "not_found");
}

async function handleProject(req: Request, env: Env, ctx: ExecutionContext, url: URL, ref: string, rest: string): Promise<Response> {
  if ((COMPAT_ACTIONS as readonly string[]).includes(rest)) {
    return handleCompat(req, env, ctx, ref, await authenticate(env, req, ref), rest as CompatAction);
  }
  const parts = rest.slice("v1/".length).split("/").filter(Boolean);

  // Unauthenticated device bootstrap.
  if (req.method === "POST" && parts[0] === "devices" && parts[1] === "claim" && parts.length === 2) {
    return json(await claimDevice(env, ref, await readJson(req, 4096)));
  }
  if (req.method === "POST" && parts[0] === "devices" && parts[1] === "token" && parts.length === 2) {
    return json(await issueDeviceToken(env, ref, await readJson(req, 4096)));
  }

  const p = await authenticate(env, req, ref);

  if (parts[0] === "spaces" && parts.length === 1) {
    if (req.method === "GET") {
      const spaces = await listSpaces(env, ref);
      return json({ spaces: p.kind === "device" ? spaces.filter((s) => p.spaces.includes(s.space)) : spaces });
    }
    if (req.method === "POST") {
      requireSecret(p);
      return json(await createSpace(env, ref, await readJson(req, 4096)), 201);
    }
  }
  if (parts[0] === "spaces" && parts.length === 3) {
    return handleSpace(req, env, ctx, url, ref, p, parts[1] as string, parts[2] as string);
  }
  if (parts[0] === "spaces" && parts.length === 4 && parts[2] === "board") {
    ctx.waitUntil(recordUsage(env, ref, 0));
    return handleBoard(req, env, url, ref, p, parts[1] as string, parts[3] as string);
  }
  if (parts[0] === "devices" && parts.length === 1) {
    requireSecret(p);
    if (req.method === "GET") return json({ devices: await listDevices(env, ref) });
    if (req.method === "POST") return json(await createDevice(env, ref, await readJson(req, 4096)), 201);
  }
  if (parts[0] === "devices" && parts.length === 2 && req.method === "DELETE") {
    requireSecret(p);
    await revokeDevice(env, ref, parts[1] as string);
    return json({ revoked: parts[1] });
  }
  throw new HttpError(404, "not found", "not_found");
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    try {
      if (url.pathname === "/health") return json({ status: "ok", version: VERSION, sha: env.GIT_SHA || "dev" });

      if (url.pathname === "/admin/projects" && req.method === "POST") {
        const token = (req.headers.get("authorization") ?? "").replace(/^Bearer\s+/i, "");
        if (!env.ADMIN_TOKEN || !token || !timingSafeEqualStr(token, env.ADMIN_TOKEN)) {
          throw new HttpError(401, "admin token required", "unauthenticated");
        }
        return json(await createProject(env, await readJson(req, 4096)), 201);
      }

      const routed = routeProject(url, env);
      if (!routed) throw new HttpError(404, "not found", "not_found");
      return await handleProject(req, env, ctx, url, routed.ref, routed.rest);
    } catch (e) {
      if (e instanceof HttpError) return errorResponse(e);
      console.error("unhandled", e);
      return json({ detail: "internal error", code: "internal" }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
