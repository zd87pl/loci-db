/** Control plane: projects, keys, spaces, devices (claim and tokens). Backed by D1. */

import { getProject, invalidateDevice } from "./auth";
import type { SpaceObject } from "./space";
import type { Env, Extents, Frame, SpaceMeta } from "./types";
import { FRAMES } from "./types";
import { validateExtents } from "./validate";
import { HttpError, SLUG, claimCode, randomAlnum, randomHex, sha256Hex, signJwt } from "./util";

const CLAIM_TTL_MS = 30 * 60_000;

function newKey(kind: "sk" | "pk", ref: string): string {
  return `loci_${kind}_${ref}_${randomHex(24)}`;
}

export async function createProject(
  env: Env,
  body: { name?: unknown; owner_email?: unknown; jurisdiction?: unknown },
): Promise<{ ref: string; name: string; jurisdiction: string; secret_key: string; publishable_key: string }> {
  const name = typeof body.name === "string" && body.name.trim() ? body.name.trim().slice(0, 128) : "LOCI project";
  const jurisdiction = body.jurisdiction === undefined ? "us" : body.jurisdiction;
  if (jurisdiction !== "us" && jurisdiction !== "eu") throw new HttpError(422, "jurisdiction must be us or eu", "invalid");
  const email = typeof body.owner_email === "string" ? body.owner_email.slice(0, 320) : null;
  const ref = randomAlnum(12);
  const now = Date.now();
  const sk = newKey("sk", ref);
  const pk = newKey("pk", ref);
  await env.DB.batch([
    env.DB.prepare("INSERT INTO projects (ref, name, owner_email, jurisdiction, token_secret, created_at) VALUES (?, ?, ?, ?, ?, ?)").bind(
      ref, name, email, jurisdiction, randomHex(32), now,
    ),
    env.DB.prepare("INSERT INTO api_keys (id, project_ref, kind, key_hash, prefix, label, created_at) VALUES (?, ?, 'secret', ?, ?, 'default', ?)").bind(
      `key_${randomHex(8)}`, ref, await sha256Hex(sk), sk.slice(0, 20), now,
    ),
    env.DB.prepare("INSERT INTO api_keys (id, project_ref, kind, key_hash, prefix, label, created_at) VALUES (?, ?, 'publishable', ?, ?, 'default', ?)").bind(
      `key_${randomHex(8)}`, ref, await sha256Hex(pk), pk.slice(0, 20), now,
    ),
  ]);
  return { ref, name, jurisdiction, secret_key: sk, publishable_key: pk };
}

export interface SpaceRow {
  space: string;
  name: string | null;
  frame: Frame;
  extents: Extents;
}

const spaceCache = new Map<string, { v: SpaceRow | null; until: number }>();

export async function getSpace(env: Env, ref: string, space: string): Promise<SpaceRow | null> {
  const k = `${ref}/${space}`;
  const c = spaceCache.get(k);
  if (c && c.until > Date.now()) return c.v;
  const r = await env.DB.prepare("SELECT space, name, frame, extents FROM spaces WHERE project_ref = ? AND space = ?")
    .bind(ref, space)
    .first<{ space: string; name: string | null; frame: Frame; extents: string }>();
  const v = r ? { ...r, extents: JSON.parse(r.extents) as Extents } : null;
  spaceCache.set(k, { v, until: Date.now() + 60_000 });
  return v;
}

function namespaceFor(env: Env, jurisdiction: string | undefined): DurableObjectNamespace<SpaceObject> {
  if (jurisdiction !== "eu") return env.SPACE;
  try {
    return env.SPACE.jurisdiction("eu");
  } catch (e) {
    // The local runtime (wrangler dev, tests) does not implement jurisdictions. Production must
    // never silently store EU data unpinned, so the fallback needs an explicit dev-only flag.
    if (env.DEV_ALLOW_UNPINNED === "1") {
      console.warn("EU jurisdiction unavailable locally; using an unpinned namespace (DEV_ALLOW_UNPINNED=1)");
      return env.SPACE;
    }
    throw e;
  }
}

export async function spaceStub(env: Env, ref: string, space: string) {
  const project = await getProject(env, ref);
  const ns = namespaceFor(env, project?.jurisdiction);
  return ns.get(ns.idFromName(`${ref}/${space}`));
}

export async function createSpace(env: Env, ref: string, body: Record<string, unknown>): Promise<SpaceRow> {
  const space = body.space;
  if (typeof space !== "string" || !SLUG.test(space)) throw new HttpError(422, "space must be a lowercase slug (a-z, 0-9, -)", "invalid");
  const frame = (body.frame ?? "room") as Frame;
  if (!FRAMES.includes(frame)) throw new HttpError(422, `frame must be one of ${FRAMES.join(", ")}`, "invalid");
  const extents = validateExtents(body.extents, frame);
  const name = typeof body.name === "string" ? body.name.slice(0, 128) : null;
  const res = await env.DB.prepare(
    "INSERT OR IGNORE INTO spaces (project_ref, space, name, frame, extents, created_at) VALUES (?, ?, ?, ?, ?, ?)",
  )
    .bind(ref, space, name, frame, JSON.stringify(extents), Date.now())
    .run();
  if (!res.meta.changes) throw new HttpError(409, `space ${space} already exists`, "conflict");
  spaceCache.delete(`${ref}/${space}`);
  const meta: SpaceMeta = { ref, space, frame, extents };
  await (await spaceStub(env, ref, space)).configure(meta);
  return { space, name, frame, extents };
}

export async function listSpaces(env: Env, ref: string): Promise<SpaceRow[]> {
  const { results } = await env.DB.prepare("SELECT space, name, frame, extents FROM spaces WHERE project_ref = ? ORDER BY space")
    .bind(ref)
    .all<{ space: string; name: string | null; frame: Frame; extents: string }>();
  return results.map((r) => ({ ...r, extents: JSON.parse(r.extents) as Extents }));
}

export async function createDevice(
  env: Env,
  ref: string,
  body: Record<string, unknown>,
): Promise<{ device_id: string; name: string; spaces: string[]; claim_code: string; claim_expires_at: number }> {
  const name = typeof body.name === "string" && body.name.trim() ? body.name.trim().slice(0, 128) : "device";
  const spaces = body.spaces;
  if (!Array.isArray(spaces) || spaces.length === 0 || spaces.length > 32 || spaces.some((s) => typeof s !== "string")) {
    throw new HttpError(422, "spaces must be a non-empty list of space slugs", "invalid");
  }
  for (const s of spaces as string[]) {
    if (!(await getSpace(env, ref, s))) throw new HttpError(422, `unknown space ${s}`, "invalid");
  }
  const id = `dev_${randomHex(8)}`;
  const code = claimCode();
  const expires = Date.now() + CLAIM_TTL_MS;
  await env.DB.prepare(
    "INSERT INTO devices (id, project_ref, name, spaces, claim_hash, claim_expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(id, ref, name, JSON.stringify(spaces), await sha256Hex(code), expires, Date.now())
    .run();
  return { device_id: id, name, spaces: spaces as string[], claim_code: code, claim_expires_at: expires };
}

/** Exchange a one-time claim code for a long-lived device secret. */
export async function claimDevice(env: Env, ref: string, body: Record<string, unknown>) {
  if (typeof body.claim_code !== "string") throw new HttpError(422, "claim_code is required", "invalid");
  const codeHash = await sha256Hex(body.claim_code.trim().toUpperCase());
  const row = await env.DB.prepare(
    "SELECT id, spaces, claim_expires_at FROM devices WHERE project_ref = ? AND claim_hash = ? AND revoked = 0",
  )
    .bind(ref, codeHash)
    .first<{ id: string; spaces: string; claim_expires_at: number }>();
  if (!row || row.claim_expires_at < Date.now()) throw new HttpError(401, "invalid or expired claim code", "unauthenticated");
  const secret = `loci_ds_${randomHex(24)}`;
  // Single use: the claim hash is cleared in the same statement that stores the secret.
  const res = await env.DB.prepare(
    "UPDATE devices SET secret_hash = ?, claim_hash = NULL, claim_expires_at = NULL, claimed_at = ? WHERE id = ? AND claim_hash = ?",
  )
    .bind(await sha256Hex(secret), Date.now(), row.id, codeHash)
    .run();
  if (!res.meta.changes) throw new HttpError(401, "claim code already used", "unauthenticated");
  invalidateDevice(row.id);
  return { device_id: row.id, device_secret: secret, spaces: JSON.parse(row.spaces) as string[] };
}

export async function issueDeviceToken(env: Env, ref: string, body: Record<string, unknown>) {
  if (typeof body.device_id !== "string" || typeof body.device_secret !== "string") {
    throw new HttpError(422, "device_id and device_secret are required", "invalid");
  }
  const row = await env.DB.prepare(
    "SELECT id, spaces, secret_hash, revoked, token_epoch FROM devices WHERE id = ? AND project_ref = ?",
  )
    .bind(body.device_id, ref)
    .first<{ id: string; spaces: string; secret_hash: string | null; revoked: number; token_epoch: number }>();
  const hash = await sha256Hex(body.device_secret);
  if (!row || row.revoked || !row.secret_hash || row.secret_hash !== hash) {
    throw new HttpError(401, "invalid device credentials", "unauthenticated");
  }
  const project = await getProject(env, ref);
  if (!project) throw new HttpError(404, "project not found", "not_found");
  const ttl = Number(env.DEVICE_TOKEN_TTL_S ?? "3600") || 3600;
  const exp = Math.floor(Date.now() / 1000) + ttl;
  const token = await signJwt({ sub: row.id, ref, ep: row.token_epoch, exp }, project.token_secret);
  await env.DB.prepare("UPDATE devices SET last_seen_at = ? WHERE id = ?").bind(Date.now(), row.id).run();
  // expires_in lets a device without a set clock schedule its refresh.
  return { token, expires_at: exp, expires_in: ttl, spaces: JSON.parse(row.spaces) as string[] };
}

export async function listDevices(env: Env, ref: string) {
  const { results } = await env.DB.prepare(
    "SELECT id, name, spaces, claimed_at, revoked, last_seen_at, created_at FROM devices WHERE project_ref = ? ORDER BY created_at",
  )
    .bind(ref)
    .all<{ id: string; name: string; spaces: string; claimed_at: number | null; revoked: number; last_seen_at: number | null; created_at: number }>();
  return results.map((r) => ({ ...r, spaces: JSON.parse(r.spaces) as string[], revoked: !!r.revoked }));
}

export async function revokeDevice(env: Env, ref: string, id: string): Promise<void> {
  const res = await env.DB.prepare(
    "UPDATE devices SET revoked = 1, token_epoch = token_epoch + 1, claim_hash = NULL WHERE id = ? AND project_ref = ?",
  )
    .bind(id, ref)
    .run();
  if (!res.meta.changes) throw new HttpError(404, "device not found", "not_found");
  invalidateDevice(id);
}

export async function recordUsage(env: Env, ref: string, writes: number): Promise<void> {
  const day = new Date().toISOString().slice(0, 10);
  await env.DB.prepare(
    `INSERT INTO usage_daily (project_ref, day, requests, writes) VALUES (?, ?, 1, ?)
     ON CONFLICT(project_ref, day) DO UPDATE SET requests = requests + 1, writes = writes + excluded.writes`,
  )
    .bind(ref, day, writes)
    .run();
}
