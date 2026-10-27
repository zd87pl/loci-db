/**
 * Credentials:
 *   loci_sk_<ref>_<48 hex>  secret key: full project access
 *   loci_pk_<ref>_<48 hex>  publishable key: read and ask
 *   <JWT>                   device token (HS256, project secret), write + read on its spaces
 * Lookups are cached per isolate for CACHE_MS, so revocation takes effect within a minute.
 */

import type { Env, Principal } from "./types";
import { HttpError, sha256Hex, verifyJwt } from "./util";

const CACHE_MS = 60_000;
const KEY_RE = /^loci_(sk|pk)_([a-z0-9]{12})_([0-9a-f]{48})$/;

interface Cached<T> {
  v: T;
  until: number;
}
const keyCache = new Map<string, Cached<{ id: string; kind: "secret" | "publishable"; ref: string } | null>>();
const projectCache = new Map<string, Cached<ProjectRow | null>>();
const deviceCache = new Map<string, Cached<DeviceRow | null>>();

export interface ProjectRow {
  ref: string;
  name: string;
  jurisdiction: "us" | "eu";
  plan: string;
  token_secret: string;
}

export interface DeviceRow {
  id: string;
  project_ref: string;
  spaces: string;
  revoked: number;
  token_epoch: number;
  secret_hash: string | null;
}

function cached<T>(m: Map<string, Cached<T>>, k: string): T | undefined {
  const c = m.get(k);
  if (c && c.until > Date.now()) return c.v;
  if (c) m.delete(k);
  return undefined;
}

function put<T>(m: Map<string, Cached<T>>, k: string, v: T): T {
  if (m.size > 5000) m.clear();
  m.set(k, { v, until: Date.now() + CACHE_MS });
  return v;
}

export function invalidateDevice(id: string): void {
  deviceCache.delete(id);
}

export async function getProject(env: Env, ref: string): Promise<ProjectRow | null> {
  const hit = cached(projectCache, ref);
  if (hit !== undefined) return hit;
  const row = await env.DB.prepare("SELECT ref, name, jurisdiction, plan, token_secret FROM projects WHERE ref = ?")
    .bind(ref)
    .first<ProjectRow>();
  return put(projectCache, ref, row ?? null);
}

export async function getDevice(env: Env, id: string): Promise<DeviceRow | null> {
  const hit = cached(deviceCache, id);
  if (hit !== undefined) return hit;
  const row = await env.DB.prepare(
    "SELECT id, project_ref, spaces, revoked, token_epoch, secret_hash FROM devices WHERE id = ?",
  )
    .bind(id)
    .first<DeviceRow>();
  return put(deviceCache, id, row ?? null);
}

export async function authenticate(env: Env, req: Request, ref: string): Promise<Principal> {
  const header = req.headers.get("authorization") ?? "";
  const m = header.match(/^Bearer\s+(\S+)$/i);
  if (!m || !m[1]) throw new HttpError(401, "missing bearer token", "unauthenticated");
  const token = m[1];

  const km = token.match(KEY_RE);
  if (km) {
    if (km[2] !== ref) throw new HttpError(403, "key belongs to another project", "forbidden");
    const hash = await sha256Hex(token);
    let row = cached(keyCache, hash);
    if (row === undefined) {
      const r = await env.DB.prepare("SELECT id, kind, project_ref AS ref FROM api_keys WHERE key_hash = ? AND revoked = 0")
        .bind(hash)
        .first<{ id: string; kind: "secret" | "publishable"; ref: string }>();
      row = put(keyCache, hash, r ?? null);
    }
    if (!row || row.ref !== ref) throw new HttpError(401, "invalid or revoked key", "unauthenticated");
    return row.kind === "secret" ? { kind: "secret", ref, keyId: row.id } : { kind: "publishable", ref, keyId: row.id };
  }

  if (token.split(".").length === 3) {
    const project = await getProject(env, ref);
    if (!project) throw new HttpError(404, "project not found", "not_found");
    const claims = await verifyJwt(token, project.token_secret, Math.floor(Date.now() / 1000));
    if (!claims || claims.ref !== ref || typeof claims.sub !== "string") {
      throw new HttpError(401, "invalid or expired device token", "unauthenticated");
    }
    const dev = await getDevice(env, claims.sub);
    if (!dev || dev.project_ref !== ref || dev.revoked || dev.token_epoch !== claims.ep) {
      throw new HttpError(401, "device revoked or token superseded", "unauthenticated");
    }
    return { kind: "device", ref, deviceId: dev.id, spaces: JSON.parse(dev.spaces) as string[] };
  }

  throw new HttpError(401, "unrecognised credential", "unauthenticated");
}

export function requireSecret(p: Principal): void {
  if (p.kind !== "secret") throw new HttpError(403, "this endpoint needs a secret key", "forbidden");
}

export function canRead(p: Principal, space: string): boolean {
  if (p.kind === "device") return p.spaces.includes(space);
  return true;
}

export function canWrite(p: Principal, space: string): boolean {
  if (p.kind === "secret") return true;
  if (p.kind === "device") return p.spaces.includes(space);
  return false;
}

export function principalKey(p: Principal): string {
  return p.kind === "device" ? `dev:${p.deviceId}` : `key:${p.keyId}`;
}
