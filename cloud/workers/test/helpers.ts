import { exports } from "cloudflare:workers";
import { expect } from "vitest";

const BASE = "http://loci.test";
export type Json = Record<string, any>;

export async function call(method: string, path: string, opts: { token?: string; body?: unknown } = {}): Promise<{ status: number; body: Json }> {
  const headers: Record<string, string> = {};
  if (opts.token) headers.authorization = `Bearer ${opts.token}`;
  if (opts.body !== undefined) headers["content-type"] = "application/json";
  const res = await exports.default.fetch(new Request(`${BASE}${path}`, {
    method, headers, body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  }));
  return { status: res.status, body: (await res.json()) as Json };
}

/** A fresh project: `base` is /p/<ref> (the SDK's base_url path), `v1` is /p/<ref>/v1. */
export async function newProject(name = "test"): Promise<{ ref: string; sk: string; pk: string; base: string; v1: string }> {
  const r = await call("POST", "/admin/projects", { token: "test-admin-token", body: { name } });
  expect(r.status).toBe(201);
  const ref = r.body.ref as string;
  return { ref, sk: r.body.secret_key, pk: r.body.publishable_key, base: `/p/${ref}`, v1: `/p/${ref}/v1` };
}
