import { exports } from "cloudflare:workers";
import { beforeAll, describe, expect, it } from "vitest";

const BASE = "http://loci.test";
type Json = Record<string, any>;

async function call(method: string, path: string, opts: { token?: string; body?: unknown } = {}): Promise<{ status: number; body: Json }> {
  const headers: Record<string, string> = {};
  if (opts.token) headers.authorization = `Bearer ${opts.token}`;
  if (opts.body !== undefined) headers["content-type"] = "application/json";
  const res = await exports.default.fetch(new Request(`${BASE}${path}`, {
    method, headers, body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  }));
  return { status: res.status, body: (await res.json()) as Json };
}

let ref = "";
let sk = "";
let pk = "";
let p = "";
let deviceId = "";
let deviceSecret = "";
let deviceToken = "";

const now = Date.now();
const obs = (label: string, x: number, z: number, t: number, extra: Json = {}) => ({ label, x, y: 0.8, z, t, confidence: 0.9, room: "kitchen", ...extra });

beforeAll(async () => {
  const r = await call("POST", "/admin/projects", { token: "test-admin-token", body: { name: "Home assistant", owner_email: "a@b.c" } });
  expect(r.status).toBe(201);
  ({ ref, secret_key: sk, publishable_key: pk } = r.body as any);
  p = `/p/${ref}/v1`;
});

describe("health and admin", () => {
  it("serves health", async () => {
    expect((await call("GET", "/health")).body.status).toBe("ok");
  });
  it("rejects project creation without the admin token", async () => {
    expect((await call("POST", "/admin/projects", { token: "nope", body: {} })).status).toBe(401);
  });
  it("issues keys bound to the project ref", () => {
    expect(ref).toMatch(/^[a-z0-9]{12}$/);
    expect(sk.startsWith(`loci_sk_${ref}_`)).toBe(true);
    expect(pk.startsWith(`loci_pk_${ref}_`)).toBe(true);
  });
  it("creates an EU-pinned project", async () => {
    const r = await call("POST", "/admin/projects", { token: "test-admin-token", body: { name: "Care home", jurisdiction: "eu" } });
    expect(r.status).toBe(201);
    expect(r.body.jurisdiction).toBe("eu");
    const space = await call("POST", `/p/${r.body.ref}/v1/spaces`, { token: r.body.secret_key, body: { space: "ward-a" } });
    expect(space.status).toBe(201);
  });
});

describe("spaces and devices", () => {
  it("creates spaces with a secret key only", async () => {
    expect((await call("POST", `${p}/spaces`, { token: sk, body: { space: "home", name: "Home" } })).status).toBe(201);
    expect((await call("POST", `${p}/spaces`, { token: sk, body: { space: "office" } })).status).toBe(201);
    expect((await call("POST", `${p}/spaces`, { token: sk, body: { space: "home" } })).status).toBe(409);
    expect((await call("POST", `${p}/spaces`, { token: pk, body: { space: "garage" } })).status).toBe(403);
    expect((await call("POST", `${p}/spaces`, { token: sk, body: { space: "Bad Slug" } })).status).toBe(422);
    const list = await call("GET", `${p}/spaces`, { token: pk });
    expect(list.body.spaces.map((s: Json) => s.space)).toEqual(["home", "office"]);
  });

  it("rejects keys from another project and malformed tokens", async () => {
    const other = await call("POST", "/admin/projects", { token: "test-admin-token", body: {} });
    expect((await call("GET", `${p}/spaces`, { token: other.body.secret_key })).status).toBe(403);
    expect((await call("GET", `${p}/spaces`)).status).toBe(401);
    expect((await call("GET", `${p}/spaces`, { token: "garbage" })).status).toBe(401);
  });

  it("provisions a device: claim code once, then secret, then a token", async () => {
    const d = await call("POST", `${p}/devices`, { token: sk, body: { name: "ESP32 pendant", spaces: ["home"] } });
    expect(d.status).toBe(201);
    expect(d.body.claim_code).toMatch(/^[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$/);
    const claim = await call("POST", `${p}/devices/claim`, { body: { claim_code: d.body.claim_code.toLowerCase() } });
    expect(claim.status).toBe(200);
    deviceId = claim.body.device_id;
    deviceSecret = claim.body.device_secret;
    expect((await call("POST", `${p}/devices/claim`, { body: { claim_code: d.body.claim_code } })).status).toBe(401);
    const tok = await call("POST", `${p}/devices/token`, { body: { device_id: deviceId, device_secret: deviceSecret } });
    expect(tok.status).toBe(200);
    expect(tok.body.spaces).toEqual(["home"]);
    deviceToken = tok.body.token;
    expect((await call("POST", `${p}/devices/token`, { body: { device_id: deviceId, device_secret: "loci_ds_wrong" } })).status).toBe(401);
    const devices = await call("GET", `${p}/devices`, { token: sk });
    expect(devices.body.devices[0].claimed_at).toBeTypeOf("number");
  });

  it("rejects devices for unknown spaces", async () => {
    expect((await call("POST", `${p}/devices`, { token: sk, body: { name: "x", spaces: ["nowhere"] } })).status).toBe(422);
  });
});

describe("observations and queries", () => {
  it("lets a device write only to its own spaces", async () => {
    const ok = await call("POST", `${p}/spaces/home/observations`, {
      token: deviceToken,
      body: { observations: [obs("cup", -1.2, 1.5, now - 600_000, { seq: 1 }), obs("bottle", -0.6, 1.2, now - 590_000, { seq: 2 })] },
    });
    expect(ok.status).toBe(201);
    expect(ok.body).toMatchObject({ accepted: 2, duplicates: 0, rev: 1 });
    expect((await call("POST", `${p}/spaces/office/observations`, { token: deviceToken, body: { observations: [obs("cup", 0, 1, now)] } })).status).toBe(403);
    expect((await call("POST", `${p}/spaces/home/observations`, { token: pk, body: { observations: [obs("cup", 0, 1, now)] } })).status).toBe(403);
    expect((await call("GET", `${p}/spaces/office/stats`, { token: deviceToken })).status).toBe(403);
  });

  it("deduplicates retried device writes by (device, seq)", async () => {
    const again = await call("POST", `${p}/spaces/home/observations`, {
      token: deviceToken,
      body: { observations: [obs("cup", -1.2, 1.5, now - 600_000, { seq: 1 })] },
    });
    expect(again.status).toBe(200);
    expect(again.body).toMatchObject({ accepted: 0, duplicates: 1, rev: 1 });
  });

  it("validates payloads", async () => {
    const bad = await call("POST", `${p}/spaces/home/observations`, { token: sk, body: { observations: [{ label: "cup", x: "a", y: 0, z: 0, t: now }] } });
    expect(bad.status).toBe(422);
    const future = await call("POST", `${p}/spaces/home/observations`, { token: sk, body: { observations: [obs("cup", 0, 1, now + 3 * 86_400_000)] } });
    expect(future.status).toBe(422);
    const big = await call("POST", `${p}/spaces/home/observations`, { token: sk, body: { observations: Array.from({ length: 51 }, (_, i) => obs("cup", 0, 1, now - i)) } });
    expect(big.status).toBe(413);
    expect((await call("POST", `${p}/spaces/missing/observations`, { token: sk, body: { observations: [obs("cup", 0, 1, now)] } })).status).toBe(404);
  });

  it("answers where with the newest sighting first", async () => {
    await call("POST", `${p}/spaces/home/observations`, {
      token: sk,
      body: { observations: [obs("cup", 1.0, 2.8, now - 60_000, { id: "host-1", embedding: [0.1, -0.4, 0.9], embedding_model: "potion-base-8M" })] },
    });
    const w = await call("GET", `${p}/spaces/home/where?label=Cup&limit=5`, { token: pk });
    expect(w.status).toBe(200);
    expect(w.body.results).toHaveLength(2);
    expect(w.body.results[0].id).toBe("host-1");
    expect(w.body.results[0].x).toBe(1.0);
  });

  it("reports the cup as moved and the bottle as new", async () => {
    const c = await call("GET", `${p}/spaces/home/changes?window_s=3600&room=kitchen`, { token: pk });
    const byLabel = Object.fromEntries(c.body.changes.map((x: Json) => [x.label, x]));
    expect(byLabel.cup.kind).toBe("moved");
    expect(byLabel.cup.distance_m).toBeGreaterThan(2);
    expect(byLabel.bottle.kind).toBe("new");
  });

  it("answers spoken questions with aliases and plurals", async () => {
    const mug = await call("POST", `${p}/spaces/home/ask`, { token: deviceToken, body: { text: "Where is my mug?" } });
    expect(mug.body.label).toBe("cup");
    expect(mug.body.answer).toMatch(/^Your mug \(I track it as a cup\) was last seen in the kitchen, 1 minute ago\./);
    const bottles = await call("POST", `${p}/spaces/home/ask`, { token: pk, body: { text: "where are the bottles" } });
    expect(bottles.body.label).toBe("bottle");
    expect(bottles.body.answer).toMatch(/^Your bottle was last seen in the kitchen/);
    const keys = await call("POST", `${p}/spaces/home/ask`, { token: pk, body: { text: "where are my keys" } });
    expect(keys.body.answer).toBe("I haven't seen your keys yet.");
    const changed = await call("POST", `${p}/spaces/home/ask`, { token: pk, body: { text: "What changed in the kitchen?" } });
    expect(changed.body.answer).toMatch(/the cup moved about 2\.\d metres/);
    const list = await call("POST", `${p}/spaces/home/ask`, { token: pk, body: { text: "what do you see" } });
    expect(list.body.answer).toMatch(/^I remember: cup in the kitchen/);
  });

  it("serves device digests: full, then deltas with removals", async () => {
    const full = await call("GET", `${p}/spaces/home/digest`, { token: deviceToken });
    expect(full.body.full).toBe(true);
    expect(full.body.objects.map((o: Json) => o.label).sort()).toEqual(["bottle", "cup"]);
    expect(full.body.objects.find((o: Json) => o.label === "cup").where).toBe("in the kitchen");
    const rev = full.body.rev;

    const none = await call("GET", `${p}/spaces/home/digest?since_rev=${rev}`, { token: deviceToken });
    expect(none.body.objects).toEqual([]);

    const bottleId = (await call("GET", `${p}/spaces/home/where?label=bottle`, { token: sk })).body.results[0].id;
    const del = await call("POST", `${p}/spaces/home/observations`, { token: sk, body: { deletes: [bottleId] } });
    expect(del.body).toMatchObject({ deleted: 1, rev: rev + 1 });
    const delta = await call("GET", `${p}/spaces/home/digest?since_rev=${rev}`, { token: deviceToken });
    expect(delta.body.removed).toEqual([{ label: "bottle", room: "kitchen" }]);
    expect(delta.body.objects).toEqual([]);

    const stats = await call("GET", `${p}/spaces/home/stats`, { token: pk });
    expect(stats.body).toMatchObject({ space: "home", observations: 2, labels: 1, rev: rev + 1 });
  });

  it("revokes a device immediately in the same isolate", async () => {
    expect((await call("DELETE", `${p}/devices/${deviceId}`, { token: sk })).status).toBe(200);
    expect((await call("GET", `${p}/spaces/home/digest`, { token: deviceToken })).status).toBe(401);
    expect((await call("POST", `${p}/devices/token`, { body: { device_id: deviceId, device_secret: deviceSecret } })).status).toBe(401);
  });
});

describe("jurisdiction safety", () => {
  it("refuses an EU project space when unpinned storage is not explicitly allowed", async () => {
    const { env } = await import("cloudflare:workers");
    const { spaceStub } = await import("../src/control");
    const r = await call("POST", "/admin/projects", { token: "test-admin-token", body: { jurisdiction: "eu" } });
    const strict = { ...env, DEV_ALLOW_UNPINNED: undefined } as typeof env;
    await expect(spaceStub(strict, r.body.ref, "ward-b")).rejects.toThrow(/[Jj]urisdiction/);
  });
});
