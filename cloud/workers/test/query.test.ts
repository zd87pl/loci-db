import { beforeAll, describe, expect, it } from "vitest";
import { call, newProject, type Json } from "./helpers";

const now = Date.now();
let P: Awaited<ReturnType<typeof newProject>>;

/** Deterministic PRNG (mulberry32) for the region property test. */
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sdkState = (extra: Json = {}): Json => ({
  x: 0.5, y: 0.5, z: 0.5, timestamp_ms: now - 1000, vector: [1, 0, 0, 0], scene_id: "s", ...extra,
});

beforeAll(async () => {
  P = await newProject("Query");
  expect((await call("POST", `${P.v1}/spaces`, { token: P.sk, body: { space: "home" } })).status).toBe(201);
  const obs = [
    { id: "cup-old", label: "cup", x: -2, y: 0.8, z: 3, t: now - 3_600_000, room: "kitchen", confidence: 0.9, embedding: [1, 0, 0] },
    { id: "cup-new", label: "Cup", x: 1, y: 0.8, z: 2, t: now - 60_000, room: "kitchen", confidence: 0.6, embedding: [0.8, 0.6, 0] },
    { id: "keys", label: "keys", x: 4, y: 0.4, z: -5, t: now - 30_000, room: "hall", confidence: 0.95, embedding: [0, 1, 0] },
    { id: "plant", label: "plant", x: 9, y: 0, z: 9, t: now - 120_000, room: "garden", confidence: 1 },
    { id: "wide", label: "mug", x: 0, y: 0, z: 0, t: now - 10_000, embedding: [1, 0, 0, 0] },
  ];
  expect((await call("POST", `${P.v1}/spaces/home/observations`, { token: P.sk, body: { observations: obs } })).status).toBe(201);
});

const q = (body: Json, token = P.pk, space = "home") => call("POST", `${P.v1}/spaces/${space}/query`, { token, body });
const ids = (r: { body: Json }) => r.body.results.map((x: Json) => x.id);

describe("v1 query", () => {
  it("returns newest first without a vector, with the where-style fields", async () => {
    const r = await q({ limit: 3 });
    expect(r.status).toBe(200);
    expect(ids(r)).toEqual(["wide", "keys", "cup-new"]);
    expect(r.body.results[0]).not.toHaveProperty("score");
    expect(r.body.results[1]).toMatchObject({ label: "keys", room: "hall", frame: "room", x: 4 });
    expect(r.body.truncated).toBe(false);
  });

  it("filters by normalised label, room, time window and confidence", async () => {
    expect(ids(await q({ label: "  CUP " }))).toEqual(["cup-new", "cup-old"]);
    expect(ids(await q({ labels: ["keys", "plant"], room: "Garden" }))).toEqual(["plant"]);
    expect(ids(await q({ time: { start: now - 3_600_000, end: now - 60_000 } }))).toEqual(["cup-new", "plant", "cup-old"]);
    expect(ids(await q({ time: { start: now - 60_000 } }))).toEqual(["wide", "keys", "cup-new"]);
    expect(ids(await q({ min_confidence: 0.9 }))).toEqual(["wide", "keys", "plant", "cup-old"]);
  });

  it("filters by a region in metres, bounds inclusive, outside the extents too", async () => {
    expect(ids(await q({ region: { lo: [-2, 0, 2], hi: [1, 1, 3] } }))).toEqual(["cup-new", "cup-old"]);
    expect(ids(await q({ region: { lo: [9, 0, 9], hi: [9, 0, 9] } }))).toEqual(["plant"]);
    expect(ids(await q({ region: { lo: [20, 20, 20], hi: [30, 30, 30] } }))).toEqual([]);
  });

  it("ranks by cosine with recency decay, skipping rows without a matching embedding", async () => {
    const r = await q({ vector: [1, 0, 0], include_embedding: true });
    expect(ids(r)).toEqual(["cup-new", "cup-old", "keys"]);
    const [first] = r.body.results;
    expect(first.similarity).toBeCloseTo(0.8, 2);
    expect(first.score).toBeLessThan(first.similarity);
    expect(first.embedding[0]).toBeCloseTo(0.8, 2); // int8-quantised
    const pure = await q({ vector: [1, 0, 0], half_life_s: 0 });
    expect(ids(pure)).toEqual(["cup-old", "cup-new", "keys"]);
    expect(pure.body.results[0].score).toBe(pure.body.results[0].similarity);
  });

  it("matches a brute-force filter for random regions", async () => {
    const rand = rng(7);
    await call("POST", `${P.v1}/spaces`, { token: P.sk, body: { space: "grid", extents: { lo: [-5, 0, -5], hi: [5, 3, 5] } } });
    const pts = Array.from({ length: 50 }, (_, i) => ({
      id: `p${i}`, label: "dot", x: rand() * 12 - 6, y: rand() * 4 - 0.5, z: rand() * 12 - 6, t: now - i,
    }));
    expect((await call("POST", `${P.v1}/spaces/grid/observations`, { token: P.sk, body: { observations: pts } })).status).toBe(201);
    for (let k = 0; k < 25; k++) {
      const a = [rand() * 12 - 6, rand() * 4 - 0.5, rand() * 12 - 6];
      const b = [rand() * 12 - 6, rand() * 4 - 0.5, rand() * 12 - 6];
      const lo = a.map((v, i) => Math.min(v, b[i]!));
      const hi = a.map((v, i) => Math.max(v, b[i]!));
      const want = pts.filter((p) => [p.x, p.y, p.z].every((v, i) => v >= lo[i]! && v <= hi[i]!)).map((p) => p.id);
      expect(ids(await q({ region: { lo, hi }, limit: 100 }, P.pk, "grid"))).toEqual(want);
    }
  });

  it("validates the body", async () => {
    for (const body of [
      [],
      { region: { lo: [1, 0, 0], hi: [0, 1, 1] } },
      { region: { lo: [0, 0], hi: [1, 1, 1] } },
      { time: { start: 10, end: 5 } },
      { time: { start: -1 } },
      { label: "cup", labels: ["cup"] },
      { labels: Array.from({ length: 17 }, (_, i) => `l${i}`) },
      { labels: [""] },
      { vector: [] },
      { vector: [1, "a"] },
      { limit: 0 },
      { limit: 1001 },
      { min_confidence: 2 },
      { half_life_s: -1 },
      { include_embedding: "yes" },
    ]) {
      const r = await q(body as Json);
      expect(r.status, JSON.stringify(body)).toBe(422);
    }
  });

  it("enforces credentials and space existence", async () => {
    expect((await call("POST", `${P.v1}/spaces/home/query`, { body: {} })).status).toBe(401);
    expect((await q({}, P.pk, "missing")).status).toBe(404);
    const other = await newProject("Other");
    expect((await q({}, other.sk)).status).toBe(403);
  });

  it("rejects a blank where label", async () => {
    expect((await call("GET", `${P.v1}/spaces/home/where?label=%20%20`, { token: P.pk })).status).toBe(422);
  });
});

describe("SDK compatibility layer", () => {
  let C: Awaited<ReturnType<typeof newProject>>;
  beforeAll(async () => {
    C = await newProject("Compat");
  });

  it("answers queries with no results before the first insert", async () => {
    const r = await call("POST", `${C.base}/query`, { token: C.pk, body: { vector: [1, 0, 0, 0] } });
    expect(r).toEqual({ status: 200, body: { results: [] } });
  });

  it("lets only writers insert, and any reader query", async () => {
    expect((await call("POST", `${C.base}/insert`, { token: C.pk, body: sdkState() })).status).toBe(403);
    expect((await call("POST", `${C.base}/insert`, { body: sdkState() })).status).toBe(401);
    const ins = await call("POST", `${C.base}/insert`, { token: C.sk, body: sdkState({ metadata: { label: "Red  Door" } }) });
    expect(ins.status).toBe(200);
    expect(ins.body.id).toMatch(/^[0-9a-f-]{36}$/);
    const r = await call("POST", `${C.base}/query`, { token: C.pk, body: { vector: [1, 0, 0, 0] } });
    expect(ids(r)).toEqual([ins.body.id]);
  });

  it("files SDK states under their metadata label and keeps unlabelled ones out of where, ask and digest", async () => {
    await call("POST", `${C.base}/insert`, { token: C.sk, body: sdkState({ vector: [0, 1, 0, 0] }) });
    const w = await call("GET", `${C.v1}/spaces/default/where?label=red%20door`, { token: C.pk });
    expect(w.body.results).toHaveLength(1);
    const digest = await call("GET", `${C.v1}/spaces/default/digest`, { token: C.pk });
    expect(digest.body.objects.map((o: Json) => o.label)).toEqual(["red door"]);
    const stats = await call("GET", `${C.v1}/spaces/default/stats`, { token: C.pk });
    expect(stats.body).toMatchObject({ observations: 2, labels: 1 });
  });

  it("fixes the vector dimension at the first insert", async () => {
    const ins = await call("POST", `${C.base}/insert`, { token: C.sk, body: sdkState({ vector: [1, 0, 0] }) });
    expect(ins.status).toBe(422);
    expect(ins.body.detail).toBe("vector must have 4 dimensions, got 3");
    expect((await call("POST", `${C.base}/query`, { token: C.sk, body: { vector: [1, 0, 0] } })).status).toBe(422);
  });

  it("clips the box to the unit cube", async () => {
    const wide = { vector: [1, 0, 0, 0], x_min: -100, x_max: 100, y_min: -1e9, y_max: 1e9 };
    expect((await call("POST", `${C.base}/query`, { token: C.sk, body: wide })).body.results).toHaveLength(2);
    const outside = { vector: [1, 0, 0, 0], x_min: 1.5, x_max: 2 };
    expect(await call("POST", `${C.base}/query`, { token: C.sk, body: outside })).toEqual({ status: 200, body: { results: [] } });
  });

  it("validates inserts like the FastAPI service did", async () => {
    for (const extra of [
      { x: 1.5 },
      { z: -0.1 },
      { timestamp_ms: -1 },
      { timestamp_ms: 1.5 },
      { vector: [] },
      { vector: [1, null, 0, 0] },
      { scene_id: "" },
      { scene_id: "x".repeat(257) },
      { scale_level: "huge" },
      { confidence: 1.1 },
      { metadata: [] },
      { metadata: { blob: "x".repeat(16 * 1024) } },
      { prev_state_id: 5 },
    ]) {
      const r = await call("POST", `${C.base}/insert`, { token: C.sk, body: sdkState(extra) });
      expect(r.status, JSON.stringify(extra).slice(0, 80)).toBe(422);
    }
    const missing = sdkState();
    delete missing.scene_id;
    expect((await call("POST", `${C.base}/insert`, { token: C.sk, body: missing })).status).toBe(422);
  });

  it("validates queries like the FastAPI service did", async () => {
    for (const extra of [
      { vector: undefined },
      { x_min: 0.8, x_max: 0.2 },
      { y_min: -2e9 },
      { time_start_ms: 10, time_end_ms: 5 },
      { limit: 0 },
      { limit: 1001 },
      { overlap_factor: 20 },
      { min_confidence: -0.1 },
      { include_vectors: "yes" },
    ]) {
      const r = await call("POST", `${C.base}/query`, { token: C.sk, body: { vector: [1, 0, 0, 0], ...extra } });
      expect(r.status, JSON.stringify(extra)).toBe(422);
    }
    // A half-open window is ignored, as before.
    const half = await call("POST", `${C.base}/query`, { token: C.sk, body: { vector: [1, 0, 0, 0], time_start_ms: now + 86_400_000 } });
    expect(half.body.results).toHaveLength(2);
  });

  it("routes only POST /insert and /query", async () => {
    expect((await call("GET", `${C.base}/query`, { token: C.sk })).status).toBe(405);
    expect((await call("POST", `${C.base}/delete`, { token: C.sk, body: {} })).status).toBe(404);
    expect((await call("POST", `${C.base}/v1/insert`, { token: C.sk, body: {} })).status).toBe(404);
  });
});
