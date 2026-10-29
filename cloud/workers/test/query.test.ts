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

describe("4D address and per-space epochs", () => {
  it("sets epoch_ms per space, 5,000 by default", async () => {
    const p = await newProject("Epochs");
    expect((await call("POST", `${p.v1}/spaces`, { token: p.sk, body: { space: "lab", epoch_ms: 1000 } })).body.epoch_ms).toBe(1000);
    expect((await call("POST", `${p.v1}/spaces`, { token: p.sk, body: { space: "home" } })).body.epoch_ms).toBe(5000);
    for (const epoch_ms of [50, 1.5, "1000", 86_400_001]) {
      expect((await call("POST", `${p.v1}/spaces`, { token: p.sk, body: { space: "bad", epoch_ms } })).status).toBe(422);
    }
    const list = await call("GET", `${p.v1}/spaces`, { token: p.pk });
    expect(list.body.spaces.map((s: Json) => [s.space, s.epoch_ms])).toEqual([["home", 5000], ["lab", 1000]]);
  });

  it("matches a brute-force filter for regions with windows inside and across epochs", async () => {
    const rand = rng(11);
    const p = await newProject("Epoch property");
    await call("POST", `${p.v1}/spaces`, { token: p.sk, body: { space: "lab", epoch_ms: 1000, extents: { lo: [-5, 0, -5], hi: [5, 3, 5] } } });
    const base = Math.floor(now / 1000) * 1000 - 10_000;
    const pts = Array.from({ length: 50 }, (_, i) => ({
      id: `q${i}`, label: "dot", x: rand() * 12 - 6, y: rand() * 4 - 0.5, z: rand() * 12 - 6, t: base + Math.floor(rand() * 5000),
    }));
    expect((await call("POST", `${p.v1}/spaces/lab/observations`, { token: p.sk, body: { observations: pts } })).status).toBe(201);
    const newestFirst = (a: { t: number; id: string }, b: { t: number; id: string }) => b.t - a.t || (a.id < b.id ? -1 : 1);
    for (let k = 0; k < 40; k++) {
      const a = [rand() * 12 - 6, rand() * 4 - 0.5, rand() * 12 - 6];
      const b = [rand() * 12 - 6, rand() * 4 - 0.5, rand() * 12 - 6];
      const lo = a.map((v, i) => Math.min(v, b[i]!));
      const hi = a.map((v, i) => Math.max(v, b[i]!));
      // Even k: a window inside one epoch; odd k: one spanning up to three epochs.
      const epoch = base + 1000 * Math.floor(rand() * 5);
      const [s, e] = k % 2 === 0
        ? [epoch + Math.floor(rand() * 500), epoch + 500 + Math.floor(rand() * 499)]
        : [base + Math.floor(rand() * 2000), base + 2000 + Math.floor(rand() * 3000)];
      const want = pts
        .filter((pt) => [pt.x, pt.y, pt.z].every((v, i) => v >= lo[i]! && v <= hi[i]!) && pt.t >= s && pt.t <= e)
        .sort(newestFirst)
        .map((pt) => pt.id);
      const r = await call("POST", `${p.v1}/spaces/lab/query`, { token: p.pk, body: { region: { lo, hi }, time: { start: s, end: e }, limit: 100 } });
      expect(ids(r), JSON.stringify({ lo, hi, s, e })).toEqual(want);
    }
  });

  it("returns each result's address", async () => {
    const [first] = (await q({ limit: 1 })).body.results;
    expect(first.address).toEqual({ epoch: Math.floor(first.t / 5000), r4: expect.any(Number), r8: expect.any(Number), r12: expect.any(Number) });
  });
});

describe("SDK compatibility layer, M1 endpoints", () => {
  let C: Awaited<ReturnType<typeof newProject>>;
  let anchor = "";
  beforeAll(async () => {
    C = await newProject("Compat M1");
    anchor = (await call("POST", `${C.base}/insert`, { token: C.sk, body: sdkState() })).body.id;
  });

  it("validates insert_batch", async () => {
    const batch = (body: Json, token = C.sk) => call("POST", `${C.base}/insert_batch`, { token, body });
    expect((await batch({ states: [sdkState()] }, C.pk)).status).toBe(403);
    expect((await batch({ states: Array.from({ length: 101 }, () => sdkState()) })).status).toBe(413);
    expect((await batch({ states: [] })).status).toBe(422);
    const bad = await batch({ states: [sdkState(), sdkState({ x: 2 })] });
    expect([bad.status, bad.body.detail]).toEqual([422, "states[1]: x must be in [0, 1]"]);
    expect((await batch({ states: [sdkState()], chains: { s: 5 } })).status).toBe(422);
    expect((await batch({ states: [sdkState()], chains: [] })).status).toBe(422);
    const dims = await batch({ states: [sdkState({ vector: [1, 0, 0] })] });
    expect([dims.status, dims.body.detail]).toEqual([422, "states[0].vector must have 4 dimensions, got 3"]);
    const ok = await batch({ states: [sdkState({ timestamp_ms: now - 500 }), sdkState({ scene_id: "t", timestamp_ms: now - 900 })] });
    expect(ok.body.ids).toHaveLength(2);
    expect(ok.body.chains).toEqual({ s: ok.body.ids[0], t: ok.body.ids[1] });
  });

  it("validates trajectory and causal_context, and answers an unknown id with no results", async () => {
    const post = (path: string, body: Json) => call("POST", `${C.base}/${path}`, { token: C.pk, body });
    expect((await post("trajectory", {})).status).toBe(422);
    expect((await post("trajectory", { state_id: anchor, steps_back: 1001 })).status).toBe(422);
    expect((await post("trajectory", { state_id: anchor, steps_forward: -1 })).status).toBe(422);
    expect((await post("causal_context", { state_id: anchor, window_ms: -1 })).status).toBe(422);
    expect((await post("causal_context", { state_id: anchor, include_vectors: "no" })).status).toBe(422);
    expect((await post("trajectory", { state_id: "nope" })).body).toEqual({ results: [] });
    expect((await post("causal_context", { state_id: "nope" })).body).toEqual({ results: [] });
    const traj = await post("trajectory", { state_id: anchor, include_vectors: false });
    expect(traj.body.results[0]).toMatchObject({ id: anchor, vector: [] });
    expect((await call("GET", `${C.base}/trajectory`, { token: C.pk })).status).toBe(405);
  });

  it("validates include_scores", async () => {
    expect((await call("POST", `${C.base}/query`, { token: C.pk, body: { vector: [1, 0, 0, 0], include_scores: 1 } })).status).toBe(422);
    const r = await call("POST", `${C.base}/query`, { token: C.pk, body: { vector: [1, 0, 0, 0] } });
    expect(r.body.results[0]).not.toHaveProperty("score");
  });
});
