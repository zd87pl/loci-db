/**
 * Golden queries shared with the Python library (tests/test_cloud_conformance.py checks the same
 * file against LocalLociClient and the SDK). Inserts go through the SDK compatibility /insert;
 * every query runs through the SDK /query and the v1 /spaces/default/query, and both must return
 * the library's keys in the library's order.
 */
import { beforeAll, describe, expect, it } from "vitest";
import goldenJson from "./conformance-golden.json";
import { call, newProject, type Json } from "./helpers";

interface GState {
  key: string; dt_ms: number; x: number; y: number; z: number; vector: number[]; scene_id: string; scale_level?: string;
  confidence: number; metadata?: Json; prev_state_id?: string; next_state_id?: string;
}
interface GQuery {
  name: string; vector: number[]; bounds?: Record<string, number>; window_dt_ms?: [number, number]; min_confidence?: number;
  labels?: string[]; limit: number; expect: string[];
}
type Links = Record<string, { prev: string | null; next: string | null }>;
interface Golden {
  states: GState[];
  queries: GQuery[];
  links: Links;
  wire: { insert_keys: string[]; query_keys: string[]; result_keys: string[] };
  scored: { query: string; scores: [string, number][] };
  batch: { states: GState[]; links: Links };
  trajectories: { anchor: string; steps_back: number; steps_forward: number; expect: string[] }[];
  causal: { anchor: string; window_ms: number; expect: string[] }[];
  addresses: { epoch_ms: number; states: { x: number; y: number; z: number; timestamp_ms: number; expect: Json }[] };
}
const G = goldenJson as unknown as Golden;
const MISSING_ID = "00000000-0000-0000-0000-000000000000";

const now = Date.now();
let project: Awaited<ReturnType<typeof newProject>>;
const keyOf: Record<string, string> = {};

/** What loci.cloud_transport._insert_payload sends. */
function insertBody(s: GState): Json {
  const body: Json = {
    x: s.x, y: s.y, z: s.z, timestamp_ms: now + s.dt_ms, vector: s.vector, scene_id: s.scene_id,
    scale_level: s.scale_level ?? "patch", confidence: s.confidence,
  };
  if (s.metadata && Object.keys(s.metadata).length) body.metadata = s.metadata;
  if (s.prev_state_id !== undefined) body.prev_state_id = s.prev_state_id;
  if (s.next_state_id !== undefined) body.next_state_id = s.next_state_id;
  return body;
}

/** What loci.cloud_transport._query_payload sends for LociClient.query(...). */
function sdkQueryBody(q: GQuery): Json {
  const body: Json = { vector: q.vector, limit: q.limit, overlap_factor: 1.2 };
  if (q.min_confidence !== undefined) body.min_confidence = q.min_confidence;
  if (q.bounds) {
    for (const a of ["x", "y", "z"]) {
      body[`${a}_min`] = q.bounds[`${a}_min`] ?? 0;
      body[`${a}_max`] = q.bounds[`${a}_max`] ?? 1;
    }
  }
  if (q.window_dt_ms) [body.time_start_ms, body.time_end_ms] = [now + q.window_dt_ms[0], now + q.window_dt_ms[1]];
  return body;
}

function v1QueryBody(q: GQuery): Json {
  const body: Json = { vector: q.vector, limit: q.limit };
  if (q.bounds) {
    const b = q.bounds;
    body.region = { lo: [b.x_min ?? 0, b.y_min ?? 0, b.z_min ?? 0], hi: [b.x_max ?? 1, b.y_max ?? 1, b.z_max ?? 1] };
  }
  if (q.window_dt_ms) body.time = { start: now + q.window_dt_ms[0], end: now + q.window_dt_ms[1] };
  if (q.min_confidence !== undefined) body.min_confidence = q.min_confidence;
  if (q.labels) body.labels = q.labels;
  return body;
}

/** Every state's links, by key. */
async function links(base: string, token: string, keys: Record<string, string>): Promise<Links> {
  const r = await call("POST", `${base}/query`, { token, body: { vector: [1, 0, 0, 0], limit: 100, include_vectors: false } });
  return Object.fromEntries(
    r.body.results.map((x: Json) => [keys[x.id], { prev: keys[x.prev_state_id] ?? x.prev_state_id, next: keys[x.next_state_id] ?? x.next_state_id }]),
  );
}

beforeAll(async () => {
  project = await newProject("Conformance");
  for (const s of G.states) {
    const r = await call("POST", `${project.base}/insert`, { token: project.sk, body: insertBody(s) });
    expect(r.status).toBe(200);
    keyOf[r.body.id] = s.key;
  }
});

describe("conformance with the Python library", () => {
  it("sends only the keys the SDK sends", () => {
    for (const s of G.states) expect(Object.keys(insertBody(s)).every((k) => G.wire.insert_keys.includes(k))).toBe(true);
    for (const q of G.queries) expect(Object.keys(sdkQueryBody(q)).every((k) => G.wire.query_keys.includes(k))).toBe(true);
  });

  it("creates the default space with unit extents on the first insert", async () => {
    const r = await call("GET", `${project.v1}/spaces`, { token: project.pk });
    expect(r.body.spaces).toEqual([
      expect.objectContaining({ space: "default", frame: "room", extents: { lo: [0, 0, 0], hi: [1, 1, 1] } }),
    ]);
  });

  it.each(G.queries.filter((q) => !q.labels))("SDK /query: $name", async (q) => {
    const r = await call("POST", `${project.base}/query`, { token: project.sk, body: sdkQueryBody(q) });
    expect(r.status).toBe(200);
    expect(r.body.results.map((x: Json) => keyOf[x.id])).toEqual(q.expect);
  });

  it.each(G.queries)("v1 /query: $name", async (q) => {
    const r = await call("POST", `${project.v1}/spaces/default/query`, { token: project.pk, body: v1QueryBody(q) });
    expect(r.status).toBe(200);
    expect(r.body.results.map((x: Json) => keyOf[x.id])).toEqual(q.expect);
  });

  it("links states within a scene as LociClient.insert does", async () => {
    expect(await links(project.base, project.sk, keyOf)).toEqual(G.links);
  });

  it("returns every WorldState field, with vectors as float32", async () => {
    const r = await call("POST", `${project.base}/query`, { token: project.sk, body: { vector: [1, 0, 0, 0], limit: 100 } });
    expect(r.body.results).toHaveLength(G.states.length);
    for (const x of r.body.results as Json[]) {
      expect(Object.keys(x).sort()).toEqual(G.wire.result_keys);
      const s = G.states.find((st) => st.key === keyOf[x.id])!;
      expect([x.x, x.y, x.z, x.timestamp_ms, x.confidence]).toEqual([s.x, s.y, s.z, now + s.dt_ms, s.confidence]);
      expect([x.scene_id, x.scale_level]).toEqual([s.scene_id, s.scale_level ?? "patch"]);
      expect(x.metadata).toEqual(s.metadata ?? {});
      expect(x.vector).toHaveLength(s.vector.length);
      x.vector.forEach((v: number, i: number) => {
        expect(v).toBeCloseTo(s.vector[i]!, 6);
        expect(Math.fround(v)).toBe(v);
      });
    }
  });

  it("omits vectors when include_vectors is false", async () => {
    const r = await call("POST", `${project.base}/query`, { token: project.sk, body: { vector: [1, 0, 0, 0], limit: 2, include_vectors: false } });
    expect(r.body.results.map((x: Json) => x.vector)).toEqual([[], []]);
  });
});

describe("conformance, M1: scores, batches, trajectories, causal context, the 4D address", () => {
  const idOf = (key: string) => Object.entries(keyOf).find(([, k]) => k === key)?.[0] ?? MISSING_ID;

  it("returns the library's similarity with include_scores", async () => {
    const q = G.queries.find((x) => x.name === G.scored.query)!;
    const r = await call("POST", `${project.base}/query`, { token: project.pk, body: { ...sdkQueryBody(q), include_scores: true } });
    expect(r.body.results.map((x: Json) => keyOf[x.id])).toEqual(G.scored.scores.map(([k]) => k));
    r.body.results.forEach((x: Json, i: number) => {
      expect(x.score).toBeCloseTo(G.scored.scores[i]![1], 6);
      expect(x.decayed_score).toBeLessThanOrEqual(x.score);
    });
  });

  it("links an insert_batch as LocalLociClient.insert_batch does", async () => {
    const r = await call("POST", `${project.base}/insert_batch`, { token: project.sk, body: { states: G.batch.states.map(insertBody) } });
    expect(r.status).toBe(200);
    G.batch.states.forEach((s, i) => (keyOf[r.body.ids[i]] = s.key));
    expect(await links(project.base, project.sk, keyOf)).toEqual(G.batch.links);
  });

  it.each(G.trajectories)("trajectory around $anchor", async (t) => {
    const body = { state_id: idOf(t.anchor), steps_back: t.steps_back, steps_forward: t.steps_forward };
    const r = await call("POST", `${project.base}/trajectory`, { token: project.pk, body });
    expect(r.status).toBe(200);
    expect(r.body.results.map((x: Json) => keyOf[x.id])).toEqual(t.expect);
  });

  it.each(G.causal)("causal context around $anchor", async (c) => {
    const r = await call("POST", `${project.base}/causal_context`, { token: project.pk, body: { state_id: idOf(c.anchor), window_ms: c.window_ms } });
    expect(r.status).toBe(200);
    expect(r.body.results.map((x: Json) => keyOf[x.id])).toEqual(c.expect);
  });

  it("links a batch split into requests, with chains, the same as one request", async () => {
    const p = await newProject("Chunked");
    const keys: Record<string, string> = {};
    for (const s of G.states) keys[(await call("POST", `${p.base}/insert`, { token: p.sk, body: insertBody(s) })).body.id] = s.key;
    // The SDK's order: (scene, time). Split inside the kitchen scene.
    const sorted = [...G.batch.states].sort((a, b) => (a.scene_id < b.scene_id ? -1 : a.scene_id > b.scene_id ? 1 : 0) || a.dt_ms - b.dt_ms);
    let chains: Record<string, string> = {};
    for (const chunk of [sorted.slice(0, 2), sorted.slice(2)]) {
      const r = await call("POST", `${p.base}/insert_batch`, { token: p.sk, body: { states: chunk.map(insertBody), chains } });
      expect(r.status).toBe(200);
      chunk.forEach((s, i) => (keys[r.body.ids[i]] = s.key));
      chains = { ...chains, ...r.body.chains };
    }
    expect(await links(p.base, p.sk, keys)).toEqual(G.batch.links);
  });

  it("stores the library's 4D address", async () => {
    const p = await newProject("Addresses");
    const want: Record<string, Json> = {};
    for (const a of G.addresses.states) {
      const body = { x: a.x, y: a.y, z: a.z, timestamp_ms: a.timestamp_ms, vector: [1], scene_id: "address" };
      want[(await call("POST", `${p.base}/insert`, { token: p.sk, body })).body.id] = a.expect;
    }
    const spaces = await call("GET", `${p.v1}/spaces`, { token: p.pk });
    expect(spaces.body.spaces[0].epoch_ms).toBe(G.addresses.epoch_ms);
    const r = await call("POST", `${p.v1}/spaces/default/query`, { token: p.pk, body: { limit: 100 } });
    expect(r.body.results).toHaveLength(G.addresses.states.length);
    for (const x of r.body.results as Json[]) expect(x.address).toEqual(want[x.id]);
  });
});
