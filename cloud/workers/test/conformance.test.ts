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
interface Golden {
  states: GState[];
  queries: GQuery[];
  links: Record<string, { prev: string | null; next: string | null }>;
  wire: { insert_keys: string[]; query_keys: string[]; result_keys: string[] };
}
const G = goldenJson as unknown as Golden;

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
    const r = await call("POST", `${project.base}/query`, { token: project.sk, body: { vector: [1, 0, 0, 0], limit: 100 } });
    const links = Object.fromEntries(
      r.body.results.map((x: Json) => [keyOf[x.id], {
        prev: keyOf[x.prev_state_id] ?? x.prev_state_id,
        next: keyOf[x.next_state_id] ?? x.next_state_id,
      }]),
    );
    expect(links).toEqual(G.links);
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
