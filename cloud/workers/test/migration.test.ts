import { runInDurableObject } from "cloudflare:test";
import { env } from "cloudflare:workers";
import { expect, it } from "vitest";
import { hilbert3 } from "../src/hilbert";
import { SpaceObject } from "../src/space";
import type { QuerySpec, SpaceMeta } from "../src/types";

const meta: SpaceMeta = { ref: "migrationref", space: "home", frame: "room", extents: { lo: [-10, -1, -10], hi: [10, 4, 10] } };
const spec: QuerySpec = {
  region: { lo: [-1, -1, -1], hi: [1, 1, 1] }, timeStart: null, timeEnd: null, labels: null, room: null, minConfidence: null,
  vector: null, halfLifeMs: 0, limit: 10, includeEmbedding: false,
};

it("upgrades a Phase 0 space object in place, once", async () => {
  const stub = env.SPACE.get(env.SPACE.idFromName("migrationref/home"));
  await runInDurableObject(stub, (_instance: SpaceObject, state) => {
    const sql = state.storage.sql;
    // Roll this object back to the Phase 0 schema, with a row written by Phase 0.
    sql.exec(`
      DROP INDEX obs_scene_t; DROP INDEX obs_h8;
      ALTER TABLE obs DROP COLUMN scene; ALTER TABLE obs DROP COLUMN scale;
      ALTER TABLE obs DROP COLUMN prev_id; ALTER TABLE obs DROP COLUMN next_id;
      DELETE FROM meta WHERE k = 'schema';
    `);
    sql.exec(
      "INSERT INTO obs (id, label, frame, x, y, z, h8, t, confidence, rev) VALUES ('old', 'cup', 'room', 0, 0, 0, ?, 1, 1, 1)",
      hilbert3([0.5, 0.2, 0.5], 8), // (0, 0, 0) in the extents below
    );
    sql.exec("INSERT INTO meta (k, v) VALUES ('rev', '1')");

    const upgraded = new SpaceObject(state, env);
    expect(() => new SpaceObject(state, env)).not.toThrow(); // already at v2: no second ALTER
    const [hit] = upgraded.query(meta, spec).hits;
    expect(hit?.obs.id).toBe("old");
    expect(hit?.sdk).toEqual({ scene_id: null, scale_level: null, prev_state_id: null, next_state_id: null });
    const ins = upgraded.insertState(meta, {
      x: 0, y: 0, z: 0, timestamp_ms: 2, vector: [1, 0], scene_id: "s", scale_level: "patch", confidence: 1, metadata: {},
      prev_state_id: null, next_state_id: null,
    }, "", null);
    expect(ins).toMatchObject({ rev: 2 });
  });
});
