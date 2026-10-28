import { runDurableObjectAlarm, runInDurableObject } from "cloudflare:test";
import { env } from "cloudflare:workers";
import { expect, it } from "vitest";
import { hilbert3, hilbert4 } from "../src/hilbert";
import { SpaceObject } from "../src/space";
import type { QuerySpec, SpaceMeta } from "../src/types";

const meta: SpaceMeta = { ref: "migrationref", space: "home", frame: "room", extents: { lo: [-10, -1, -10], hi: [10, 4, 10] }, epoch_ms: 5000 };
const spec: QuerySpec = {
  region: { lo: [-1, -1, -1], hi: [1, 1, 1] }, timeStart: null, timeEnd: null, labels: null, room: null, minConfidence: null,
  vector: null, halfLifeMs: 0, limit: 10, includeEmbedding: false,
};

/** Roll an object back to the Phase 0 schema and write Phase 0 rows at (0, 0, 0), t = 1..n. */
function rollBackToPhase0(sql: SqlStorage, rows: number): void {
  sql.exec(`
    DROP INDEX obs_scene_t; DROP INDEX obs_h8; DROP INDEX obs_epoch_r12; DROP INDEX obs_t;
    ALTER TABLE obs DROP COLUMN scene; ALTER TABLE obs DROP COLUMN scale;
    ALTER TABLE obs DROP COLUMN prev_id; ALTER TABLE obs DROP COLUMN next_id;
    ALTER TABLE obs DROP COLUMN epoch; ALTER TABLE obs DROP COLUMN r4; ALTER TABLE obs DROP COLUMN r8; ALTER TABLE obs DROP COLUMN r12;
    DELETE FROM meta WHERE k IN ('schema', 'address');
  `);
  // Phase 0 stored the meta without epoch_ms.
  const { epoch_ms: _, ...phase0Meta } = meta;
  sql.exec("INSERT INTO meta (k, v) VALUES ('space', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", JSON.stringify(phase0Meta));
  for (let i = 1; i <= rows; i++) {
    sql.exec(
      "INSERT INTO obs (id, label, frame, x, y, z, h8, t, confidence, rev) VALUES (?, 'cup', 'room', 0, 0, 0, ?, ?, 1, 1)",
      `old${i}`, hilbert3([0.5, 0.2, 0.5], 8), i, // (0, 0, 0) in the extents above
    );
  }
  sql.exec("INSERT INTO meta (k, v) VALUES ('rev', '1') ON CONFLICT(k) DO UPDATE SET v = excluded.v");
}

it("upgrades a Phase 0 space object in place, once, and backfills the 4D address", async () => {
  const stub = env.SPACE.get(env.SPACE.idFromName("migrationref/home"));
  await runInDurableObject(stub, (_instance: SpaceObject, state) => {
    rollBackToPhase0(state.storage.sql, 1);

    const upgraded = new SpaceObject(state, env);
    expect(() => new SpaceObject(state, env)).not.toThrow(); // already at v3: no second ALTER
    const [hit] = upgraded.query(meta, spec).hits;
    expect(hit?.obs.id).toBe("old1");
    expect(hit?.sdk).toEqual({ scene_id: null, scale_level: null, prev_state_id: null, next_state_id: null });
    const p: [number, number, number, number] = [0.5, 0.2, 0.5, 1 / 5000];
    expect(hit?.address).toEqual({ epoch: 0, r4: hilbert4(p, 4), r8: hilbert4(p, 8), r12: hilbert4(p, 12) });
    const ins = upgraded.insertState(meta, {
      x: 0, y: 0, z: 0, timestamp_ms: 2, vector: [1, 0], scene_id: "s", scale_level: "patch", confidence: 1, metadata: {},
      prev_state_id: null, next_state_id: null,
    }, "", null);
    expect(ins).toMatchObject({ rev: 2 });
  });
});

it("continues a large backfill from alarms, using the 3D path until it finishes", async () => {
  const stub = env.SPACE.get(env.SPACE.idFromName("migrationref/big"));
  const perPass = SpaceObject.backfillPerPass;
  SpaceObject.backfillPerPass = 2;
  try {
    await runInDurableObject(stub, (_instance: SpaceObject, state) => {
      rollBackToPhase0(state.storage.sql, 5);
      const upgraded = new SpaceObject(state, env);
      const pending = state.storage.sql.exec("SELECT COUNT(*) AS n FROM obs WHERE r12 IS NULL").one();
      expect(Number(pending.n)).toBe(3);
      // A one-epoch window still finds every row while the address is incomplete.
      expect(upgraded.query(meta, { ...spec, timeStart: 0, timeEnd: 10 }).hits).toHaveLength(5);
    });
    // The runtime may already have fired some of them; run any still pending.
    for (let i = 0; i < 5 && (await runDurableObjectAlarm(stub)); i++);
    await runInDurableObject(stub, (instance: SpaceObject, state) => {
      expect(Number(state.storage.sql.exec("SELECT COUNT(*) AS n FROM obs WHERE r12 IS NULL").one().n)).toBe(0);
      expect(String(state.storage.sql.exec("SELECT v FROM meta WHERE k = 'address'").one().v)).toBe("ready");
      expect(instance.query(meta, { ...spec, timeStart: 0, timeEnd: 10 }).hits).toHaveLength(5);
    });
  } finally {
    SpaceObject.backfillPerPass = perPass;
  }
});
