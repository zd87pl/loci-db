import { exports } from "cloudflare:workers";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { boardMemory, memorySummary, plain } from "../src/board";
import { isHallucination, pcmFromWav, resample, setAiForTests, wavFromPcm } from "../src/speech";
import type { Observation } from "../src/types";
import { HttpError, errorResponse } from "../src/util";
import { call, newProject } from "./helpers";

const NOW = Date.now();

/** Workers AI stand-in: TTS answers with a WAV of `ttsRate` (one sample per character), STT with `heard`. */
const fakeAi = {
  ttsRate: 16_000,
  heard: "where is my cup",
  calls: [] as { model: string; inputs: Record<string, unknown> }[],
  async run(model: string, inputs: Record<string, unknown>): Promise<unknown> {
    this.calls.push({ model, inputs });
    if (model.includes("aura")) {
      const n = String(inputs.text).length * (this.ttsRate / 1000);
      return new Blob([wavFromPcm(new Uint8Array(n * 2).fill(1), this.ttsRate)]).stream();
    }
    return { text: this.heard };
  },
};

async function raw(method: string, path: string, opts: { token?: string; body?: BodyInit; json?: unknown } = {}) {
  const headers: Record<string, string> = {};
  if (opts.token) headers.authorization = `Bearer ${opts.token}`;
  let body = opts.body;
  if (opts.json !== undefined) {
    headers["content-type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  return exports.default.fetch(new Request(`http://loci.test${path}`, { method, headers, body }));
}

function obs(over: Partial<Record<string, unknown>> = {}) {
  return { label: "cup", frame: "image", x: 0.2, y: 0.5, z: 0.5, t: NOW - 60_000, confidence: 0.9, ...over };
}

let v1 = "";
let sk = "";
let dev = "";
let devToken = "";
let otherToken = "";

async function device(name: string): Promise<{ id: string; token: string; expiresIn: number }> {
  const d = await call("POST", `${v1}/devices`, { token: sk, body: { name, spaces: ["home"] } });
  const c = await call("POST", `${v1}/devices/claim`, { body: { claim_code: d.body.claim_code } });
  const t = await call("POST", `${v1}/devices/token`, {
    body: { device_id: c.body.device_id, device_secret: c.body.device_secret },
  });
  return { id: c.body.device_id, token: t.body.token, expiresIn: t.body.expires_in };
}

beforeAll(async () => {
  const p = await newProject("boards");
  v1 = p.v1;
  sk = p.sk;
  expect((await call("POST", `${v1}/spaces`, { token: sk, body: { space: "home", frame: "room" } })).status).toBe(201);
  const a = await device("p4");
  dev = a.id;
  devToken = a.token;
  expect(a.expiresIn).toBe(3600);
  otherToken = (await device("s3")).token;
  setAiForTests(fakeAi);
});

afterEach(() => {
  fakeAi.ttsRate = 16_000;
  fakeAi.heard = "where is my cup";
});

describe("upsert", () => {
  const write = (token: string, body: unknown) => call("POST", `${v1}/spaces/home/observations`, { token, body });

  it("rewrites a board's own sighting in place and digest sees it", async () => {
    const id = `d_${dev}_3_9f2c_1`;
    const first = await write(devToken, { observations: [obs({ id, room: "kitchen" })], upsert: true });
    expect(first.status).toBe(201);
    expect(first.body).toMatchObject({ accepted: 1, updated: 0, conflicts: [] });
    const moved = await write(devToken, { observations: [obs({ id, room: "kitchen", x: 0.8, t: NOW - 1000 })], upsert: true });
    expect(moved.status).toBe(201);
    expect(moved.body).toMatchObject({ accepted: 0, updated: 1, duplicates: 0, conflicts: [] });
    expect(moved.body.rev).toBe(first.body.rev + 1);
    const where = await call("GET", `${v1}/spaces/home/where?label=cup`, { token: devToken });
    expect(where.body.results.filter((r: Observation) => r.id === id)).toHaveLength(1);
    expect(where.body.results[0]).toMatchObject({ id, x: 0.8, t: NOW - 1000, device_id: dev });
    const digest = await call("GET", `${v1}/spaces/home/digest?since_rev=${first.body.rev}`, { token: devToken });
    expect(digest.body.objects.map((o: { label: string }) => o.label)).toContain("cup");

    // Same content again: a duplicate, no new revision.
    const again = await write(devToken, { observations: [obs({ id, room: "kitchen", x: 0.8, t: NOW - 1000 })], upsert: true });
    expect(again.status).toBe(200);
    expect(again.body).toMatchObject({ updated: 0, duplicates: 1, rev: moved.body.rev });
  });

  it("refuses to rewrite another writer's row, a different label or room, or an older time", async () => {
    const id = `d_${dev}_3_9f2c_2`;
    await write(devToken, { observations: [obs({ id, label: "bottle", room: "kitchen", t: NOW - 5000 })], upsert: true });
    const cases = [
      { token: otherToken, o: obs({ id, label: "bottle", room: "kitchen", x: 0.9 }) },
      { token: sk, o: obs({ id, label: "bottle", room: "kitchen", x: 0.9 }) },
      { token: devToken, o: obs({ id, label: "book", room: "kitchen" }) },
      { token: devToken, o: obs({ id, label: "bottle", room: "hall" }) },
      { token: devToken, o: obs({ id, label: "bottle", room: "kitchen", t: NOW - 9000 }) },
    ];
    for (const c of cases) {
      const r = await write(c.token, { observations: [c.o], upsert: true });
      expect(r.body).toMatchObject({ accepted: 0, updated: 0, conflicts: [id] });
    }
    const row = (await call("GET", `${v1}/spaces/home/where?label=bottle`, { token: sk })).body.results[0];
    expect(row).toMatchObject({ id, x: 0.2, room: "kitchen" });
  });

  it("takes the demo host's mirror payload (secret key, h_<sighting id>, last_room)", async () => {
    const h = { id: "h_5f0c", label: "laptop", frame: "image", x: 0.5, y: 0.4, z: 0.3, t: NOW - 7000, confidence: 0.8,
      metadata: { scene_id: "laptop", merge_count: 0, last_room: "Office" } };
    expect((await write(sk, { observations: [h] })).body.accepted).toBe(1);
    const merged = { ...h, x: 0.55, t: NOW - 2000, metadata: { ...h.metadata, merge_count: 1 } };
    const r = await write(sk, { observations: [merged], upsert: true });
    expect(r.body).toMatchObject({ updated: 1, conflicts: [] });
    const ask = await call("POST", `${v1}/spaces/home/ask`, { token: sk, body: { text: "where is my laptop" } });
    expect(ask.body.answer).toBe("Your laptop was last seen in the middle of the camera view, in the Office, just now.");
  });

  it("leaves known ids alone without the flag and rejects a non-boolean flag", async () => {
    const id = `d_${dev}_3_9f2c_3`;
    await write(devToken, { observations: [obs({ id, label: "remote" })] });
    const r = await write(devToken, { observations: [obs({ id, label: "remote", x: 0.9 })] });
    expect(r.body).toMatchObject({ accepted: 0, duplicates: 1, updated: 0 });
    expect((await write(devToken, { observations: [obs({ id, label: "remote" })], upsert: "yes" })).status).toBe(422);
  });
});

describe("board routes", () => {
  it("serves the cache JSON the firmware reads: one object per label, no people, compact, stable rev", async () => {
    const t = NOW - 30_000;
    await call("POST", `${v1}/spaces/home/observations`, {
      token: devToken,
      body: { observations: [obs({ label: "person", t }), obs({ label: "keys", room: "hall", t, id: `d_${dev}_1_a_1` })] },
    });
    const res = await raw("GET", `${v1}/spaces/home/board/memory`, { token: devToken });
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).not.toMatch(/": /); // the firmware's reader does not skip spaces after a colon
    const m = JSON.parse(text);
    const labels = m.objects.map((o: { label: string }) => o.label);
    expect(labels).not.toContain("person");
    expect(new Set(labels).size).toBe(labels.length);
    expect(m.objects.find((o: { label: string }) => o.label === "keys")).toMatchObject({ where: "in the hall", room: "hall" });
    expect(m.count).toBe(m.objects.length);
    expect(m.rev).toMatch(/^[0-9a-f]{8}$/);
    expect(m.summary.startsWith("I'm offline right now. The last things I remember: ")).toBe(true);

    // A time-only refresh of a sighting does not change what the board would say.
    await call("POST", `${v1}/spaces/home/observations`, {
      token: devToken, body: { observations: [obs({ label: "keys", room: "hall", t: NOW - 100, id: `d_${dev}_1_a_1` })], upsert: true },
    });
    const m2 = await (await raw("GET", `${v1}/spaces/home/board/memory`, { token: devToken })).json<{ rev: string }>();
    expect(m2.rev).toBe(m.rev);
  });

  it("answers a typed question as 16 kHz PCM with the text in headers, and caches the speech", async () => {
    await call("POST", `${v1}/spaces/home/observations`, {
      token: devToken, body: { observations: [obs({ label: "mug", room: "kitchen", t: NOW - 2000, id: `d_${dev}_1_a_2` })] },
    });
    fakeAi.calls.length = 0;
    const res = await raw("POST", `${v1}/spaces/home/board/ask`, { token: devToken, json: { text: "where is my mug" } });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("audio/L16; rate=16000; channels=1");
    const answer = decodeURIComponent(res.headers.get("x-answer") ?? "");
    expect(answer).toBe("Your mug was last seen in the kitchen, just now.");
    expect(res.headers.get("x-intent")).toBe("where");
    expect(res.headers.get("x-label")).toBe("mug");
    const pcm = new Uint8Array(await res.arrayBuffer());
    expect(pcm.length).toBe(answer.length * 16 * 2);
    expect(fakeAi.calls.map((c) => c.inputs)).toEqual([
      { text: answer, speaker: "asteria", encoding: "linear16", container: "wav", sample_rate: 16000 },
    ]);

    const again = await raw("POST", `${v1}/spaces/home/board/ask`, { token: devToken, json: { text: "where is my mug" } });
    expect((await again.arrayBuffer()).byteLength).toBe(pcm.length);
    expect(fakeAi.calls).toHaveLength(1); // spoken from the cache
  });

  it("resamples speech that comes back at another rate", async () => {
    fakeAi.ttsRate = 24_000;
    const res = await raw("GET", `${v1}/spaces/home/board/phrase.pcm?text=${encodeURIComponent("Hold the button again.")}`, { token: devToken });
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBe("Hold the button again.".length * 16 * 2);
    expect((await raw("GET", `${v1}/spaces/home/board/phrase.pcm?text=`, { token: devToken })).status).toBe(422);
    const long = "x".repeat(161);
    expect((await raw("GET", `${v1}/spaces/home/board/phrase.pcm?text=${long}`, { token: devToken })).status).toBe(422);
  });

  it("transcribes push-to-talk audio and answers it; silence and Whisper's stock phrases are 'not heard'", async () => {
    const clip = new Uint8Array(16_000 * 2); // 1 s
    const res = await raw("POST", `${v1}/spaces/home/board/voice`, { token: devToken, body: clip });
    expect(res.status).toBe(200);
    expect(decodeURIComponent(res.headers.get("x-transcription") ?? "")).toBe("where is my cup");
    expect(res.headers.get("x-answer")).toBeTruthy();
    const stt = fakeAi.calls.filter((c) => c.model.includes("whisper")).at(-1)!;
    expect(stt.inputs).toMatchObject({ language: "en", vad_filter: true });
    expect(atob(String(stt.inputs.audio)).slice(0, 4)).toBe("RIFF");

    fakeAi.heard = " Thank you. ";
    const quiet = await raw("POST", `${v1}/spaces/home/board/voice`, { token: devToken, body: clip });
    expect(quiet.status).toBe(422);
    expect(await quiet.json()).toMatchObject({ code: "not_heard" });
    expect((await raw("POST", `${v1}/spaces/home/board/voice`, { token: devToken, body: new Uint8Array(100) })).status).toBe(422);
    expect((await raw("POST", `${v1}/spaces/home/board/voice`, { token: devToken, body: new Uint8Array(330_000) })).status).toBe(413);
  });

  it("needs a credential for the space and speaks the summary", async () => {
    expect((await raw("GET", `${v1}/spaces/home/board/memory`)).status).toBe(401);
    expect((await raw("GET", `${v1}/spaces/nowhere/board/memory`, { token: devToken })).status).toBe(404);
    const res = await raw("GET", `${v1}/spaces/home/board/memory.pcm`, { token: devToken });
    expect(res.status).toBe(200);
    expect(res.headers.get("x-memory-rev")).toMatch(/^[0-9a-f]{8}$/);
  });
});

describe("helpers", () => {
  it("keeps board strings plain", () => {
    expect(plain(' my "big"\\cup\n ', 31)).toBe("my big cup");
    expect(plain("x".repeat(40), 31)).toHaveLength(31);
  });

  it("builds the cache from the newest sighting per label", async () => {
    const o = (label: string, t: number, extra: Partial<Observation> = {}): Observation => ({
      id: label + t, label, frame: "image", room: null, x: 0.5, y: 0.5, z: 0.5, t, confidence: 1, device_id: null,
      metadata: {}, media: null, rev: 1, ...extra,
    });
    const m = await boardMemory([o("cup", 2000, { metadata: { last_room: "living_room" } }), o("cup", 1000), o("book", 500)], 62_000);
    expect(m.objects).toEqual([
      { label: "cup", where: "in the middle of the camera view, in the living room", room: "living room", last_seen_ms: 2000, age_s: 60 },
      { label: "book", where: "in the middle of the camera view", room: null, last_seen_ms: 500, age_s: 62 },
    ]);
    expect((await boardMemory([], 0)).rev).toBe("empty");
    expect(memorySummary([])).toBe("I'm offline right now, and I don't remember any objects yet.");
  });

  it("parses WAV, resamples and spots Whisper's stock phrases", () => {
    const wav = wavFromPcm(new Uint8Array(48_000).fill(2), 24_000);
    const { pcm, rate } = pcmFromWav(wav);
    expect([pcm.length, rate]).toEqual([24_000, 24_000]);
    expect(resample(pcm, 24_000, 16_000).length).toBe(16_000);
    expect(pcmFromWav(new Uint8Array([1, 0, 2, 0, 3])).pcm.length).toBe(2);
    expect(isHallucination("Thanks for watching!")).toBe(true);
    expect(isHallucination("")).toBe(true);
    expect(isHallucination("where is my cup")).toBe(false);
  });

  it("tells clients when to retry a rate-limited call", () => {
    expect(errorResponse(new HttpError(429, "slow down", "rate_limited")).headers.get("retry-after")).toBe("60");
  });

  it("reports the deployed commit on /health", async () => {
    const r = await call("GET", "/health");
    expect(r.body).toMatchObject({ status: "ok", sha: "dev" });
  });

  it("hedges low-confidence answers without a capital in mid-sentence", async () => {
    await call("POST", `${v1}/spaces/home/observations`, {
      token: devToken, body: { observations: [obs({ label: "umbrella", room: "hall", confidence: 0.4, t: NOW - 1000 })] },
    });
    const r = await call("POST", `${v1}/spaces/home/ask`, { token: devToken, body: { text: "where is my umbrella" } });
    expect(r.body.answer).toBe("I think your umbrella was last seen in the hall, just now.");
  });
});
