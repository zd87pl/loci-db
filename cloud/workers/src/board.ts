/**
 * Board routes: what an ESP32 board needs to answer from the cloud, in the formats the demo
 * host's /api/esp32/* routes use, so the firmware's cache and playback code works against both.
 *
 *   GET  spaces/<s>/board/memory            {"rev","count","objects":[{label,where,room,last_seen_ms,age_s}],"summary","now_ms"}
 *   GET  spaces/<s>/board/memory.pcm        the summary, spoken
 *   GET  spaces/<s>/board/phrase.pcm?text=  any short sentence, spoken (cached)
 *   POST spaces/<s>/board/ask               {"text"} → spoken answer; X-Answer, X-Intent, X-Label
 *   POST spaces/<s>/board/voice             raw PCM16 (16 kHz mono, ≤ 10 s) → spoken answer; X-Transcription too
 *
 * Audio is audio/L16; rate=16000; channels=1. The firmware reads JSON with strstr() and does not
 * unescape, so strings here never carry quotes, backslashes or control characters.
 */

import { roomName, wherePhrase } from "./ask";
import { canRead, principalKey } from "./auth";
import { getSpace, spaceStub } from "./control";
import { AUDIO_TYPE, SAMPLE_RATE, VOICE, stt, tts } from "./speech";
import type { Env, Observation, Principal } from "./types";
import { HttpError, json, readJson, sha256Hex } from "./util";

const MAX_OBJECTS = 12;
const MAX_LABEL = 31;
const MAX_PHRASE = 160;
const MAX_VOICE_BYTES = 10 * SAMPLE_RATE * 2;
const MIN_VOICE_BYTES = SAMPLE_RATE / 5; // 0.1 s
const MAX_HEADER_TEXT = 400;

/** The part of SpaceObject.ask a board uses (the RPC type of the full result is not inferable). */
type AskReply = { answer: string; intent: string; label: string | null };

/** Safe for the board's string reader: no quotes, backslashes or control characters. */
export function plain(s: string, max: number): string {
  // eslint-disable-next-line no-control-regex
  return s.replace(/["\\\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, max);
}

export interface BoardObject {
  label: string;
  where: string;
  room: string | null;
  last_seen_ms: number;
  age_s: number;
}

export interface BoardMemory {
  rev: string;
  count: number;
  objects: BoardObject[];
  summary: string;
  now_ms: number;
}

export function memorySummary(objects: BoardObject[], maxItems = 6): string {
  if (!objects.length) return "I'm offline right now, and I don't remember any objects yet.";
  const parts = objects.slice(0, maxItems).map((o) => `${o.label} ${o.where}`);
  const more = objects.length - maxItems;
  return `I'm offline right now. The last things I remember: ${parts.join("; ")}${more > 0 ? `, and ${more} more` : ""}.`;
}

/** Newest sighting per label in the board's cache shape; `rev` changes only when what it would say changes. */
export async function boardMemory(latest: Observation[], nowMs: number): Promise<BoardMemory> {
  const objects: BoardObject[] = [];
  const seen = new Set<string>();
  for (const o of latest) {
    const label = plain(o.label, MAX_LABEL);
    if (!label || seen.has(label)) continue;
    seen.add(label);
    const lastRoom = typeof o.metadata?.last_room === "string" ? o.metadata.last_room : null;
    const room = o.room ?? lastRoom;
    objects.push({
      label,
      where: plain(wherePhrase(o), 120) || "somewhere nearby",
      room: room ? plain(roomName(room), 48) || null : null,
      last_seen_ms: o.t,
      age_s: Math.max(0, Math.round((nowMs - o.t) / 1000)),
    });
    if (objects.length >= MAX_OBJECTS) break;
  }
  const rev = objects.length
    ? (await sha256Hex(objects.map((o) => `${o.label}@${o.where}@${o.room ?? ""}`).sort().join("|"))).slice(0, 8)
    : "empty";
  return { rev, count: objects.length, objects, summary: memorySummary(objects), now_ms: nowMs };
}

function headerText(s: string): string {
  return encodeURIComponent(s.slice(0, MAX_HEADER_TEXT));
}

async function aiLimit(env: Env, key: string): Promise<void> {
  if (!env.AI_LIMITER) return;
  const { success } = await env.AI_LIMITER.limit({ key });
  if (!success) throw new HttpError(429, "speech rate limit exceeded", "rate_limited");
}

async function readPcm(req: Request): Promise<Uint8Array> {
  const len = Number(req.headers.get("content-length") ?? "0");
  if (len > MAX_VOICE_BYTES) throw new HttpError(413, `audio exceeds ${MAX_VOICE_BYTES} bytes (10 s)`, "too_large");
  const pcm = new Uint8Array(await req.arrayBuffer());
  if (pcm.length > MAX_VOICE_BYTES) throw new HttpError(413, `audio exceeds ${MAX_VOICE_BYTES} bytes (10 s)`, "too_large");
  if (pcm.length < MIN_VOICE_BYTES) throw new HttpError(422, "audio is shorter than 0.1 s", "invalid");
  return pcm.length % 2 ? pcm.subarray(0, pcm.length - 1) : pcm;
}

export async function handleBoard(
  req: Request, env: Env, url: URL, ref: string, p: Principal, space: string, sub: string,
): Promise<Response> {
  const row = await getSpace(env, ref, space);
  if (!row) throw new HttpError(404, `space ${space} not found`, "not_found");
  if (!canRead(p, space)) throw new HttpError(403, "not allowed to read this space", "forbidden");
  const stub = await spaceStub(env, ref, space);
  const limitKey = `${ref}:${principalKey(p)}`;

  // Spoken text, from the space's cache when the same sentence was spoken before (cache hits are free).
  const speak = async (text: string, headers: Record<string, string> = {}): Promise<Response> => {
    const key = await sha256Hex(`${VOICE}|${text}`);
    let pcm = await stub.ttsGet(key);
    if (!pcm) {
      await aiLimit(env, limitKey);
      pcm = await tts(env, text);
      await stub.ttsPut(key, pcm);
    }
    const seconds = (pcm.byteLength / (SAMPLE_RATE * 2)).toFixed(2);
    return new Response(pcm, { headers: { "content-type": AUDIO_TYPE, "x-audio-seconds": seconds, ...headers } });
  };
  const answerHeaders = (r: AskReply) => ({
    "x-answer": headerText(r.answer),
    "x-intent": r.intent,
    ...(r.label ? { "x-label": headerText(r.label) } : {}),
  });

  if (sub === "memory" && req.method === "GET") {
    return json(await boardMemory(await stub.newestPerLabel(MAX_OBJECTS * 2), Date.now()));
  }
  if (sub === "memory.pcm" && req.method === "GET") {
    const m = await boardMemory(await stub.newestPerLabel(MAX_OBJECTS * 2), Date.now());
    return speak(m.summary, { "x-memory-rev": m.rev });
  }
  if (sub === "phrase.pcm" && req.method === "GET") {
    const text = plain(url.searchParams.get("text") ?? "", MAX_PHRASE + 1);
    if (!text || text.length > MAX_PHRASE) throw new HttpError(422, `text must be 1-${MAX_PHRASE} characters`, "invalid");
    return speak(text);
  }
  if (sub === "ask" && req.method === "POST") {
    const body = await readJson<{ text?: unknown }>(req, 8 * 1024);
    if (typeof body.text !== "string" || !body.text.trim() || body.text.length > 500) {
      throw new HttpError(422, "text must be 1-500 characters", "invalid");
    }
    const r = (await stub.ask(body.text)) as unknown as AskReply;
    return speak(r.answer, answerHeaders(r));
  }
  if (sub === "voice" && req.method === "POST") {
    const pcm = await readPcm(req);
    await aiLimit(env, limitKey);
    const heard = await stt(env, pcm, await stub.labels());
    if (!heard) return json({ detail: "nothing was heard", code: "not_heard", transcription: "" }, 422);
    const r = (await stub.ask(heard)) as unknown as AskReply;
    return speak(r.answer, { ...answerHeaders(r), "x-transcription": headerText(heard) });
  }
  throw new HttpError(404, "not found", "not_found");
}
