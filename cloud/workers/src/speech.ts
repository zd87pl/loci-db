/**
 * Speech for the boards, on Workers AI: text to 16 kHz mono PCM16 (the boards' speaker format,
 * the same bytes the demo host's /api/esp32/*.pcm routes return) and push-to-talk PCM to text.
 *
 *   TTS: @cf/deepgram/aura-1, asked for linear16 at 16 kHz in a WAV container; the header is
 *        read so a different rate is resampled instead of played at the wrong speed.
 *   STT: @cf/openai/whisper-large-v3-turbo on a WAV built from the board's raw PCM.
 */

import type { Env } from "./types";
import { HttpError } from "./util";

export const SAMPLE_RATE = 16_000;
export const AUDIO_TYPE = `audio/L16; rate=${SAMPLE_RATE}; channels=1`;
export const TTS_MODEL = "@cf/deepgram/aura-1";
export const STT_MODEL = "@cf/openai/whisper-large-v3-turbo";
/** An adult female voice, like the host's en-US-JennyNeural. */
export const VOICE = "asteria";

/** What the code needs from the AI binding (tests install a stub with setAiForTests). */
export interface AiRunner {
  run(model: string, inputs: Record<string, unknown>): Promise<unknown>;
}

let testAi: AiRunner | null = null;

/** Tests only: replace Workers AI (never reached from tests, see vitest.config.ts). */
export function setAiForTests(ai: AiRunner | null): void {
  testAi = ai;
}

function ai(env: Env): AiRunner {
  const a = testAi ?? (env.AI as unknown as AiRunner | undefined);
  if (!a) throw new HttpError(503, "speech is not configured on this deployment", "unavailable");
  return a;
}

// Whisper's stock phrases for silence or a bumped button (same list as the host,
// demo_spatial/app/voice_pipeline.py): a transcript that is only one of these is "nothing said".
const STT_HALLUCINATIONS = new Set([
  "thank you", "thank you very much", "thanks", "thanks for watching", "thank you for watching",
  "thank you so much", "you", "bye", "bye bye", "okay", "oh", "uh", "um", "hmm", "so", "the end",
  "please subscribe", "subtitles by the amara org community",
]);

export function isHallucination(text: string): boolean {
  const t = text.toLowerCase().replace(/[^\w' ]+/g, " ").split(/\s+/).filter(Boolean).join(" ");
  return !t || STT_HALLUCINATIONS.has(t);
}

async function bytesOf(out: unknown): Promise<Uint8Array> {
  if (out instanceof ReadableStream) return new Uint8Array(await new Response(out).arrayBuffer());
  if (out instanceof ArrayBuffer) return new Uint8Array(out);
  if (out instanceof Uint8Array) return out;
  const audio = (out as { audio?: unknown } | null)?.audio;
  if (typeof audio === "string") return Uint8Array.from(atob(audio), (c) => c.charCodeAt(0));
  throw new HttpError(502, "speech service returned no audio", "upstream");
}

function u32(b: Uint8Array, i: number): number {
  return (b[i]! | (b[i + 1]! << 8) | (b[i + 2]! << 16) | (b[i + 3]! << 24)) >>> 0;
}

/** PCM16 samples and rate from a WAV file; raw bytes are taken as 16 kHz mono PCM16. */
export function pcmFromWav(b: Uint8Array): { pcm: Int16Array; rate: number } {
  const tag = (i: number) => String.fromCharCode(b[i]!, b[i + 1]!, b[i + 2]!, b[i + 3]!);
  const even = (n: number) => n - (n % 2);
  if (b.length < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
    return { pcm: new Int16Array(b.buffer.slice(b.byteOffset, b.byteOffset + even(b.length))), rate: SAMPLE_RATE };
  }
  let rate = SAMPLE_RATE;
  let channels = 1;
  let i = 12;
  while (i + 8 <= b.length) {
    const id = tag(i);
    const size = u32(b, i + 4);
    if (id === "fmt ") {
      channels = b[i + 10]! | (b[i + 11]! << 8);
      rate = u32(b, i + 12);
    } else if (id === "data") {
      // Streamed WAVs may carry a placeholder size: take what is there.
      const end = size === 0 || size === 0xffffffff || i + 8 + size > b.length ? b.length : i + 8 + size;
      const raw = new Int16Array(b.buffer.slice(b.byteOffset + i + 8, b.byteOffset + i + 8 + even(end - i - 8)));
      if (channels <= 1) return { pcm: raw, rate };
      const mono = new Int16Array(Math.floor(raw.length / channels));
      for (let k = 0; k < mono.length; k++) mono[k] = raw[k * channels]!;
      return { pcm: mono, rate };
    }
    i += 8 + size + (size % 2);
  }
  throw new HttpError(502, "speech service returned a WAV without audio", "upstream");
}

/** Linear-interpolation resample of mono PCM16 (speech only, so no filter). */
export function resample(pcm: Int16Array, from: number, to: number): Int16Array {
  if (from === to || pcm.length === 0) return pcm;
  const out = new Int16Array(Math.max(1, Math.round((pcm.length * to) / from)));
  const step = from / to;
  for (let k = 0; k < out.length; k++) {
    const pos = k * step;
    const j = Math.floor(pos);
    const a = pcm[Math.min(j, pcm.length - 1)]!;
    const c = pcm[Math.min(j + 1, pcm.length - 1)]!;
    out[k] = Math.round(a + (c - a) * (pos - j));
  }
  return out;
}

/** WAV container around mono PCM16, for Whisper. */
export function wavFromPcm(pcm: Uint8Array, rate = SAMPLE_RATE): Uint8Array {
  const out = new Uint8Array(44 + pcm.length);
  const v = new DataView(out.buffer);
  const ascii = (o: number, s: string) => {
    for (let k = 0; k < 4; k++) out[o + k] = s.charCodeAt(k);
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + pcm.length, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ascii(36, "data");
  v.setUint32(40, pcm.length, true);
  out.set(pcm, 44);
  return out;
}

function base64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Speak `text`: 16 kHz mono PCM16 bytes. */
export async function tts(env: Env, text: string): Promise<Uint8Array> {
  const out = await ai(env).run(TTS_MODEL, {
    text, speaker: VOICE, encoding: "linear16", container: "wav", sample_rate: SAMPLE_RATE,
  });
  const { pcm, rate } = pcmFromWav(await bytesOf(out));
  const fixed = resample(pcm, rate, SAMPLE_RATE);
  return new Uint8Array(fixed.buffer, fixed.byteOffset, fixed.byteLength);
}

/** Transcribe one push-to-talk clip (raw 16 kHz mono PCM16); "" when nothing was said. */
export async function stt(env: Env, pcm: Uint8Array, labels: string[]): Promise<string> {
  const inputs: Record<string, unknown> = { audio: base64(wavFromPcm(pcm)), language: "en", vad_filter: true };
  // Remembered objects first, like the host's vocabulary prompt, so "where is my cup" is heard right.
  if (labels.length) inputs.initial_prompt = `Where is my ${labels.slice(0, 20).join(", ")}?`;
  const out = (await ai(env).run(STT_MODEL, inputs)) as { text?: unknown } | null;
  const text = typeof out?.text === "string" ? out.text.trim() : "";
  return isHallucination(text) ? "" : text;
}
