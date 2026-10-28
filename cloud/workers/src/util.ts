/** Small helpers: JSON responses, random ids, hashing, HS256 JWTs. */

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly code = "error",
  ) {
    super(detail);
  }
}

export function json(body: unknown, status = 200, headers: HeadersInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

export function errorResponse(e: HttpError): Response {
  // The rate limiters count per 60 s window.
  return json({ detail: e.detail, code: e.code }, e.status, e.status === 429 ? { "retry-after": "60" } : {});
}

const enc = new TextEncoder();

export function randomHex(bytes: number): string {
  const b = new Uint8Array(bytes);
  crypto.getRandomValues(b);
  return [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
}

/** Lowercase alphanumeric id, e.g. project refs. */
export function randomAlnum(len: number): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  const b = new Uint8Array(len);
  crypto.getRandomValues(b);
  return [...b].map((x) => alphabet[x % alphabet.length]).join("");
}

/** Human-typable claim code: 4-4-4 uppercase, no ambiguous characters. */
export function claimCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const b = new Uint8Array(12);
  crypto.getRandomValues(b);
  const s = [...b].map((x) => alphabet[x % alphabet.length]).join("");
  return `${s.slice(0, 4)}-${s.slice(4, 8)}-${s.slice(8, 12)}`;
}

export async function sha256Hex(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", enc.encode(s));
  return [...new Uint8Array(d)].map((x) => x.toString(16).padStart(2, "0")).join("");
}

export function timingSafeEqualStr(a: string, b: string): boolean {
  const ab = enc.encode(a);
  const bb = enc.encode(b);
  if (ab.length !== bb.length) return false;
  let diff = 0;
  for (let i = 0; i < ab.length; i++) diff |= (ab[i] ?? 0) ^ (bb[i] ?? 0);
  return diff === 0;
}

function b64url(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlDecode(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? "" : "=".repeat(4 - (s.length % 4));
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/") + pad);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

async function hmacKey(secretHex: string): Promise<CryptoKey> {
  const raw = Uint8Array.from(secretHex.match(/../g) ?? [], (h) => parseInt(h, 16));
  return crypto.subtle.importKey("raw", raw, { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function signJwt(payload: Record<string, unknown>, secretHex: string): Promise<string> {
  const head = b64url(enc.encode(JSON.stringify({ alg: "HS256", typ: "JWT" })));
  const body = b64url(enc.encode(JSON.stringify(payload)));
  const sig = await crypto.subtle.sign("HMAC", await hmacKey(secretHex), enc.encode(`${head}.${body}`));
  return `${head}.${body}.${b64url(new Uint8Array(sig))}`;
}

/** Verify signature and expiry; returns the payload or null. */
export async function verifyJwt(token: string, secretHex: string, nowS: number): Promise<Record<string, unknown> | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [head, body, sig] = parts as [string, string, string];
  let header: { alg?: string };
  let payload: Record<string, unknown>;
  try {
    header = JSON.parse(new TextDecoder().decode(b64urlDecode(head)));
    payload = JSON.parse(new TextDecoder().decode(b64urlDecode(body)));
  } catch {
    return null;
  }
  if (header.alg !== "HS256") return null;
  const ok = await crypto.subtle.verify("HMAC", await hmacKey(secretHex), b64urlDecode(sig), enc.encode(`${head}.${body}`));
  if (!ok) return null;
  if (typeof payload.exp !== "number" || payload.exp <= nowS) return null;
  return payload;
}

export async function readJson<T>(req: Request, maxBytes = 512 * 1024): Promise<T> {
  const len = Number(req.headers.get("content-length") ?? "0");
  if (len > maxBytes) throw new HttpError(413, `body exceeds ${maxBytes} bytes`, "too_large");
  const text = await req.text();
  if (text.length > maxBytes) throw new HttpError(413, `body exceeds ${maxBytes} bytes`, "too_large");
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new HttpError(400, "body is not valid JSON", "bad_json");
  }
}

export const SLUG = /^[a-z0-9][a-z0-9-]{0,62}$/;
