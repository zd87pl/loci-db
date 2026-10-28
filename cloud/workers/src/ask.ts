/**
 * Minimal natural-language layer for Phase 0: intent parsing, label resolution
 * against tracked labels, and answers phrased for speech. The richer phrasing of
 * demo_spatial/app/phrasing.py (walk-in perspective, landmarks) moves here later.
 */

import type { Change, Observation } from "./types";

export type Intent =
  | { kind: "where"; noun: string; room: string | null }
  | { kind: "changes"; room: string | null }
  | { kind: "list"; room: string | null }
  | { kind: "unknown" };

const ALIASES: Record<string, string> = {
  mug: "cup",
  glass: "cup",
  phone: "cell phone",
  cellphone: "cell phone",
  mobile: "cell phone",
  smartphone: "cell phone",
  sofa: "couch",
  television: "tv",
  computer: "laptop",
  notebook: "laptop",
  plant: "potted plant",
  bag: "handbag",
  purse: "handbag",
  rucksack: "backpack",
};

function cleanRoom(s: string | undefined): string | null {
  if (!s) return null;
  return s.trim().toLowerCase().replace(/^the\s+/, "").replace(/\s+/g, "_");
}

export function parseIntent(text: string): Intent {
  const t = text.toLowerCase().replace(/[?.!,]/g, " ").replace(/\s+/g, " ").trim();
  let m = t.match(/^what(?:'s| has| have)? changed(?: in (?:the )?(.+))?$/);
  if (m) return { kind: "changes", room: cleanRoom(m[1]) };
  m = t.match(/^what (?:do you see|is here|objects?(?: do you see)?|is (?:in|around)(?: the)?(.*))\s*(?:in (?:the )?(.+))?$/);
  if (m) return { kind: "list", room: cleanRoom(m[2] ?? (m[1]?.trim() || undefined)) };
  m = t.match(/^(?:where(?:'s| is| are| did i (?:leave|put))|find|have you seen)\s+(?:my |the |a |an |our )?(.+?)(?: in (?:the )?([a-z _]+))?$/);
  if (m && m[1]) return { kind: "where", noun: m[1].trim(), room: cleanRoom(m[2]) };
  return { kind: "unknown" };
}

export function singular(w: string): string {
  if (w.endsWith("ies") && w.length > 4) return `${w.slice(0, -3)}y`;
  if (/(ches|shes|xes|sses)$/.test(w)) return w.slice(0, -2);
  if (w.endsWith("s") && !w.endsWith("ss") && w.length > 3) return w.slice(0, -1);
  return w;
}

/** Resolve a spoken noun onto a tracked label: exact, singular, alias. */
export function resolveLabel(noun: string, tracked: string[]): { label: string | null; spoken: string } {
  const spoken = noun.trim().toLowerCase();
  const set = new Set(tracked);
  const candidates = [spoken, singular(spoken), ALIASES[spoken], ALIASES[singular(spoken)]].filter(Boolean) as string[];
  for (const c of candidates) if (set.has(c)) return { label: c, spoken };
  const alias = ALIASES[singular(spoken)] ?? ALIASES[spoken];
  return { label: alias ?? null, spoken };
}

export function roomName(room: string | null): string {
  return room ? room.replace(/_/g, " ") : "";
}

export function age(nowMs: number, t: number): string {
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} minute${m === 1 ? "" : "s"} ago`;
  const h = Math.round(m / 60);
  if (h < 36) return `${h} hour${h === 1 ? "" : "s"} ago`;
  const d = Math.round(h / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/**
 * Where-phrase without time, safe to cache on a device. Camera-view sightings name the room the
 * writer was in when it knows it (`metadata.last_room`, sent by the demo host's mirror).
 */
export function wherePhrase(o: {
  room: string | null; frame: string; x: number; y: number; z: number; metadata?: Record<string, unknown> | null;
}): string {
  if (o.room) return `in the ${roomName(o.room)}`;
  const last = o.metadata?.last_room;
  const inRoom = typeof last === "string" && last.trim() ? `, in the ${roomName(last.trim())}` : "";
  if (o.frame === "camera") {
    const side = o.x < -0.3 ? "to the left" : o.x > 0.3 ? "to the right" : "straight ahead";
    return `about ${o.z.toFixed(1)} metres ${side} of where the camera was${inRoom}`;
  }
  if (o.frame === "image") {
    const side = o.x < 0.33 ? "on the left" : o.x > 0.67 ? "on the right" : "in the middle";
    return `${side} of the camera view${inRoom}`;
  }
  return `at ${o.x.toFixed(1)}, ${o.y.toFixed(1)}, ${o.z.toFixed(1)}`;
}

export function whereAnswer(spoken: string, label: string, obs: Observation[], nowMs: number): string {
  const latest = obs[0];
  if (!latest) return `I haven't seen your ${spoken} yet.`;
  const named = spoken === label || singular(spoken) === label ? label : `${spoken} (I track it as a ${label})`;
  const lead = latest.confidence < 0.6 ? "I think your" : "Your";
  let text = `${lead} ${named} was last seen ${wherePhrase(latest)}, ${age(nowMs, latest.t)}.`;
  const earlierRoom = obs.find((o) => o.room && o.room !== latest.room);
  if (earlierRoom) text += ` Before that it was in the ${roomName(earlierRoom.room)}, ${age(nowMs, earlierRoom.t)}.`;
  return text;
}

export function changesAnswer(changes: Change[], room: string | null): string {
  const where = room ? ` in the ${roomName(room)}` : "";
  const moved = changes.filter((c) => c.kind === "moved");
  const fresh = changes.filter((c) => c.kind === "new");
  if (moved.length === 0 && fresh.length === 0) return `Nothing has changed${where} recently.`;
  const parts: string[] = [];
  for (const c of moved.slice(0, 4)) parts.push(`the ${c.label} moved about ${(c.distance_m ?? 0).toFixed(1)} metres`);
  if (fresh.length) parts.push(`new: ${fresh.slice(0, 5).map((c) => c.label).join(", ")}`);
  return `Recent changes${where}: ${parts.join("; ")}.`;
}

export function listAnswer(objects: Observation[], room: string | null, nowMs: number): string {
  const where = room ? ` in the ${roomName(room)}` : "";
  if (objects.length === 0) return `I haven't seen anything${where} yet.`;
  const items = objects.slice(0, 8).map((o) => `${o.label} ${o.room && !room ? `in the ${roomName(o.room)}` : ""} (${age(nowMs, o.t)})`.replace(/\s+/g, " "));
  return `I remember${where}: ${items.join("; ")}.`;
}
