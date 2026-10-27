import type { SpaceObject } from "./space";

export interface Env {
  DB: D1Database;
  SPACE: DurableObjectNamespace<SpaceObject>;
  WRITE_LIMITER?: RateLimit;
  USAGE?: AnalyticsEngineDataset;
  ADMIN_TOKEN?: string;
  PROJECT_HOST_SUFFIX?: string;
  DEVICE_TOKEN_TTL_S?: string;
  /** Dev/test only: allow EU projects where the runtime lacks jurisdiction support. Never set in production. */
  DEV_ALLOW_UNPINNED?: string;
}

export type Frame = "room" | "camera" | "image" | "map" | "geo";
export const FRAMES: readonly Frame[] = ["room", "camera", "image", "map", "geo"];

export interface Extents {
  lo: [number, number, number];
  hi: [number, number, number];
}

export interface SpaceMeta {
  ref: string;
  space: string;
  frame: Frame;
  extents: Extents;
}

/** One observation as accepted by POST /spaces/:space/observations. */
export interface ObservationIn {
  id?: string;
  seq?: number;
  label: string;
  frame?: Frame;
  room?: string | null;
  x: number;
  y: number;
  z: number;
  t: number;
  confidence?: number;
  embedding?: number[];
  embedding_model?: string;
  metadata?: Record<string, unknown>;
  media?: string;
}

/** One observation as returned by queries (no embedding). */
export interface Observation {
  id: string;
  label: string;
  frame: Frame;
  room: string | null;
  x: number;
  y: number;
  z: number;
  t: number;
  confidence: number;
  device_id: string | null;
  metadata: Record<string, unknown>;
  media: string | null;
  rev: number;
}

export interface InsertResult {
  accepted: number;
  duplicates: number;
  deleted: number;
  ids: string[];
  rev: number;
}

export interface Change {
  kind: "new" | "moved" | "seen";
  label: string;
  room: string | null;
  t: number;
  first_seen: number;
  position: [number, number, number];
  previous_position: [number, number, number] | null;
  distance_m: number | null;
}

export interface DigestObject {
  label: string;
  room: string | null;
  frame: Frame;
  x: number;
  y: number;
  z: number;
  t: number;
  confidence: number;
  where: string;
}

export interface Digest {
  rev: number;
  full: boolean;
  objects: DigestObject[];
  removed: { label: string; room: string | null }[];
}

export type Principal =
  | { kind: "secret"; ref: string; keyId: string }
  | { kind: "publishable"; ref: string; keyId: string }
  | { kind: "device"; ref: string; deviceId: string; spaces: string[] };
