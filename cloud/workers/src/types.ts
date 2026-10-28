import type { SpaceObject } from "./space";

export interface Env {
  DB: D1Database;
  SPACE: DurableObjectNamespace<SpaceObject>;
  WRITE_LIMITER?: RateLimit;
  QUERY_LIMITER?: RateLimit;
  /** Workers AI calls (TTS, STT) per principal; speech cache hits do not count. */
  AI_LIMITER?: RateLimit;
  AI?: Ai;
  USAGE?: AnalyticsEngineDataset;
  /** Commit deployed, set by the deploy workflow (`--var GIT_SHA:<sha>`); shown on /health. */
  GIT_SHA?: string;
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

/** A validated POST /spaces/:space/query (also the core of the SDK's /query). */
export interface QuerySpec {
  /** Inclusive box in the space frame. */
  region: Extents | null;
  /** Inclusive event-time bounds, epoch ms. */
  timeStart: number | null;
  timeEnd: number | null;
  labels: string[] | null;
  room: string | null;
  minConfidence: number | null;
  /** Rank by cosine similarity to this vector (decayed by age); without it, newest first. */
  vector: number[] | null;
  /** Recency half-life applied to similarity; 0 ranks by similarity alone. */
  halfLifeMs: number;
  limit: number;
  includeEmbedding: boolean;
}

/** WorldState fields stored for observations written through the SDK compatibility layer. */
export interface SdkFields {
  scene_id: string | null;
  scale_level: string | null;
  prev_state_id: string | null;
  next_state_id: string | null;
}

export interface QueryHit {
  obs: Observation;
  sdk: SdkFields;
  /** Raw cosine similarity and the decayed ranking score; null when not ranked by vector. */
  similarity: number | null;
  score: number | null;
  embedding: number[] | null;
}

export interface QueryResult {
  hits: QueryHit[];
  /** True when a vector query stopped at the scan cap before seeing every matching row. */
  truncated: boolean;
}

/** One SDK WorldState as accepted by the compatibility /insert. */
export interface SdkStateIn {
  x: number;
  y: number;
  z: number;
  timestamp_ms: number;
  vector: number[];
  scene_id: string;
  scale_level: string;
  confidence: number;
  metadata: Record<string, unknown>;
  prev_state_id: string | null;
  next_state_id: string | null;
}

export interface InsertResult {
  accepted: number;
  duplicates: number;
  /** Known ids rewritten by an upsert. */
  updated: number;
  /** Ids an upsert was not allowed to rewrite (another writer's row, a different label or room, an older t). */
  conflicts: string[];
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
