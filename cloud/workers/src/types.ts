import type { SpaceObject } from "./space";

export interface Env {
  DB: D1Database;
  SPACE: DurableObjectNamespace<SpaceObject>;
  WRITE_LIMITER?: RateLimit;
  QUERY_LIMITER?: RateLimit;
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
  /** Logical epoch length: the unit of the 4D address's time normalisation, as in the library. */
  epoch_ms: number;
}

export const DEFAULT_EPOCH_MS = 5000;

/** The library's 4D address: Hilbert indices of (x, y, z, t within the epoch) at orders 4, 8 and 12. */
export interface Address {
  epoch: number;
  r4: number;
  r8: number;
  r12: number;
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
  /** Null only while a Phase 0 row waits for the address backfill. */
  address: Address | null;
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

/** Result of the SDK's insert_batch: ids in input order, and the last id per scene for the next chunk. */
export interface BatchResult {
  ids: string[];
  chains: Record<string, string>;
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
