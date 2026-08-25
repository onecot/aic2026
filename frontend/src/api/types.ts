/**
 * Typed mirrors of the backend HTTP contracts (src/aic2026/api.py).
 *
 * These describe the current API only; do not invent fields the backend
 * does not return. Optional fields (`null`) mirror the backend exactly.
 */

export type BackendTask = "kis" | "trake";

/** GET /api/health */
export interface HealthPayload {
  status: "ready" | "starting";
  model: boolean;
  index: boolean;
  frame_catalog: boolean;
}

/** GET /api/system */
export interface SystemModelInfo {
  name: string;
  backend: string;
}

export interface SystemIndexInfo {
  id: string;
  total_keyframes: number;
  total_videos: number;
  embedding_dimension: number;
  similarity: string;
}

export interface SystemKeyframesInfo {
  version: string | null;
  total_keyframes: number;
  total_videos: number;
}

export interface SystemVlmInfo {
  enabled: boolean;
  model: string | null;
  load_policy: string;
}

export interface SystemFeatures {
  visual_embedding: boolean;
  ocr: boolean;
  asr: boolean;
  metadata_search: boolean;
  temporal_navigation: boolean;
}

export interface SystemPayload {
  status: "ready";
  retrieval: {
    model: SystemModelInfo;
    index: SystemIndexInfo;
    keyframes: SystemKeyframesInfo;
  };
  features: SystemFeatures;
  generation: { vlm: SystemVlmInfo };
}

/** One ranked candidate (POST /api/search, KIS). */
export interface SearchHit {
  rank: number;
  video_id: string;
  frame_index: number;
  keyframe_ordinal: number;
  pts_time: number;
  score: number;
}

/** One temporal window candidate of a TRAKE event. */
export interface TraKeWindow {
  video_id: string;
  frame_start: number;
  frame_end: number;
  score: number;
  rank: number;
}

/** One TRAKE event with ranked predictions and windows. */
export interface TraKeEvent {
  event_id: string;
  event_text: string;
  predictions: SearchHit[];
  windows: TraKeWindow[];
}

export interface SearchTiming {
  total_ms: number;
  encode_ms: number | null;
  retrieve_ms: number | null;
}

export interface SearchSignals {
  visual_embedding: boolean;
  ocr: boolean;
  asr: boolean;
  metadata: boolean;
  temporal_reranking: boolean;
}

export interface SearchMeta {
  task: string;
  returned: number;
  timing: SearchTiming;
  signals: SearchSignals;
}

/** POST /api/search */
export interface SearchResponse {
  query: string;
  results: SearchHit[];
  events?: TraKeEvent[] | null;
  meta?: SearchMeta | null;
}

export interface SearchRequestBody {
  query: string;
  task: BackendTask;
  limit: number;
}

/** GET /api/videos/{video_id} */
export interface VideoMetadata {
  video_id: string;
  frame_count: number;
  first_frame_index: number;
  last_frame_index: number;
}

/** One indexed keyframe while browsing a video. */
export interface FrameRecord {
  keyframe_ordinal: number;
  frame_index: number;
  pts_time: number;
  fps: number;
}

/** GET /api/videos/{video_id}/frames */
export interface VideoFramesResponse {
  video_id: string;
  frames: FrameRecord[];
  total: number;
}

export interface FramesAroundParams {
  center: number;
  radius?: number;
  limit?: number;
}

export interface FramesPageParams {
  offset?: number;
  limit?: number;
}
