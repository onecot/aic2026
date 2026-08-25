import type {
  FramesAroundParams,
  FramesPageParams,
  HealthPayload,
  SearchRequestBody,
  SearchResponse,
  SystemPayload,
  VideoFramesResponse,
  VideoMetadata,
} from "./types";

/**
 * Base URL for all API calls.
 *
 * Defaults to same-origin ("") so the Vite dev proxy (or a production
 * reverse proxy) forwards /api to the FastAPI backend, which does not send
 * CORS headers. Set VITE_API_URL to an absolute origin to call a backend
 * that explicitly allows cross-origin requests.
 */
export const API_BASE: string = import.meta.env.VITE_API_URL ?? "";

/** Error kind surfaced by the client. */
export type ApiErrorKind = "http" | "network";

/** Normalized API failure. The message is UI-safe; raw details stay in console. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  /** HTTP status code; undefined for network failures. */
  readonly status?: number;

  constructor(message: string, kind: ApiErrorKind, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

function friendlyMessage(status: number): string {
  if (status === 400) return "Invalid request.";
  if (status === 404) return "Not found.";
  if (status === 422) return "Invalid request body.";
  if (status === 503) return "Retrieval backend is starting or busy.";
  if (status === 405 || status === 406) return "Endpoint not supported.";
  return `Request failed (HTTP ${status}).`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (cause) {
    // Network-level failure (backend down, DNS, CORS block). Raw cause is
    // logged; the UI only sees the friendly message.
    console.error(`[api] ${init?.method ?? "GET"} ${path} failed`, cause);
    throw new ApiError("Cannot reach the retrieval backend.", "network");
  }
  if (!response.ok) {
    const raw = await response.text().catch(() => "");
    console.error(
      `[api] ${init?.method ?? "GET"} ${path} -> ${response.status}`,
      raw.slice(0, 500),
    );
    throw new ApiError(friendlyMessage(response.status), "http", response.status);
  }
  return (await response.json()) as T;
}

/** GET /api/health */
export async function health(signal?: AbortSignal): Promise<HealthPayload> {
  // The starting state is reported via HTTP 503 with a flat body; treat it
  // as data instead of an error so the UI can poll until READY.
  try {
    const response = await fetch(`${API_BASE}/api/health`, { signal });
    if (!response.ok && response.status !== 503) {
      throw new ApiError(friendlyMessage(response.status), "http", response.status);
    }
    return (await response.json()) as HealthPayload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    console.error("[api] health check failed", error);
    throw new ApiError("Cannot reach the retrieval backend.", "network");
  }
}

/** GET /api/system */
export function getSystem(signal?: AbortSignal): Promise<SystemPayload> {
  return requestJson<SystemPayload>("/api/system", { signal });
}

/** POST /api/search */
export async function search(
  body: SearchRequestBody,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  return requestJson<SearchResponse>("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** POST /api/search/similar (Query-by-Image visual similarity search) */
export async function searchSimilar(
  body: { video_id: string; frame_index: number; limit?: number },
  signal?: AbortSignal,
): Promise<SearchResponse> {
  return requestJson<SearchResponse>("/api/search/similar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** GET /api/videos/{video_id} */
export function getVideo(videoId: string, signal?: AbortSignal): Promise<VideoMetadata> {
  return requestJson<VideoMetadata>(
    `/api/videos/${encodeURIComponent(videoId)}`,
    { signal },
  );
}

/**
 * GET /api/videos/{video_id}/frames around a center keyframe.
 * Pass page params instead to list from the start of the video.
 */
export function getFrames(
  videoId: string,
  params: FramesAroundParams | FramesPageParams = {},
  signal?: AbortSignal,
): Promise<VideoFramesResponse> {
  const query = new URLSearchParams();
  if ("center" in params) {
    query.set("center", String(params.center));
    if (params.radius !== undefined) query.set("radius", String(params.radius));
    if (params.limit !== undefined) query.set("limit", String(params.limit));
  } else {
    if (params.offset !== undefined) query.set("offset", String(params.offset));
    if (params.limit !== undefined) query.set("limit", String(params.limit));
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return requestJson<VideoFramesResponse>(
    `/api/videos/${encodeURIComponent(videoId)}/frames${suffix}`,
    { signal },
  );
}

/**
 * Build the <img> URL for one indexed keyframe. Pure URL construction —
 * no request is made here; the browser loads it when rendered.
 */
export function getFrameUrl(videoId: string, frameIndex: number): string {
  return `${API_BASE}/api/frames/${encodeURIComponent(videoId)}/${frameIndex}`;
}
