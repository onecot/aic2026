import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, API_BASE, getFrameUrl, getFrames, health, search } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("health", () => {
  it("parses the ready payload", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: "ready", model: true, index: true, frame_catalog: true }),
    );
    const payload = await health();
    expect(payload).toEqual({
      status: "ready",
      model: true,
      index: true,
      frame_catalog: true,
    });
    expect(fetchMock.mock.calls[0][0]).toBe(`${API_BASE}/api/health`);
  });

  it("treats the starting 503 flat body as data, not an error", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        { status: "starting", model: false, index: false, frame_catalog: false },
        503,
      ),
    );
    const payload = await health();
    expect(payload.status).toBe("starting");
  });

  it("maps network failure to a friendly ApiError", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    const error = await health().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).kind).toBe("network");
    expect((error as ApiError).message).not.toContain("TypeError");
  });

  it("surfaces unexpected HTTP errors with the status code", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "boom" }, 500));
    const error = await health().catch((e: unknown) => e);
    expect((error as ApiError).status).toBe(500);
  });
});

describe("search", () => {
  it("posts the exact backend request body and parses hits", async () => {
    const response = {
      query: "red boat",
      results: [
        {
          rank: 1,
          video_id: "L21_V001",
          frame_index: 10,
          keyframe_ordinal: 1,
          pts_time: 4.0,
          score: 0.42,
        },
      ],
      meta: {
        task: "kis",
        returned: 1,
        timing: { total_ms: 12.5, encode_ms: 9.1, retrieve_ms: 3.2 },
        signals: {
          visual_embedding: true,
          ocr: false,
          asr: false,
          metadata: false,
          temporal_reranking: false,
        },
      },
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(response));

    const payload = await search({ query: "red boat", task: "kis", limit: 100 });

    expect(payload.results[0]?.video_id).toBe("L21_V001");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/search`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      query: "red boat",
      task: "kis",
      limit: 100,
    });
  });

  it("maps 400/404/422/503 to friendly messages without raw details", async () => {
    for (const status of [400, 404, 422, 503]) {
      fetchMock.mockResolvedValueOnce(
        jsonResponse({ detail: "very internal stack info" }, status),
      );
      const error = await search(
        { query: "x", task: "trake", limit: 5 },
      ).catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(status);
      expect((error as ApiError).message).not.toContain("stack");
    }
  });
});

describe("getFrames", () => {
  it("builds center/radius/limit query parameters", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ video_id: "V", frames: [], total: 0 }),
    );
    await getFrames("V", { center: 120, radius: 12, limit: 25 });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/api/videos/V/frames?");
    expect(url).toContain("center=120");
    expect(url).toContain("radius=12");
    expect(url).toContain("limit=25");
  });

  it("builds offset paging when no center is given", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ video_id: "V", frames: [], total: 0 }),
    );
    await getFrames("V", { offset: 40, limit: 20 });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("offset=40");
    expect(url).toContain("limit=20");
    expect(url).not.toContain("center=");
  });
});

describe("getFrameUrl", () => {
  it("constructs the image URL without fetching", () => {
    expect(getFrameUrl("L21_V001", 606)).toBe(`${API_BASE}/api/frames/L21_V001/606`);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
