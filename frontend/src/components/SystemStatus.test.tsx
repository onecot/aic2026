import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SystemStatus } from "./SystemStatus";
import type { SystemPayload } from "../api/types";

const payload: SystemPayload = {
  status: "ready",
  retrieval: {
    model: { name: "jina-clip-v2", backend: "jina-clip" },
    index: {
      id: "jina-v2-full",
      total_keyframes: 148322,
      total_videos: 873,
      embedding_dimension: 1024,
      similarity: "cosine",
    },
    keyframes: { version: "adaptive_sbd_v1", total_keyframes: 148322, total_videos: 873 },
  },
  features: {
    visual_embedding: true,
    ocr: false,
    asr: false,
    metadata_search: false,
    temporal_navigation: true,
  },
  generation: { vlm: { enabled: false, model: "jinaai/jina-vlm-mlx", load_policy: "lazy" } },
};

describe("SystemStatus", () => {
  it("renders the always-visible values from the API payload", () => {
    render(<SystemStatus system={payload} />);
    expect(screen.getByText("READY")).toBeInTheDocument();
    expect(screen.getByText("jina-clip-v2")).toBeInTheDocument();
    expect(screen.getByText(/148K frames/)).toBeInTheDocument();
    expect(screen.getByText(/873 videos/)).toBeInTheDocument();
    expect(screen.getByText(/adaptive_sbd_v1/)).toBeInTheDocument();
  });

  it("shows the details panel with retrieval/keyframe/feature/generation sections", async () => {
    const user = userEvent.setup();
    render(<SystemStatus system={payload} />);
    await user.click(screen.getByRole("button", { name: "System" }));
    const dialog = screen.getByRole("dialog", { name: "System details" });
    expect(dialog).toHaveTextContent("jina-v2-full");
    expect(dialog).toHaveTextContent("1024");
    expect(dialog).toHaveTextContent("cosine");
    expect(dialog).toHaveTextContent("adaptive_sbd_v1");
    expect(dialog).toHaveTextContent("jinaai/jina-vlm-mlx");
    // feature flags rendered from the payload (ocr off, visual on)
    expect(dialog).toHaveTextContent(/ocr\s*off/);
    expect(dialog).toHaveTextContent(/visual embedding\s*on/);
  });

  it("does not invent a version when the API returns none", () => {
    const noVersion = {
      ...payload,
      retrieval: {
        ...payload.retrieval,
        keyframes: { version: null, total_keyframes: 10, total_videos: 1 },
      },
    };
    render(<SystemStatus system={noVersion} />);
    expect(screen.getByText(/unknown keyframe set/)).toBeInTheDocument();
  });
});
