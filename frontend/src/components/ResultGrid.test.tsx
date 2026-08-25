import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ResultGrid } from "./ResultGrid";
import type { SearchHit } from "../api/types";
import { useUiStore } from "../state/uiStore";

const hits: SearchHit[] = [11, 22, 33].map((i, idx) => ({
  rank: idx + 1,
  video_id: `L21_V00${idx + 1}`,
  frame_index: i,
  keyframe_ordinal: idx + 1,
  pts_time: i * 0.04,
  score: 0.5 - idx * 0.1,
}));

describe("ResultGrid", () => {
  it("renders every hit with its server-assigned rank, in order", () => {
    render(<ResultGrid hits={hits} />);
    const ranks = screen.getAllByRole("button").map((b) => b.textContent ?? "");
    expect(ranks[0]).toContain("#1");
    expect(screen.getByAltText(/rank 3/)).toHaveAttribute(
      "src",
      "/api/frames/L21_V003/33",
    );
  });

  it("marks the clicked result as selected in the shared store", async () => {
    const user = userEvent.setup();
    render(<ResultGrid hits={hits} />);
    await user.click(screen.getByTitle(/L21_V002 frame 22/));
    const selected = useUiStore.getState().selected;
    expect(selected).toMatchObject({ videoId: "L21_V002", frameIndex: 22, rank: 2 });
    expect(screen.getByTitle(/L21_V002 frame 22/)).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the empty hint when there are no hits", () => {
    render(<ResultGrid hits={[]} emptyHint="nothing here" />);
    expect(screen.getByText("nothing here")).toBeInTheDocument();
  });
});
