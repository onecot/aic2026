import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrakeResults } from "./TrakeResults";
import type { TraKeEvent } from "../../api/types";
import { initialUiState, useUiStore } from "../../state/uiStore";

const events: TraKeEvent[] = [
  {
    event_id: "E1",
    event_text: "a man enters the room",
    predictions: [{ rank: 1, video_id: "L10_V001", frame_index: 1200, keyframe_ordinal: 5, pts_time: 48, score: 0.9 }],
    windows: [{ video_id: "L10_V001", frame_start: 1200, frame_end: 1300, score: 0.9, rank: 1 }],
  },
  {
    event_id: "E2",
    event_text: "lights on",
    predictions: [{ rank: 1, video_id: "L10_V001", frame_index: 2450, keyframe_ordinal: 9, pts_time: 98, score: 0.8 }],
    windows: [{ video_id: "L10_V001", frame_start: 2400, frame_end: 2450, score: 0.8, rank: 1 }],
  },
];

function seedTrake() {
  useUiStore.setState({ ...initialUiState() });
  const id = useUiStore.getState().newQuery("trake");
  useUiStore.getState().setEventCount(2);
  return id;
}

describe("TrakeResults", () => {
  it("adds one window as a single sequence prediction with N anchors", async () => {
    const user = userEvent.setup();
    const queryId = seedTrake();
    render(<TrakeResults events={events} />);

    const firstEvent = within(screen.getByRole("region", { name: "a man enters the room" }));
    await user.click(firstEvent.getByRole("button", { name: "Add prediction" }));
    const query = useUiStore.getState().queries[queryId]!;
    expect(query.predictions).toHaveLength(1);
    expect(query.predictions[0]).toEqual({
      videoId: "L10_V001",
      // one anchor per requested event, temporal order preserved
      frameIndices: [1200, 2450],
    });

    // identical assembled sequences are treated as duplicates -> "Added"
    expect(firstEvent.getByRole("button", { name: "Added ✓" })).toBeDisabled();
    expect(useUiStore.getState().queries[queryId]!.predictions).toHaveLength(1);
  });

  it("inspecting the window card targets its representative anchor", async () => {
    const user = userEvent.setup();
    seedTrake();
    render(<TrakeResults events={events} />);
    await user.click(
      within(screen.getByRole("region", { name: "a man enters the room" })).getByTitle(
        /Inspect window L10_V001/,
      ),
    );
    expect(useUiStore.getState().selected).toMatchObject({
      videoId: "L10_V001",
      frameIndex: 1200,
    });
  });
});
