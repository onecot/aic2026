import { describe, expect, it } from "vitest";
import type { FrameRecord } from "../api/types";
import { MAX_PREDICTIONS, type TrakePrediction } from "./submission";
import { initialUiState, useUiStore } from "./uiStore";

function frame(index: number): FrameRecord {
  return { keyframe_ordinal: index + 1, frame_index: index * 10, pts_time: index, fps: 25 };
}

const hit = {
  videoId: "L21_V001",
  frameIndex: 606,
  rank: 1,
  score: 0.9,
  keyframeOrdinal: 12,
  ptsTime: 24.2,
};

function freshSession() {
  useUiStore.setState({ ...initialUiState() });
  useUiStore.getState().ensureDefaultQuery();
}

describe("query workspaces", () => {
  it("ensureDefaultQuery creates Q01 as the active KIS query", () => {
    useUiStore.setState({ ...initialUiState() });
    expect(useUiStore.getState().activeQueryId).toBeNull();
    useUiStore.getState().ensureDefaultQuery();
    expect(useUiStore.getState().activeQueryId).toBe("Q01");
    expect(useUiStore.getState().queries["Q01"]?.task).toBe("kis");
  });

  it("newQuery allocates sequential ids and activates the new query", () => {
    freshSession();
    const id = useUiStore.getState().newQuery("trake");
    expect(id).toBe("Q02");
    expect(useUiStore.getState().activeQueryId).toBe("Q02");
    expect(useUiStore.getState().queries["Q02"]?.task).toBe("trake");
  });

  it("switching queries restores that query's draft, search, and queue", () => {
    freshSession();
    const s = useUiStore.getState();
    s.setDraftQuery("red boat");
    s.submitSearch();
    s.selectFrame(hit);
    s.addPredictionFromSelected();

    const q2 = s.newQuery("qa");
    expect(useUiStore.getState().selected).toBeNull();
    expect(useUiStore.getState().queries[q2]?.predictions).toHaveLength(0);

    useUiStore.getState().setActiveQuery("Q01");
    const restored = useUiStore.getState().queries["Q01"];
    expect(restored?.draftQuery).toBe("red boat");
    expect(restored?.lastSearch).toEqual({ query: "red boat", limit: 100 });
    expect(restored?.predictions).toHaveLength(1);
  });

  it("setTask reuses the most recent query of that task or creates one", () => {
    freshSession();
    useUiStore.getState().newQuery("qa"); // Q02 active
    useUiStore.getState().setTask("kis"); // back to Q01
    expect(useUiStore.getState().activeQueryId).toBe("Q01");
    useUiStore.getState().setTask("trake"); // none exists -> created
    const active = useUiStore.getState().activeQueryId!;
    expect(useUiStore.getState().queries[active]?.task).toBe("trake");
  });
});

describe("search submission", () => {
  it("stores trimmed query per active query; ignores blank input", () => {
    freshSession();
    useUiStore.getState().setDraftQuery("  a red boat  ");
    useUiStore.getState().submitSearch();
    expect(useUiStore.getState().queries["Q01"]?.lastSearch).toEqual({
      query: "a red boat",
      limit: 100,
    });

    useUiStore.getState().setDraftQuery("   ");
    useUiStore.getState().submitSearch();
    expect(useUiStore.getState().queries["Q01"]?.lastSearch?.query).toBe("a red boat");
  });
});

describe("selection + temporal navigation", () => {
  it("selectFrame stores the inspected frame; clearInspector resets", () => {
    freshSession();
    useUiStore.getState().selectFrame(hit);
    expect(useUiStore.getState().selected).toEqual(hit);
    useUiStore.getState().clearInspector();
    expect(useUiStore.getState().selected).toBeNull();
  });

  it("stepTemporal moves within loaded neighbors and clamps at edges", () => {
    freshSession();
    const s = useUiStore.getState();
    s.setTemporalFrames([frame(0), frame(1), frame(2), frame(3)]);
    s.selectFrame({ videoId: "V", frameIndex: 10 });
    s.stepTemporal(1);
    expect(useUiStore.getState().selected?.frameIndex).toBe(20);
    s.stepTemporal(1);
    s.stepTemporal(1);
    s.stepTemporal(1); // clamped at end
    expect(useUiStore.getState().selected?.frameIndex).toBe(30);
    s.stepTemporal(-1);
    expect(useUiStore.getState().selected?.frameIndex).toBe(20);
  });
});

describe("prediction queue", () => {
  it("adds predictions in user order from the inspected frame", () => {
    freshSession();
    const s = useUiStore.getState();
    s.selectFrame(hit);
    expect(s.addPredictionFromSelected()).toBe(true);
    s.selectFrame({ ...hit, videoId: "L22_V009", frameIndex: 42 });
    s.addPredictionFromSelected();

    const predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions.map((p) => ("frameIndex" in p ? p.frameIndex : -1))).toEqual([606, 42]);
  });

  it("rejects duplicate predictions without changing order", () => {
    freshSession();
    const s = useUiStore.getState();
    s.selectFrame(hit);
    s.addPredictionFromSelected();
    s.selectFrame({ ...hit, videoId: "L22_V009", frameIndex: 42 });
    s.addPredictionFromSelected();

    s.selectFrame(hit); // duplicate
    expect(s.addPredictionFromSelected()).toBe(false);
    const predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions).toHaveLength(2);
    expect(predictions[0]).toMatchObject({ videoId: "L21_V001" });
    expect(useUiStore.getState().notice?.message).toMatch(/Already added/);
  });

  it("blocks the 101st prediction with a clear notice, never truncating", () => {
    freshSession();
    const s = useUiStore.getState();
    for (let i = 0; i < MAX_PREDICTIONS; i += 1) {
      s.selectFrame({ videoId: `V${i}`, frameIndex: i });
      expect(s.addPredictionFromSelected()).toBe(true);
    }
    s.selectFrame({ videoId: "OVERFLOW", frameIndex: 999 });
    expect(s.addPredictionFromSelected()).toBe(false);
    const predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions).toHaveLength(MAX_PREDICTIONS);
    expect(useUiStore.getState().notice?.message).toMatch(/max 100/);
  });

  it("removes by index and moves rows up/down", () => {
    freshSession();
    const s = useUiStore.getState();
    [1, 2, 3].forEach((n) => {
      s.selectFrame({ videoId: `V${n}`, frameIndex: n });
      s.addPredictionFromSelected();
    });
    s.movePrediction(2, -1); // V3 moves up
    let predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions.map((p) => ("videoId" in p ? p.videoId : ""))).toEqual(["V1", "V3", "V2"]);
    s.movePrediction(0, -1); // no-op at top
    predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions[0]).toMatchObject({ videoId: "V1" });

    useUiStore.getState().removePrediction(1);
    predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions.map((p) => ("videoId" in p ? p.videoId : ""))).toEqual(["V1", "V2"]);
  });

  it("adds the TRAKE window as an editable initial suggestion only", () => {
    freshSession();
    useUiStore.getState().newQuery("trake"); // Q02
    const s = useUiStore.getState();

    // N unknown until the search has been run -> refuse to guess
    expect(
      s.addTraKeSuggestion(
        { video_id: "L10_V001", frame_start: 0, frame_end: 10, score: 1, rank: 1 },
        [{ predictions: [] }],
      ),
    ).toBe(false);
    expect(useUiStore.getState().notice?.message).toMatch(/event count N is unknown/);

    s.setEventCount(3);
    const events = [
      { predictions: [{ video_id: "L10_V001", frame_index: 1850 }] },
      { predictions: [{ video_id: "L99_V000", frame_index: 5 }] }, // other video -> slot stays unset
      { predictions: [{ video_id: "L10_V001", frame_index: 1200 }] },
    ];
    expect(
      s.addTraKeSuggestion(
        { video_id: "L10_V001", frame_start: 1200, frame_end: 1850, score: 0.9, rank: 1 },
        events,
      ),
    ).toBe(true);
    const prediction = useUiStore.getState().queries["Q02"]!
      .predictions[0] as { videoId: string; frameIndices: (number | null)[] };
    // positional per event, NOT sorted, NOT completed by invention:
    expect(prediction.frameIndices).toEqual([1850, null, 1200]);

    // identical suggestion is a duplicate
    expect(
      useUiStore.getState().addTraKeSuggestion(
        { video_id: "L10_V001", frame_start: 1200, frame_end: 1850, score: 0.9, rank: 1 },
        events,
      ),
    ).toBe(false);

    // all-empty suggestion invents nothing and adds nothing
    expect(
      useUiStore.getState().addTraKeSuggestion(
        { video_id: "L77_V000", frame_start: 1, frame_end: 2, score: 0.5, rank: 2 },
        events,
      ),
    ).toBe(false);
    expect(useUiStore.getState().notice?.message).toMatch(/nothing invented/);
  });

  it("replaces individual event anchors without touching other events or order", () => {
    freshSession();
    const s = useUiStore.getState();
    s.newQuery("trake"); // Q02 active
    useUiStore.setState({
      queries: {
        ...useUiStore.getState().queries,
        Q02: {
          ...useUiStore.getState().queries["Q02"]!,
          eventCount: 2,
          predictions: [{ videoId: "V", frameIndices: [100, null] }],
        },
      },
    });

    // fill event 2 via pick flow
    s.chooseTraKeAnchor(0, 1);
    expect(useUiStore.getState().trakePick).toEqual({ predictionIndex: 0, eventIndex: 1 });
    s.setTraKeAnchor("Q02", 0, 1, 250);
    let prediction = useUiStore.getState().queries["Q02"]!.predictions[0] as TrakePrediction;
    expect(prediction.frameIndices).toEqual([100, 250]);
    expect(useUiStore.getState().trakePick).toBeNull(); // picking finished

    // replace event 2 again; event 1 untouched
    s.setTraKeAnchor("Q02", 0, 1, 400);
    prediction = useUiStore
      .getState()
      .queries["Q02"]!.predictions[0] as TrakePrediction;
    expect(prediction.frameIndices).toEqual([100, 400]);

    // reset event 1; event 2 untouched
    s.setTraKeAnchor("Q02", 0, 0, null);
    prediction = useUiStore
      .getState()
      .queries["Q02"]!.predictions[0] as TrakePrediction;
    expect(prediction.frameIndices).toEqual([null, 400]);

    // out-of-bounds writes are no-ops
    s.setTraKeAnchor("Q02", 0, 5, 999);
    prediction = useUiStore
      .getState()
      .queries["Q02"]!.predictions[0] as TrakePrediction;
    expect(prediction.frameIndices).toEqual([null, 400]);
  });
});

describe("QA answers are per-prediction", () => {
  it("editing one answer leaves other predictions untouched", () => {
    freshSession();
    const s = useUiStore.getState();
    s.newQuery("qa"); // Q02 active
    s.selectFrame({ ...hit, videoId: "A" });
    s.addPredictionFromSelected();
    s.selectFrame({ ...hit, videoId: "B" });
    s.addPredictionFromSelected();

    useUiStore.getState().setQaAnswer("Q02", 1, "Bát Tràng");
    const predictions = useUiStore.getState().queries["Q02"]!.predictions;
    expect(predictions[0]).not.toHaveProperty("answer", "Bát Tràng");
    expect(predictions[1]).toMatchObject({ answer: "Bát Tràng" });
  });

  it("captures the requested TRAKE event count per query", () => {
    freshSession();
    useUiStore.getState().newQuery("trake"); // Q02
    useUiStore.getState().setEventCount(4);
    expect(useUiStore.getState().queries["Q02"]?.eventCount).toBe(4);
    useUiStore.getState().setEventCount(4); // idempotent
    expect(useUiStore.getState().queries["Q02"]?.eventCount).toBe(4);
  });
});
