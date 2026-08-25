import { describe, expect, it } from "vitest";
import {
  ANSWER_MAX_LENGTH,
  predictionIssue,
  predictionKey,
  queryStatus,
  suggestTraKeAnchors,
  validateQueries,
  type QueryWorkspace,
} from "./submission";

function query(
  id: string,
  task: QueryWorkspace["task"],
  predictions: QueryWorkspace["predictions"],
  eventCount: number | null = null,
): QueryWorkspace {
  return { id, task, draftQuery: "", lastSearch: null, eventCount, predictions };
}

describe("prediction keys", () => {
  it("identifies duplicates by video+frame (kis/qa) or full sequence (trake)", () => {
    expect(predictionKey("kis", { videoId: "V", frameIndex: 5 })).toBe("V::5");
    expect(predictionKey("qa", { videoId: "V", frameIndex: 5, answer: "" })).toBe("V::5");
    const seq = { videoId: "V", frameIndices: [1, null] };
    expect(predictionKey("trake", seq)).toBe("V::1,_");
    expect(predictionKey("trake", { videoId: "V", frameIndices: [null, 1] })).not.toBe(
      predictionKey("trake", seq),
    ); // position matters, not just the set of values
  });
});

describe("suggestTraKeAnchors", () => {
  it("fills one slot per requested event, positionally, without inventing frames", () => {
    const events = [
      { predictions: [{ video_id: "V", frame_index: 300 }, { video_id: "W", frame_index: 9 }] },
      { predictions: [{ video_id: "Z", frame_index: 7 }] }, // no match in V -> stays unset
      { predictions: [{ video_id: "V", frame_index: 100 }] },
    ];
    expect(suggestTraKeAnchors("V", events, 3)).toEqual([300, null, 100]);
  });

  it("never reorders or deduplicates slots and never falls back to window anchors", () => {
    const events = [
      { predictions: [{ video_id: "V", frame_index: 500 }] },
      { predictions: [{ video_id: "V", frame_index: 500 }] }, // duplicate is honest
      { predictions: [] },
    ];
    expect(suggestTraKeAnchors("V", events, 4)).toEqual([500, 500, null, null]);
  });
});

describe("predictionIssue", () => {
  it("kis rows are valid by construction", () => {
    expect(predictionIssue("kis", { videoId: "V", frameIndex: 1 }, null)).toBeNull();
  });

  it("qa rows need a non-empty answer of at most 100 characters", () => {
    expect(predictionIssue("qa", { videoId: "V", frameIndex: 1, answer: "" }, null)).toBe(
      "missing answer",
    );
    expect(
      predictionIssue(
        "qa",
        { videoId: "V", frameIndex: 1, answer: "x".repeat(ANSWER_MAX_LENGTH) },
        null,
      ),
    ).toBeNull();
    expect(
      predictionIssue(
        "qa",
        { videoId: "V", frameIndex: 1, answer: "x".repeat(ANSWER_MAX_LENGTH + 1) },
        null,
      ),
    ).toMatch(/longer than 100/);
  });

  it("trake rows need exactly N chosen anchors in temporal order", () => {
    expect(
      predictionIssue("trake", { videoId: "V", frameIndices: [1, 2] }, null),
    ).toMatch(/unknown/);
    // slot-count mismatch (structural)
    expect(
      predictionIssue("trake", { videoId: "V", frameIndices: [1, 2, 3] }, 4),
    ).toMatch(/3\/4/);
    // unset slots
    expect(
      predictionIssue("trake", { videoId: "V", frameIndices: [1200] }, 2),
    ).toMatch(/1\/2 event frames/);
    expect(
      predictionIssue("trake", { videoId: "V", frameIndices: [1200, null] }, 2),
    ).toMatch(/1\/2 event frames/);
    // temporal order over chosen values
    expect(
      predictionIssue("trake", { videoId: "V", frameIndices: [500, 300] }, 2),
    ).toMatch(/temporal order/);
    expect(predictionIssue("trake", { videoId: "V", frameIndices: [300, 500] }, 2)).toBeNull();
  });
});

describe("queryStatus", () => {
  it("follows the EMPTY / IN_PROGRESS / READY model", () => {
    expect(queryStatus("kis", [], null)).toBe("empty");
    // BTC sets a maximum, not a minimum: 12 valid KIS rows are ready.
    expect(queryStatus("kis", [{ videoId: "V", frameIndex: 1 }], null)).toBe("ready");
    expect(
      queryStatus("qa", [{ videoId: "V", frameIndex: 1, answer: "" }], null),
    ).toBe("in_progress");
    expect(queryStatus("trake", [{ videoId: "V", frameIndices: [1] }], 2)).toBe("in_progress");
    expect(queryStatus("trake", [{ videoId: "V", frameIndices: [null, 5] }], 2)).toBe("in_progress");
    expect(queryStatus("trake", [{ videoId: "V", frameIndices: [1, 2] }], 2)).toBe("ready");
  });
});

describe("validateQueries", () => {
  it("aggregates per-row issues with links and counts ready queries", () => {
    const result = validateQueries([
      query("Q01", "kis", [
        { videoId: "V1", frameIndex: 1 },
        { videoId: "V2", frameIndex: 2 },
      ]),
      query("Q02", "qa", [
        { videoId: "V3", frameIndex: 3, answer: "ok" },
        { videoId: "V4", frameIndex: 4, answer: "" },
      ]),
      query("Q03", "trake", [{ videoId: "V5", frameIndices: [1, null, null, null] }], 4),
      query("Q04", "kis", []),
    ]);

    expect(result.startedByTask).toMatchObject({ kis: 2, qa: 1, trake: 1 });
    expect(result.readyByTask).toMatchObject({ kis: 1, qa: 0, trake: 0 });
    expect(result.blocked).toBe(false);

    const q02Issues = result.issues.filter((i) => i.queryId === "Q02");
    expect(q02Issues[0]).toMatchObject({
      predictionIndex: 1,
      message: "prediction #2: missing answer",
      severity: "warning",
    });
    expect(result.issues.some((i) => i.message === "no predictions")).toBe(true);
    expect(result.issues.some((i) => i.message.includes("1/4"))).toBe(true);
  });
});
