import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictionQueue } from "./PredictionQueue";
import type { QueryWorkspace } from "../../state/submission";
import { initialUiState, useUiStore } from "../../state/uiStore";

function seed(queries: QueryWorkspace[], activeId: string) {
  useUiStore.setState({
    ...initialUiState(),
    queries: Object.fromEntries(queries.map((q) => [q.id, q])),
    queryOrder: queries.map((q) => q.id),
    activeQueryId: activeId,
  });
}

const kis: QueryWorkspace = {
  id: "Q01",
  task: "kis",
  draftQuery: "",
  lastSearch: null,
  eventCount: null,
  predictions: [
    { videoId: "L29_V005", frameIndex: 23221 },
    { videoId: "L21_V018", frameIndex: 8113 },
  ],
};

describe("PredictionQueue", () => {
  it("renders rows in user order with the counter and saved stamp", () => {
    seed([kis], "Q01");
    useUiStore.setState({ savedAt: Date.now() });
    render(<PredictionQueue />);
    expect(screen.getByText(/Q01 · KIS/)).toBeInTheDocument();
    expect(screen.getByText("2 / 100 predictions")).toBeInTheDocument();
    expect(screen.getByText("L29_V005 / 23221")).toBeInTheDocument();
    expect(screen.getByText(/last saved/)).toBeInTheDocument();
  });

  it("removes a prediction via its row button", async () => {
    const user = userEvent.setup();
    seed([kis], "Q01");
    render(<PredictionQueue />);
    await user.click(screen.getByRole("button", { name: "Remove prediction 1" }));
    expect(useUiStore.getState().queries["Q01"]?.predictions).toHaveLength(1);
    expect(useUiStore.getState().queries["Q01"]?.predictions[0]).toMatchObject({
      videoId: "L21_V018",
    });
  });

  it("edits QA answers per row and shows overflow without truncating", async () => {
    const user = userEvent.setup();
    seed(
      [
        {
          ...kis,
          id: "Q02",
          task: "qa",
          predictions: [
            { videoId: "V", frameIndex: 1, answer: "" },
            { videoId: "W", frameIndex: 2, answer: "" },
          ],
        },
      ],
      "Q02",
    );
    render(<PredictionQueue />);
    await user.type(screen.getByLabelText("Answer for prediction 1"), "Bát Tràng");
    const predictions = useUiStore.getState().queries["Q02"]!.predictions;
    expect(predictions[0]).toMatchObject({ answer: "Bát Tràng" });
    expect(predictions[1]).toMatchObject({ answer: "" }); // independent

    useUiStore.setState({
      queries: {
        Q02: {
          ...useUiStore.getState().queries["Q02"]!,
          predictions: [{ videoId: "V", frameIndex: 1, answer: "x".repeat(101) }],
        },
      },
    });
    expect(await screen.findByText(/101\/100/)).toBeInTheDocument();
    expect(useUiStore.getState().queries["Q02"]!.predictions[0]).toMatchObject({
      answer: "x".repeat(101), // never silently truncated
    });
  });

  it("reopens a queue row in the inspector on click", async () => {
    const user = userEvent.setup();
    seed([kis], "Q01");
    render(<PredictionQueue />);
    await user.click(screen.getByText("L21_V018 / 8113"));
    expect(useUiStore.getState().selected).toMatchObject({
      videoId: "L21_V018",
      frameIndex: 8113,
    });
  });

  it("shows an empty-state hint and flags incomplete TRAKE rows", () => {
    seed(
      [
        {
          ...kis,
          id: "Q03",
          task: "trake",
          eventCount: 4,
          predictions: [{ videoId: "L10_V001", frameIndices: [1200, null, null, null] }],
        },
      ],
      "Q03",
    );
    render(<PredictionQueue />);
    expect(screen.getByText(/1\/4 event frames/)).toBeInTheDocument();
    expect(screen.getByText(/Event 1/)).toBeInTheDocument();
    expect(screen.getByText("1200")).toBeInTheDocument();
    expect(screen.getAllByText(/unset/)).toHaveLength(3);
  });

  it("supports per-event anchor editing: choose, replace, reset, inspect", async () => {
    const user = userEvent.setup();
    seed(
      [
        {
          ...kis,
          id: "Q04",
          task: "trake",
          eventCount: 2,
          predictions: [{ videoId: "L10_V001", frameIndices: [1200, 2450] }],
        },
      ],
      "Q04",
    );
    render(<PredictionQueue />);

    // Choose enters picking mode for that slot
    await user.click(screen.getByRole("button", { name: "Choose frame for L10_V001 event 1" }));
    let state = useUiStore.getState();
    expect(state.trakePick).toEqual({ predictionIndex: 0, eventIndex: 0 });
    expect(screen.getByText(/Set as Event 1/)).toBeInTheDocument();

    // Inspect opens the current anchor in the shared inspector target
    await user.click(screen.getByRole("button", { name: "1200" }));
    expect(useUiStore.getState().selected).toMatchObject({
      videoId: "L10_V001",
      frameIndex: 1200,
    });

    // Reset clears exactly one slot; the other event keeps its value
    await user.click(screen.getByRole("button", { name: "Reset L10_V001 event 2" }));
    const prediction = useUiStore.getState().queries["Q04"]!.predictions[0];
    expect(prediction).toEqual({ videoId: "L10_V001", frameIndices: [1200, null] });
    expect(useUiStore.getState().trakePick).toBeNull();

    // set slots offer "Change"; unset slots offer "Choose"
    expect(
      screen.getByRole("button", { name: "Choose frame for L10_V001 event 1" }),
    ).toHaveTextContent("Change");
    expect(
      screen.getByRole("button", { name: "Choose frame for L10_V001 event 2" }),
    ).toHaveTextContent("Choose");
    // and the remaining value remains inspectable
    expect(screen.getByRole("button", { name: "1200" })).toBeInTheDocument();
  });
});
