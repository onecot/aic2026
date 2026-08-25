import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ResultGrid } from "./ResultGrid";
import type { QueryWorkspace } from "../state/submission";
import { initialUiState, useUiStore } from "../state/uiStore";

function seed(query: QueryWorkspace) {
  useUiStore.setState({
    ...initialUiState(),
    queries: { [query.id]: query },
    queryOrder: [query.id],
    activeQueryId: query.id,
  });
}

const kis: QueryWorkspace = {
  id: "Q01",
  task: "kis",
  draftQuery: "",
  lastSearch: null,
  eventCount: null,
  predictions: [
    { videoId: "L21_V001", frameIndex: 11, score: 0.5 },
    { videoId: "L21_V002", frameIndex: 22, score: 0.4 },
    { videoId: "L21_V003", frameIndex: 33, score: 0.3 },
  ],
};

/** Minimal stub good enough for jsdom's incomplete native DataTransfer. */
function stubDataTransfer() {
  const store = new Map<string, string>();
  return {
    setData: (fmt: string, value: string) => store.set(fmt, value),
    getData: (fmt: string) => store.get(fmt) ?? "",
  } as unknown as DataTransfer;
}

describe("ResultGrid", () => {
  it("renders every prediction with its live, position-derived rank, in order", () => {
    seed(kis);
    render(<ResultGrid />);
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
    expect(screen.getByText("#3")).toBeInTheDocument();
    expect(screen.getByAltText(/rank 3/)).toHaveAttribute("src", "/api/frames/L21_V003/33");
  });

  it("marks the clicked result as selected in the shared store", async () => {
    const user = userEvent.setup();
    seed(kis);
    render(<ResultGrid />);
    await user.click(screen.getByTitle(/L21_V002 frame 22/));
    const selected = useUiStore.getState().selected;
    expect(selected).toMatchObject({ videoId: "L21_V002", frameIndex: 22 });
    expect(screen.getByTitle(/L21_V002 frame 22/)).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the empty hint when there are no predictions", () => {
    seed({ ...kis, predictions: [] });
    render(<ResultGrid emptyHint="nothing here" />);
    expect(screen.getByText("nothing here")).toBeInTheDocument();
  });

  it("removes a prediction via its card button", async () => {
    const user = userEvent.setup();
    seed(kis);
    render(<ResultGrid />);
    await user.click(screen.getByRole("button", { name: "Remove prediction 1" }));
    const predictions = useUiStore.getState().queries["Q01"]?.predictions;
    expect(predictions).toHaveLength(2);
    expect(predictions?.[0]).toMatchObject({ videoId: "L21_V002" });
  });

  it("edits QA answers per card, independently, without truncating overflow", async () => {
    const user = userEvent.setup();
    seed({
      ...kis,
      id: "Q02",
      task: "qa",
      predictions: [
        { videoId: "V", frameIndex: 1, answer: "" },
        { videoId: "W", frameIndex: 2, answer: "" },
      ],
    });
    render(<ResultGrid />);
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
    expect(await screen.findByText("101/100")).toBeInTheDocument();
    expect(useUiStore.getState().queries["Q02"]!.predictions[0]).toMatchObject({
      answer: "x".repeat(101), // never silently truncated
    });
  });

  it("moves a card to an exact position via its rank input", async () => {
    const user = userEvent.setup();
    seed(kis);
    render(<ResultGrid />);
    const rankInput = screen.getByLabelText("Rank for prediction 1"); // currently L21_V001
    await user.clear(rankInput);
    await user.type(rankInput, "3");
    fireEvent.blur(rankInput);
    const predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions.map((p) => ("videoId" in p ? p.videoId : ""))).toEqual([
      "L21_V002",
      "L21_V003",
      "L21_V001",
    ]);
  });

  it("reorders via drag-and-drop", () => {
    seed(kis);
    render(<ResultGrid />);
    // One inspect-button per card (has the "· rank N" title) — closest
    // draggable ancestor is that card's own container, in DOM/array order.
    const cards = screen
      .getAllByTitle(/· rank \d/)
      .map((b) => b.closest("[draggable]")) as HTMLElement[];
    const dataTransfer = stubDataTransfer();
    fireEvent.dragStart(cards[0]!, { dataTransfer });
    fireEvent.dragOver(cards[2]!, { dataTransfer });
    fireEvent.drop(cards[2]!, { dataTransfer });
    const predictions = useUiStore.getState().queries["Q01"]!.predictions;
    expect(predictions.map((p) => ("videoId" in p ? p.videoId : ""))).toEqual([
      "L21_V002",
      "L21_V003",
      "L21_V001",
    ]);
  });
});
