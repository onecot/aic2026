import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SubmissionView } from "./SubmissionView";
import type { QueryWorkspace } from "../../state/submission";
import { initialUiState, useUiStore } from "../../state/uiStore";

function seed(queries: QueryWorkspace[]) {
  useUiStore.setState({
    ...initialUiState(),
    queries: Object.fromEntries(queries.map((q) => [q.id, q])),
    queryOrder: queries.map((q) => [q.id]).flat(),
    activeQueryId: queries[0]?.id ?? null,
    view: "submission",
  });
}

const queries: QueryWorkspace[] = [
  {
    id: "Q01", task: "kis", draftQuery: "", lastSearch: null, eventCount: null,
    predictions: [{ videoId: "V1", frameIndex: 1 }],
  },
  {
    id: "Q02", task: "qa", draftQuery: "", lastSearch: null, eventCount: null,
    predictions: [{ videoId: "V2", frameIndex: 2, answer: "" }], // missing answer
  },
  {
    id: "Q03", task: "trake", draftQuery: "", lastSearch: null, eventCount: null,
    predictions: [], // empty
  },
];

describe("SubmissionView", () => {
  it("summarizes per-task readiness and lists every query with status", async () => {
    seed(queries);
    render(<SubmissionView />);
    expect(screen.getByText(/1 \/ 1 queries ready/)).toBeInTheDocument(); // KIS
    expect(screen.getAllByText(/0 \/ 1 queries ready/)).toHaveLength(2); // QA + TRAKE
    expect(screen.getByText("Q01")).toBeInTheDocument();
    expect(screen.getByText("in_progress")).toBeInTheDocument(); // Q02
    expect(screen.getByText("empty")).toBeInTheDocument(); // Q03
  });

  it("validates and links issues directly to the affected prediction", async () => {
    const user = userEvent.setup();
    seed(queries);
    render(<SubmissionView />);

    await user.click(screen.getByRole("button", { name: "Validate submission" }));
    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent("STATUS: VALID (not exported)");
    expect(status).toHaveTextContent("Q02 prediction #1: missing answer");
    expect(status).toHaveTextContent("Q03 no predictions");

    // Clicking the issue jumps to that exact query + prediction.
    await user.click(screen.getByText("⚠ Q02 prediction #1: missing answer"));
    const state = useUiStore.getState();
    expect(state.activeQueryId).toBe("Q02");
    expect(state.view).toBe("workspace");
    expect(state.selected).toMatchObject({ videoId: "V2", frameIndex: 2 });
  });

  it("Review incomplete jumps to the first query needing attention", async () => {
    const user = userEvent.setup();
    seed(queries);
    useUiStore.setState({ activeQueryId: "Q01", view: "submission" });
    render(<SubmissionView />);
    await user.click(screen.getByRole("button", { name: "Review incomplete" }));
    // Q01 is ready; Q02 has an incomplete row -> target
    expect(useUiStore.getState().activeQueryId).toBe("Q02");
    expect(useUiStore.getState().view).toBe("workspace");
  });

  it("opening a query row returns to its workspace without touching selections", async () => {
    const user = userEvent.setup();
    seed(queries);
    render(<SubmissionView />);
    await user.click(screen.getByTitle("Open Q03"));
    expect(useUiStore.getState().activeQueryId).toBe("Q03");
    expect(useUiStore.getState().view).toBe("workspace");
  });
});
