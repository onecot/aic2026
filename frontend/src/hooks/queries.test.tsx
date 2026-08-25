import { describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { useSearch } from "./queries";
import { initialUiState, useUiStore } from "../state/uiStore";

function Probe() {
  const { pending } = useSearch(true);
  return <span data-testid="pending">{String(pending)}</span>;
}

function renderProbe() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  );
}

describe("useSearch pending state", () => {
  // Regression: `query.isPending` is true for a disabled, never-fetched
  // query in TanStack Query v5, which kept the search button stuck on
  // "Searching…" before the first submit.
  it("is false before any search has been submitted (disabled query)", () => {
    useUiStore.setState({ ...initialUiState() });
    renderProbe();
    expect(screen.getByTestId("pending").textContent).toBe("false");
  });

  it("is true only while the request is in flight, then resets on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ query: "boat", results: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    try {
      useUiStore.setState({ ...initialUiState() });
      useUiStore.getState().ensureDefaultQuery();
      renderProbe();
      expect(screen.getByTestId("pending").textContent).toBe("false");

      useUiStore.getState().setDraftQuery("boat");
      useUiStore.getState().submitSearch();

      await waitFor(() =>
        expect(screen.getByTestId("pending").textContent).toBe("true"),
      );
      await waitFor(() =>
        expect(screen.getByTestId("pending").textContent).toBe("false"),
      );
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect((fetchMock.mock.calls[0] as unknown[])[0]).toBe("/api/search");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("resets to false after an error as well", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response("{}", { status: 503 }));
    vi.stubGlobal("fetch", fetchMock);
    try {
      useUiStore.setState({ ...initialUiState() });
      useUiStore.getState().ensureDefaultQuery();
      renderProbe();
      useUiStore.getState().setDraftQuery("boat");
      useUiStore.getState().submitSearch();

      await waitFor(() => expect(fetchMock).toHaveBeenCalled());
      await waitFor(() =>
        expect(screen.getByTestId("pending").textContent).toBe("false"),
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
