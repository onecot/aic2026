import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError } from "./api/client";
import { loadSession } from "./state/persistence";
import { useUiStore } from "./state/uiStore";
import App from "./App";
import "./index.css";

// Restore the persisted prediction workspace (if any) before first render.
const session = loadSession();
if (session !== null) {
  useUiStore.setState({
    queries: Object.fromEntries(session.queries.map((q) => [q.id, q])),
    queryOrder: session.queries.map((q) => q.id),
    activeQueryId: session.activeQueryId,
    savedAt: Date.parse(session.savedAt) || null,
  });
}
useUiStore.getState().ensureDefaultQuery();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // HTTP errors are meaningful states (404/503/…); never blind-retry them.
      retry: (failureCount, error) =>
        error instanceof ApiError ? false : failureCount < 2,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
