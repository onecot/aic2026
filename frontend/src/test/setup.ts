import "@testing-library/jest-dom/vitest";

import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Reset the zustand store between tests so ordering never matters.
import { initialUiState, useUiStore } from "../state/uiStore";

afterEach(() => {
  cleanup();
  useUiStore.setState({ ...initialUiState() });
});
