import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type { FrameRecord, SearchHit } from "../api/types";
import type { TraKeWindow } from "../api/types";
import {
  MAX_PREDICTIONS,
  predictionKey,
  suggestTraKeAnchors,
  type KisPrediction,
  type Prediction,
  type QueryWorkspace,
  type Task,
  type TrakePrediction,
} from "./submission";

/** Backend task sent to POST /api/search for a UI task. */
export function backendTask(task: Task): "kis" | "trake" {
  return task === "trake" ? "trake" : "kis";
}

/** The frame currently under inspection (drives inspector + temporal strip). */
export interface SelectedFrame {
  videoId: string;
  frameIndex: number;
  rank?: number;
  score?: number;
  keyframeOrdinal?: number;
  ptsTime?: number;
}

export type View = "workspace" | "submission";

export interface UiState {
  /** All query workspaces by id; order tracked separately. */
  queries: Record<string, QueryWorkspace>;
  queryOrder: string[];
  activeQueryId: string | null;
  view: View;
  selected: SelectedFrame | null;
  /** Ordered neighbor frames around the selection (kept in sync by TemporalStrip). */
  temporalFrames: FrameRecord[];
  /** Transient user-facing message (limit reached, duplicates, …). */
  notice: { message: string; at: number } | null;
  savedAt: number | null;
  gridCols: 3 | 4 | 6;
  layoutMode: "flat" | "grouped";
  /**
   * Active TRAKE anchor-picking target (prediction row + event slot in the
   * ACTIVE query). While set, the inspector's primary action assigns the
   * inspected frame to that event.
   */
  trakePick: { predictionIndex: number; eventIndex: number } | null;

  // query management
  ensureDefaultQuery: () => void;
  newQuery: (task: Task) => string;
  setActiveQuery: (id: string) => void;
  /** Switch to the most recent query of a task, creating one if needed. */
  setTask: (task: Task) => void;
  setView: (view: View) => void;
  setGridCols: (cols: 3 | 4 | 6) => void;
  setLayoutMode: (mode: "flat" | "grouped") => void;

  // search (active query scoped)
  setDraftQuery: (query: string) => void;
  submitSearch: (limit?: number) => void;
  setEventCount: (count: number) => void;

  // inspector
  selectFrame: (frame: SelectedFrame) => void;
  clearInspector: () => void;
  setTemporalFrames: (frames: FrameRecord[]) => void;
  stepTemporal: (direction: -1 | 1) => void;

  // prediction queue (active query scoped)
  chooseTraKeAnchor: (predictionIndex: number, eventIndex: number) => void;
  cancelTraKePick: () => void;
  /**
   * Replace or clear one positional event anchor of a TRAKE prediction.
   * Never reorders slots and never touches other events.
   */
  setTraKeAnchor: (
    queryId: string,
    predictionIndex: number,
    eventIndex: number,
    frameIndex: number | null,
  ) => void;
  /** Add one window as an initial editable SUGGESTION for a TRAKE query. */
  addTraKeSuggestion: (window: TraKeWindow, events: Parameters<typeof suggestTraKeAnchors>[1]) => boolean;
  /**
   * Replace the active kis/qa query's predictions with fresh search hits
   * (rank order, capped at MAX_PREDICTIONS). No-op for trake queries.
   */
  syncPredictionsFromSearch: (hits: SearchHit[]) => void;
  removePrediction: (index: number) => void;
  movePrediction: (index: number, direction: -1 | 1) => void;
  /** Move one prediction to an arbitrary target index (splice-move). */
  reorderPrediction: (from: number, to: number) => void;
  setQaAnswer: (queryId: string, index: number, answer: string) => void;
  setSavedAt: (savedAt: number) => void;
  notify: (message: string) => void;
}

function nextQueryId(queryOrder: string[]): string {
  let n = 1;
  while (queryOrder.includes(`Q${String(n).padStart(2, "0")}`)) n += 1;
  return `Q${String(n).padStart(2, "0")}`;
}

function emptyQuery(id: string, task: Task): QueryWorkspace {
  return {
    id,
    task,
    draftQuery: "",
    lastSearch: null,
    eventCount: null,
    predictions: [],
  };
}

/** Initial (fresh-session) state factory — also the hydration baseline. */
export function initialUiState(): Omit<
  UiState,
  | "ensureDefaultQuery"
  | "chooseTraKeAnchor"
  | "cancelTraKePick"
  | "setTraKeAnchor"
  | "newQuery"
  | "setActiveQuery"
  | "setTask"
  | "setView"
  | "setDraftQuery"
  | "submitSearch"
  | "setEventCount"
  | "selectFrame"
  | "clearInspector"
  | "setTemporalFrames"
  | "stepTemporal"
  | "addTraKeSuggestion"
  | "syncPredictionsFromSearch"
  | "removePrediction"
  | "movePrediction"
  | "reorderPrediction"
  | "setQaAnswer"
  | "setSavedAt"
  | "setGridCols"
  | "setLayoutMode"
  | "notify"
> {
  return {
    queries: {},
    queryOrder: [],
    activeQueryId: null,
    view: "workspace",
    selected: null,
    temporalFrames: [],
    notice: null,
    savedAt: null,
    gridCols: 4,
    layoutMode: "flat",
    trakePick: null,
  };
}

export const useUiStore = create<UiState>()(
  subscribeWithSelector((set, get) => ({
  ...initialUiState(),

  ensureDefaultQuery: () => {
    const { activeQueryId, queries } = get();
    if (activeQueryId !== null && queries[activeQueryId] !== undefined) return;
    if (get().queryOrder.length > 0) {
      set({ activeQueryId: get().queryOrder[0]! });
      return;
    }
    const id = nextQueryId([]);
    set({
      queries: { [id]: emptyQuery(id, "kis") },
      queryOrder: [id],
      activeQueryId: id,
    });
  },

  newQuery: (task) => {
    const { queryOrder } = get();
    const id = nextQueryId(queryOrder);
    set((state) => ({
      queries: { ...state.queries, [id]: emptyQuery(id, task) },
      queryOrder: [...state.queryOrder, id],
      activeQueryId: id,
      selected: null,
    }));
    return id;
  },

  setActiveQuery: (id) => {
    if (get().queries[id] === undefined) return;
    set({ activeQueryId: id, selected: null });
  },

  setTask: (task) => {
    const { activeQueryId, queries, queryOrder } = get();
    const active = activeQueryId !== null ? queries[activeQueryId] : undefined;
    if (active !== undefined && active.task === task) return;

    for (let i = queryOrder.length - 1; i >= 0; i -= 1) {
      const id = queryOrder[i]!;
      if (queries[id]?.task === task) {
        set({ activeQueryId: id, selected: null });
        return;
      }
    }
    const id = nextQueryId(queryOrder);
    set((state) => ({
      queries: { ...state.queries, [id]: emptyQuery(id, task) },
      queryOrder: [...state.queryOrder, id],
      activeQueryId: id,
      selected: null,
    }));
  },

  setView: (view) => set({ view }),

  setDraftQuery: (draftQuery) =>
    set((state) => {
      const active = state.activeQueryId;
      if (active === null || state.queries[active] === undefined) return {};
      return {
        queries: {
          ...state.queries,
          [active]: { ...state.queries[active]!, draftQuery },
        },
      };
    }),

  submitSearch: (limit = 100) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined) return {};
      const trimmed = query.draftQuery.trim();
      if (trimmed.length === 0) return {};
      return {
        queries: {
          ...state.queries,
          [active!]: { ...query, lastSearch: { query: trimmed, limit } },
        },
      };
    }),

  setEventCount: (count) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined || query.eventCount === count) return {};
      return {
        queries: {
          ...state.queries,
          [active!]: { ...query, eventCount: count },
        },
      };
    }),

  selectFrame: (selected) => set({ selected }),
  clearInspector: () => set({ selected: null }),

  setTemporalFrames: (temporalFrames) => set({ temporalFrames }),

  stepTemporal: (direction) => {
    const { selected, temporalFrames } = get();
    if (selected === null || temporalFrames.length === 0) return;
    const index = temporalFrames.findIndex(
      (f) => f.frame_index === selected.frameIndex,
    );
    if (index === -1) return;
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= temporalFrames.length) return;
    const next = temporalFrames[nextIndex];
    set({
      selected: {
        ...selected,
        frameIndex: next!.frame_index,
        keyframeOrdinal: next!.keyframe_ordinal,
        ptsTime: next!.pts_time,
        rank: undefined,
        score: undefined,
      },
    });
  },

  chooseTraKeAnchor: (predictionIndex, eventIndex) =>
    set({ trakePick: { predictionIndex, eventIndex } }),

  cancelTraKePick: () => set({ trakePick: null }),

  setTraKeAnchor: (queryId, predictionIndex, eventIndex, frameIndex) =>
    set((state) => {
      const query = state.queries[queryId];
      if (query === undefined || query.task !== "trake") return {};
      const prediction = query.predictions[predictionIndex] as TrakePrediction | undefined;
      if (
        prediction === undefined ||
        "answer" in prediction ||
        eventIndex < 0 ||
        eventIndex >= prediction.frameIndices.length
      )
        return {};
      // Replace exactly one positional slot; other slots keep their values.
      const slots = [...prediction.frameIndices];
      slots[eventIndex] = frameIndex;
      const predictions = [...query.predictions];
      predictions[predictionIndex] = { videoId: prediction.videoId, frameIndices: slots };
      return {
        queries: { ...state.queries, [queryId]: { ...query, predictions } },
        trakePick: null,
      };
    }),

  addTraKeSuggestion: (window, events) => {
    const state = get();
    const active = state.activeQueryId;
    const query = active !== null ? state.queries[active] : undefined;
    if (query === undefined || query.task !== "trake") return false;

    const n = query.eventCount;
    if (n === null || n < 1) {
      set({
        notice: {
          message: "Run the TRAKE search first — the requested event count N is unknown.",
          at: Date.now(),
        },
      });
      return false;
    }
    if (query.predictions.length >= MAX_PREDICTIONS) {
      set({
        notice: {
          message: `Limit reached: max ${MAX_PREDICTIONS} predictions per BTC query.`,
          at: Date.now(),
        },
      });
      return false;
    }

    // Initial suggestion only: per-event candidates in this video; missing
    // events stay unset (`null`) for the user to fill via the inspector.
    const slots = suggestTraKeAnchors(window.video_id, events, n);
    if (slots.every((v) => v === null)) {
      set({
        notice: {
          message:
            "No candidate frames for this video — no prediction was added (nothing invented).",
          at: Date.now(),
        },
      });
      return false;
    }

    const prediction: TrakePrediction = { videoId: window.video_id, frameIndices: slots };
    const key = predictionKey("trake", prediction);
    if (query.predictions.some((p) => predictionKey("trake", p) === key)) {
      set({ notice: { message: "Already added to this query's queue.", at: Date.now() } });
      return false;
    }
    set((s) => ({
      queries: {
        ...s.queries,
        [active!]: { ...query, predictions: [...query.predictions, prediction] },
      },
    }));
    return true;
  },

  syncPredictionsFromSearch: (hits) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined || query.task === "trake") return {};
      const capped = hits.slice(0, MAX_PREDICTIONS);
      const predictions: Prediction[] = capped.map((hit) =>
        query.task === "qa"
          ? { videoId: hit.video_id, frameIndex: hit.frame_index, score: hit.score, answer: "" }
          : ({ videoId: hit.video_id, frameIndex: hit.frame_index, score: hit.score } satisfies KisPrediction),
      );
      return {
        queries: { ...state.queries, [active!]: { ...query, predictions } },
      };
    }),

  removePrediction: (index) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined || index < 0 || index >= query.predictions.length) return {};
      return {
        queries: {
          ...state.queries,
          [active!]: {
            ...query,
            predictions: query.predictions.filter((_, i) => i !== index),
          },
        },
      };
    }),

  movePrediction: (index, direction) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined) return {};
      const target = index + direction;
      if (target < 0 || target >= query.predictions.length) return {};
      const predictions = [...query.predictions];
      const [moved] = predictions.splice(index, 1);
      predictions.splice(target, 0, moved!);
      return {
        queries: { ...state.queries, [active!]: { ...query, predictions } },
      };
    }),

  reorderPrediction: (from, to) =>
    set((state) => {
      const active = state.activeQueryId;
      const query = active !== null ? state.queries[active] : undefined;
      if (query === undefined) return {};
      const count = query.predictions.length;
      if (from === to || from < 0 || from >= count || to < 0 || to >= count) return {};
      const predictions = [...query.predictions];
      const [moved] = predictions.splice(from, 1);
      predictions.splice(to, 0, moved!);
      return {
        queries: { ...state.queries, [active!]: { ...query, predictions } },
      };
    }),

  setQaAnswer: (queryId, index, answer) =>
    set((state) => {
      const query = state.queries[queryId];
      if (query === undefined || query.task !== "qa") return {};
      if (index < 0 || index >= query.predictions.length) return {};
      const predictions = [...query.predictions];
      const target = predictions[index] as Prediction;
      predictions[index] = { ...(target as object), answer } as QaPredictionLike;
      return {
        queries: { ...state.queries, [queryId]: { ...query, predictions } },
      };
    }),

  setSavedAt: (savedAt) => set({ savedAt }),
  setGridCols: (gridCols) => set({ gridCols }),
  setLayoutMode: (layoutMode) => set({ layoutMode }),
  notify: (message) => set({ notice: { message, at: Date.now() } }),
  })));

type QaPredictionLike = Prediction & { answer: string };

/** Convenience selector: the active query workspace (or undefined). */
export function useActiveQuery(): QueryWorkspace | undefined {
  return useUiStore((s) => (s.activeQueryId !== null ? s.queries[s.activeQueryId] : undefined));
}
