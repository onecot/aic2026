import { useEffect } from "react";
import { FrameInspector, TEMPORAL_LIMIT, TEMPORAL_RADIUS } from "./components/FrameInspector";
import { QueryToolbar } from "./components/QueryToolbar";
import { ResultGrid } from "./components/ResultGrid";
import { SearchBar, } from "./components/SearchBar";
import { SystemStatus } from "./components/SystemStatus";
import { TemporalStrip } from "./components/TemporalStrip";
import { PredictionQueue } from "./features/queue/PredictionQueue";
import { SubmissionView } from "./features/submission/SubmissionView";
import { TrakeResults } from "./features/trake/TrakeResults";
import { useBackendStatus, useSearch, useSystem, useTemporalFrames } from "./hooks/queries";
import { focusSearchInput } from "./lib/focusSearch";
import { saveSession, serializeSession } from "./state/persistence";
import { useUiStore } from "./state/uiStore";

function GateScreen({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3">
      {children}
    </div>
  );
}

/** Task-primary action for the inspected frame ("S" key / Select button). */
function primarySelect(): void {
  const state = useUiStore.getState();
  if (state.selected === null) return;
  const task = state.activeQueryId !== null ? state.queries[state.activeQueryId]?.task : undefined;
  if (task === "kis" || task === "qa") state.addPredictionFromSelected();
}

/** Debounced autosave of the prediction workspace to localStorage. */
function useWorkspaceAutosave(): void {
  useEffect(() => {
    let timer: number | undefined;
    const unsubscribe = useUiStore.subscribe(
      (s) => JSON.stringify([s.queries, s.queryOrder, s.activeQueryId]),
      (json) => {
        window.clearTimeout(timer);
        timer = window.setTimeout(() => {
          const [queries, queryOrder, activeQueryId] = JSON.parse(json) as [
            Parameters<typeof serializeSession>[0],
            string[],
            string | null,
          ];
          saveSession(serializeSession(queries, queryOrder, activeQueryId));
          useUiStore.getState().setSavedAt(Date.now());
        }, 400);
      },
    );
    return () => {
      unsubscribe();
      window.clearTimeout(timer);
    };
  }, []);
}

export default function App() {
  const backend = useBackendStatus();
  const system = useSystem(backend.status === "ready");
  const view = useUiStore((s) => s.view);
  const selected = useUiStore((s) => s.selected);
  const selectFrame = useUiStore((s) => s.selectFrame);
  const setTemporalFrames = useUiStore((s) => s.setTemporalFrames);
  const stepTemporal = useUiStore((s) => s.stepTemporal);
  const clearInspector = useUiStore((s) => s.clearInspector);
  const setEventCount = useUiStore((s) => s.setEventCount);

  const searchState = useSearch(backend.status === "ready");

  const temporalQuery = useTemporalFrames(
    selected?.videoId ?? null,
    selected ? { center: selected.frameIndex, radius: TEMPORAL_RADIUS, limit: TEMPORAL_LIMIT } : null,
  );

  useWorkspaceAutosave();

  useEffect(() => {
    if (temporalQuery.data?.frames) setTemporalFrames(temporalQuery.data.frames);
  }, [temporalQuery.data, setTemporalFrames]);

  // Capture the requested event count N for TRAKE validation.
  const events = searchState.data?.events ?? [];
  const activeTask =
    useUiStore((s) => (s.activeQueryId !== null ? s.queries[s.activeQueryId]?.task : undefined)) ??
    "kis";
  useEffect(() => {
    if (activeTask === "trake" && events.length > 0) setEventCount(events.length);
  }, [activeTask, events.length, setEventCount]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const inField =
        target !== null &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT");
      if (event.key === "/" && !inField) {
        event.preventDefault();
        focusSearchInput();
      } else if (event.key === "Escape") {
        if (inField) target?.blur();
        useUiStore.getState().cancelTraKePick();
        clearInspector();
      } else if (!inField && event.key === "ArrowLeft") {
        event.preventDefault();
        stepTemporal(-1);
      } else if (!inField && event.key === "ArrowRight") {
        event.preventDefault();
        stepTemporal(1);
      } else if (!inField && (event.key === "s" || event.key === "S")) {
        primarySelect();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepTemporal, clearInspector]);

  if (backend.status === "unavailable") {
    return (
      <GateScreen>
        <p>Retrieval backend unavailable.</p>
        <button
          type="button"
          onClick={backend.retry}
          className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-black"
        >
          Retry
        </button>
      </GateScreen>
    );
  }
  if (backend.status !== "ready") {
    return (
      <GateScreen>
        <p className="text-dim">Retrieval backend is starting…</p>
      </GateScreen>
    );
  }

  const meta = searchState.data?.meta ?? null;
  const hits = searchState.data?.results ?? [];

  let resultsArea: React.ReactNode;
  if (searchState.pending && searchState.data === undefined) {
    resultsArea = <p className="p-6 text-sm text-dim">Searching…</p>;
  } else if (searchState.error !== null && searchState.data === undefined) {
    resultsArea = (
      <div className="p-6 text-sm">
        <p>{searchState.error.message}</p>
        <p className="mt-1 text-xs text-dim">
          The query was not changed; press Search to try again.
        </p>
      </div>
    );
  } else if (activeTask === "trake") {
    resultsArea = <TrakeResults events={events} />;
  } else {
    resultsArea = <ResultGrid hits={hits} />;
  }

  if (view === "submission") {
    return (
      <div className="flex h-full min-h-0 flex-col">
        <SystemStatus system={system.data} />
        <QueryToolbar />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <SubmissionView />
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <SystemStatus system={system.data} />
      <QueryToolbar />
      <SearchBar pending={searchState.pending} meta={meta} />

      <main className="flex min-h-0 flex-1">
        <section aria-label="Results" className="min-w-0 flex-1 overflow-y-auto border-r border-edge">
          {resultsArea}
        </section>
        <aside aria-label="Inspector" className="hidden w-[340px] shrink-0 md:block">
          <FrameInspector onSelectPrimary={primarySelect} />
        </aside>
      </main>

      <footer className="border-t border-edge">
        {selected !== null && temporalQuery.data && (
          <div className="border-b border-edge">
            <TemporalStrip
              videoId={selected.videoId}
              frames={temporalQuery.data.frames}
              current={
                temporalQuery.data.frames.find(
                  (f) => f.frame_index === selected.frameIndex,
                ) ?? null
              }
              onSelect={(frame) =>
                selectFrame({
                  videoId: selected.videoId,
                  frameIndex: frame.frame_index,
                  keyframeOrdinal: frame.keyframe_ordinal,
                  ptsTime: frame.pts_time,
                })
              }
            />
          </div>
        )}
        <PredictionQueue />
      </footer>
    </div>
  );
}
