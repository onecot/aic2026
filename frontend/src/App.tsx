import { useEffect, useState } from "react";
import { FrameInspector, TEMPORAL_LIMIT, TEMPORAL_RADIUS } from "./components/FrameInspector";
import { QueryToolbar } from "./components/QueryToolbar";
import { ResultGrid } from "./components/ResultGrid";
import { SearchBar } from "./components/SearchBar";
import { ShortcutsModal } from "./components/ShortcutsModal";
import { SystemStatus } from "./components/SystemStatus";
import { TemporalStrip } from "./components/TemporalStrip";
import { Toast } from "./components/Toast";
import { VideoBrowserModal } from "./components/VideoBrowserModal";
import { PredictionQueue } from "./features/queue/PredictionQueue";
import { SubmissionView } from "./features/submission/SubmissionView";
import { TrakeResults } from "./features/trake/TrakeResults";
import { useBackendStatus, useSearch, useSystem, useTemporalFrames } from "./hooks/queries";
import { searchSimilar } from "./api/client";
import { focusSearchInput } from "./lib/focusSearch";
import { saveSession, serializeSession } from "./state/persistence";
import { useUiStore, useActiveQuery } from "./state/uiStore";

function GateScreen({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3">
      {children}
    </div>
  );
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
  const removePrediction = useUiStore((s) => s.removePrediction);
  const notify = useUiStore((s) => s.notify);
  const activeQuery = useActiveQuery();

  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [browserVideoId, setBrowserVideoId] = useState<string | null>(null);

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

  // kis/qa: a completed search IS the displayed, exportable prediction list —
  // no manual "Add" step. Fires once per genuinely new search (react-query
  // keeps `data` referentially stable across re-renders of the same key).
  useEffect(() => {
    if (activeTask !== "trake" && searchState.data) {
      useUiStore.getState().syncPredictionsFromSearch(searchState.data.results);
    }
  }, [activeTask, searchState.data]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const inField =
        target !== null &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT");

      if (event.key === "/" && !inField) {
        event.preventDefault();
        focusSearchInput();
      } else if (event.key === "?" && !inField) {
        event.preventDefault();
        setShortcutsOpen((prev) => !prev);
      } else if (event.key === "Escape") {
        if (inField) target?.blur();
        useUiStore.getState().cancelTraKePick();
        clearInspector();
        setShortcutsOpen(false);
        setBrowserVideoId(null);
      } else if (!inField && event.key === "ArrowLeft") {
        event.preventDefault();
        stepTemporal(-1);
      } else if (!inField && event.key === "ArrowRight") {
        event.preventDefault();
        stepTemporal(1);
      } else if (!inField && (event.key === "j" || event.key === "ArrowDown")) {
        // Step down card selection
        event.preventDefault();
        const preds = (activeQuery?.predictions ?? []).filter((p): p is { videoId: string; frameIndex: number; score?: number } => typeof (p as any).frameIndex === "number");
        if (preds.length === 0) return;
        const currIdx = selected ? preds.findIndex((p) => p.videoId === selected.videoId && p.frameIndex === selected.frameIndex) : -1;
        const nextIdx = currIdx === -1 ? 0 : Math.min(currIdx + 1, preds.length - 1);
        const p = preds[nextIdx];
        if (p) selectFrame({ videoId: p.videoId, frameIndex: p.frameIndex, score: p.score });
      } else if (!inField && (event.key === "k" || event.key === "ArrowUp")) {
        // Step up card selection
        event.preventDefault();
        const preds = (activeQuery?.predictions ?? []).filter((p): p is { videoId: string; frameIndex: number; score?: number } => typeof (p as any).frameIndex === "number");
        if (preds.length === 0) return;
        const currIdx = selected ? preds.findIndex((p) => p.videoId === selected.videoId && p.frameIndex === selected.frameIndex) : -1;
        const prevIdx = currIdx === -1 ? 0 : Math.max(currIdx - 1, 0);
        const p = preds[prevIdx];
        if (p) selectFrame({ videoId: p.videoId, frameIndex: p.frameIndex, score: p.score });
      } else if (!inField && (event.key === "d" || event.key === "Delete")) {
        // Delete selected card
        const preds = (activeQuery?.predictions ?? []).filter((p): p is { videoId: string; frameIndex: number; score?: number } => typeof (p as any).frameIndex === "number");
        if (!selected || preds.length === 0) return;
        const currIdx = preds.findIndex((p) => p.videoId === selected.videoId && p.frameIndex === selected.frameIndex);
        if (currIdx !== -1) {
          event.preventDefault();
          removePrediction(currIdx);
          clearInspector();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepTemporal, clearInspector, activeQuery?.predictions, selected, selectFrame, removePrediction]);

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

  const handleSearchSimilar = async (videoId: string, frameIndex: number) => {
    try {
      notify(`Đang quét các khung hình tương tự ${videoId} f${frameIndex}…`);
      const res = await searchSimilar({ video_id: videoId, frame_index: frameIndex, limit: 100 });
      if (res.results.length > 0) {
        useUiStore.getState().syncPredictionsFromSearch(res.results);
        notify(`✅ Tìm thấy ${res.results.length} khung hình thị giác tương tự!`);
      } else {
        notify("Không tìm thấy khung hình tương tự phù hợp.");
      }
    } catch (err) {
      notify(`Lỗi khi tìm ảnh tương tự: ${String(err)}`);
    }
  };

  let resultsArea: React.ReactNode;
  if (searchState.pending && searchState.data === undefined) {
    resultsArea = <p className="p-6 text-sm text-dim">Đang tìm kiếm…</p>;
  } else if (searchState.error !== null && searchState.data === undefined) {
    resultsArea = (
      <div className="p-6 text-sm">
        <p>{searchState.error.message}</p>
        <p className="mt-1 text-xs text-dim">
          Câu tìm kiếm chưa thay đổi; bấm Search để thử lại.
        </p>
      </div>
    );
  } else if (activeTask === "trake") {
    resultsArea = <TrakeResults events={events} />;
  } else {
    resultsArea = (
      <ResultGrid
        onOpenVideoBrowser={(vid) => setBrowserVideoId(vid)}
        onSearchSimilar={handleSearchSimilar}
      />
    );
  }

  if (view === "submission") {
    return (
      <div className="flex h-full min-h-0 flex-col">
        <SystemStatus system={system.data} />
        <QueryToolbar onOpenShortcuts={() => setShortcutsOpen(true)} />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <SubmissionView />
        </main>
        <Toast />
        <ShortcutsModal isOpen={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
        <VideoBrowserModal videoId={browserVideoId} onClose={() => setBrowserVideoId(null)} />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <SystemStatus system={system.data} />
      <QueryToolbar onOpenShortcuts={() => setShortcutsOpen(true)} />
      <SearchBar pending={searchState.pending} meta={meta} />

      <main className="flex min-h-0 flex-1">
        <section aria-label="Results" className="min-w-0 flex-1 overflow-y-auto border-r border-edge">
          {resultsArea}
        </section>
        <aside aria-label="Inspector" className="hidden w-[340px] shrink-0 md:block">
          <FrameInspector />
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

      <Toast />
      <ShortcutsModal isOpen={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      <VideoBrowserModal videoId={browserVideoId} onClose={() => setBrowserVideoId(null)} />
    </div>
  );
}
