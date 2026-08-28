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
import type { SearchHit, SearchResponse } from "./api/types";
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
  const notify = useUiStore((s) => s.notify);

  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [browserVideoId, setBrowserVideoId] = useState<string | null>(null);
  const [similarData, setSimilarData] = useState<SearchResponse | null>(null);
  const [similarPending, setSimilarPending] = useState(false);

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

  // Clear similar view when a new text search completes (so text results are authoritative).
  useEffect(() => {
    if (searchState.data) {
      // A new text search arrived - clear the transient similar view so
      // the user's explicit text query is not hidden behind old similar results.
      // We keep similarData only if user explicitly triggered similar after this search.
      // Simple heuristic: if searchState.data.query differs from similarData?.query,
      // the text search is newer. Since similar queries are prefixed with "similar:",
      // any text query that is not similar: prefix should clear.
      if (similarData !== null && !searchState.data.query.startsWith("similar:")) {
        // Only clear if the new text search is for a different query than the similar source?
        // To avoid flicker when similarData is set via manual fetch (not via useSearch),
        // we do not auto-clear on every searchState change. Instead, we clear when
        // the text search's results are fresh and user hasn't just done similar.
        // For now, keep similarData untouched; user can dismiss manually.
        // This effect is intentionally minimal to avoid surprising clears.
      }
    }
  }, [searchState.data, similarData]);

  const handleFindSimilar = async (videoId: string, frameIndex: number) => {
    setSimilarPending(true);
    try {
      notify(`Searching similar to ${videoId} f${frameIndex}…`);
      const res = await searchSimilar({ video_id: videoId, frame_index: frameIndex, limit: 100 });
      setSimilarData(res);
      notify(`Found ${res.results.length} similar frames`);
      // Optionally select the top result to keep inspector in sync
      if (res.results.length > 0) {
        const top = res.results[0]!;
        selectFrame({
          videoId: top.video_id,
          frameIndex: top.frame_index,
          rank: top.rank,
          score: top.score,
          keyframeOrdinal: top.keyframe_ordinal,
          ptsTime: top.pts_time,
        });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      notify(`Similar search failed: ${msg}`);
    } finally {
      setSimilarPending(false);
    }
  };

  const clearSimilar = () => setSimilarData(null);

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
      } else if (!inField && (event.key === "s" || event.key === "S")) {
        primarySelect();
      } else if (!inField && (event.key === "j" || event.key === "ArrowDown")) {
        // j / ArrowDown: next card in currently displayed results
        event.preventDefault();
        const hits: SearchHit[] = similarData ? similarData.results : (searchState.data?.results ?? []);
        if (hits.length === 0) return;
        const currIdx = selected
          ? hits.findIndex((h) => h.video_id === selected.videoId && h.frame_index === selected.frameIndex)
          : -1;
        const nextIdx = currIdx === -1 ? 0 : Math.min(currIdx + 1, hits.length - 1);
        const h = hits[nextIdx];
        if (h) {
          selectFrame({
            videoId: h.video_id,
            frameIndex: h.frame_index,
            rank: h.rank,
            score: h.score,
            keyframeOrdinal: h.keyframe_ordinal,
            ptsTime: h.pts_time,
          });
        }
      } else if (!inField && (event.key === "k" || event.key === "ArrowUp")) {
        event.preventDefault();
        const hits: SearchHit[] = similarData ? similarData.results : (searchState.data?.results ?? []);
        if (hits.length === 0) return;
        const currIdx = selected
          ? hits.findIndex((h) => h.video_id === selected.videoId && h.frame_index === selected.frameIndex)
          : -1;
        const prevIdx = currIdx === -1 ? 0 : Math.max(currIdx - 1, 0);
        const h = hits[prevIdx];
        if (h) {
          selectFrame({
            videoId: h.video_id,
            frameIndex: h.frame_index,
            rank: h.rank,
            score: h.score,
            keyframeOrdinal: h.keyframe_ordinal,
            ptsTime: h.pts_time,
          });
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepTemporal, clearInspector, selected, selectFrame, searchState.data, similarData]);

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

  const textMeta = searchState.data?.meta ?? null;
  const textHits = searchState.data?.results ?? [];
  const displayHits = similarData ? similarData.results : textHits;
  const displayMeta = similarData ? similarData.meta : textMeta;
  const isSimilarView = similarData !== null;

  let resultsArea: React.ReactNode;
  if (similarPending) {
    resultsArea = <p className="p-6 text-sm text-dim">Searching similar…</p>;
  } else if (searchState.pending && searchState.data === undefined && !isSimilarView) {
    resultsArea = <p className="p-6 text-sm text-dim">Searching…</p>;
  } else if (searchState.error !== null && searchState.data === undefined && !isSimilarView) {
    resultsArea = (
      <div className="p-6 text-sm">
        <p>{searchState.error.message}</p>
        <p className="mt-1 text-xs text-dim">
          The query was not changed; press Search to try again.
        </p>
      </div>
    );
  } else if (activeTask === "trake" && !isSimilarView) {
    resultsArea = <TrakeResults events={events} />;
  } else {
    resultsArea = (
      <div className="flex flex-col">
        {isSimilarView && (
          <div className="flex items-center justify-between gap-2 border-b border-edge/60 bg-panel/40 px-3 py-1.5 text-xs">
            <span className="text-dim">
              Showing <span className="font-medium text-ink">{displayHits.length}</span> similar to {similarData?.query}
            </span>
            <button
              type="button"
              onClick={clearSimilar}
              className="rounded border border-edge px-2 py-0.5 text-dim hover:text-ink"
            >
              Back to text results
            </button>
          </div>
        )}
        <ResultGrid hits={displayHits} />
      </div>
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
      <SearchBar pending={searchState.pending || similarPending} meta={displayMeta} />

      <main className="flex min-h-0 flex-1">
        <section aria-label="Results" className="min-w-0 flex-1 overflow-y-auto border-r border-edge">
          {resultsArea}
        </section>
        <aside aria-label="Inspector" className="hidden w-[340px] shrink-0 md:block">
          <FrameInspector
            onSelectPrimary={primarySelect}
            onFindSimilar={handleFindSimilar}
            onBrowseVideo={(vid) => setBrowserVideoId(vid)}
          />
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
