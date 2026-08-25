import { getFrameUrl } from "../api/client";
import { formatScore, formatSeconds } from "../lib/format";
import { predictionKey } from "../state/submission";
import { useActiveQuery, useUiStore } from "../state/uiStore";
import { useTemporalFrames, useVideo } from "../hooks/queries";
import { MetadataPanel } from "./MetadataPanel";

export const TEMPORAL_RADIUS = 2000;
export const TEMPORAL_LIMIT = 25;

/**
 * Large view of the selected frame with metadata and temporal controls.
 * KIS/QA predictions come straight from search results (edited in the
 * results grid); TRAKE sequences are still assembled from window cards
 * here via anchor picking.
 */
export function FrameInspector() {
  const selected = useUiStore((s) => s.selected);
  const stepTemporal = useUiStore((s) => s.stepTemporal);
  const clearInspector = useUiStore((s) => s.clearInspector);
  const removePrediction = useUiStore((s) => s.removePrediction);
  const activeQuery = useActiveQuery();
  const pick = useUiStore((s) => s.trakePick);
  const setTraKeAnchor = useUiStore((s) => s.setTraKeAnchor);

  const videoQuery = useVideo(selected?.videoId ?? null);
  const temporalQuery = useTemporalFrames(
    selected?.videoId ?? null,
    selected ? { center: selected.frameIndex, radius: TEMPORAL_RADIUS, limit: TEMPORAL_LIMIT } : null,
  );
  const record =
    temporalQuery.data?.frames.find((f) => f.frame_index === selected?.frameIndex) ?? null;

  if (selected === null || activeQuery === undefined) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-dim">
        Select a result to inspect it.
      </div>
    );
  }

  const task = activeQuery.task;
  const currentKey =
    task === "trake" || !("frameIndex" in (selected as object))
      ? null
      : predictionKey(task, {
          videoId: selected.videoId,
          frameIndex: selected.frameIndex,
        });
  const addedIndex =
    currentKey === null
      ? -1
      : activeQuery.predictions.findIndex((p) => predictionKey(task, p) === currentKey);
  const added = addedIndex !== -1;

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 overflow-y-auto p-3">
      <img
        src={getFrameUrl(selected.videoId, selected.frameIndex)}
        alt={`${selected.videoId} frame ${selected.frameIndex}`}
        className="w-full rounded border border-edge bg-black object-contain"
      />
      <MetadataPanel
        entries={[
          ["video", selected.videoId],
          ["frame", String(selected.frameIndex)],
          [
            "keyframe #",
            String(selected.keyframeOrdinal ?? record?.keyframe_ordinal ?? "—"),
          ],
          ["time", formatSeconds(record?.pts_time ?? selected.ptsTime ?? 0)],
          ["fps", record ? String(record.fps) : "—"],
          ...(selected.rank !== undefined ? ([["rank", `#${selected.rank}`]] as [string, string][]) : []),
          ...(selected.score !== undefined
            ? ([["score", formatScore(selected.score)]] as [string, string][])
            : []),
          ...(videoQuery.data
            ? ([
                [
                  "video keys",
                  `${videoQuery.data.first_frame_index}–${videoQuery.data.last_frame_index} (${videoQuery.data.frame_count})`,
                ],
              ] as [string, string][])
            : []),
        ]}
      />
      <div className="mt-auto flex gap-2 pt-1">
        <button
          type="button"
          onClick={() => stepTemporal(-1)}
          className="rounded border border-edge px-2 py-1 text-xs hover:border-accent"
          aria-label="Previous temporal frame"
        >
          ◀ Prev
        </button>
        <button
          type="button"
          onClick={() => stepTemporal(1)}
          className="rounded border border-edge px-2 py-1 text-xs hover:border-accent"
          aria-label="Next temporal frame"
        >
          Next ▶
        </button>
        {task === "trake" &&
          (pick !== null && selected !== null ? (
            <button
              type="button"
              onClick={() =>
                setTraKeAnchor(activeQuery.id, pick.predictionIndex, pick.eventIndex, selected.frameIndex)
              }
              title="Replace this event's anchor with the inspected frame"
              className="ml-auto rounded bg-accent px-3 py-1 text-xs font-medium text-black"
            >
              Set as Event {pick.eventIndex + 1}
            </button>
          ) : (
            <span className="ml-auto text-xs text-dim">
              “Choose” an event slot in the queue to set its frame here.
            </span>
          ))}
        <button
          type="button"
          onClick={() => removePrediction(addedIndex)}
          disabled={!added}
          className={`rounded border border-edge px-2 py-1 text-xs text-dim enabled:hover:border-red-500 enabled:hover:text-red-400 disabled:opacity-30 ${
            task === "trake" ? "" : "ml-auto"
          }`}
        >
          Remove prediction
        </button>
        <button
          type="button"
          onClick={clearInspector}
          className="rounded border border-edge px-2 py-1 text-xs text-dim hover:text-ink"
          aria-label="Clear inspector"
        >
          Esc ✕
        </button>
      </div>
    </div>
  );
}
