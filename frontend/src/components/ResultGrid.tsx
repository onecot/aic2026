import { getFrameUrl } from "../api/client";
import type { SearchHit } from "../api/types";
import { formatScore } from "../lib/format";
import { predictionKey } from "../state/submission";
import { useUiStore, useActiveQuery } from "../state/uiStore";

interface ResultCardProps {
  hit: SearchHit;
  selected: boolean;
  added: boolean;
  onSelect: (hit: SearchHit) => void;
}

/** One ranked keyframe candidate. Rank-preserving; image dominant. */
export function ResultCard({ hit, selected, added, onSelect }: ResultCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(hit)}
      aria-pressed={selected}
      title={`${hit.video_id} frame ${hit.frame_index} · rank ${hit.rank} · score ${formatScore(hit.score)}`}
      className={`group relative block overflow-hidden rounded border text-left transition-colors ${
        selected
          ? "border-accent ring-2 ring-accent"
          : "border-edge hover:border-dim"
      }`}
    >
      <img
        src={getFrameUrl(hit.video_id, hit.frame_index)}
        alt={`rank ${hit.rank}: ${hit.video_id} frame ${hit.frame_index}`}
        loading="lazy"
        decoding="async"
        className="aspect-video w-full bg-black object-cover"
      />
      <span className="absolute left-1 top-1 rounded bg-black/75 px-1.5 py-0.5 text-xs font-semibold tabular-nums">
        #{hit.rank}
      </span>
      {added && (
        <span
          aria-label="Already in prediction queue"
          className="absolute right-1 top-1 rounded bg-accent px-1.5 py-0.5 text-xs font-bold text-black"
        >
          ✓
        </span>
      )}
      <span className="absolute inset-x-0 bottom-0 flex items-baseline justify-between gap-2 bg-gradient-to-t from-black/85 to-transparent px-1.5 pb-1 pt-4 text-[11px] leading-tight">
        <span className="truncate tabular-nums">
          {hit.video_id} · f{hit.frame_index}
        </span>
        <span className="tabular-nums text-dim">{formatScore(hit.score)}</span>
      </span>
    </button>
  );
}

interface ResultGridProps {
  hits: SearchHit[];
  emptyHint?: string;
}

/** Rank-preserving candidate grid (no client-side reordering/grouping). */
export function ResultGrid({ hits, emptyHint = "No results yet." }: ResultGridProps) {
  const selected = useUiStore((s) => s.selected);
  const selectFrame = useUiStore((s) => s.selectFrame);
  const activeQuery = useActiveQuery();

  if (hits.length === 0) {
    return <p className="p-6 text-sm text-dim">{emptyHint}</p>;
  }
  return (
    <div className="grid grid-cols-3 gap-2 p-3 lg:grid-cols-4 2xl:grid-cols-5">
      {hits.map((hit) => (
        <ResultCard
          key={`${hit.video_id}-${hit.frame_index}`}
          hit={hit}
          selected={
            selected !== null &&
            selected.videoId === hit.video_id &&
            selected.frameIndex === hit.frame_index
          }
          added={
            activeQuery !== undefined &&
            activeQuery.predictions.some(
              (p) =>
                predictionKey(activeQuery.task, p) ===
                predictionKey("kis", { videoId: hit.video_id, frameIndex: hit.frame_index }),
            )
          }
          onSelect={(h) =>
            selectFrame({
              videoId: h.video_id,
              frameIndex: h.frame_index,
              rank: h.rank,
              score: h.score,
              keyframeOrdinal: h.keyframe_ordinal,
              ptsTime: h.pts_time,
            })
          }
        />
      ))}
    </div>
  );
}
