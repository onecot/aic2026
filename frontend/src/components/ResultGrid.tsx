import { useState, type ChangeEvent, type DragEvent } from "react";
import { getFrameUrl } from "../api/client";
import { formatScore } from "../lib/format";
import { ANSWER_MAX_LENGTH, type KisPrediction, type QaPrediction, type Task } from "../state/submission";
import { useUiStore, useActiveQuery } from "../state/uiStore";

type FlatPrediction = KisPrediction | QaPrediction;

interface ResultCardProps {
  prediction: FlatPrediction;
  index: number;
  count: number;
  task: Task;
  selected: boolean;
  onSelect: (prediction: FlatPrediction) => void;
  onRemove: (index: number) => void;
  onReorder: (from: number, to: number) => void;
  onAnswerChange: (index: number, answer: string) => void;
  onSearchSimilar?: (videoId: string, frameIndex: number) => void;
}

/**
 * One ranked keyframe prediction card: rank is position-derived, and the
 * card is both the "displayed keyframe" AND the exported submission row.
 * Includes Hover Scrubbing and Visual Similarity Search ("Tìm tương tự").
 */
export function ResultCard({
  prediction,
  index,
  count,
  task,
  selected,
  onSelect,
  onRemove,
  onReorder,
  onAnswerChange,
  onSearchSimilar,
}: ResultCardProps) {
  const { videoId, frameIndex, score } = prediction;
  const [scrubPercent, setScrubPercent] = useState<number | null>(null);

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData("text/plain", String(index));
  };
  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const from = Number(e.dataTransfer.getData("text/plain"));
    if (!Number.isNaN(from)) onReorder(from, index);
  };
  const handleRankChange = (e: ChangeEvent<HTMLInputElement>) => {
    const rank = Math.trunc(Number(e.target.value));
    if (!Number.isFinite(rank)) return;
    const target = Math.min(Math.max(rank - 1, 0), count - 1);
    onReorder(index, target);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setScrubPercent(Math.round((x / rect.width) * 100));
  };

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      className={`group relative flex flex-col overflow-hidden rounded border transition-all ${
        selected ? "border-accent ring-2 ring-accent shadow-lg shadow-accent/20" : "border-edge hover:border-dim"
      }`}
    >
      <div
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setScrubPercent(null)}
        className="relative block w-full text-left"
      >
        <button
          type="button"
          onClick={() => onSelect(prediction)}
          aria-pressed={selected}
          title={`${videoId} frame ${frameIndex} · rank ${index + 1}${
            score !== undefined ? ` · score ${formatScore(score)}` : ""
          }`}
          className="relative block w-full text-left focus:outline-none"
        >
          <img
            src={getFrameUrl(videoId, frameIndex)}
            alt={`rank ${index + 1}: ${videoId} frame ${frameIndex}`}
            loading="lazy"
            decoding="async"
            className="aspect-video w-full bg-black object-cover transition-transform duration-200 group-hover:scale-[1.02]"
          />
          <span className="absolute left-1 top-1 rounded bg-black/80 px-1.5 py-0.5 text-xs font-semibold tabular-nums text-white">
            #{index + 1}
          </span>
          <span className="absolute inset-x-0 bottom-0 flex items-baseline justify-between gap-2 bg-gradient-to-t from-black/90 to-transparent px-1.5 pb-1 pt-4 text-[11px] leading-tight">
            <span className="truncate tabular-nums font-medium text-white">
              {videoId} · f{frameIndex}
            </span>
            {score !== undefined ? (
              <span
                className={`rounded px-1.5 py-0.2 text-[10px] font-semibold tabular-nums ${
                  Number(score) >= 0.16
                    ? "bg-emerald-900/80 text-emerald-300 border border-emerald-500/40"
                    : Number(score) >= 0.13
                    ? "bg-amber-900/80 text-amber-300 border border-amber-500/40"
                    : "bg-black/80 text-dim border border-edge/60"
                }`}
              >
                {formatScore(score)}
              </span>
            ) : (
              <span className="tabular-nums text-dim">—</span>
            )}
          </span>
        </button>

        {/* Hover Scrubber Bar Indicator */}
        {scrubPercent !== null && (
          <div className="absolute inset-x-0 bottom-0 h-1 bg-black/60">
            <div
              className="h-full bg-accent transition-all duration-75"
              style={{ width: `${scrubPercent}%` }}
            />
          </div>
        )}
      </div>

      <div className="flex items-center gap-1 border-t border-edge/60 bg-panel/40 px-1.5 py-1 text-[11px]">
        <label className="flex items-center gap-1 text-dim">
          rank
          <input
            type="number"
            min={1}
            max={count}
            value={index + 1}
            onChange={handleRankChange}
            aria-label={`Rank for prediction ${index + 1}`}
            className="w-11 rounded border border-edge bg-panel px-1 py-0.5 text-ink outline-none focus:border-accent"
          />
        </label>

        {onSearchSimilar && (
          <button
            type="button"
            onClick={() => onSearchSimilar(videoId, frameIndex)}
            title="Tìm kiếm các khung hình tương tự ảnh này (Query-by-Image)"
            className="rounded border border-edge/80 px-1.5 py-0.5 text-[10px] text-accent hover:bg-accent/20 hover:border-accent transition-colors"
          >
            🔍 Tương tự
          </button>
        )}

        <button
          type="button"
          onClick={() => onRemove(index)}
          aria-label={`Remove prediction ${index + 1}`}
          className="ml-auto rounded border border-edge px-1.5 py-0.5 text-dim hover:border-red-500 hover:text-red-400 text-[10px]"
        >
          Xóa
        </button>
      </div>

      {task === "qa" && (
        <div className="flex flex-col gap-0.5 border-t border-edge/60 px-1.5 py-1">
          <input
            value={(prediction as QaPrediction).answer}
            onChange={(e) => onAnswerChange(index, e.target.value)}
            placeholder="answer…"
            aria-label={`Answer for prediction ${index + 1}`}
            spellCheck={false}
            className={`w-full rounded border bg-panel px-1.5 py-0.5 text-[11px] outline-none focus:border-accent ${
              (prediction as QaPrediction).answer.length > ANSWER_MAX_LENGTH ? "border-red-500" : "border-edge"
            }`}
          />
          <span
            className={`text-[10px] tabular-nums ${
              (prediction as QaPrediction).answer.length > ANSWER_MAX_LENGTH ? "text-red-400" : "text-dim"
            }`}
          >
            {(prediction as QaPrediction).answer.length}/{ANSWER_MAX_LENGTH}
          </span>
        </div>
      )}
    </div>
  );
}

interface ResultGridProps {
  emptyHint?: string;
  onOpenVideoBrowser?: (videoId: string) => void;
  onSearchSimilar?: (videoId: string, frameIndex: number) => void;
}

/**
 * Editable, exportable keyframe list for kis/qa: supports both Flat Ranked Grid
 * and Grouped by Video modes.
 */
export function ResultGrid({
  emptyHint = "No results yet.",
  onOpenVideoBrowser,
  onSearchSimilar,
}: ResultGridProps) {
  const selected = useUiStore((s) => s.selected);
  const selectFrame = useUiStore((s) => s.selectFrame);
  const removePrediction = useUiStore((s) => s.removePrediction);
  const reorderPrediction = useUiStore((s) => s.reorderPrediction);
  const setQaAnswer = useUiStore((s) => s.setQaAnswer);
  const gridCols = useUiStore((s) => s.gridCols);
  const setGridCols = useUiStore((s) => s.setGridCols);
  const layoutMode = useUiStore((s) => s.layoutMode);
  const setLayoutMode = useUiStore((s) => s.setLayoutMode);
  const activeQuery = useActiveQuery();

  const [videoFilter, setVideoFilter] = useState("");
  const [minScore, setMinScore] = useState(0);

  if (activeQuery === undefined || activeQuery.task === "trake") return null;

  const predictions = activeQuery.predictions as FlatPrediction[];
  if (predictions.length === 0) {
    return <p className="p-6 text-sm text-dim">{emptyHint}</p>;
  }

  const filtered = predictions
    .map((p, idx) => ({ p, originalIndex: idx }))
    .filter(({ p }) => {
      if (videoFilter && !p.videoId.toLowerCase().includes(videoFilter.toLowerCase())) {
        return false;
      }
      if (minScore > 0 && p.score !== undefined && Number(p.score) < minScore) {
        return false;
      }
      return true;
    });

  // Group items by videoId and sort each group chronologically
  const videoGroups = (() => {
    const map = new Map<string, Array<{ p: FlatPrediction; originalIndex: number }>>();
    for (const item of filtered) {
      const list = map.get(item.p.videoId) ?? [];
      list.push(item);
      map.set(item.p.videoId, list);
    }
    return Array.from(map.entries()).map(([vid, items]) => ({
      videoId: vid,
      items: [...items].sort((a, b) => a.p.frameIndex - b.p.frameIndex),
      bestScore: Math.max(...items.map((i) => i.p.score ?? 0)),
    }));
  })();

  const gridClass =
    gridCols === 3
      ? "grid grid-cols-2 md:grid-cols-3 gap-2.5 p-3"
      : gridCols === 6
      ? "grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-1.5 p-2"
      : "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5 gap-2 p-3";

  return (
    <div className="flex flex-col">
      {/* Grid Toolbar: count, filter, layout & column zoom */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/60 bg-panel/40 px-3 py-1.5 text-xs text-dim">
        <div className="flex items-center gap-2">
          <span className="font-medium">
            {filtered.length}/{predictions.length} khung hình ({videoGroups.length} videos)
          </span>
          <input
            type="text"
            value={videoFilter}
            onChange={(e) => setVideoFilter(e.target.value)}
            placeholder="Lọc video ID (ví dụ: V014)…"
            aria-label="Lọc theo video ID"
            className="w-44 rounded border border-edge bg-panel px-2 py-0.5 text-[11px] text-ink outline-none focus:border-accent"
          />
          {videoFilter && (
            <button
              type="button"
              onClick={() => setVideoFilter("")}
              className="text-[10px] text-dim hover:text-ink"
            >
              ✕ Bỏ lọc
            </button>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Layout Mode Toggle */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setLayoutMode("flat")}
              className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                layoutMode === "flat"
                  ? "bg-accent font-medium text-black"
                  : "border border-edge/80 text-dim hover:text-ink"
              }`}
            >
              Lưới phẳng
            </button>
            <button
              type="button"
              onClick={() => setLayoutMode("grouped")}
              className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                layoutMode === "grouped"
                  ? "bg-accent font-medium text-black"
                  : "border border-edge/80 text-dim hover:text-ink"
              }`}
            >
              Gom theo Video
            </button>
          </div>

          {/* Min Score Filter */}
          <div className="flex items-center gap-1 border-l border-edge/60 pl-2">
            <span className="text-[11px] text-dim/80">Điểm:</span>
            {(
              [
                { score: 0, label: "Tất cả" },
                { score: 0.13, label: "≥ 0.13" },
                { score: 0.16, label: "≥ 0.16" },
              ] as const
            ).map(({ score, label }) => (
              <button
                key={label}
                type="button"
                onClick={() => setMinScore(score)}
                className={`rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                  minScore === score
                    ? "bg-accent/20 border border-accent text-accent font-semibold"
                    : "border border-edge/80 text-dim hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Column Zoom */}
          <div className="flex items-center gap-1 border-l border-edge/60 pl-2">
            <span className="text-[11px] text-dim/80">Lưới:</span>
            {(
              [
                { cols: 3, label: "Lớn (3)" },
                { cols: 4, label: "Vừa (4)" },
                { cols: 6, label: "Nhỏ (6)" },
              ] as const
            ).map(({ cols, label }) => (
              <button
                key={cols}
                type="button"
                onClick={() => setGridCols(cols)}
                className={`rounded px-2 py-0.5 text-[11px] transition-colors ${
                  gridCols === cols
                    ? "bg-accent font-medium text-black"
                    : "border border-edge/80 text-dim hover:border-dim hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Render based on layoutMode */}
      {layoutMode === "grouped" ? (
        <div className="flex flex-col divide-y divide-edge/50 p-3 gap-4">
          {videoGroups.map((group) => (
            <div key={group.videoId} className="flex flex-col gap-2 pt-2 first:pt-0">
              <div className="flex items-center justify-between bg-panel/60 px-3 py-1.5 rounded border border-edge/60">
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-bold text-ink flex items-center gap-1.5">
                    <span>🎬</span> {group.videoId}
                  </span>
                  <span className="rounded bg-edge/50 px-1.5 py-0.5 text-[10px] text-dim">
                    {group.items.length} frame tìm thấy
                  </span>
                  <span className="text-[11px] text-dim">
                    Điểm cao nhất: <strong className="text-accent">{formatScore(group.bestScore)}</strong>
                  </span>
                </div>
                {onOpenVideoBrowser && (
                  <button
                    type="button"
                    onClick={() => onOpenVideoBrowser(group.videoId)}
                    className="rounded border border-edge/80 bg-panel px-2 py-0.5 text-[11px] text-dim hover:border-accent hover:text-ink transition-colors flex items-center gap-1"
                  >
                    <span>🔍</span> Duyệt toàn bộ video này
                  </button>
                )}
              </div>

              <div className={gridClass}>
                {group.items.map(({ p: prediction, originalIndex }) => (
                  <ResultCard
                    key={`${prediction.videoId}-${prediction.frameIndex}`}
                    prediction={prediction}
                    index={originalIndex}
                    count={predictions.length}
                    task={activeQuery.task}
                    selected={
                      selected !== null &&
                      selected.videoId === prediction.videoId &&
                      selected.frameIndex === prediction.frameIndex
                    }
                    onSelect={(p) => selectFrame({ videoId: p.videoId, frameIndex: p.frameIndex, score: p.score })}
                    onRemove={removePrediction}
                    onReorder={reorderPrediction}
                    onAnswerChange={(i, answer) => setQaAnswer(activeQuery.id, i, answer)}
                    onSearchSimilar={onSearchSimilar}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className={gridClass}>
          {filtered.map(({ p: prediction, originalIndex }) => (
            <ResultCard
              key={`${prediction.videoId}-${prediction.frameIndex}`}
              prediction={prediction}
              index={originalIndex}
              count={predictions.length}
              task={activeQuery.task}
              selected={
                selected !== null &&
                selected.videoId === prediction.videoId &&
                selected.frameIndex === prediction.frameIndex
              }
              onSelect={(p) => selectFrame({ videoId: p.videoId, frameIndex: p.frameIndex, score: p.score })}
              onRemove={removePrediction}
              onReorder={reorderPrediction}
              onAnswerChange={(i, answer) => setQaAnswer(activeQuery.id, i, answer)}
              onSearchSimilar={onSearchSimilar}
            />
          ))}
        </div>
      )}
    </div>
  );
}
