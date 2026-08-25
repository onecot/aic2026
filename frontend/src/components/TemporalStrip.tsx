import { useEffect, useRef, useState } from "react";
import type { FrameRecord } from "../api/types";
import { getFrameUrl } from "../api/client";
import { formatSeconds } from "../lib/format";

interface TemporalStripProps {
  videoId: string;
  frames: FrameRecord[];
  current: FrameRecord | null;
  onSelect: (frame: FrameRecord) => void;
}

/**
 * Horizontal strip of indexed keyframes around the inspected frame.
 * Includes video playback simulation to inspect motion sequences.
 */
export function TemporalStrip({ videoId, frames, current, onSelect }: TemporalStripProps) {
  const currentRef = useRef<HTMLButtonElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    currentRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [current?.frame_index]);

  useEffect(() => {
    if (!isPlaying || frames.length === 0) return;
    const interval = setInterval(() => {
      const currentIndex = frames.findIndex((f) => f.frame_index === current?.frame_index);
      if (currentIndex === -1 || currentIndex >= frames.length - 1) {
        setIsPlaying(false);
      } else {
        const next = frames[currentIndex + 1];
        if (next) onSelect(next);
      }
    }, 450);
    return () => clearInterval(interval);
  }, [isPlaying, frames, current?.frame_index, onSelect]);

  const handleStep = (direction: -1 | 1) => {
    setIsPlaying(false);
    const currentIndex = frames.findIndex((f) => f.frame_index === current?.frame_index);
    if (currentIndex === -1) return;
    const next = frames[currentIndex + direction];
    if (next) onSelect(next);
  };

  return (
    <section aria-label="Temporal context" className="flex items-center gap-2 overflow-x-auto px-3 py-2 border-t border-edge/60 bg-panel/30">
      {/* Playback Simulation Toolbar */}
      <div className="flex shrink-0 items-center gap-1 border-r border-edge/60 pr-2">
        <button
          type="button"
          onClick={() => handleStep(-1)}
          title="Khung hình trước (◀)"
          className="rounded border border-edge bg-panel px-1.5 py-1 text-xs text-dim hover:text-ink hover:border-accent"
        >
          ◀
        </button>
        <button
          type="button"
          onClick={() => setIsPlaying(!isPlaying)}
          title={isPlaying ? "Dừng (Pause)" : "Phát mô phỏng video (Play)"}
          className={`rounded border px-2 py-1 text-xs font-semibold ${
            isPlaying
              ? "bg-amber-500 text-black border-amber-400"
              : "bg-panel text-ink border-edge hover:border-accent"
          }`}
        >
          {isPlaying ? "⏸ Dừng" : "▶ Phát"}
        </button>
        <button
          type="button"
          onClick={() => handleStep(1)}
          title="Khung hình sau (▶)"
          className="rounded border border-edge bg-panel px-1.5 py-1 text-xs text-dim hover:text-ink hover:border-accent"
        >
          ▶
        </button>
      </div>

      {frames.map((frame) => {
        const isCurrent = current !== null && frame.frame_index === current.frame_index;
        return (
          <button
            key={frame.frame_index}
            ref={isCurrent ? currentRef : undefined}
            type="button"
            onClick={() => {
              setIsPlaying(false);
              onSelect(frame);
            }}
            aria-current={isCurrent ? "true" : undefined}
            title={`frame ${frame.frame_index} · ${formatSeconds(frame.pts_time)}`}
            className={`relative shrink-0 overflow-hidden rounded border transition-all ${
              isCurrent
                ? "border-accent ring-2 ring-accent scale-105"
                : "border-edge opacity-75 hover:opacity-100"
            }`}
          >
            <img
              src={getFrameUrl(videoId, frame.frame_index)}
              alt={`frame ${frame.frame_index}`}
              loading="lazy"
              decoding="async"
              width={96}
              height={54}
              className="h-[54px] w-24 bg-black object-cover"
            />
            <span className="absolute bottom-0 right-0 bg-black/80 px-1 text-[9px] tabular-nums text-white">
              {formatSeconds(frame.pts_time)}
            </span>
          </button>
        );
      })}
    </section>
  );
}
