import { useEffect, useRef } from "react";
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
 * The backend returns a small window (center/radius), so the browser only
 * loads nearby thumbnails.
 */
export function TemporalStrip({ videoId, frames, current, onSelect }: TemporalStripProps) {
  const currentRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    currentRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [current?.frame_index]);

  return (
    <section aria-label="Temporal context" className="flex items-stretch gap-2 overflow-x-auto px-3 py-2">
      {frames.map((frame) => {
        const isCurrent = current !== null && frame.frame_index === current.frame_index;
        return (
          <button
            key={frame.frame_index}
            ref={isCurrent ? currentRef : undefined}
            type="button"
            onClick={() => onSelect(frame)}
            aria-current={isCurrent ? "true" : undefined}
            title={`frame ${frame.frame_index} · ${formatSeconds(frame.pts_time)}`}
            className={`relative shrink-0 overflow-hidden rounded border ${
              isCurrent
                ? "border-accent ring-2 ring-accent"
                : "border-edge opacity-80 hover:opacity-100"
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
            <span className="absolute bottom-0 right-0 bg-black/75 px-1 text-[10px] tabular-nums">
              {formatSeconds(frame.pts_time)}
            </span>
          </button>
        );
      })}
    </section>
  );
}
