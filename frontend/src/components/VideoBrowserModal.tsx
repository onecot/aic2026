import { useState } from "react";
import { getFrameUrl } from "../api/client";
import { formatSeconds } from "../lib/format";
import { useTemporalFrames, useVideo } from "../hooks/queries";
import { useUiStore } from "../state/uiStore";

interface VideoBrowserModalProps {
  videoId: string | null;
  onClose: () => void;
}

export function VideoBrowserModal({ videoId, onClose }: VideoBrowserModalProps) {
  const selectFrame = useUiStore((s) => s.selectFrame);
  const notify = useUiStore((s) => s.notify);

  const videoQuery = useVideo(videoId);
  const framesQuery = useTemporalFrames(
    videoId,
    videoId ? { offset: 0, limit: 300 } : null,
  );

  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null);

  if (!videoId) return null;

  const frames = framesQuery.data?.frames ?? [];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="video-browser-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4"
      onClick={onClose}
    >
      <div
        className="flex h-[85vh] w-full max-w-5xl flex-col rounded-xl border border-edge bg-panel shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-edge/60 px-5 py-3 bg-panel/80">
          <div>
            <h2 id="video-browser-title" className="text-sm font-bold text-ink flex items-center gap-2">
              <span>🎬</span> Browse Video: <span className="font-mono text-accent">{videoId}</span>
            </h2>
            <p className="text-xs text-dim">
              {frames.length} indexed keyframes · range {videoQuery.data?.first_frame_index ?? 0} – {videoQuery.data?.last_frame_index ?? "—"}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-dim hover:text-ink text-base"
          >
            ✕
          </button>
        </div>

        {/* Frames Grid */}
        <div className="flex-1 overflow-y-auto p-4">
          {framesQuery.isLoading ? (
            <div className="flex h-full items-center justify-center text-sm text-dim">
              Loading keyframes...
            </div>
          ) : frames.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-dim">
              No keyframes found for this video.
            </div>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
              {frames.map((frame) => {
                const isSelected = selectedFrameIndex === frame.frame_index;
                return (
                  <button
                    key={frame.frame_index}
                    type="button"
                    onClick={() => {
                      setSelectedFrameIndex(frame.frame_index);
                      selectFrame({
                        videoId,
                        frameIndex: frame.frame_index,
                        keyframeOrdinal: frame.keyframe_ordinal,
                        ptsTime: frame.pts_time,
                      });
                      notify(`Selected frame ${frame.frame_index} of ${videoId}`);
                    }}
                    className={`group relative flex flex-col overflow-hidden rounded border text-left transition-all ${
                      isSelected
                        ? "border-accent ring-2 ring-accent scale-105 shadow-lg shadow-accent/20"
                        : "border-edge hover:border-accent hover:opacity-100 opacity-85"
                    }`}
                  >
                    <img
                      src={getFrameUrl(videoId, frame.frame_index)}
                      alt={`frame ${frame.frame_index}`}
                      loading="lazy"
                      className="aspect-video w-full bg-black object-cover"
                    />
                    <div className="flex items-center justify-between bg-panel/90 px-1.5 py-1 text-[10px] tabular-nums">
                      <span className="font-mono font-medium text-ink">#{frame.frame_index}</span>
                      <span className="text-dim">{formatSeconds(frame.pts_time)}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-edge/60 px-5 py-2.5 bg-panel/60 text-xs">
          <span className="text-dim">
            Click any frame to inspect it in the main workspace.
          </span>
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-accent px-4 py-1.5 text-xs font-semibold text-black hover:opacity-90 transition-opacity"
          >
            Close (Esc)
          </button>
        </div>
      </div>
    </div>
  );
}
