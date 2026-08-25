import type { TraKeEvent, TraKeWindow } from "../../api/types";
import { getFrameUrl } from "../../api/client";
import { formatScore } from "../../lib/format";
import { predictionKey, suggestTraKeAnchors } from "../../state/submission";
import { useActiveQuery, useUiStore } from "../../state/uiStore";

interface WindowCardProps {
  event: TraKeEvent;
  events: TraKeEvent[];
  window: TraKeWindow;
  added: boolean;
}

/** One candidate window: representative frame + range + sequence add. */
export function TraKeWindowCard({ event, events, window: w, added }: WindowCardProps) {
  const selectFrame = useUiStore((s) => s.selectFrame);
  const addTraKeSuggestion = useUiStore((s) => s.addTraKeSuggestion);

  // frame_start is an indexed keyframe of this window, so its image URL is
  // always valid; the end frame is shown only as metadata.
  const representative = w.frame_start;

  return (
    <div
      className={`w-40 shrink-0 overflow-hidden rounded border ${
        added ? "border-accent" : "border-edge"
      }`}
    >
      <button
        type="button"
        title={`Inspect window ${w.video_id} ${w.frame_start}–${w.frame_end}`}
        onClick={() =>
          selectFrame({
            videoId: w.video_id,
            frameIndex: representative,
            rank: w.rank,
            score: w.score,
          })
        }
        className="block w-full text-left"
      >
        <img
          src={getFrameUrl(w.video_id, representative)}
          alt={`${event.event_id} window rank ${w.rank}: ${w.video_id} frames ${w.frame_start}–${w.frame_end}`}
          loading="lazy"
          decoding="async"
          className="aspect-video w-full bg-black object-cover"
        />
        <span className="flex items-baseline justify-between px-1.5 py-1 text-[11px] tabular-nums">
          <span>
            {w.frame_start}–{w.frame_end}
          </span>
          <span className="text-dim">{formatScore(w.score)}</span>
        </span>
      </button>
      <button
        type="button"
        disabled={added}
        title={
          added
            ? "Already in the queue"
            : "Store an editable initial suggestion (one slot per event)"
        }
        onClick={() => addTraKeSuggestion(w, events)}
        className={`w-full py-1 text-center text-xs ${
          added
            ? "bg-accent/15 font-medium text-accent"
            : "border-t border-edge text-dim hover:text-ink"
        }`}
      >
        {added ? "Added ✓" : "Add prediction"}
      </button>
    </div>
  );
}

/** Event-centric result view for the TRAKE task (replaces the flat grid). */
export function TrakeResults({ events }: { events: TraKeEvent[] }) {
  const activeQuery = useActiveQuery();

  if (events.length === 0) {
    return <p className="p-6 text-sm text-dim">No events returned.</p>;
  }
  return (
    <div className="space-y-3 p-3">
      <p className="text-xs text-dim">
        {events.length} requested events · “Add prediction” stores an
        editable initial suggestion; set or replace each event anchor from
        the queue via the inspector.
      </p>
      {events.map((event) => (
        <section key={event.event_id} aria-label={event.event_text}>
          <h3 className="mb-1 text-xs uppercase tracking-wide text-dim">
            Event {event.event_id.replace(/^E/i, "")} · {event.event_text}
          </h3>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {event.windows.map((w) => {
              // Prospective suggestion for this window (used only to mark
              // already-added duplicates); never stored as final data.
              const slots = suggestTraKeAnchors(
                w.video_id,
                events,
                activeQuery?.eventCount ?? events.length,
              );
              const added =
                activeQuery !== undefined &&
                activeQuery.predictions.some(
                  (p) =>
                    predictionKey("trake", p) ===
                    `${w.video_id}::${slots.map((v) => (v === null ? "_" : String(v))).join(",")}`,
                );
              return (
                <TraKeWindowCard
                  key={`${event.event_id}-${w.rank}-${w.video_id}-${w.frame_start}`}
                  event={event}
                  events={events}
                  window={w}
                  added={added}
                />
              );
            })}
            {event.windows.length === 0 && (
              <span className="text-xs text-dim">No windows.</span>
            )}
          </div>
        </section>
      ))}
    </div>
  );
}
