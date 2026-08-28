import { useEffect } from "react";
import {
  MAX_PREDICTIONS,
  ANSWER_MAX_LENGTH,
  predictionIssue,
  statusGlyph,
  type QueryWorkspace,
} from "../../state/submission";
import { useActiveQuery, useUiStore } from "../../state/uiStore";

function formatClock(ms: number): string {
  return new Date(ms).toLocaleTimeString();
}

/** Persistent bottom panel: the active query's prediction queue. */
export function PredictionQueue() {
  const query = useActiveQuery();
  const savedAt = useUiStore((s) => s.savedAt);
  const selectFrame = useUiStore((s) => s.selectFrame);
  const removePrediction = useUiStore((s) => s.removePrediction);
  const movePrediction = useUiStore((s) => s.movePrediction);
  const setQaAnswer = useUiStore((s) => s.setQaAnswer);
  const pick = useUiStore((s) => s.trakePick);
  const chooseTraKeAnchor = useUiStore((s) => s.chooseTraKeAnchor);
  const setTraKeAnchor = useUiStore((s) => s.setTraKeAnchor);

  // Auto-clear transient notices (limit reached, duplicate add).
  // Visual display is via global Toast; timer still clears the store so
  // subsequent identical notices can be re-triggered.
  const notice = useUiStore((s) => s.notice);
  useEffect(() => {
    if (notice === null) return;
    const timer = window.setTimeout(() => {
      useUiStore.setState({ notice: null });
    }, 4000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  if (query === undefined) return null;
  const count = query.predictions.length;

  return (
    <section aria-label="Prediction queue" className="max-h-[38vh] overflow-y-auto">
      <header className="flex items-baseline gap-3 border-b border-edge px-4 py-1.5 text-xs">
        <span className="font-semibold tracking-wide">
          {query.id} · {query.task.toUpperCase()}
        </span>
        <span className="text-dim tabular-nums">
          {count} / {MAX_PREDICTIONS} predictions
        </span>
        {savedAt !== null && (
          <span className="ml-auto text-dim" role="status">
            Saved · last saved {formatClock(savedAt)}
          </span>
        )}
      </header>

      {count === 0 ? (
        <p className="px-4 py-2 text-xs text-dim">
          Queue empty — inspect candidates and press “Add prediction”.
        </p>
      ) : (
        <ol className="px-4 py-1 text-xs">
          {query.predictions.map((prediction, index) => {
            const issue = predictionIssue(query.task, prediction, query.eventCount);
            return (
              <li
                key={`${query.id}-${index}`}
                className={`flex items-start gap-2 border-b border-edge/50 py-1 last:border-0 ${
                  issue !== null ? "text-red-400" : ""
                }`}
              >
                <span className="w-5 shrink-0 text-right text-dim tabular-nums">{index + 1}</span>

                {query.task !== "trake" && "frameIndex" in prediction && (
                  <button
                    type="button"
                    onClick={() =>
                      selectFrame({
                        videoId: prediction.videoId,
                        frameIndex: prediction.frameIndex,
                      })
                    }
                    className="tabular-nums hover:text-accent"
                    title="Inspect this prediction"
                  >
                    {prediction.videoId} / {prediction.frameIndex}
                  </button>
                )}
                {query.task === "trake" && "frameIndices" in prediction && (
                  <span className="flex min-w-0 flex-1 flex-col gap-1">
                    <span className="tabular-nums">{prediction.videoId}</span>
                    {prediction.frameIndices.map((value, eventIdx) => {
                      const picking =
                        pick !== null &&
                        pick.predictionIndex === index &&
                        pick.eventIndex === eventIdx;
                      return (
                        <span key={eventIdx} className="flex items-center gap-1.5">
                          <span className="text-dim">Event {eventIdx + 1}</span>
                          {value !== null ? (
                            <button
                              type="button"
                              onClick={() =>
                                selectFrame({ videoId: prediction.videoId, frameIndex: value })
                              }
                              title={`Inspect anchor ${value}`}
                              className="tabular-nums hover:text-accent"
                            >
                              {value}
                            </button>
                          ) : (
                            <span className="text-dim">— unset</span>
                          )}
                          <button
                            type="button"
                            onClick={() => chooseTraKeAnchor(index, eventIdx)}
                            aria-label={`Choose frame for ${prediction.videoId} event ${eventIdx + 1}`}
                            className={`rounded border px-1 ${
                              picking
                                ? "border-accent text-accent"
                                : "border-edge text-dim hover:text-ink"
                            }`}
                          >
                            {value !== null ? "Change" : "Choose"}
                          </button>
                          {value !== null && (
                            <button
                              type="button"
                              onClick={() => setTraKeAnchor(query.id, index, eventIdx, null)}
                              aria-label={`Reset ${prediction.videoId} event ${eventIdx + 1}`}
                              className="text-dim hover:text-red-400"
                            >
                              ✕
                            </button>
                          )}
                          {picking && (
                            <span className="text-accent">
                              ← inspect a frame, then “Set as Event {eventIdx + 1}”
                            </span>
                          )}
                        </span>
                      );
                    })}
                    {issue !== null && <span>— {issue}</span>}
                  </span>
                )}

                {query.task === "qa" && "answer" in prediction && (
                  <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <input
                      value={prediction.answer}
                      onChange={(e) => setQaAnswer(query.id, index, e.target.value)}
                      placeholder="answer…"
                      aria-label={`Answer for prediction ${index + 1}`}
                      spellCheck={false}
                      className={`w-full max-w-xs rounded border bg-panel px-1.5 py-0.5 outline-none focus:border-accent ${
                        prediction.answer.length > ANSWER_MAX_LENGTH
                          ? "border-red-500"
                          : "border-edge"
                      }`}
                    />
                    <span
                      className={`tabular-nums ${
                        prediction.answer.length > ANSWER_MAX_LENGTH ? "text-red-400" : "text-dim"
                      }`}
                    >
                      {prediction.answer.length}/{ANSWER_MAX_LENGTH}
                      {issue !== null ? ` — ${issue}` : ""}
                    </span>
                  </span>
                )}

                <span className="ml-auto flex shrink-0 items-center gap-1 text-dim">
                  <button
                    type="button"
                    onClick={() => movePrediction(index, -1)}
                    disabled={index === 0}
                    aria-label={`Move prediction ${index + 1} up`}
                    className="px-0.5 hover:text-ink disabled:opacity-30"
                  >
                    ▲
                  </button>
                  <button
                    type="button"
                    onClick={() => movePrediction(index, 1)}
                    disabled={index === count - 1}
                    aria-label={`Move prediction ${index + 1} down`}
                    className="px-0.5 hover:text-ink disabled:opacity-30"
                  >
                    ▼
                  </button>
                  <button
                    type="button"
                    onClick={() => removePrediction(index)}
                    aria-label={`Remove prediction ${index + 1}`}
                    className="rounded border border-edge px-1.5 hover:border-red-500 hover:text-red-400"
                  >
                    Remove
                  </button>
                </span>
              </li>
            );
          })}
        </ol>
      )}
      <p className="px-4 pb-1 text-[10px] text-dim">
        status {statusGlyph(count === 0 ? "empty" : issueFree(query) ? "ready" : "in_progress")} · order
        is user-controlled (move ▲▼); BTC allows at most {MAX_PREDICTIONS} rows per query
      </p>
    </section>
  );
}

function issueFree(query: QueryWorkspace): boolean {
  return query.predictions.every(
    (p) => predictionIssue(query.task, p, query.eventCount) === null,
  );
}
