import { useState } from "react";
import {
  predictionIssue,
  queryStatus,
  statusGlyph,
  validateQueries,
  type QueryWorkspace,
  type Task,
} from "../../state/submission";
import { useUiStore } from "../../state/uiStore";

const TASKS: Task[] = ["kis", "qa", "trake"];

function statusOf(query: QueryWorkspace) {
  return queryStatus(query.task, query.predictions, query.eventCount);
}

function isIncomplete(query: QueryWorkspace): boolean {
  if (query.predictions.length === 0) return true;
  return query.predictions.some(
    (p) => predictionIssue(query.task, p, query.eventCount) !== null,
  );
}

/**
 * Submission workspace: cross-query summary, validation, and jump-to-issue
 * navigation. Export is intentionally absent — the backend does not expose
 * a submission endpoint yet.
 */
export function SubmissionView() {
  const queries = useUiStore((s) => s.queries);
  const queryOrder = useUiStore((s) => s.queryOrder);
  const setActiveQuery = useUiStore((s) => s.setActiveQuery);
  const setView = useUiStore((s) => s.setView);
  const selectFrame = useUiStore((s) => s.selectFrame);
  const [result, setResult] = useState<ReturnType<typeof validateQueries> | null>(null);

  const ordered: QueryWorkspace[] = queryOrder.flatMap((id) => {
    const q = queries[id];
    return q !== undefined ? [q] : [];
  });

  /** Jump back into a query; row-specific issues also open the inspector. */
  const jumpToQuery = (queryId: string, issueIndex?: number) => {
    const query = queries[queryId];
    if (query === undefined) return;
    setActiveQuery(queryId);
    setView("workspace");
    if (issueIndex !== undefined) {
      const prediction = query.predictions[issueIndex];
      if (prediction !== undefined) {
        const firstAnchor =
          "frameIndices" in prediction
            ? (prediction.frameIndices.find((v) => v !== null) ?? 0)
            : prediction.frameIndex;
        selectFrame({
          videoId: prediction.videoId,
          frameIndex: firstAnchor,
        });
      }
    }
  };

  return (
    <section aria-label="Submission workspace" className="mx-auto max-w-3xl space-y-4 p-4 text-sm">
      <header className="flex items-center gap-3">
        <h2 className="text-base font-semibold">Submission</h2>
        <button
          type="button"
          onClick={() => {
            const target = ordered.find(isIncomplete);
            if (target !== undefined) jumpToQuery(target.id);
          }}
          className="rounded border border-edge px-2 py-0.5 text-xs text-dim hover:border-accent hover:text-ink"
        >
          Review incomplete
        </button>
        <button
          type="button"
          onClick={() => setResult(validateQueries(ordered))}
          className="rounded bg-accent px-2.5 py-0.5 text-xs font-medium text-black"
        >
          Validate submission
        </button>
      </header>

      <div className="space-y-1">
        {TASKS.filter((task) => ordered.some((q) => q.task === task)).map((task) => {
          const group = ordered.filter((q) => q.task === task);
          const ready = group.filter((q) => statusOf(q) === "ready").length;
          return (
            <p key={task} className="text-xs text-dim">
              <span className="uppercase tracking-wide text-ink">{task}</span>{" "}
              {ready} / {group.length} queries ready
            </p>
          );
        })}
      </div>

      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-dim">
            <th className="py-1 font-normal">query</th>
            <th className="font-normal">task</th>
            <th className="font-normal">predictions</th>
            <th className="font-normal">status</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((query) => {
            const status = statusOf(query);
            return (
              <tr key={query.id} className="border-t border-edge/60">
                <td className="py-1 tabular-nums">{query.id}</td>
                <td className="uppercase">{query.task}</td>
                <td className="tabular-nums">{query.predictions.length}</td>
                <td>
                  <button
                    type="button"
                    onClick={() => jumpToQuery(query.id)}
                    className="hover:text-accent"
                    title={`Open ${query.id}`}
                  >
                    <span aria-hidden>{statusGlyph(status)}</span> {status}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {result !== null && (
        <div className="space-y-2 rounded border border-edge bg-panel p-3 text-xs" role="status">
          <h3 className="text-sm font-semibold">Submission validation</h3>
          {TASKS.filter((task) => result.startedByTask[task] > 0).map((task) => (
            <div key={task}>
              <p className="uppercase tracking-wide text-dim">{task}</p>
              <p>
                ✓ {result.readyByTask[task]} / {result.startedByTask[task]} queries valid
              </p>
              {result.issues
                .filter((issue) => issue.task === task)
                .map((issue, i) => (
                  <button
                    key={`${issue.queryId}-${i}`}
                    type="button"
                    onClick={() =>
                      jumpToQuery(issue.queryId, issue.predictionIndex)
                    }
                    className={`block text-left ${
                      issue.severity === "error" ? "text-red-400" : "text-yellow-500"
                    } hover:text-accent`}
                  >
                    {issue.severity === "error" ? "✗" : "⚠"} {issue.queryId}{" "}
                    {issue.message}
                  </button>
                ))}
            </div>
          ))}
          <p className={result.blocked ? "text-red-400" : "text-accent"}>
            STATUS: {result.blocked ? "BLOCKED" : "VALID (not exported)"}
          </p>
          <p className="text-dim">
            Export is not available yet — the backend does not expose a
            submission endpoint.
          </p>
        </div>
      )}
    </section>
  );
}
