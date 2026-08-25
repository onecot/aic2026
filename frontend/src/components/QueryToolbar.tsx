import { statusGlyph, queryStatus } from "../state/submission";
import { useActiveQuery, useUiStore } from "../state/uiStore";

/**
 * Answers "which query am I working on?" and moves between views.
 * The task tabs live in SearchBar; this bar handles exact query switching,
 * creating queries, and Workspace/Submission navigation.
 */
export function QueryToolbar() {
  const active = useActiveQuery();
  const queries = useUiStore((s) => s.queries);
  const queryOrder = useUiStore((s) => s.queryOrder);
  const view = useUiStore((s) => s.view);
  const setActiveQuery = useUiStore((s) => s.setActiveQuery);
  const newQuery = useUiStore((s) => s.newQuery);
  const setView = useUiStore((s) => s.setView);

  return (
    <div className="flex items-center gap-2 border-b border-edge px-4 py-1.5 text-sm">
      <select
        aria-label="Active query"
        value={active?.id ?? ""}
        onChange={(e) => setActiveQuery(e.target.value)}
        className="rounded border border-edge bg-panel px-2 py-0.5 text-xs outline-none focus:border-accent"
      >
        {queryOrder.length === 0 && <option value="">no query</option>}
        {queryOrder.map((id) => {
          const q = queries[id]!;
          const status = queryStatus(q.task, q.predictions, q.eventCount);
          return (
            <option key={id} value={id}>
              {id} · {q.task.toUpperCase()} · {statusGlyph(status)} {q.predictions.length}
            </option>
          );
        })}
      </select>
      <button
        type="button"
        onClick={() => newQuery(active?.task ?? "kis")}
        className="rounded border border-edge px-2 py-0.5 text-xs text-dim hover:border-accent hover:text-ink"
      >
        + New query
      </button>

      <div className="ml-auto flex gap-1" role="tablist" aria-label="View">
        {(["workspace", "submission"] as const).map((v) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            onClick={() => setView(v)}
            className={`rounded px-2 py-0.5 text-xs capitalize ${
              view === v ? "bg-accent font-medium text-black" : "text-dim hover:text-ink"
            }`}
          >
            {v}
          </button>
        ))}
      </div>
    </div>
  );
}
