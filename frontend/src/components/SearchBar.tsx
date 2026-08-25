import { useRef } from "react";
import type { SearchMeta } from "../api/types";
import { formatMs } from "../lib/format";
import type { Task } from "../state/submission";
import { useActiveQuery, useUiStore } from "../state/uiStore";

const TASKS: { id: Task; label: string }[] = [
  { id: "kis", label: "KIS" },
  { id: "qa", label: "QA" },
  { id: "trake", label: "TRAKE" },
];

interface SearchBarProps {
  pending: boolean;
  meta: SearchMeta | null | undefined;
}

/** Task tabs, query input, submit button and the telemetry line. */
export function SearchBar({ pending, meta }: SearchBarProps) {
  const activeQuery = useActiveQuery();
  const task: Task = activeQuery?.task ?? "kis";
  const setTask = useUiStore((s) => s.setTask);
  const setDraftQuery = useUiStore((s) => s.setDraftQuery);
  const submitSearch = useUiStore((s) => s.submitSearch);
  const draftQuery = activeQuery?.draftQuery ?? "";
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <section aria-label="Search" className="border-b border-edge px-4 pb-2 pt-2">
      <div role="tablist" aria-label="Task" className="mb-2 flex gap-1">
        {TASKS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={task === id}
            onClick={() => setTask(id)}
            className={`rounded px-3 py-1 text-xs font-medium uppercase tracking-wide ${
              task === id
                ? "bg-accent text-black"
                : "border border-transparent text-dim hover:text-ink"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submitSearch();
        }}
      >
        <input
          id="query-input"
          ref={inputRef}
          value={draftQuery}
          onChange={(e) => setDraftQuery(e.target.value)}
          placeholder={task === "trake" ? "(1) first event… (2) second event…" : "Describe what you are looking for…"}
          spellCheck={false}
          aria-label="Query"
          className="min-w-0 flex-1 rounded border border-edge bg-panel px-3 py-1.5 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={pending || draftQuery.trim().length === 0}
          className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-black disabled:opacity-40"
        >
          {pending ? "Searching…" : "Search"}
        </button>
      </form>

      {meta && (
        <p className="mt-1 flex items-baseline gap-2 text-xs text-dim">
          <span>
            {meta.returned} {meta.task === "trake" ? "events" : "results"} · {formatMs(meta.timing.total_ms)}
          </span>
          <details className="ml-auto">
            <summary className="cursor-pointer select-none hover:text-ink">telemetry</summary>
            <span className="tabular-nums">
              encode {meta.timing.encode_ms !== null ? `${formatMs(meta.timing.encode_ms)}` : "—"}
              {" · retrieve "}
              {meta.timing.retrieve_ms !== null ? `${formatMs(meta.timing.retrieve_ms)}` : "—"}
              {" · task "}
              {meta.task}
            </span>
            <span className="ml-2 inline-flex gap-1">
              {Object.entries(meta.signals).map(([signal, active]) => (
                <span
                  key={signal}
                  title={`signal ${active ? "active" : "inactive"}: ${signal}`}
                  className={`rounded px-1 ${
                    active ? "bg-accent/20 text-[--color-accent]" : "bg-panel text-dim"
                  }`}
                >
                  {signal}
                </span>
              ))}
            </span>
          </details>
        </p>
      )}
    </section>
  );
}
