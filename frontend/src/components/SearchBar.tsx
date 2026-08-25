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
        <div className="relative min-w-0 flex-1">
          <input
            id="query-input"
            ref={inputRef}
            value={draftQuery}
            onChange={(e) => setDraftQuery(e.target.value)}
            placeholder={
              task === "trake"
                ? "Ví dụ: (1) người đi bộ (2) xe máy rẽ phải…"
                : task === "qa"
                ? "Ví dụ: người đang làm gì? có bao nhiêu xe?…"
                : "Nhập nội dung cần tìm kiếm (nhấn / để focus)…"
            }
            spellCheck={false}
            aria-label="Query"
            className="w-full rounded border border-edge bg-panel px-3 py-1.5 pr-8 text-sm outline-none focus:border-accent"
          />
          {draftQuery.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setDraftQuery("");
                inputRef.current?.focus();
              }}
              title="Xóa nhanh (Clear)"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-dim hover:text-ink text-sm px-1 py-0.5"
            >
              ✕
            </button>
          )}
        </div>
        <button
          type="submit"
          disabled={pending || draftQuery.trim().length === 0}
          className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-black disabled:opacity-40 hover:opacity-90 transition-opacity"
        >
          {pending ? "Đang tìm…" : "Search"}
        </button>
      </form>

      {/* Quick Example Chips */}
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-dim">
        <span className="text-dim/80">Gợi ý mẫu:</span>
        {(task === "trake"
          ? [
              "(1) người đi bộ (2) xe máy dừng lại",
              "(1) xe ô tô chạy (2) người qua đường",
              "(1) học sinh chào cờ (2) học sinh vào lớp",
            ]
          : task === "qa"
          ? [
              "người đang làm gì?",
              "có bao nhiêu chiếc xe?",
              "người mặc áo màu gì?",
              "địa điểm này ở đâu?",
            ]
          : [
              "người phụ nữ mặc áo dài",
              "xe ô tô trên đường phố",
              "học sinh trong lớp học",
              "ngôi nhà và cây xanh",
            ]
        ).map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setDraftQuery(example);
              inputRef.current?.focus();
            }}
            className="rounded border border-edge/80 bg-panel/70 px-2 py-0.5 text-dim hover:border-accent hover:text-ink transition-colors"
          >
            {example}
          </button>
        ))}
      </div>

      {meta && (
        <div className="mt-1 flex items-baseline gap-2 text-xs text-dim">
          <span>
            {meta.returned} {meta.task === "trake" ? "sự kiện" : "kết quả"} · {formatMs(meta.timing.total_ms)}
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
        </div>
      )}
    </section>
  );
}
