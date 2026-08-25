import { useState } from "react";
import type { SystemPayload } from "../api/types";
import { compactCount } from "../lib/format";

interface SystemStatusProps {
  system: SystemPayload | undefined;
}

function Flag({ value }: { value: boolean }) {
  return (
    <span className={value ? "text-[--color-accent]" : "text-dim"}>
      {value ? "on" : "off"}
    </span>
  );
}

function DetailRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-0.5">
      <span className="text-dim">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

/** Header: backend identity from /api/system plus a details toggle. */
export function SystemStatus({ system }: SystemStatusProps) {
  const [open, setOpen] = useState(false);
  if (system === undefined) {
    return (
      <header className="flex h-14 items-center border-b border-edge px-4 text-sm text-dim">
        AIC 2026 · loading system metadata…
      </header>
    );
  }
  const { retrieval, features, generation } = system;
  return (
    <header className="relative border-b border-edge px-4 py-2">
      <div className="flex items-center gap-3 text-sm">
        <span className="font-semibold tracking-wide">AIC 2026</span>
        <span className="inline-flex items-center gap-1.5 text-dim">
          <span aria-hidden className="inline-block size-2 rounded-full bg-accent" />
          READY
        </span>
        <span>{retrieval.model.name}</span>
        <span className="text-dim">
          {compactCount(retrieval.keyframes.total_keyframes)} frames
          {" · "}
          {retrieval.keyframes.total_videos} videos
          {" · "}
          {retrieval.keyframes.version ?? "unknown keyframe set"}
        </span>
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="ml-auto rounded border border-edge px-2 py-0.5 text-xs hover:border-accent"
        >
          System
        </button>
      </div>

      {open && (
        <div
          role="dialog"
          aria-label="System details"
          className="absolute right-4 top-full z-20 mt-1 w-80 space-y-3 rounded border border-edge bg-panel p-3 text-xs shadow-xl"
        >
          <section>
            <h3 className="mb-1 font-semibold uppercase tracking-wide text-dim">Retrieval</h3>
            <DetailRow label="model">{retrieval.model.name} ({retrieval.model.backend})</DetailRow>
            <DetailRow label="index id">{retrieval.index.id}</DetailRow>
            <DetailRow label="dimension">{retrieval.index.embedding_dimension}</DetailRow>
            <DetailRow label="similarity">{retrieval.index.similarity}</DetailRow>
            <DetailRow label="keyframes">{compactCount(retrieval.index.total_keyframes)}</DetailRow>
            <DetailRow label="videos">{retrieval.index.total_videos}</DetailRow>
          </section>
          <section>
            <h3 className="mb-1 font-semibold uppercase tracking-wide text-dim">Keyframes</h3>
            <DetailRow label="version">{retrieval.keyframes.version ?? "—"}</DetailRow>
            <DetailRow label="total">{retrieval.keyframes.total_keyframes}</DetailRow>
            <DetailRow label="videos">{retrieval.keyframes.total_videos}</DetailRow>
          </section>
          <section>
            <h3 className="mb-1 font-semibold uppercase tracking-wide text-dim">Features</h3>
            <DetailRow label="visual embedding"><Flag value={features.visual_embedding} /></DetailRow>
            <DetailRow label="ocr"><Flag value={features.ocr} /></DetailRow>
            <DetailRow label="asr"><Flag value={features.asr} /></DetailRow>
            <DetailRow label="metadata search"><Flag value={features.metadata_search} /></DetailRow>
            <DetailRow label="temporal navigation"><Flag value={features.temporal_navigation} /></DetailRow>
          </section>
          <section>
            <h3 className="mb-1 font-semibold uppercase tracking-wide text-dim">Generation</h3>
            <DetailRow label="vlm model">{generation.vlm.model ?? "—"}</DetailRow>
            <DetailRow label="state">
              {generation.vlm.enabled ? "enabled" : `disabled (${generation.vlm.load_policy})`}
            </DetailRow>
          </section>
        </div>
      )}
    </header>
  );
}
