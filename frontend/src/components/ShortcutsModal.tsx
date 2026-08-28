import { useEffect } from "react";

interface ShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SHORTCUTS = [
  { key: "/", label: "Focus search", desc: "Jump cursor to search input" },
  { key: "j / ↓", label: "Next frame", desc: "Select next keyframe card" },
  { key: "k / ↑", label: "Prev frame", desc: "Select previous keyframe card" },
  { key: "◀ / ▶", label: "Timeline", desc: "Step temporal neighbor frames" },
  { key: "Delete", label: "Remove", desc: "Remove selected frame from queue (KIS/QA)" },
  { key: "Esc", label: "Close / Clear", desc: "Close inspector or browser, clear selection" },
  { key: "?", label: "Shortcuts", desc: "Toggle this help" },
];

export function ShortcutsModal({ isOpen, onClose }: ShortcutsModalProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="shortcuts-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-xl border border-edge bg-panel p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-edge/60 pb-3">
          <h2 id="shortcuts-title" className="text-sm font-bold text-ink flex items-center gap-2">
            <span>⌨️</span> Keyboard Shortcuts
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-dim hover:text-ink text-sm"
          >
            ✕
          </button>
        </div>

        <div className="mt-3 flex flex-col divide-y divide-edge/40 text-xs">
          {SHORTCUTS.map(({ key, label, desc }) => (
            <div key={key} className="flex items-center justify-between py-2">
              <div className="flex flex-col">
                <span className="font-semibold text-ink">{label}</span>
                <span className="text-dim text-[11px]">{desc}</span>
              </div>
              <kbd className="rounded border border-edge bg-black/40 px-2 py-0.5 font-mono text-[11px] text-accent font-medium shadow-inner">
                {key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-accent px-4 py-1.5 text-xs font-semibold text-black hover:opacity-90 transition-opacity"
          >
            Got it (Esc)
          </button>
        </div>
      </div>
    </div>
  );
}
