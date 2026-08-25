import { useEffect, useState } from "react";
import { useUiStore } from "../state/uiStore";

export function Toast() {
  const notice = useUiStore((s) => s.notice);
  const [visible, setVisible] = useState(false);
  const [currentNotice, setCurrentNotice] = useState<{ message: string; at: number } | null>(null);

  useEffect(() => {
    if (notice) {
      setCurrentNotice(notice);
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [notice]);

  if (!visible || !currentNotice) return null;

  return (
    <aside
      aria-live="polite"
      aria-label="Notification"
      className="fixed bottom-5 right-5 z-50 flex items-center gap-2.5 rounded-lg border border-accent/40 bg-panel/95 px-4 py-2.5 text-xs text-ink shadow-2xl backdrop-blur-md transition-all duration-300 animate-in fade-in slide-in-from-bottom-3"
    >
      <span className="flex h-2 w-2 rounded-full bg-accent animate-pulse" />
      <span>{currentNotice.message}</span>
      <button
        type="button"
        onClick={() => setVisible(false)}
        aria-label="Close notification"
        className="ml-2 text-dim hover:text-ink text-sm"
      >
        ✕
      </button>
    </aside>
  );
}
