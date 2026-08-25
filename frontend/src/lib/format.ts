/** Compact display helpers for dense UI text. */

/** 148322 -> "148K"; 873 -> "873". */
export function compactCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 10_000) return `${Math.round(value / 1000)}K`;
  return String(value);
}

/** Milliseconds -> short human string: 399.6 -> "400 ms". */
export function formatMs(ms: number): string {
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms)} ms`;
}

/** Presentation-time seconds -> "mm:ss.d" (e.g. 63.52 -> "1:03.5"). */
export function formatSeconds(seconds: number): string {
  const total = Math.max(0, seconds);
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

/** Score -> fixed 3-decimals string without sign noise. */
export function formatScore(score?: number | null): string {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return "-";
  return Number(score).toFixed(3);
}
