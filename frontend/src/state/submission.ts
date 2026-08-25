/**
 * Submission-domain model: query workspaces, prediction queues, statuses,
 * and validation rules for the BTC submission format.
 *
 * BTC contracts enforced here:
 *   KIS    row = <video_id>, <frame_id>              (max 100 rows/query)
 *   QA     row = <video_id>, <frame_id>, <answer>    (max 100 rows/query, answer <= 100 chars)
 *   TRAKE  row = <video_id>, <frame_id_1>..N         (N == requested events, temporal order)
 *
 * Pure functions only — no framework or storage dependencies.
 */

export const MAX_PREDICTIONS = 100;
export const ANSWER_MAX_LENGTH = 100;

export type Task = "kis" | "qa" | "trake";

/** KIS prediction: one ranked frame. */
export interface KisPrediction {
  videoId: string;
  frameIndex: number;
}

/** QA prediction: one frame plus its own independent answer. */
export interface QaPrediction {
  videoId: string;
  frameIndex: number;
  answer: string;
}

/**
 * TRAKE prediction: one event alignment for a single video.
 *
 * `frameIndices` is POSITIONAL: slot i holds the anchor frame for event
 * i+1; `null` means that event's frame has not been chosen yet. The array
 * always has exactly N slots once created (N = requested event count), so
 * incomplete alignments are represented explicitly instead of being
 * silently shortened or invented.
 */
export interface TrakePrediction {
  videoId: string;
  frameIndices: (number | null)[];
}

export type Prediction = KisPrediction | QaPrediction | TrakePrediction;

/** Per-query working state (the unit the user switches between). */
export interface QueryWorkspace {
  id: string;
  task: Task;
  draftQuery: string;
  lastSearch: { query: string; limit: number } | null;
  /** Requested event count N for TRAKE, captured from the search response. */
  eventCount: number | null;
  /** Queue in user order (order added; move up/down allowed). */
  predictions: Prediction[];
}

export type QueryStatus = "empty" | "in_progress" | "ready" | "error";

/* ------------------------------------------------------------------ keys */

/** Stable identity used for duplicate detection and "Added" badges. */
export function predictionKey(task: Task, p: Prediction): string {
  if (task === "trake") {
    const t = p as TrakePrediction;
    return `${t.videoId}::${t.frameIndices.map((v) => (v === null ? "_" : String(v))).join(",")}`;
  }
  const k = p as KisPrediction;
  return `${k.videoId}::${k.frameIndex}`;
}

/**
 * Initial TRAKE anchor SUGGESTION for one window: the best candidate of
 * each requested event inside that video, placed at its own event slot.
 *
 * This is explicitly NOT a final prediction:
 * - events without a candidate in this video leave their slot `null`;
 * - no fallback frames are invented;
 * - slots are never reordered or deduplicated — whatever the retrieval
 *   ranking says per event is what lands in that event's slot, and
 *   validation flags non-monotonic suggestions honestly.
 */
export function suggestTraKeAnchors(
  videoId: string,
  events: Array<{ predictions: Array<{ video_id: string; frame_index: number }> }>,
  eventCount: number,
): (number | null)[] {
  if (!Number.isInteger(eventCount) || eventCount < 1) return [];
  const slots: (number | null)[] = Array.from({ length: eventCount }, () => null);
  for (let i = 0; i < Math.min(eventCount, events.length); i += 1) {
    const match = events[i]!.predictions.find((p) => p.video_id === videoId);
    if (match !== undefined) slots[i] = match.frame_index;
  }
  return slots;
}

/* --------------------------------------------------------------- validity */

/** Per-row validation message, or null when the row is valid. */
export function predictionIssue(
  task: Task,
  p: Prediction,
  eventCount: number | null,
): string | null {
  if (task === "qa") {
    const q = p as QaPrediction;
    const answer = q.answer.trim();
    if (answer.length === 0) return "missing answer";
    if (q.answer.length > ANSWER_MAX_LENGTH)
      return `answer longer than ${ANSWER_MAX_LENGTH} characters`;
    return null;
  }
  if (task === "trake") {
    const t = p as TrakePrediction;
    if (eventCount === null) return "event count unknown (run the TRAKE search)";
    if (t.frameIndices.length !== eventCount)
      return `${t.frameIndices.length}/${eventCount} event frames`;
    const chosen = t.frameIndices.filter((v) => v !== null).length;
    if (chosen < eventCount) return `${chosen}/${eventCount} event frames`;
    for (let i = 1; i < t.frameIndices.length; i += 1) {
      const prev = t.frameIndices[i - 1];
      const cur = t.frameIndices[i];
      // Post-guard: every slot is set when `chosen === eventCount`.
      if (prev === null || cur === null || cur <= prev)
        return "event frames not in temporal order";
    }
    return null;
  }
  return null; // KIS rows are valid by construction
}

/** Coarse queue status for list views. */
export function queryStatus(
  task: Task,
  predictions: Prediction[],
  eventCount: number | null,
): QueryStatus {
  if (predictions.length === 0) return "empty";
  if (predictions.length > MAX_PREDICTIONS) return "error";
  const invalid = predictions.some(
    (p) => predictionIssue(task, p, eventCount) !== null,
  );
  return invalid ? "in_progress" : "ready";
}

export function statusGlyph(status: QueryStatus): string {
  switch (status) {
    case "ready":
      return "✓";
    case "in_progress":
      return "⚠";
    case "error":
      return "✗";
    default:
      return "·";
  }
}

/* ------------------------------------------------------------- validation */

export interface SessionIssue {
  queryId: string;
  task: Task;
  /** Prediction index when the issue is row-specific. */
  predictionIndex?: number;
  message: string;
  severity: "warning" | "error";
}

export interface ValidationResult {
  issues: SessionIssue[];
  /** True when submission must not proceed. */
  blocked: boolean;
  readyByTask: Record<Task, number>;
  startedByTask: Record<Task, number>;
}

const emptyCounts = (): Record<Task, number> => ({ kis: 0, qa: 0, trake: 0 });

/**
 * Validate all queries before submission.
 *
 * Blocked when any structural error exists (>100 rows, unknown TRAKE event
 * count). Empty queries and incomplete rows are warnings only — BTC sets a
 * maximum per query, not a minimum.
 */
export function validateQueries(queries: QueryWorkspace[]): ValidationResult {
  const issues: SessionIssue[] = [];
  const readyByTask = emptyCounts();
  const startedByTask = emptyCounts();

  for (const query of queries) {
    startedByTask[query.task] += 1;
    const status = queryStatus(query.task, query.predictions, query.eventCount);

    if (status === "empty") {
      issues.push({
        queryId: query.id,
        task: query.task,
        message: "no predictions",
        severity: "warning",
      });
      continue;
    }

    if (query.predictions.length > MAX_PREDICTIONS) {
      issues.push({
        queryId: query.id,
        task: query.task,
        message: `${query.predictions.length} rows exceed the ${MAX_PREDICTIONS}-row maximum`,
        severity: "error",
      });
      continue;
    }

    let rowsValid = true;
    query.predictions.forEach((p, index) => {
      const rowIssue = predictionIssue(query.task, p, query.eventCount);
      if (rowIssue !== null) {
        rowsValid = false;
        issues.push({
          queryId: query.id,
          task: query.task,
          predictionIndex: index,
          message: `prediction #${index + 1}: ${rowIssue}`,
          severity: "warning",
        });
      }
    });
    if (rowsValid && status === "ready") readyByTask[query.task] += 1;
  }

  const blocked = issues.some((issue) => issue.severity === "error");
  return { issues, blocked, readyByTask, startedByTask };
}
