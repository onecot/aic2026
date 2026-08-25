/**
 * Local session persistence (browser localStorage), versioned schema.
 *
 * Survives reload: query workspaces (draft queries, last search, TRAKE
 * event counts, prediction queues incl. QA answers), active query.
 * Not persisted: inspector selection, TanStack cache.
 *
 * Malformed or wrong-version payloads are discarded whole — never partially
 * applied — and reported via console warning.
 */
import type { Prediction, QueryWorkspace, Task } from "./submission";
import { MAX_PREDICTIONS } from "./submission";

export const PERSISTENCE_KEY = "aic2026.workspace.v1";
/**
 * v2: TRAKE anchors are positional slots that may be `null` (event frame
 * not chosen yet). v1 payloads (dense numeric arrays from the automatic
 * assembly heuristic) remain readable and normalize on the next save.
 */
export const PERSISTENCE_VERSION = 2;
const READABLE_VERSIONS = [1, PERSISTENCE_VERSION];

export interface PersistedSession {
  version: number;
  savedAt: string;
  activeQueryId: string | null;
  queries: QueryWorkspace[];
}

function isTask(value: unknown): value is Task {
  return value === "kis" || value === "qa" || value === "trake";
}

function isPredictionShape(task: Task, p: unknown): boolean {
  if (typeof p !== "object" || p === null) return false;
  const rec = p as Record<string, unknown>;
  if (typeof rec["videoId"] !== "string") return false;
  if (task === "trake") {
    return (
      Array.isArray(rec["frameIndices"]) &&
      rec["frameIndices"].every(
        // v2 slots are `number | null`; legacy v1 rows are dense numbers.
        (n) => n === null || typeof n === "number",
      )
    );
  }
  if (task === "qa") return typeof rec["answer"] === "string";
  return true;
}

function parseQuery(raw: unknown): QueryWorkspace | null {
  if (typeof raw !== "object" || raw === null) return null;
  const q = raw as Record<string, unknown>;
  if (typeof q["id"] !== "string" || !isTask(q["task"])) return null;
  const task = q["task"];
  if (!Array.isArray(q["predictions"])) return null;
  if (q["predictions"].length > MAX_PREDICTIONS) return null;
  const predictions = q["predictions"] as unknown[];
  if (!predictions.every((p) => isPredictionShape(task, p))) return null;

  let lastSearch: QueryWorkspace["lastSearch"] = null;
  if (q["lastSearch"] !== null && q["lastSearch"] !== undefined) {
    const ls = q["lastSearch"] as Record<string, unknown>;
    if (
      typeof ls["query"] !== "string" ||
      typeof ls["limit"] !== "number"
    )
      return null;
    lastSearch = { query: ls["query"], limit: ls["limit"] };
  }
  const eventCount =
    q["eventCount"] === null || q["eventCount"] === undefined
      ? null
      : typeof q["eventCount"] === "number"
        ? q["eventCount"]
        : null;

  return {
    id: q["id"],
    task,
    draftQuery: typeof q["draftQuery"] === "string" ? q["draftQuery"] : "",
    lastSearch,
    eventCount,
    predictions: predictions as Prediction[],
  };
}

/** Serialize the persisted slice from store-shaped data. */
export function serializeSession(
  queries: Record<string, QueryWorkspace>,
  queryOrder: string[],
  activeQueryId: string | null,
  savedAt: Date = new Date(),
): PersistedSession {
  return {
    version: PERSISTENCE_VERSION,
    savedAt: savedAt.toISOString(),
    activeQueryId,
    queries: queryOrder
      .map((id) => queries[id])
      .filter((q): q is QueryWorkspace => q !== undefined),
  };
}

/**
 * Parse a stored payload. Returns null for anything malformed (bad JSON,
 * wrong version, wrong shapes) so callers fall back to a fresh session.
 */
export function parseSession(json: string): PersistedSession | null {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    console.warn("[persistence] discarding corrupted workspace payload");
    return null;
  }
  if (typeof raw !== "object" || raw === null) return null;
  const doc = raw as Record<string, unknown>;
  if (!READABLE_VERSIONS.includes(doc["version"] as number)) {
    console.warn(
      `[persistence] discarding workspace with schema version ${String(doc["version"])}`,
    );
    return null;
  }
  if (doc["activeQueryId"] !== null && typeof doc["activeQueryId"] !== "string")
    return null;
  if (!Array.isArray(doc["queries"])) return null;

  const queries: QueryWorkspace[] = [];
  for (const entry of doc["queries"]) {
    const parsed = parseQuery(entry);
    if (parsed === null) {
      console.warn("[persistence] discarding malformed workspace payload");
      return null;
    }
    queries.push(parsed);
  }

  const activeQueryId =
    doc["activeQueryId"] !== null &&
    queries.some((q) => q.id === doc["activeQueryId"])
      ? (doc["activeQueryId"] as string)
      : (queries[0]?.id ?? null);

  const savedAt = typeof doc["savedAt"] === "string" ? doc["savedAt"] : new Date().toISOString();
  return { version: PERSISTENCE_VERSION, savedAt, activeQueryId, queries };
}

/** Load the persisted session from localStorage; null when absent/invalid. */
export function loadSession(storage: Storage = globalThis.localStorage): PersistedSession | null {
  let json: string | null = null;
  try {
    json = storage.getItem(PERSISTENCE_KEY);
  } catch (error) {
    console.warn("[persistence] storage unavailable", error);
    return null;
  }
  if (json === null) return null;
  return parseSession(json);
}

/** Atomically write the session. Throws only on unusable storage. */
export function saveSession(session: PersistedSession, storage: Storage = globalThis.localStorage): void {
  try {
    storage.setItem(PERSISTENCE_KEY, JSON.stringify(session));
  } catch (error) {
    // Private-mode quota errors must not crash the app mid-interaction.
    console.warn("[persistence] could not save workspace", error);
  }
}

export function clearSession(storage: Storage = globalThis.localStorage): void {
  try {
    storage.removeItem(PERSISTENCE_KEY);
  } catch {
    // ignore: nothing to clear
  }
}
