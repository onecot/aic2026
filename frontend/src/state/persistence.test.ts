import { describe, expect, it, vi } from "vitest";
import {
  PERSISTENCE_VERSION,
  parseSession,
  serializeSession,
  type PersistedSession,
} from "./persistence";
import type { QueryWorkspace } from "./submission";

function sampleQueries(): QueryWorkspace[] {
  return [
    {
      id: "Q01",
      task: "kis",
      draftQuery: "red boat",
      lastSearch: { query: "red boat", limit: 100 },
      eventCount: null,
      predictions: [{ videoId: "L29_V005", frameIndex: 23221 }],
    },
    {
      id: "Q02",
      task: "qa",
      draftQuery: "",
      lastSearch: null,
      eventCount: null,
      predictions: [
        { videoId: "V", frameIndex: 1, answer: "Bát Tràng" },
        { videoId: "W", frameIndex: 2, answer: "" },
      ],
    },
    {
      id: "Q03",
      task: "trake",
      draftQuery: "(1) a (2) b",
      lastSearch: { query: "(1) a (2) b", limit: 100 },
      eventCount: 3,
      // v2: positional slots, partially edited by the user
      predictions: [{ videoId: "L10_V001", frameIndices: [1200, null, 2450] }],
    },
  ];
}

describe("persistence round-trip", () => {
  it("serializes and restores the full workspace", () => {
    const queries = sampleQueries();
    const doc = serializeSession(
      Object.fromEntries(queries.map((q) => [q.id, q])),
      ["Q01", "Q02", "Q03"],
      "Q02",
    );
    expect(doc.version).toBe(PERSISTENCE_VERSION);

    const restored = parseSession(JSON.stringify(doc));
    expect(restored).toEqual(doc);
    expect(restored?.activeQueryId).toBe("Q02");
    expect(restored?.queries[1]?.predictions[0]).toEqual({
      videoId: "V",
      frameIndex: 1,
      answer: "Bát Tràng",
    });
    // edited TRAKE sequence with an unset middle slot survives exactly
    expect(restored?.queries[2]?.predictions[0]).toEqual({
      videoId: "L10_V001",
      frameIndices: [1200, null, 2450],
    });
  });

  it("reads legacy v1 workspaces and upgrades them to v2 on the next save", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const legacy = {
      version: 1,
      savedAt: "2026-08-26T00:00:00.000Z",
      activeQueryId: "Q01",
      queries: [
        {
          id: "Q01",
          task: "trake",
          draftQuery: "",
          lastSearch: null,
          eventCount: 2,
          predictions: [{ videoId: "L10_V001", frameIndices: [1200, 1850] }], // dense v1
        },
      ],
    };
    const restored = parseSession(JSON.stringify(legacy));
    expect(warn).not.toHaveBeenCalled();
    expect(restored?.version).toBe(2);
    expect(restored?.queries[0]?.predictions[0]).toEqual({
      videoId: "L10_V001",
      frameIndices: [1200, 1850],
    });

    // re-saving writes a current-version document
    const next = serializeSession(
      Object.fromEntries((restored?.queries ?? []).map((q) => [q.id, q])),
      ["Q01"],
      "Q01",
    );
    expect(next.version).toBe(2);
    warn.mockRestore();
  });

  it("rejects TRAKE slots that are neither numbers nor null", () => {
    const base = serializeSession(
      Object.fromEntries(sampleQueries().map((q) => [q.id, q])),
      ["Q01", "Q02", "Q03"],
      "Q03",
    );
    const bad = {
      ...base,
      queries: [
        ...base.queries.slice(0, 2),
        {
          ...base.queries[2]!,
          predictions: [{ videoId: "L10_V001", frameIndices: ["1200" as unknown] }],
        },
      ],
    };
    expect(parseSession(JSON.stringify(bad))).toBeNull();
  });
});

describe("malformed persisted state is discarded whole", () => {
  it("rejects corrupted JSON", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    expect(parseSession("{not json")).toBeNull();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("rejects unknown schema versions", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const doc = serializeSession({}, [], null);
    const mutated = JSON.stringify({ ...doc, version: 99 });
    expect(parseSession(mutated)).toBeNull();
    warn.mockRestore();
  });

  it("rejects wrong prediction shapes per task", () => {
    const base: PersistedSession = serializeSession(
      Object.fromEntries(sampleQueries().map((q) => [q.id, q])),
      ["Q01", "Q02", "Q03"],
      "Q01",
    );
    const cases = [
      // kis row missing videoId
      { ...base, queries: [{ ...base.queries[0]!, predictions: [{ frameIndex: 1 }] }] },
      // qa row with non-string answer
      {
        ...base,
        queries: [
          base.queries[0]!,
          {
            ...base.queries[1]!,
            predictions: [{ videoId: "V", frameIndex: 1, answer: 5 }],
          },
        ],
      },
      // trake row with non-numeric indices
      {
        ...base,
        queries: [
          base.queries[0]!,
          base.queries[1]!,
          { ...base.queries[2]!, predictions: [{ videoId: "V", frameIndices: ["x"] }] },
        ],
      },
      // over-limit queue
      {
        ...base,
        queries: [
          {
            ...base.queries[0]!,
            predictions: Array.from({ length: 101 }, (_, i) => ({
              videoId: `V${i}`,
              frameIndex: i,
            })),
          },
        ],
      },
    ];
    for (const doc of cases) {
      expect(parseSession(JSON.stringify(doc))).toBeNull();
    }
  });

  it("falls back to the first query when activeQueryId is unknown", () => {
    const doc = serializeSession(
      Object.fromEntries(sampleQueries().map((q) => [q.id, q])),
      ["Q01", "Q02"],
      "Q99",
    );
    const restored = parseSession(JSON.stringify(doc));
    expect(restored?.activeQueryId).toBe("Q01");
  });
});
