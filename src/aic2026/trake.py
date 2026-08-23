"""Minimal end-to-end TRAKE evidence prediction.

Wires the existing retrieval stages for the TRAKE task:

    TRAKE query text
        -> parse E<number> events (or numbered-list fallback)
        -> per-event text encoder -> IndexRetriever.search(top_k)
        -> ranked frame candidates + ranked temporal windows
        -> TRAKE prediction artifact (manifest.json + predictions.jsonl)

Every event is encoded and retrieved independently; the whole multi-event
query is never embedded as one vector. Prediction uses only the query text:
``gt_events`` / ``gt_frame_start`` / ``gt_frame_end`` / ``gt_answer`` are
evaluation metadata and are never read.

This module reuses :class:`aic2026.retrieval.IndexRetriever` /
``RetrievalCandidate`` for frame search and the task-independent primitives in
:mod:`aic2026.retrieval_candidates` (``group_candidates_by_video``,
``group_windows``, ``sort_by_score``, ``WindowCandidate``) for temporal
windows. It does not implement scoring, reranking, or LLM/VLM reasoning; the
artifact carries ranked evidence for manual notebook inspection only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from re import MULTILINE, compile
from typing import Any

from aic2026.embeddings import EmbeddingEncoder
from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask, load_eval_set
from aic2026.qa.qa_types import QueryTextField
from aic2026.retrieval import RetrievalCandidate
from aic2026.retrieval_candidates import (
    WindowCandidate,
    build_window_candidates,
    group_candidates_by_video,
)
from aic2026.search import SearchEngine
from aic2026.task_support import (
    default_run_id,
    resolve_query_text_field,
    utc_now_iso,
)
from aic2026.text_query import TextQuery

TRAKE_PREDICTION_SCHEMA_VERSION = "1.0"

#: One ``E<number>: event text`` line (each event is a single line).
_EVENT_LINE_PATTERN = compile(r"^E(\d+):\s*(.+?)\s*$", MULTILINE)

#: Numbered-list marker used as a fallback, e.g. ``(1) event text (2) ...``.
_NUMBERED_EVENT_PATTERN = compile(r"\((\d+)\)\s*")


class TrakeEventParseError(ValueError):
    """Raised when a TRAKE query text has no parseable event list."""


@dataclass(frozen=True)
class TrakeEvent:
    """One parsed TRAKE event derived from query text only.

    Expected input:
        - event_id: stable label such as ``E1``.
        - event_text: the text after the label, preserved verbatim.

    Returns:
        - A validated TRAKE event.

    Raises:
        - ValueError: if either field is empty.
    """

    event_id: str
    event_text: str

    def __post_init__(self) -> None:
        for name, value in (("event_id", self.event_id), ("event_text", self.event_text)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Serialize this event to a JSON-safe dictionary."""
        return {"event_id": self.event_id, "event_text": self.event_text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrakeEvent:
        """Deserialize one TRAKE event."""
        return cls(event_id=str(data["event_id"]), event_text=str(data["event_text"]))


@dataclass(frozen=True)
class TrakeEventResult:
    """Ranked evidence for one TRAKE event.

    Expected input:
        - event_id / event_text: the source event identity.
        - predictions: ranked frame candidates (``RetrievalCandidate``
          values, already sorted by retrieval score).
        - windows: ranked temporal window candidates (``WindowCandidate``
          values) covering the retrieved frames.

    Returns:
        - A validated per-event result.

    Raises:
        - ValueError: if any field violates the contract.
    """

    event_id: str
    event_text: str
    predictions: tuple[RetrievalCandidate, ...]
    windows: tuple[WindowCandidate, ...]

    def __post_init__(self) -> None:
        for name, value in (("event_id", self.event_id), ("event_text", self.event_text)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if not all(isinstance(item, RetrievalCandidate) for item in self.predictions):
            raise ValueError("predictions must contain only RetrievalCandidate values")
        if not all(isinstance(item, WindowCandidate) for item in self.windows):
            raise ValueError("windows must contain only WindowCandidate values")
        for label, candidates in (
            ("predictions", self.predictions),
            ("windows", self.windows),
        ):
            previous = 0
            for item in candidates:
                if item.rank <= previous:
                    raise ValueError(f"{label} ranks must be strictly increasing")
                previous = item.rank

    def to_dict(self) -> dict[str, Any]:
        """Serialize this event result to a JSON-safe dictionary."""
        return {
            "event_id": self.event_id,
            "event_text": self.event_text,
            "predictions": [_retrieval_candidate_to_dict(candidate) for candidate in self.predictions],
            "windows": [_window_candidate_to_dict(window) for window in self.windows],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrakeEventResult:
        """Deserialize one event result."""
        return cls(
            event_id=str(data["event_id"]),
            event_text=str(data["event_text"]),
            predictions=tuple(
                _retrieval_candidate_from_dict(item) for item in data["predictions"]
            ),
            windows=tuple(_window_candidate_from_dict(item) for item in data["windows"]),
        )


@dataclass(frozen=True)
class TrakeQueryResult:
    """Ranked per-event evidence for one TRAKE query.

    Expected input:
        - query_id: canonical query identifier.
        - query_text: the resolved query text that was parsed.
        - events: one :class:`TrakeEventResult` per parsed event, in query-text
          order.

    Returns:
        - A validated per-query TRAKE result.

    Raises:
        - ValueError: if any field violates the contract.
    """

    query_id: str
    query_text: str
    events: tuple[TrakeEventResult, ...]

    def __post_init__(self) -> None:
        for name, value in (("query_id", self.query_id), ("query_text", self.query_text)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if not self.events:
            raise ValueError("a TRAKE query result must contain at least one event")
        if not all(isinstance(event, TrakeEventResult) for event in self.events):
            raise ValueError("events must contain only TrakeEventResult values")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this query result to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrakeQueryResult:
        """Deserialize one query result."""
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            events=tuple(TrakeEventResult.from_dict(item) for item in data["events"]),
        )


@dataclass(frozen=True)
class TrakePredictionManifest:
    """Run-level metadata for one reproducible TRAKE prediction run.

    Expected input:
        - schema_version: must equal ``TRAKE_PREDICTION_SCHEMA_VERSION``.
        - run_id / created_at: provenance strings.
        - evaluation_set_path / evaluation_set_query_count /
          processed_query_count: input identity and processing coverage.
        - query_text_field: ``query_text_vi`` or ``query_text_en``.
        - top_k: frame retrieval cutoff per event.
        - window_max_gap: temporal window grouping gap, or ``None``.
        - embedding_space_compatible / embedding_space_override_used:
          embedding-space diagnostics.
        - index_artifact_path / index_manifest / model_embedding_space /
          model_config: artifact provenance.
        - selected_query_ids: explicit subset, or ``None``.

    Returns:
        - A validated TRAKE prediction manifest.

    Raises:
        - ValueError: if any field violates the contract.
    """

    schema_version: str
    run_id: str
    created_at: str
    evaluation_set_path: str
    evaluation_set_query_count: int
    processed_query_count: int
    query_text_field: QueryTextField
    top_k: int
    window_max_gap: int | None
    embedding_space_compatible: bool
    embedding_space_override_used: bool
    index_artifact_path: str
    index_manifest: dict[str, Any]
    model_embedding_space: dict[str, Any]
    model_config: dict[str, Any] | None
    selected_query_ids: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self.schema_version != TRAKE_PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported TRAKE prediction schema version {self.schema_version!r}; "
                f"expected {TRAKE_PREDICTION_SCHEMA_VERSION!r}"
            )
        for name, value in (("run_id", self.run_id), ("created_at", self.created_at)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.evaluation_set_query_count < 1:
            raise ValueError("evaluation_set_query_count must be >= 1")
        if self.processed_query_count < 1:
            raise ValueError("processed_query_count must be >= 1")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
        if self.window_max_gap is not None and (
            not isinstance(self.window_max_gap, int)
            or isinstance(self.window_max_gap, bool)
            or self.window_max_gap < 1
        ):
            raise ValueError("window_max_gap must be a positive integer or None")
        for name, value in (
            ("embedding_space_compatible", self.embedding_space_compatible),
            ("embedding_space_override_used", self.embedding_space_override_used),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if self.embedding_space_override_used and self.embedding_space_compatible:
            raise ValueError("embedding_space_override_used cannot be true when spaces are compatible")
        if not isinstance(self.index_manifest, dict) or not isinstance(self.model_embedding_space, dict):
            raise ValueError("index_manifest and model_embedding_space must be dictionaries")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "evaluation_set_path": self.evaluation_set_path,
            "evaluation_set_query_count": self.evaluation_set_query_count,
            "processed_query_count": self.processed_query_count,
            "query_text_field": self.query_text_field,
            "top_k": self.top_k,
            "window_max_gap": self.window_max_gap,
            "embedding_space_compatible": self.embedding_space_compatible,
            "embedding_space_override_used": self.embedding_space_override_used,
            "index_artifact_path": self.index_artifact_path,
            "index_manifest": self.index_manifest,
            "model_embedding_space": self.model_embedding_space,
            "model_config": self.model_config,
            "selected_query_ids": list(self.selected_query_ids) if self.selected_query_ids is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrakePredictionManifest:
        """Deserialize and validate a manifest dictionary."""
        selected = data.get("selected_query_ids")
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            evaluation_set_path=str(data["evaluation_set_path"]),
            evaluation_set_query_count=int(data["evaluation_set_query_count"]),
            processed_query_count=int(data["processed_query_count"]),
            query_text_field=data["query_text_field"],
            top_k=int(data["top_k"]),
            window_max_gap=data.get("window_max_gap"),
            embedding_space_compatible=bool(data["embedding_space_compatible"]),
            embedding_space_override_used=bool(data["embedding_space_override_used"]),
            index_artifact_path=str(data["index_artifact_path"]),
            index_manifest=dict(data["index_manifest"]),
            model_embedding_space=dict(data["model_embedding_space"]),
            model_config=dict(data["model_config"]) if data.get("model_config") is not None else None,
            selected_query_ids=tuple(selected) if selected is not None else None,
        )


@dataclass(frozen=True)
class TrakePredictionArtifact:
    """One complete machine-readable TRAKE prediction artifact.

    Expected input:
        - manifest: run-level metadata; ``processed_query_count`` must equal
          the number of query results.
        - query_results: one result per processed query, in query order.

    Returns:
        - A validated TRAKE prediction artifact.

    Raises:
        - ValueError: if the manifest/result counts disagree.
    """

    manifest: TrakePredictionManifest
    query_results: tuple[TrakeQueryResult, ...]

    def __post_init__(self) -> None:
        if self.manifest.processed_query_count != len(self.query_results):
            raise ValueError("manifest.processed_query_count must equal len(query_results)")


@dataclass(frozen=True)
class TrakePredictionRunnerConfig:
    """Configuration for one reproducible TRAKE prediction run.

    Expected input:
        - evaluation_set_path: CSV path consumed by ``load_eval_set``.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - model_config: optional JSON-serializable model configuration.
        - top_k: frame retrieval cutoff per event.
        - query_text_field: ``query_text_vi`` (default) or ``query_text_en``.
        - window_max_gap: optional positive gap for temporal window grouping;
          ``None`` disables window candidates.
        - query_subset_ids: optional explicit subset of query ids.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only. Disabled by default.
        - run_id: optional stable run id; auto-generated when omitted.

    Returns:
        - A validated TRAKE runner configuration.

    Raises:
        - ValueError: if required fields are invalid.
    """

    evaluation_set_path: Path
    index_artifact_dir: Path
    prediction_output_dir: Path
    model_config: dict[str, Any] | None = None
    top_k: int = 100
    query_text_field: QueryTextField = "query_text_vi"
    window_max_gap: int | None = 60
    query_subset_ids: tuple[str, ...] | None = None
    allow_embedding_space_mismatch: bool = False
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
        if self.window_max_gap is not None and (
            not isinstance(self.window_max_gap, int)
            or isinstance(self.window_max_gap, bool)
            or self.window_max_gap < 1
        ):
            raise ValueError("window_max_gap must be a positive integer or None")
        if not isinstance(self.allow_embedding_space_mismatch, bool):
            raise ValueError("allow_embedding_space_mismatch must be a boolean")
        if self.query_subset_ids is not None:
            if len(self.query_subset_ids) == 0:
                raise ValueError("query_subset_ids must be None or a non-empty tuple")
            if any(not isinstance(query_id, str) or query_id.strip() == "" for query_id in self.query_subset_ids):
                raise ValueError("query_subset_ids must contain only non-empty strings")


def parse_events(query_text: str) -> list[TrakeEvent]:
    """Parse ordered TRAKE events from query text only.

    The primary format is one ``E<number>: event text`` line per event. As a
    deterministic fallback, a numbered list ``(1) event (2) event ...`` in the
    query text is accepted (the format used by the shipped TRAKE rows). Event
    order always follows the query text; ``gt_events`` is never consulted.

    Expected input:
        - query_text: the resolved TRAKE query text.

    Returns:
        - Parsed events in text order, with the label text preserved verbatim.

    Raises:
        - TrakeEventParseError: if no parseable events are found.
    """
    if not isinstance(query_text, str) or query_text.strip() == "":
        raise TrakeEventParseError("query text is empty; cannot parse TRAKE events")
    labeled = [
        (int(number), text) for number, text in _EVENT_LINE_PATTERN.findall(query_text)
    ]
    if labeled:
        return [TrakeEvent(event_id=f"E{number}", event_text=text) for number, text in labeled]
    numbered = _parse_numbered_events(query_text)
    if numbered:
        return numbered
    raise TrakeEventParseError(
        "no TRAKE events found in query text; expected 'E<number>: event text' "
        "lines or a '(number) event' numbered list"
    )


def select_trake_queries(
    evaluation_set: EvaluationSet,
    subset_ids: Sequence[str] | None = None,
) -> tuple[EvaluationQuery, ...]:
    """Return ordered TRAKE queries, optionally restricted to explicit ids.

    Expected input:
        - evaluation_set: a validated evaluation set.
        - subset_ids: optional explicit subset of query ids to keep, in
          query order (subset membership only; order follows the CSV).

    Returns:
        - The selected TRAKE queries in CSV row order.

    Raises:
        - ValueError: if a requested id is unknown or no query is selected.
    """
    trake_queries = tuple(evaluation_set.iter_queries(EvaluationTask.TRAKE))
    if subset_ids is None:
        return trake_queries
    expected = {query.query_id for query in trake_queries}
    requested = set(subset_ids)
    unknown = requested - expected
    if unknown:
        raise ValueError(f"query_subset_ids contain unknown query ids: {sorted(unknown)}")
    selected = tuple(query for query in trake_queries if query.query_id in requested)
    if not selected:
        raise ValueError("query_subset_ids selected no TRAKE queries")
    return selected


def run_trake_prediction(
    config: TrakePredictionRunnerConfig,
    encoder: EmbeddingEncoder,
) -> TrakePredictionArtifact:
    """Run per-event TRAKE prediction for all or an explicit subset of queries.

    Expected input:
        - config: validated run configuration.
        - encoder: text encoder implementing the embedding contract.

    Returns:
        - The TRAKE prediction artifact, written to
          ``config.prediction_output_dir`` (``manifest.json`` +
          ``predictions.jsonl``).

    Raises:
        - ValueError: if the evaluation data has no TRAKE queries, subset ids
          are invalid, a query's events cannot be parsed, or embedding spaces
          mismatch.
        - FileNotFoundError: if configured input files are absent.
    """
    evaluation_set = load_eval_set(config.evaluation_set_path)
    all_trake_queries = tuple(evaluation_set.iter_queries(EvaluationTask.TRAKE))
    if not all_trake_queries:
        raise ValueError("evaluation set contains no TRAKE queries")

    selected = select_trake_queries(evaluation_set, config.query_subset_ids)
    queries = tuple(
        TextQuery(query_id=query.query_id, query_text=resolve_query_text_field(query.query_text_vi, query.query_text_en, config.query_text_field))
        for query in selected
    )
    return run_trake_prediction_for_queries(
        queries,
        encoder=encoder,
        index_artifact_dir=config.index_artifact_dir,
        prediction_output_dir=config.prediction_output_dir,
        top_k=config.top_k,
        query_text_field=config.query_text_field,
        window_max_gap=config.window_max_gap,
        model_config=config.model_config,
        allow_embedding_space_mismatch=config.allow_embedding_space_mismatch,
        run_id=config.run_id,
        query_source_path=Path(config.evaluation_set_path),
        query_source_count=len(all_trake_queries),
        selected_query_ids=config.query_subset_ids,
    )


def run_trake_prediction_for_queries(
    queries: tuple[TextQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_dir: Path,
    top_k: int = 100,
    query_text_field: QueryTextField = "query_text_vi",
    window_max_gap: int | None = 60,
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
    query_source_path: Path,
    query_source_count: int,
    selected_query_ids: tuple[str, ...] | None = None,
) -> TrakePredictionArtifact:
    """Run per-event TRAKE prediction for source-resolved text queries.

    This is the source-independent TRAKE prediction core: it consumes queries
    whose text is already resolved (no evaluation CSV is read), parses events
    from each query text only, and reuses the existing retrieval stack
    unchanged.

    Expected input:
        - queries: at least one resolved text query, in processing order.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - top_k: frame retrieval cutoff per event.
        - query_text_field: provenance label recorded in the manifest.
        - window_max_gap: temporal window grouping gap; ``None`` disables
          window candidates.
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.
        - query_source_path: identity of the query source, recorded as
          artifact provenance.
        - query_source_count: total queries available at the source for this
          task (>= number processed).
        - selected_query_ids: optional explicit subset recorded in the
          manifest for provenance.

    Returns:
        - The TRAKE prediction artifact written to ``prediction_output_dir``.

    Raises:
        - ValueError: if there are no queries, the source count is invalid,
          a query's events cannot be parsed, or embedding spaces mismatch
          under strict mode.
        - FileNotFoundError: if the index artifact is absent.
    """
    if not queries:
        raise ValueError("queries must contain at least one TRAKE query")
    if not isinstance(query_source_count, int) or isinstance(query_source_count, bool) or query_source_count < 1:
        raise ValueError("query_source_count must be a positive integer")
    if len(queries) > query_source_count:
        raise ValueError("query_source_count must be >= the number of queries")

    engine = SearchEngine.from_parts(
        encoder,
        index_artifact_dir,
        top_k=top_k,
        allow_embedding_space_mismatch=allow_embedding_space_mismatch,
    )
    results: list[TrakeQueryResult] = []
    for query in queries:
        results.append(
            search_trake(engine, query, top_k=top_k, window_max_gap=window_max_gap)
        )

    artifact = TrakePredictionArtifact(
        manifest=TrakePredictionManifest(
            schema_version=TRAKE_PREDICTION_SCHEMA_VERSION,
            run_id=run_id or default_run_id("trake"),
            created_at=utc_now_iso(),
            evaluation_set_path=str(Path(query_source_path).resolve()),
            evaluation_set_query_count=query_source_count,
            processed_query_count=len(results),
            query_text_field=query_text_field,
            top_k=top_k,
            window_max_gap=window_max_gap,
            embedding_space_compatible=engine.spaces_compatible,
            embedding_space_override_used=engine.override_used,
            index_artifact_path=str(Path(engine.retriever.artifact_dir).resolve()),
            index_manifest=engine.manifest.to_dict(),
            model_embedding_space=engine.embedding_space.to_dict(),
            model_config=model_config,
            selected_query_ids=selected_query_ids,
        ),
        query_results=tuple(results),
    )
    write_trake_artifact(prediction_output_dir, artifact)
    return artifact


def search_trake(
    engine: SearchEngine,
    query: TextQuery | str,
    *,
    top_k: int | None = None,
    window_max_gap: int | None = 60,
    query_id: str = "query",
) -> TrakeQueryResult:
    """Compute one TRAKE query's per-event evidence entirely in memory.

    This is the single-query entry point for interactive use: no evaluation
    CSV and no artifact writes. Events are parsed from the query text; each
    event runs an independent retrieval over the engine's index and gets its
    own ranked temporal windows.

    Expected input:
        - engine: a constructed :class:`aic2026.search.SearchEngine`.
        - query: a :class:`~aic2026.text_query.TextQuery` or plain text
          containing an event list (``E<number>: text`` lines or a
          ``(n) event ...`` numbered list).
        - top_k: optional per-event cutoff; defaults to the engine's.
        - window_max_gap: temporal window grouping gap; ``None`` disables
          window candidates.
        - query_id: identifier used when ``query`` is plain text.

    Returns:
        - The typed TRAKE result with per-event candidates and windows.

    Raises:
        - ValueError: if events cannot be parsed, inputs are invalid, or
          embedding spaces mismatch under strict mode.
    """
    if isinstance(query, str):
        query = TextQuery(query_id=query_id, query_text=query)
    events = parse_events(query.query_text)
    event_results = [
        _retrieve_event(engine=engine, event=event, top_k=top_k, window_max_gap=window_max_gap)
        for event in events
    ]
    return TrakeQueryResult(query_id=query.query_id, query_text=query.query_text, events=tuple(event_results))


def build_event_windows(
    retrieval: Sequence[RetrievalCandidate],
    max_gap: int | None,
) -> tuple[WindowCandidate, ...]:
    """Group one event's retrieved frames into ranked temporal windows.

    Thin task-level wrapper over
    :func:`aic2026.retrieval_candidates.build_window_candidates`, the single
    shared window implementation used by QA and TRAKE.

    Expected input:
        - retrieval: ranked frame-level retrieval candidates for one event.
        - max_gap: maximum frame-index distance inside one window; ``None``
          disables window generation.

    Returns:
        - Window candidates ranked by score (best window first); empty when
          disabled or when there are no candidates.

    Raises:
        - No exceptions.
    """
    return build_window_candidates(group_candidates_by_video(retrieval), max_gap)


def write_trake_artifact(output_dir: Path | str, artifact: TrakePredictionArtifact) -> None:
    """Write one TRAKE prediction artifact directory atomically.

    Expected input:
        - output_dir: target directory for ``manifest.json`` and
          ``predictions.jsonl``.
        - artifact: typed TRAKE prediction artifact.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    predictions_path = output_path / "predictions.jsonl"
    temp_manifest = manifest_path.with_name("manifest.json.tmp")
    temp_predictions = predictions_path.with_name("predictions.jsonl.tmp")
    with temp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(artifact.manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    with temp_predictions.open("w", encoding="utf-8") as handle:
        for query_result in artifact.query_results:
            handle.write(json.dumps(query_result.to_dict(), ensure_ascii=False) + "\n")
    temp_manifest.replace(manifest_path)
    temp_predictions.replace(predictions_path)


def read_trake_artifact(output_dir: Path | str) -> TrakePredictionArtifact:
    """Read and validate one TRAKE prediction artifact directory.

    Expected input:
        - output_dir: path to a directory written by
          :func:`write_trake_artifact`.

    Returns:
        - The validated TRAKE prediction artifact. Notebooks can inspect it
          as ``artifact.query_results[i].to_dict()["events"][j]``.

    Raises:
        - FileNotFoundError: if the artifact files are absent.
        - ValueError: if the payload violates the artifact contract.
    """
    output_path = Path(output_dir)
    with (output_path / "manifest.json").open(encoding="utf-8") as handle:
        manifest = TrakePredictionManifest.from_dict(json.load(handle))
    results: list[TrakeQueryResult] = []
    with (output_path / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(TrakeQueryResult.from_dict(json.loads(line)))
    return TrakePredictionArtifact(manifest=manifest, query_results=tuple(results))


def _parse_numbered_events(query_text: str) -> list[TrakeEvent]:
    """Parse a ``(number) event ...`` numbered list into ordered events."""
    parts = _NUMBERED_EVENT_PATTERN.split(query_text)
    events: list[TrakeEvent] = []
    for index in range(1, len(parts) - 1, 2):
        number = parts[index]
        text = parts[index + 1]
        cleaned = _clean_event_text(text)
        if not number.strip() or not cleaned:
            continue
        events.append(TrakeEvent(event_id=f"E{int(number)}", event_text=cleaned))
    return events


def _clean_event_text(raw: str) -> str:
    """Strip list separators (commas/semicolons) from the end of event text."""
    text = raw.strip()
    while text and text[-1] in ",;:":
        text = text[:-1].rstrip()
    return text


def _retrieve_event(
    engine: SearchEngine,
    event: TrakeEvent,
    top_k: int | None,
    window_max_gap: int | None,
) -> TrakeEventResult:
    """Run one independent event-level retrieval and build its evidence."""
    retrieval = engine.search_text(event.event_text, top_k=top_k)
    windows = build_window_candidates(group_candidates_by_video(retrieval), window_max_gap)
    return TrakeEventResult(
        event_id=event.event_id,
        event_text=event.event_text,
        predictions=tuple(retrieval),
        windows=windows,
    )


def _retrieval_candidate_to_dict(candidate: RetrievalCandidate) -> dict[str, Any]:
    """Serialize a retrieval candidate to JSON-safe fields."""
    return {
        "rank": candidate.rank,
        "video_id": candidate.video_id,
        "keyframe_ordinal": candidate.keyframe_ordinal,
        "frame_index": candidate.frame_index,
        "pts_time": float(candidate.pts_time),
        "score": float(candidate.score),
    }


def _retrieval_candidate_from_dict(data: dict[str, Any]) -> RetrievalCandidate:
    """Deserialize a retrieval candidate from JSON-safe fields."""
    return RetrievalCandidate(
        rank=int(data["rank"]),
        video_id=str(data["video_id"]),
        keyframe_ordinal=int(data["keyframe_ordinal"]),
        frame_index=int(data["frame_index"]),
        pts_time=float(data["pts_time"]),
        score=float(data["score"]),
    )


def _window_candidate_to_dict(candidate: WindowCandidate) -> dict[str, Any]:
    """Serialize a window candidate to JSON-safe fields."""
    return {
        "video_id": candidate.video_id,
        "frame_start": candidate.frame_start,
        "frame_end": candidate.frame_end,
        "score": float(candidate.score),
        "rank": candidate.rank,
    }


def _window_candidate_from_dict(data: dict[str, Any]) -> WindowCandidate:
    """Deserialize a window candidate from JSON-safe fields."""
    return WindowCandidate(
        video_id=str(data["video_id"]),
        frame_start=int(data["frame_start"]),
        frame_end=int(data["frame_end"]),
        score=float(data["score"]),
        rank=int(data["rank"]),
    )