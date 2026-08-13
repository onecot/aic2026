"""Typed, read-only access to the team evaluation CSV.

This module separates the physical ``eval_set.csv`` representation from the
canonical records evaluators consume. It intentionally contains no scoring or
retrieval logic.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aic2026.canonical import VideoId

EVALUATION_SCHEMA_VERSION = "1.0"

_REQUIRED_COLUMNS = (
    "query_id",
    "task",
    "query_text_vi",
    "query_text_en",
    "gt_video_id",
    "gt_frame_start",
    "gt_frame_end",
    "gt_answer",
    "gt_events",
    "annotator",
    "note",
)


class EvaluationDatasetError(ValueError):
    """Base error for malformed evaluation-dataset input."""


class EvaluationTask(str, Enum):
    """Task values accepted by the evaluation CSV contract."""

    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"


@dataclass(frozen=True)
class RawEvaluationRow:
    """One untyped CSV row before canonical validation and conversion.

    All fields are preserved as strings because CSV has no native null or
    numeric representation. This type is deliberately separate from
    :class:`EvaluationQuery`.
    """

    query_id: str
    task: str
    query_text_vi: str
    query_text_en: str
    gt_video_id: str
    gt_frame_start: str
    gt_frame_end: str
    gt_answer: str
    gt_events: str
    annotator: str
    note: str


@dataclass(frozen=True)
class FrameRange:
    """Inclusive original-video frame range for one ground-truth segment.

    Expected input:
        - start: non-negative inclusive original-video frame index.
        - end: non-negative inclusive original-video frame index at or after
          ``start``.

    Returns:
        - A validated frame range.

    Raises:
        - ValueError: if either endpoint is invalid or the range is reversed.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError("frame range start must be a non-negative integer")
        if not isinstance(self.end, int) or isinstance(self.end, bool) or self.end < self.start:
            raise ValueError("frame range end must be an integer >= start")


@dataclass(frozen=True)
class EventAnnotation:
    """The non-empty TRAKE event annotation stored in ``gt_events``.

    The supplied CSV currently contains no non-empty ``gt_events`` values, so
    no delimiter or nested event schema can be derived without inventing one.
    The entire field is therefore preserved as one typed annotation string.
    Future data with a defined structured event syntax requires an explicit
    contract version update rather than an undocumented parser assumption.
    """

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or self.text.strip() == "":
            raise ValueError("event annotation text must be a non-empty string")


@dataclass(frozen=True)
class AnnotationMetadata:
    """Useful provenance fields supplied by the team annotation CSV."""

    annotator: str
    note: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.annotator, str) or self.annotator.strip() == "":
            raise ValueError("annotator must be a non-empty string")
        if self.note is not None and (not isinstance(self.note, str) or self.note.strip() == ""):
            raise ValueError("note must be a non-empty string or None")


@dataclass(frozen=True)
class EvaluationQuery:
    """Canonical task-aware evaluation query.

    Expected input:
        - query_id: unique stable query identifier.
        - task: KIS, QA, or TRAKE.
        - query_text_vi: required Vietnamese query text.
        - query_text_en: optional English query text.
        - ground_truth_video_id: video containing the annotated result.
        - ground_truth_frame_range: inclusive original-video frame segment.
        - answer: required only for QA.
        - event: required only for TRAKE.
        - annotation: useful annotation provenance.

    Returns:
        - A validated, CSV-independent evaluation record.

    Raises:
        - ValueError: if shared or task-specific fields violate the contract.
    """

    query_id: str
    task: EvaluationTask
    query_text_vi: str
    query_text_en: str | None
    ground_truth_video_id: VideoId
    ground_truth_frame_range: FrameRange
    answer: str | None
    event: EventAnnotation | None
    annotation: AnnotationMetadata

    def __post_init__(self) -> None:
        for name, value in (("query_id", self.query_id), ("query_text_vi", self.query_text_vi)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.query_text_en is not None and (
            not isinstance(self.query_text_en, str) or self.query_text_en.strip() == ""
        ):
            raise ValueError("query_text_en must be a non-empty string or None")
        if not isinstance(self.task, EvaluationTask):
            raise ValueError("task must be an EvaluationTask")
        if not isinstance(self.ground_truth_video_id, VideoId):
            raise ValueError("ground_truth_video_id must be a VideoId")
        if not isinstance(self.ground_truth_frame_range, FrameRange):
            raise ValueError("ground_truth_frame_range must be a FrameRange")
        if not isinstance(self.annotation, AnnotationMetadata):
            raise ValueError("annotation must be AnnotationMetadata")
        if self.task is EvaluationTask.QA:
            if not isinstance(self.answer, str) or self.answer.strip() == "":
                raise ValueError("QA queries require a non-empty answer")
            if self.event is not None:
                raise ValueError("QA queries must not define a TRAKE event")
        elif self.task is EvaluationTask.TRAKE:
            if not isinstance(self.event, EventAnnotation):
                raise ValueError("TRAKE queries require a non-empty event annotation")
            if self.answer is not None:
                raise ValueError("TRAKE queries must not define a QA answer")
        else:
            if self.answer is not None or self.event is not None:
                raise ValueError("KIS queries must not define QA answers or TRAKE events")


@dataclass(frozen=True)
class EvaluationSet:
    """Deterministic, read-only collection of canonical evaluation queries."""

    queries: tuple[EvaluationQuery, ...]

    def __post_init__(self) -> None:
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("evaluation query_ids must be unique")

    def __len__(self) -> int:
        """Return the number of queries in CSV row order."""
        return len(self.queries)

    def iter_queries(self, task: EvaluationTask | str | None = None) -> Iterator[EvaluationQuery]:
        """Yield queries in CSV row order, optionally restricted to one task.

        Expected input:
            - task: optional task enum or case-insensitive task string.

        Returns:
            - Iterator of matching canonical queries.

        Raises:
            - ValueError: if ``task`` is not KIS, QA, or TRAKE.
        """
        selected_task = _parse_task(task) if task is not None else None
        for query in self.queries:
            if selected_task is None or query.task is selected_task:
                yield query

    def get_query(self, query_id: str) -> EvaluationQuery:
        """Return a query by its unique identifier.

        Expected input:
            - query_id: a canonical query identifier.

        Returns:
            - The matching query.

        Raises:
            - KeyError: if no query has the requested identifier.
        """
        for query in self.queries:
            if query.query_id == query_id:
                return query
        raise KeyError(f"evaluation query not found: {query_id!r}")

    def task_counts(self) -> dict[EvaluationTask, int]:
        """Return the count of canonical queries for each populated task."""
        return dict(Counter(query.task for query in self.queries))


def load_eval_set(path: Path | str) -> EvaluationSet:
    """Load and validate one team evaluation CSV without modifying it.

    Expected input:
        - path: path to an ``eval_set.csv``-format file.

    Returns:
        - Read-only canonical evaluation queries in CSV row order.

    Raises:
        - FileNotFoundError: if ``path`` does not exist.
        - EvaluationDatasetError: if the header, row values, or task-specific
          fields violate the evaluation contract.
    """
    csv_path = Path(path)
    raw_rows = _read_raw_rows(csv_path)
    try:
        queries = tuple(_canonicalize_row(row, row_number) for row_number, row in enumerate(raw_rows, start=2))
        return EvaluationSet(queries=queries)
    except ValueError as exc:
        raise EvaluationDatasetError(str(exc)) from exc


def _read_raw_rows(path: Path) -> list[RawEvaluationRow]:
    """Read exact CSV columns into raw string rows before canonicalization."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != _REQUIRED_COLUMNS:
            raise EvaluationDatasetError(
                f"evaluation CSV columns must be {_REQUIRED_COLUMNS}, got {fieldnames}"
            )
        rows: list[RawEvaluationRow] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise EvaluationDatasetError(f"row {row_number}: expected exactly {_REQUIRED_COLUMNS} columns")
            rows.append(RawEvaluationRow(**row))
    return rows


def _canonicalize_row(row: RawEvaluationRow, row_number: int) -> EvaluationQuery:
    """Convert one raw CSV row to a validated canonical query."""
    try:
        task = _parse_task(row.task)
        answer = _optional_text(row.gt_answer)
        event_text = _optional_text(row.gt_events)
        event = EventAnnotation(event_text) if event_text is not None else None
        return EvaluationQuery(
            query_id=_required_text(row.query_id, "query_id"),
            task=task,
            query_text_vi=_required_text(row.query_text_vi, "query_text_vi"),
            query_text_en=_optional_text(row.query_text_en),
            ground_truth_video_id=VideoId(_required_text(row.gt_video_id, "gt_video_id")),
            ground_truth_frame_range=FrameRange(
                start=_parse_frame(row.gt_frame_start, "gt_frame_start"),
                end=_parse_frame(row.gt_frame_end, "gt_frame_end"),
            ),
            answer=answer,
            event=event,
            annotation=AnnotationMetadata(
                annotator=_required_text(row.annotator, "annotator"),
                note=_optional_text(row.note),
            ),
        )
    except ValueError as exc:
        raise EvaluationDatasetError(f"row {row_number}: {exc}") from exc


def _parse_task(value: EvaluationTask | str) -> EvaluationTask:
    """Normalize and validate a task enum or case-insensitive CSV string."""
    if isinstance(value, EvaluationTask):
        return value
    if not isinstance(value, str):
        raise ValueError("task must be one of: kis, qa, trake")
    try:
        return EvaluationTask(value.strip().lower())
    except ValueError as exc:
        raise ValueError("task must be one of: kis, qa, trake") from exc


def _required_text(value: str, field_name: str) -> str:
    """Return a stripped required CSV text value or raise a clear error."""
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _optional_text(value: str) -> str | None:
    """Normalize a blank CSV cell to None while preserving non-blank text."""
    normalized = value.strip()
    return normalized or None


def _parse_frame(value: str, field_name: str) -> int:
    """Parse a non-negative integer original-video frame index."""
    normalized = _required_text(value, field_name)
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed
