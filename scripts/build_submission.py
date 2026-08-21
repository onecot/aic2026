"""BTC submission exporter for AIC2026 experimental queries.

Reads the actual ``query-*-<task>.txt`` files in a query directory, matches
each query to exactly one prediction result in the corresponding artifact
(``kis`` / ``qa`` / ``trake``), and produces per-query CSV files in BTC
submission format.  Packaging produces a ``submission.zip`` that contains the
``submission/`` directory itself.

Matching is strict:

1. exact ``query_text`` equality;
2. whitespace/newline-normalized equality.

Vague substring matching is never used.  If a query has no unique match, or
the match is ambiguous, the build fails loudly and generates nothing.

This module only converts existing prediction artifacts into a valid BTC
submission.  It does not run retrieval, does not generate answers, and does
not implement any new ranking framework.

Usage::

    python scripts/build_submission.py \\
        --query-dir data/sotuyen/thunghiem \\
        --kis-artifact data/interim/predictions/kis-openclip-vs-btc-batch1-full.json \\
        --qa-artifact data/interim/qa/qa-jina-real-test \\
        --trake-artifact data/interim/trake/trake-real-test \\
        --output-dir submission

No Modal runs.  No GPU runs.  No changes to core prediction pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aic2026.btc_queries import (
    QueryFileInfo,
    enumerate_query_files,
    normalize_query_text,
    parse_query_filename,
    read_query_text,
)
from aic2026.kis_prediction_runner import QueryPredictionResult, read_kis_prediction_artifact
from aic2026.qa.qa_prediction_runner import read_qa_artifact
from aic2026.qa.qa_types import QaPredictionArtifact, QaQueryResult
from aic2026.trake import (
    TrakePredictionArtifact,
    TrakeQueryResult,
    parse_events,
    read_trake_artifact,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KIS_MAX_ROWS = 100
QA_MAX_ROWS = 100
QA_ANSWER_MAX_CHARS = 100
TRAKE_MAX_ROWS = 100


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_kis_artifacts(path: Path) -> tuple[QueryPredictionResult, ...]:
    """Load KIS prediction results.

    Expected input:
        - path: a KIS prediction artifact JSON file, or a directory
          containing one (``*.json``).

    Returns:
        - Tuple of validated KIS query results.

    Raises:
        - FileNotFoundError: if the file/directory does not exist.
        - ValueError: if the artifact is invalid.
    """
    if path.is_dir():
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"no JSON prediction files found in {path}")
        path = json_files[0]
    artifact = read_kis_prediction_artifact(path)
    return artifact.query_results


def load_qa_artifact(path: Path) -> QaPredictionArtifact:
    """Load a QA prediction artifact directory.

    Expected input:
        - path: a QA prediction artifact directory with ``manifest.json``
          and ``predictions.jsonl``.

    Returns:
        - The validated QA prediction artifact.

    Raises:
        - FileNotFoundError: if the directory or files are missing.
        - ValueError: if the artifact is invalid.
    """
    return read_qa_artifact(path)


def load_trake_artifact(path: Path) -> TrakePredictionArtifact:
    """Load a TRAKE prediction artifact directory.

    Expected input:
        - path: a TRAKE prediction artifact directory with ``manifest.json``
          and ``predictions.jsonl``.

    Returns:
        - The validated TRAKE prediction artifact.

    Raises:
        - FileNotFoundError: if the directory or files are missing.
        - ValueError: if the artifact is invalid.
    """
    return read_trake_artifact(path)


# ---------------------------------------------------------------------------
# Query-to-artifact matching (strict)
# ---------------------------------------------------------------------------


class QueryMatchError(ValueError):
    """Raised when a BTC query cannot be uniquely matched to an artifact."""


def find_unique_match(
    query_text: str,
    results: tuple[Any, ...],
    get_text: Callable[[Any], str],
) -> tuple[Any, str]:
    """Find exactly one artifact result for a query text.

    Matching priority:
        1. exact ``query_text`` equality;
        2. whitespace/newline-normalized equality.

    Expected input:
        - query_text: the exact text read from a BTC query file.
        - results: tuple of artifact query results.
        - get_text: callable returning the ``query_text`` of a result.

    Returns:
        - ``(result, method)`` where ``method`` is ``"exact"`` or
          ``"normalized"``.

    Raises:
        - QueryMatchError: if there is no match or the match is ambiguous.
    """
    stripped = query_text.strip()
    exact = [r for r in results if get_text(r).strip() == stripped]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        raise QueryMatchError(
            f"ambiguous exact match ({len(exact)} candidates)"
        )

    normalized = normalize_query_text(query_text)
    normalized_candidates = [
        r for r in results if normalize_query_text(get_text(r)) == normalized
    ]
    if len(normalized_candidates) == 1:
        return normalized_candidates[0], "normalized"
    if len(normalized_candidates) > 1:
        raise QueryMatchError(
            f"ambiguous normalized match ({len(normalized_candidates)} candidates)"
        )
    raise QueryMatchError("no exact or whitespace-normalized match")


# ---------------------------------------------------------------------------
# KIS export
# ---------------------------------------------------------------------------


def export_kis_csv(result: QueryPredictionResult, output_path: Path) -> None:
    """Write a KIS submission CSV from ranked predictions.

    Expected input:
        - result: the ranked KIS predictions for one query.
        - output_path: target CSV file path.

    Returns:
        - None.

    Raises:
        - ValueError: if no predictions are available.
        - OSError: if writing fails.
    """
    if not result.predictions:
        raise ValueError(f"no KIS predictions for query {result.query_id!r}")

    rows: list[list[str]] = []
    for pred in result.predictions[:KIS_MAX_ROWS]:
        rows.append([pred.video_id, str(pred.frame_index)])

    _validate_kis_rows(rows)
    _write_csv(rows, output_path)


def _validate_kis_rows(rows: list[list[str]]) -> None:
    """Validate KIS CSV rows before writing."""
    for idx, row in enumerate(rows):
        if len(row) != 2:
            raise ValueError(f"KIS row {idx}: expected 2 columns, got {len(row)}")
        video_id = row[0]
        if not isinstance(video_id, str) or video_id == "":
            raise ValueError(f"KIS row {idx}: video_id must be a non-empty string")
        if video_id.endswith(".mp4"):
            raise ValueError(f"KIS row {idx}: video_id must not end with .mp4")
        try:
            int(row[1])
        except (ValueError, TypeError):
            raise ValueError(f"KIS row {idx}: frame_index must be an integer")


# ---------------------------------------------------------------------------
# QA export
# ---------------------------------------------------------------------------


def export_qa_csv(query_result: QaQueryResult, output_path: Path) -> None:
    """Write a QA submission CSV from predictions and generated answer.

    The evidence frame associated with the generated answer is used.  If
    ``answer_evidence`` is present, its first prediction is used; otherwise
    the first ranked prediction is used.  Answers are never truncated or
    otherwise modified.

    Expected input:
        - query_result: the QA predictions for one query.
        - output_path: target CSV file path.

    Returns:
        - None.

    Raises:
        - ValueError: if no predictions/answer are available, or the answer
          exceeds 100 characters.
        - OSError: if writing fails.
    """
    answer = query_result.answer.strip()
    if not answer:
        raise ValueError(f"no answer for query {query_result.query_id!r}")

    evidence: Any = None
    if query_result.answer_evidence:
        evidence = query_result.answer_evidence[0]
    elif query_result.predictions:
        evidence = query_result.predictions[0]

    if evidence is None:
        raise ValueError(
            f"no evidence frame for query {query_result.query_id!r}"
        )

    rows: list[list[str]] = [
        [evidence.video_id, str(evidence.frame_index), answer]
    ]

    _validate_qa_rows(rows)
    _write_csv(rows, output_path)


def _validate_qa_rows(rows: list[list[str]]) -> None:
    """Validate QA CSV rows before writing."""
    for idx, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(f"QA row {idx}: expected 3 columns, got {len(row)}")
        video_id = row[0]
        if not isinstance(video_id, str) or video_id == "":
            raise ValueError(f"QA row {idx}: video_id must be a non-empty string")
        if video_id.endswith(".mp4"):
            raise ValueError(f"QA row {idx}: video_id must not end with .mp4")
        try:
            int(row[1])
        except (ValueError, TypeError):
            raise ValueError(f"QA row {idx}: frame_index must be an integer")
        answer = row[2]
        if not answer:
            raise ValueError(f"QA row {idx}: answer must be non-empty")
        if len(answer) > QA_ANSWER_MAX_CHARS:
            raise ValueError(
                f"QA row {idx}: answer exceeds {QA_ANSWER_MAX_CHARS} chars "
                f"({len(answer)} chars)"
            )


# ---------------------------------------------------------------------------
# TRAKE export
# ---------------------------------------------------------------------------


def _best_sequence_for_video(
    candidates_by_event: list[list[tuple[int, float]]],
) -> tuple[list[int], float] | None:
    """Find the highest-scoring strictly-increasing frame sequence.

    Expected input:
        - candidates_by_event: one list of ``(frame_index, score)`` per event,
          in event order.

    Returns:
        - ``(frames, total_score)`` with one frame per event, strictly
          increasing, maximizing total score; ``None`` when no complete
          sequence exists.
    """
    if not candidates_by_event or not candidates_by_event[0]:
        return None
    states: dict[int, tuple[float, list[int]]] = {}
    for frame, score in candidates_by_event[0]:
        states[frame] = (score, [frame])

    for candidates in candidates_by_event[1:]:
        next_states: dict[int, tuple[float, list[int]]] = {}
        for frame, score in candidates:
            best_prev: tuple[float, list[int]] | None = None
            for prev_frame, (prev_score, prev_path) in states.items():
                if prev_frame < frame and (
                    best_prev is None or prev_score > best_prev[0]
                ):
                    best_prev = (prev_score, prev_path)
            if best_prev is None:
                continue
            total = best_prev[0] + score
            path = best_prev[1] + [frame]
            if frame not in next_states or total > next_states[frame][0]:
                next_states[frame] = (total, path)
        states = next_states
        if not states:
            return None

    total, path = max(states.values(), key=lambda item: item[0])
    return path, total


def select_trake_rows(
    query_result: TrakeQueryResult,
    max_rows: int = TRAKE_MAX_ROWS,
) -> list[list[str]]:
    """Select valid complete TRAKE submission rows from event predictions.

    For every video that has candidates in all events, the highest-scoring
    complete sequence (one frame per event, same video, strictly increasing
    frame indices) is produced.  Sequences are ranked by total retrieval
    score (descending) and at most ``max_rows`` are returned.  Partial
    sequences are never emitted.

    Expected input:
        - query_result: the TRAKE predictions with per-event candidates.
        - max_rows: maximum number of rows to return.

    Returns:
        - A list of valid submission rows, each being
          ``[video_id, frame_E1, frame_E2, ..., frame_EN]``.

    Raises:
        - ValueError: if the query has no events.
    """
    if not query_result.events:
        raise ValueError(f"no events for TRAKE query {query_result.query_id!r}")

    num_events = len(query_result.events)

    per_event_by_video: list[dict[str, list[tuple[int, float]]]] = []
    for event in query_result.events:
        by_video: dict[str, list[tuple[int, float]]] = {}
        for pred in event.predictions:
            by_video.setdefault(pred.video_id, []).append(
                (pred.frame_index, float(pred.score))
            )
        per_event_by_video.append(by_video)

    candidate_videos = set(per_event_by_video[0])
    for by_video in per_event_by_video[1:]:
        candidate_videos &= set(by_video)

    ranked: list[tuple[list[str], float]] = []
    for video_id in sorted(candidate_videos):
        candidates_by_event = [
            per_event_by_video[e][video_id] for e in range(num_events)
        ]
        sequence = _best_sequence_for_video(candidates_by_event)
        if sequence is None:
            continue
        frames, total_score = sequence
        row = [video_id] + [str(frame) for frame in frames]
        ranked.append((row, total_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [row for row, _score in ranked[:max_rows]]


def export_trake_csv(query_result: TrakeQueryResult, output_path: Path) -> None:
    """Write a TRAKE submission CSV from chained event predictions.

    Expected input:
        - query_result: the TRAKE predictions for one query.
        - output_path: target CSV file path.

    Returns:
        - None.

    Raises:
        - ValueError: if no valid rows can be generated.
        - OSError: if writing fails.
    """
    rows = select_trake_rows(query_result)
    if not rows:
        raise ValueError(
            f"no valid TRAKE rows for query {query_result.query_id!r}"
        )
    _validate_trake_rows(rows, num_events=len(query_result.events))
    _write_csv(rows, output_path)


def _validate_trake_rows(rows: list[list[str]], num_events: int) -> None:
    """Validate TRAKE CSV rows before writing."""
    expected_cols = num_events + 1  # video_id + N frame indices
    for idx, row in enumerate(rows):
        if len(row) != expected_cols:
            raise ValueError(
                f"TRAKE row {idx}: expected {expected_cols} columns "
                f"(video_id + {num_events} frames), got {len(row)}"
            )
        video_id = row[0]
        if not isinstance(video_id, str) or video_id == "":
            raise ValueError(
                f"TRAKE row {idx}: video_id must be a non-empty string"
            )
        if video_id.endswith(".mp4"):
            raise ValueError(
                f"TRAKE row {idx}: video_id must not end with .mp4"
            )
        frame_indices: list[int] = []
        for col_idx in range(1, len(row)):
            try:
                frame_indices.append(int(row[col_idx]))
            except (ValueError, TypeError):
                raise ValueError(
                    f"TRAKE row {idx}, col {col_idx}: frame_index must be "
                    "an integer"
                )
        for i in range(1, len(frame_indices)):
            if frame_indices[i] <= frame_indices[i - 1]:
                raise ValueError(
                    f"TRAKE row {idx}: frames must be strictly increasing "
                    f"(got {frame_indices[i - 1]} >= {frame_indices[i]})"
                )


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def _write_csv(rows: list[list[str]], output_path: Path) -> None:
    """Write rows to a CSV file (no header, UTF-8, comma-delimited).

    Expected input:
        - rows: list of row lists.
        - output_path: target file path.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
    temp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Actual-file validation (re-reads every generated CSV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvValidation:
    """Validation result for one generated CSV file.

    Expected input:
        - filename: CSV basename.
        - task: ``kis``, ``qa``, or ``trake``.
        - expected_columns: required column count per row.
        - row_count: number of rows read.
        - column_counts: set of column counts actually seen.
        - invalid_rows: 0-based indices of invalid rows.
        - valid: whether the file fully conforms.
    """

    filename: str
    task: str
    expected_columns: int
    row_count: int
    column_counts: set[int]
    invalid_rows: list[int]
    valid: bool


def _csv_row_is_valid(
    row: list[str],
    task: str,
    expected_columns: int,
) -> bool:
    """Validate one CSV row against the BTC rules for its task."""
    if len(row) != expected_columns:
        return False
    video_id = row[0]
    if video_id == "" or video_id.endswith(".mp4"):
        return False
    if task == "qa":
        frame_idx = row[1]
        answer = row[2]
        try:
            int(frame_idx)
        except (ValueError, TypeError):
            return False
        if not answer or len(answer) > QA_ANSWER_MAX_CHARS:
            return False
        return True
    try:
        frame_indices = [int(cell) for cell in row[1:]]
    except (ValueError, TypeError):
        return False
    if task == "trake":
        for i in range(1, len(frame_indices)):
            if frame_indices[i] <= frame_indices[i - 1]:
                return False
    return True


def validate_csv_file(
    path: Path,
    task: str,
    expected_columns: int,
) -> CsvValidation:
    """Re-open one generated CSV and validate every actual row.

    Expected input:
        - path: the CSV file to validate.
        - task: ``kis``, ``qa``, or ``trake``.
        - expected_columns: required column count per row.

    Returns:
        - A ``CsvValidation`` describing the actual file content.

    Raises:
        - FileNotFoundError: if the file does not exist.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    column_counts = {len(row) for row in rows}
    invalid_rows = [
        idx
        for idx, row in enumerate(rows)
        if not _csv_row_is_valid(row, task, expected_columns)
    ]
    valid = (
        len(rows) <= KIS_MAX_ROWS
        and len(rows) > 0
        and column_counts == {expected_columns}
        and not invalid_rows
    )
    return CsvValidation(
        filename=path.name,
        task=task,
        expected_columns=expected_columns,
        row_count=len(rows),
        column_counts=column_counts,
        invalid_rows=invalid_rows,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Zip packaging and inspection
# ---------------------------------------------------------------------------


def create_submission_zip(submission_dir: Path, zip_path: Path) -> None:
    """Create a zip archive containing the submission directory.

    The zip contains the ``submission/`` directory itself (not just the CSV
    files inside it).

    Expected input:
        - submission_dir: the ``submission/`` directory path.
        - zip_path: target zip file path.

    Returns:
        - None.

    Raises:
        - FileNotFoundError: if the submission directory does not exist.
        - OSError: if writing fails.
    """
    if not submission_dir.is_dir():
        raise FileNotFoundError(
            f"submission directory not found: {submission_dir}"
        )
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{submission_dir.name}/", "")
        for csv_file in sorted(submission_dir.glob("*.csv")):
            arcname = f"{submission_dir.name}/{csv_file.name}"
            zf.write(csv_file, arcname)
    temp_zip.replace(zip_path)


def inspect_submission_zip(
    zip_path: Path,
    expected_csv_names: set[str],
) -> dict[str, Any]:
    """Inspect and validate a generated submission zip.

    Expected input:
        - zip_path: the generated ``submission.zip``.
        - expected_csv_names: the set of CSV basenames that must be present
          inside ``submission/``.

    Returns:
        - A dict with ``entries``, ``has_submission_dir``, ``missing``,
          ``unexpected``, and ``valid``.

    Raises:
        - FileNotFoundError: if the zip does not exist.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"submission zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.namelist()

    prefix = "submission/"
    has_submission_dir = any(name == prefix or name.startswith(prefix) for name in entries)
    csv_names = {
        name[len(prefix):]
        for name in entries
        if name.startswith(prefix) and name.endswith(".csv")
    }
    missing = sorted(expected_csv_names - csv_names)
    unexpected = sorted(csv_names - expected_csv_names)
    valid = (
        has_submission_dir
        and not missing
        and not unexpected
        and len(csv_names) == len(expected_csv_names)
    )
    return {
        "entries": entries,
        "has_submission_dir": has_submission_dir,
        "missing": missing,
        "unexpected": unexpected,
        "valid": valid,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _result_id(result: Any) -> str:
    """Return the query id of an artifact result for reporting."""
    return str(getattr(result, "query_id", "?"))


def build_submission(
    query_dir: Path,
    output_dir: Path,
    kis_artifact: Path | None = None,
    qa_artifact: Path | None = None,
    trake_artifact: Path | None = None,
) -> dict[str, Any]:
    """Build the BTC submission from actual query files and prediction artifacts.

    Expected input:
        - query_dir: directory containing ``query-*-<task>.txt`` files.
        - output_dir: output directory for ``submission/`` and
          ``submission.zip``.
        - kis_artifact: optional KIS prediction artifact path.
        - qa_artifact: optional QA prediction artifact directory.
        - trake_artifact: optional TRAKE prediction artifact directory.

    Returns:
        - A summary dict with the mapping report, counts, validation results,
          and zip inspection.

    Raises:
        - FileNotFoundError: if required files are missing.
        - ValueError: if artifacts are invalid, queries cannot be matched
          uniquely, or any generated file fails validation.
    """
    query_files = enumerate_query_files(query_dir)

    kis_results: tuple[QueryPredictionResult, ...] | None = None
    qa_art: QaPredictionArtifact | None = None
    trake_art: TrakePredictionArtifact | None = None

    if kis_artifact is not None:
        logger.info("Loading KIS artifact: %s", kis_artifact)
        kis_results = load_kis_artifacts(kis_artifact)
        logger.info("Loaded %d KIS query results", len(kis_results))
    if qa_artifact is not None:
        logger.info("Loading QA artifact: %s", qa_artifact)
        qa_art = load_qa_artifact(qa_artifact)
        logger.info("Loaded %d QA query results", len(qa_art.query_results))
    if trake_artifact is not None:
        logger.info("Loading TRAKE artifact: %s", trake_artifact)
        trake_art = load_trake_artifact(trake_artifact)
        logger.info("Loaded %d TRAKE query results", len(trake_art.query_results))

    # ------------------------------------------------------------------
    # Mapping phase: match every actual query before generating anything.
    # ------------------------------------------------------------------
    mapping: list[dict[str, str]] = []
    matched: dict[str, tuple[Any, str]] = {}
    errors: list[str] = []
    counts = {"kis": 0, "qa": 0, "trake": 0}
    totals = {"kis": 0, "qa": 0, "trake": 0}

    for info in query_files:
        totals[info.task] += 1
        query_text = read_query_text(info.path)
        try:
            if info.task == "kis":
                if kis_results is None:
                    raise QueryMatchError("KIS artifact not provided")
                result, method = find_unique_match(
                    query_text, kis_results, lambda r: r.query_text
                )
            elif info.task == "qa":
                if qa_art is None:
                    raise QueryMatchError("QA artifact not provided")
                result, method = find_unique_match(
                    query_text, qa_art.query_results, lambda r: r.query_text
                )
            else:
                if trake_art is None:
                    raise QueryMatchError("TRAKE artifact not provided")
                result, method = find_unique_match(
                    query_text, trake_art.query_results, lambda r: r.query_text
                )
        except QueryMatchError as exc:
            errors.append(f"{info.path.name}: {exc}")
            mapping.append(
                {
                    "query_file": info.path.name,
                    "task": info.task,
                    "matched_id": "-",
                    "method": "NO MATCH",
                    "error": str(exc),
                }
            )
            continue

        matched[info.path.name] = (result, method)
        counts[info.task] += 1
        mapping.append(
            {
                "query_file": info.path.name,
                "task": info.task,
                "matched_id": _result_id(result),
                "method": method,
                "error": "",
            }
        )

    # Print mapping report + summary before generating anything.
    print("query_file | task | matched_prediction_id | match_method")
    for row in mapping:
        print(
            f"{row['query_file']} | {row['task']} | {row['matched_id']} | {row['method']}"
        )
    print("")
    print("Matching summary:")
    for task in ("kis", "qa", "trake"):
        print(f"{task.upper()}: {counts[task]}/{totals[task]} matched")
    print("")

    if errors:
        for error in errors:
            logger.error("error: %s", error)
        raise ValueError(
            f"{len(errors)} query file(s) could not be matched uniquely; "
            "no submission generated. See errors above."
        )

    # ------------------------------------------------------------------
    # Generation phase.
    # ------------------------------------------------------------------
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for info in query_files:
        result, _method = matched[info.path.name]
        query_text = read_query_text(info.path)
        csv_path = submission_dir / info.csv_name

        if info.task == "kis":
            export_kis_csv(result, csv_path)
            expected_columns = 2
        elif info.task == "qa":
            export_qa_csv(result, csv_path)
            expected_columns = 3
        else:
            num_events = len(parse_events(query_text))
            artifact_events = len(result.events)
            if num_events != artifact_events:
                raise ValueError(
                    f"{info.path.name}: query requests {num_events} events "
                    f"but the artifact has {artifact_events}"
                )
            export_trake_csv(result, csv_path)
            expected_columns = num_events + 1

        generated.append(
            {
                "query_file": info.path.name,
                "task": info.task,
                "matched_id": _result_id(result),
                "csv_name": csv_path.name,
                "csv_path": csv_path,
                "expected_columns": expected_columns,
            }
        )

    # ------------------------------------------------------------------
    # Validation phase: re-open every generated CSV and validate actual rows.
    # ------------------------------------------------------------------
    print("Per-file validation:")
    print("QUERY FILE | TASK | MATCHED | CSV | ROWS | VALID")
    validations: list[CsvValidation] = []
    all_valid = True
    for item in generated:
        validation = validate_csv_file(
            item["csv_path"], item["task"], item["expected_columns"]
        )
        validations.append(validation)
        all_valid = all_valid and validation.valid
        print(
            f"{item['query_file']} | {item['task']} | {item['matched_id']} | "
            f"{item['csv_name']} | {validation.row_count} | "
            f"{'VALID' if validation.valid else 'INVALID'}"
        )
        if item["task"] == "trake":
            print(
                f"  {item['csv_name']}: events={item['expected_columns'] - 1} "
                f"expected_columns={item['expected_columns']} "
                f"actual_columns={sorted(validation.column_counts)} "
                f"invalid_rows={validation.invalid_rows}"
            )

    if not all_valid:
        for validation in validations:
            if not validation.valid:
                logger.error(
                    "invalid CSV %s: rows=%d expected_cols=%d actual_cols=%s "
                    "invalid_rows=%s",
                    validation.filename,
                    validation.row_count,
                    validation.expected_columns,
                    sorted(validation.column_counts),
                    validation.invalid_rows,
                )
        raise ValueError("generated CSVs failed validation; submission is NOT valid")

    # ------------------------------------------------------------------
    # Packaging phase: zip + inspection.
    # ------------------------------------------------------------------
    zip_path = output_dir / "submission.zip"
    create_submission_zip(submission_dir, zip_path)

    expected_csv_names = {item["csv_name"] for item in generated}
    zip_info = inspect_submission_zip(zip_path, expected_csv_names)
    print("")
    print("submission.zip contents:")
    for entry in zip_info["entries"]:
        print(f"  {entry}")
    print("")
    if not zip_info["valid"]:
        logger.error(
            "zip validation failed: has_submission_dir=%s missing=%s unexpected=%s",
            zip_info["has_submission_dir"],
            zip_info["missing"],
            zip_info["unexpected"],
        )
        raise ValueError("submission.zip failed validation; submission is NOT valid")

    return {
        "query_files": [info.path.name for info in query_files],
        "mapping": mapping,
        "counts": counts,
        "totals": totals,
        "generated": [
            {
                "query_file": item["query_file"],
                "task": item["task"],
                "matched_id": item["matched_id"],
                "csv_name": item["csv_name"],
                "expected_columns": item["expected_columns"],
            }
            for item in generated
        ],
        "validations": [
            {
                "filename": v.filename,
                "task": v.task,
                "expected_columns": v.expected_columns,
                "row_count": v.row_count,
                "column_counts": sorted(v.column_counts),
                "invalid_rows": v.invalid_rows,
                "valid": v.valid,
            }
            for v in validations
        ],
        "zip_path": str(zip_path),
        "zip_info": zip_info,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the BTC submission builder.

    Expected input:
        - argv: optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        - Exit code (0 = success).

    Raises:
        - SystemExit: on argument parsing errors.
    """
    parser = argparse.ArgumentParser(
        description="Build BTC submission CSV files from prediction artifacts.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query-*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for submission/ and submission.zip.",
    )
    parser.add_argument(
        "--kis-artifact",
        type=Path,
        default=None,
        help="Path to KIS prediction artifact (JSON file or directory).",
    )
    parser.add_argument(
        "--qa-artifact",
        type=Path,
        default=None,
        help="Path to QA prediction artifact directory.",
    )
    parser.add_argument(
        "--trake-artifact",
        type=Path,
        default=None,
        help="Path to TRAKE prediction artifact directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = build_submission(
            query_dir=args.query_dir,
            output_dir=args.output_dir,
            kis_artifact=args.kis_artifact,
            qa_artifact=args.qa_artifact,
            trake_artifact=args.trake_artifact,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("")
    print(f"KIS queries:  {result['counts']['kis']}")
    print(f"QA queries:   {result['counts']['qa']}")
    print(f"TRAKE queries: {result['counts']['trake']}")
    print(f"Total CSVs:   {len(result['generated'])}")
    print(f"Submission:   {result['zip_path']}")
    print("")
    print("SUBMISSION VALID: yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())