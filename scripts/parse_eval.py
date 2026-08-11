"""Convert team annotation CSV exports into validated evaluation JSONL records.

Usage example:
    python -m scripts.parse_eval --csv eval.csv --out data/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.data.loader import load_index
from src.utils.config import load_config


_REQUIRED_COLUMNS = {
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
}
_VALID_TASKS = {"kis", "qa", "trake"}
_EVENT_PATTERN = re.compile(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for annotation conversion.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed CSV, output, config, and optional index-override paths.
    Raises: SystemExit if the supplied CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Convert an annotation CSV export into validated evaluation JSONL."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Annotation CSV export.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML configuration file used to find the index.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional index path overriding index.path in the configuration.",
    )
    return parser.parse_args(arguments)


def get_cell(row: dict[str | None, str | None], column: str) -> str:
    """Return one CSV cell as a string, replacing absent values with an empty string.

    Expected input: ``row`` is a ``csv.DictReader`` row and ``column`` is a header name.
    Returns: The cell value, or ``""`` when the cell is absent or empty.
    Raises: None.
    """
    value = row.get(column)
    return value if isinstance(value, str) else ""


def describe_row(row_number: int, query_id: str) -> str:
    """Create a consistent location prefix for validation messages.

    Expected input: ``row_number`` is the one-based physical CSV row number and
    ``query_id`` is the stripped query identifier, possibly empty.
    Returns: A human-readable row and query-ID location string.
    Raises: None.
    """
    return f"row {row_number} (query_id={query_id!r})"


def parse_nonnegative_int(
    value: str,
    field_name: str,
    location: str,
    errors: list[str],
) -> int | None:
    """Parse a required non-negative integer cell and record a validation error on failure.

    Expected input: ``value`` is a CSV cell, ``field_name`` identifies that cell, and
    ``errors`` is a mutable list for messages.
    Returns: The parsed integer, or ``None`` if the cell is not an integer or is negative.
    Raises: None.
    """
    try:
        parsed = int(value.strip())
    except ValueError:
        errors.append(f"{location}: {field_name} must be an integer >= 0.")
        return None
    if parsed < 0:
        errors.append(f"{location}: {field_name} must be an integer >= 0.")
        return None
    return parsed


def parse_events(value: str, location: str, errors: list[str]) -> list[list[int]] | None:
    """Parse semicolon-separated inclusive event ranges from a CSV cell.

    Expected input: ``value`` follows ``s-e;s-e;...`` with integer endpoints, while
    ``errors`` is a mutable list for messages.
    Returns: Parsed ``[[start, end], ...]`` ranges, or ``None`` if any range is invalid.
    Raises: None.
    """
    parts = value.split(";")
    if not value.strip() or any(not part.strip() for part in parts):
        errors.append(f"{location}: gt_events must contain at least one 's-e' pair.")
        return None

    events: list[list[int]] = []
    is_valid = True
    for part in parts:
        match = _EVENT_PATTERN.fullmatch(part)
        if match is None:
            errors.append(f"{location}: invalid gt_events pair {part!r}; expected 's-e'.")
            is_valid = False
            continue
        start, end = (int(endpoint) for endpoint in match.groups())
        if start > end:
            errors.append(
                f"{location}: gt_events pair {part!r} has start greater than end."
            )
            is_valid = False
            continue
        events.append([start, end])
    return events if is_valid and events else None


def validate_rows(
    rows: Sequence[dict[str | None, str | None]], video_ids: set[str]
) -> tuple[list[dict[str, Any]], list[str], Counter[str]]:
    """Validate annotation rows and construct records for rows without errors.

    Expected input: ``rows`` are CSV rows in source order and ``video_ids`` contains every
    valid video identifier from the loaded index.
    Returns: ``(records, errors, task_counts)`` where records use the evaluation JSONL
        schema, errors contains every row-level validation message, and task counts cover
        all input rows with a recognized task.
    Raises: None.
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    task_counts: Counter[str] = Counter()
    seen_query_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        query_id = get_cell(row, "query_id").strip()
        task = get_cell(row, "task").strip()
        location = describe_row(row_number, query_id)
        row_error_count = len(errors)

        if not query_id:
            errors.append(f"{location}: query_id must be non-empty.")
        elif query_id in seen_query_ids:
            errors.append(f"{location}: query_id must be unique.")
        else:
            seen_query_ids.add(query_id)

        if task not in _VALID_TASKS:
            errors.append(f"{location}: task must be one of {sorted(_VALID_TASKS)}.")
        else:
            task_counts[task] += 1

        query_text_en = get_cell(row, "query_text_en")
        if not query_text_en.strip():
            errors.append(f"{location}: query_text_en must be non-empty.")

        video_id = get_cell(row, "gt_video_id").strip()
        if video_id not in video_ids:
            errors.append(
                f"{location}: gt_video_id {video_id!r} does not exist in the index."
            )

        record: dict[str, Any] = {
            "query_id": query_id,
            "task": task,
            "query_text_vi": get_cell(row, "query_text_vi"),
            "query_text_en": query_text_en,
            "gt_video_id": video_id,
        }

        if task in {"kis", "qa"}:
            start = parse_nonnegative_int(
                get_cell(row, "gt_frame_start"), "gt_frame_start", location, errors
            )
            end = parse_nonnegative_int(
                get_cell(row, "gt_frame_end"), "gt_frame_end", location, errors
            )
            if start is not None and end is not None:
                if start > end:
                    errors.append(
                        f"{location}: gt_frame_start must not exceed gt_frame_end."
                    )
                else:
                    record["gt_frame_range"] = [start, end]

            if task == "qa":
                answer = get_cell(row, "gt_answer")
                if not answer.strip():
                    errors.append(f"{location}: gt_answer must be non-empty for qa.")
                else:
                    record["gt_answer"] = answer

        elif task == "trake":
            events = parse_events(get_cell(row, "gt_events"), location, errors)
            if events is not None:
                record["gt_events"] = events

        if len(errors) == row_error_count:
            records.append(record)

    return records, errors, task_counts


def write_jsonl(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Write evaluation records as UTF-8 JSON Lines in source order.

    Expected input: ``records`` contains JSON-serializable evaluation objects and
    ``output_path`` identifies the destination JSONL file.
    Returns: None.
    Raises: OSError if the destination directory or file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(arguments: Sequence[str] | None = None) -> None:
    """Validate an annotation CSV against the index and write its evaluation JSONL.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``;
    the CSV has the documented annotation headers and the configured index is readable.
    Returns: None.
    Raises: SystemExit with a non-zero status when CSV validation fails; FileNotFoundError
        if an input is missing; OSError if the output cannot be written.
    """
    args = parse_args(arguments)
    config = load_config(args.config)
    index_path = args.index or config["index"]["path"]
    _, vids, _ = load_index(index_path)
    video_ids = set(vids.tolist())

    with args.csv.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(_REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            print(
                f"CSV is missing required columns: {', '.join(missing_columns)}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        rows = list(reader)

    records, errors, task_counts = validate_rows(rows, video_ids)
    print(f"total_rows: {len(rows)}")
    for task in sorted(_VALID_TASKS):
        print(f"{task}_rows: {task_counts[task]}")
    print(f"errors: {len(errors)}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    write_jsonl(records, args.out)
    print(f"output_path: {args.out}")


if __name__ == "__main__":
    main()
