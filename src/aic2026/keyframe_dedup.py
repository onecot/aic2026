"""Deduplicated keyframe artifact preprocessing.

The BTC-provided ``frames.ndjson`` may contain duplicate ``(video_id,
frame_index)`` records (a record is keyed by ``(keyframe.video_id.value,
frame_index)``, the 0-based position in the original encoded video). The
embedding splitter intentionally rejects such duplicates (see
``aic2026.embedding_split``), so experiments need a reproducible
preprocessing step that produces a canonical, deduplicated ``frames.ndjson``
before split -> embed -> index.

This module is a pure preprocessing stage:

- it reads a canonical ``frames.ndjson``;
- keeps the first occurrence of each ``(video_id, frame_index)`` key;
- preserves the original ordering of the remaining records;
- never modifies the input artifact;
- writes the deduplicated output atomically;
- optionally writes a JSON ``DedupReport`` atomically.

It does not touch retrieval, evaluation, scoring, index building, or
embedding logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.canonical import KeyframeTiming

#: Schema version of the dedup report.
DEDUP_REPORT_SCHEMA_VERSION = "1.0"

#: Filename suggestion for the report inside an artifact directory.
DEDUP_REPORT_FILENAME = "dedup_report.json"

#: Strategy used when a duplicate key is encountered.
KEEP_FIRST_STRATEGY = "keep_first"

logger = logging.getLogger(__name__)


class DedupError(RuntimeError):
    """Raised when the deduplicated output fails post-write validation."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _frame_key(frame: KeyframeTiming) -> tuple[str, int]:
    """Return the identity key ``(video_id, frame_index)`` of a record."""
    return (frame.keyframe.video_id.value, frame.frame_index)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


@dataclass(frozen=True)
class DedupReport:
    """Summary of one deduplication run.

    Fields:
        - source_path: the input ``frames.ndjson`` path.
        - output_path: the written deduplicated ``frames.ndjson`` path.
        - original_records: number of non-empty records in the input.
        - unique_records: number of records written to the output.
        - removed_duplicates: ``original_records - unique_records``.
        - duplicate_examples: the first duplicate keys ``(video_id,
          frame_index)`` (first-seen order, one entry per key).
        - strategy: the deduplication strategy (``keep_first``).
    """

    source_path: str
    output_path: str
    original_records: int
    unique_records: int
    removed_duplicates: int
    duplicate_examples: tuple[tuple[str, int], ...]
    strategy: str

    def __post_init__(self) -> None:
        _require(isinstance(self.original_records, int) and self.original_records >= 0, "original_records must be >= 0")
        _require(isinstance(self.unique_records, int) and self.unique_records >= 0, "unique_records must be >= 0")
        _require(isinstance(self.removed_duplicates, int) and self.removed_duplicates >= 0, "removed_duplicates must be >= 0")
        _require(
            self.unique_records + self.removed_duplicates == self.original_records,
            "unique_records + removed_duplicates must equal original_records",
        )
        _require(self.strategy == KEEP_FIRST_STRATEGY, f"unsupported strategy {self.strategy!r}")
        _require(
            all(isinstance(vid, str) and isinstance(idx, int) for vid, idx in self.duplicate_examples),
            "duplicate_examples must be (video_id, frame_index) pairs",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Returns:
            - Dict with the schema version and all report fields.
        """
        return {
            "schema_version": DEDUP_REPORT_SCHEMA_VERSION,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "original_records": self.original_records,
            "unique_records": self.unique_records,
            "removed_duplicates": self.removed_duplicates,
            "duplicate_examples": [[vid, idx] for vid, idx in self.duplicate_examples],
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DedupReport":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid or the schema version is
              unsupported.
        """
        schema_version = data.get("schema_version")
        _require(
            schema_version == DEDUP_REPORT_SCHEMA_VERSION,
            f"unsupported dedup report schema version {schema_version!r}",
        )
        return cls(
            source_path=data["source_path"],
            output_path=data["output_path"],
            original_records=int(data["original_records"]),
            unique_records=int(data["unique_records"]),
            removed_duplicates=int(data["removed_duplicates"]),
            duplicate_examples=tuple((vid, int(idx)) for vid, idx in data["duplicate_examples"]),
            strategy=data["strategy"],
        )


def deduplicate_keyframes(
    input_path: Path | str,
    output_path: Path | str,
    report_path: Path | str | None = None,
    *,
    max_examples: int = 5,
) -> DedupReport:
    """Deduplicate a canonical ``frames.ndjson`` keeping the first occurrence.

    A record is keyed by ``(keyframe.video_id.value, frame_index)``. The first
    occurrence of each key is kept; later occurrences are dropped. The
    relative order of the kept records is unchanged. The input file is never
    modified. The output is written atomically (via a same-directory temp
    file); on any error no partial output is left behind. Empty lines in the
    input are skipped (both for counting and output), matching the canonical
    readers.

    Expected input:
        - input_path: path to a canonical ``frames.ndjson``.
        - output_path: target path for the deduplicated ``frames.ndjson``.
        - report_path: optional path for the JSON :class:`DedupReport`.
        - max_examples: number of duplicate keys to record as examples.

    Returns:
        - A :class:`DedupReport` summarizing the run.

    Raises:
        - FileNotFoundError: if ``input_path`` does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the canonical contract.
        - DedupError: if the written output fails post-write validation.
    """
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 0:
        raise ValueError(f"max_examples must be a non-negative integer, got {max_examples!r}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input frames file does not exist: {input_path}")

    seen: set[tuple[str, int]] = set()
    example_keys: list[tuple[str, int]] = []
    example_seen: set[tuple[str, int]] = set()
    kept_lines: list[str] = []
    original_records = 0
    removed_duplicates = 0

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            original_records += 1
            frame = KeyframeTiming.from_dict(json.loads(line))
            key = _frame_key(frame)
            if key in seen:
                removed_duplicates += 1
                if len(example_keys) < max_examples and key not in example_seen:
                    example_keys.append(key)
                    example_seen.add(key)
                continue
            seen.add(key)
            kept_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.writelines(kept_lines)
        tmp.replace(output_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    # Post-write validation: the output must be a readable canonical file with
    # exactly the kept row count, and it must contain no duplicate keys.
    written = 0
    read_keys: set[tuple[str, int]] = set()
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            written += 1
            key = _frame_key(KeyframeTiming.from_dict(json.loads(line)))
            if key in read_keys:
                raise DedupError(f"output still contains duplicate key {key!r}: {output_path}")
            read_keys.add(key)
    if written != len(kept_lines):
        raise DedupError(
            f"output row count ({written}) does not match expected ({len(kept_lines)}): {output_path}"
        )

    report = DedupReport(
        source_path=str(input_path.resolve()),
        output_path=str(output_path.resolve()),
        original_records=original_records,
        unique_records=len(kept_lines),
        removed_duplicates=removed_duplicates,
        duplicate_examples=tuple(example_keys),
        strategy=KEEP_FIRST_STRATEGY,
    )
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, report.to_dict())

    logger.info(
        "deduplicated %d records -> %d (removed %d) strategy=%s output=%s",
        report.original_records,
        report.unique_records,
        report.removed_duplicates,
        report.strategy,
        output_path,
    )
    return report
