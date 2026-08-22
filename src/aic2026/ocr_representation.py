"""OCR representation-v2 experiment (Phase 5B).

Deterministic cleaning + salient-unit representation built FROM THE EXISTING
artifacts without rerunning OCR:

    merged OCR records (read-only)
        |
    video-local persistent-line detection
        |
    per-chunk unit reconstruction (original OCR line boundaries)
        |
    conservative garbage filtering
        |
    CLEANED_CONCAT text   /   individual UNIT texts (<= units_per_chunk)
        |
    Jina CLIP v2 text embeddings (elsewhere; this module is text-only)

Transformation rules (all pre-declared, deterministic, inspectable):

Persistent lines
    A normalized OCR line is *persistent* for one video when it appears in
    at least ``persistent_min_coverage`` of that video's ``ok`` records that
    carry any text, and in at least ``persistent_min_records`` such records.
    Detection is strictly VIDEO-LOCAL: corpus frequency never causes
    removal, so identical overlay text in other videos is untouched.

Garbage units
    A unit is dropped when it normalizes to nothing, or when it contains no
    letter character (Unicode letters incl. Vietnamese) and at most one
    alphanumeric character. Nothing else is removed; short normal words,
    digits-as-evidence ("2024"), and accented text are preserved.

Units
    Units are the original per-line OCR texts of the chunk's source records,
    in frame order, deduplicated within the chunk by normalized form
    (first-seen raw text kept - mirrors the Phase 1 aggregation rule), then
    filtered, then capped at ``units_per_chunk`` in first-seen order.

Provenance
    Every removed piece is recorded with a machine-readable reason
    (``persistent`` or ``garbage``); original chunk identity is preserved.

Empty results
    A chunk whose units are all removed yields an empty cleaned text and no
    units; callers decide visibility, and the diagnostics report counts.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_generation import KeyframeOcr, normalize_ocr_text

logger = logging.getLogger(__name__)


class RepresentationError(RuntimeError):
    """Raised when the representation-v2 transformation cannot be built."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _has_letter(text: str) -> bool:
    """Return whether ``text`` contains at least one Unicode letter."""
    return any(char.isalpha() for char in text)


@dataclass(frozen=True)
class RepresentationConfig:
    """Pre-declared thresholds for the representation-v2 transform.

    Fields:
        - persistent_min_coverage: fraction of a video's text-bearing ``ok``
          records that must contain a normalized line for it to count as
          persistent.
        - persistent_min_records: minimum number of text-bearing records a
          line must appear in (guards small videos).
        - units_per_chunk: maximum number of surviving units kept per chunk
          (first-seen order; an engineering compute bound, not tuned).
    """

    persistent_min_coverage: float = 0.5
    persistent_min_records: int = 3
    units_per_chunk: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.persistent_min_coverage, (int, float)) or isinstance(
            self.persistent_min_coverage, bool
        ) or not 0.0 < self.persistent_min_coverage <= 1.0:
            raise ValueError(
                f"persistent_min_coverage must be in (0, 1], got {self.persistent_min_coverage!r}"
            )
        for name in ("persistent_min_records", "units_per_chunk"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be >= 1, got {value!r}")


@dataclass(frozen=True)
class RemovedPiece:
    """One removed text piece with its reason.

    Fields:
        - text: the raw piece text.
        - reason: ``"persistent"`` or ``"garbage"``.
    """

    text: str
    reason: str


@dataclass(frozen=True)
class CleanedChunk:
    """One OCR chunk after cleaning, with full provenance.

    Fields:
        - chunk_id / video_id / representative_frame_index / frame_start /
          frame_end / source_count: copied unchanged from the source chunk.
        - text: cleaned concatenation (``" ".join`` of surviving pieces);
          empty when everything was removed.
        - units: surviving unit texts in first-seen order (<= config cap).
        - removed: removed pieces in encounter order.
    """

    chunk_id: str
    video_id: str
    representative_frame_index: int
    frame_start: int
    frame_end: int
    source_count: int
    text: str
    units: tuple[str, ...]
    removed: tuple[RemovedPiece, ...] = field(default_factory=tuple)


def collect_video_persistent_lines(
    records: Sequence[KeyframeOcr],
    *,
    min_coverage: float,
    min_records: int,
) -> set[str]:
    """Return the persistent normalized OCR line texts of ONE video.

    Expected input:
        - records: all ``ok`` OCR records of a single video.
        - min_coverage: required fraction of text-bearing records containing
          the line.
        - min_records: required minimum number of text-bearing records
          containing the line.

    Returns:
        - Set of normalized line texts considered persistent overlays for
          this video.

    Raises:
        - No exceptions (an empty record list yields an empty set).
    """
    text_bearing = [record for record in records if record.status == "ok" and record.ocr_lines]
    if not text_bearing:
        return set()
    threshold = max(min_records, min_coverage * len(text_bearing))
    counts: Counter[str] = Counter()
    for record in text_bearing:
        for line in record.ocr_lines:
            normalized = normalize_ocr_text(line.text)
            if normalized:
                counts[normalized] += 1
    return {line for line, count in counts.items() if count >= threshold}


def _chunk_source_records(
    chunk: OcrChunk,
    records_by_ordinal: dict[tuple[str, int], KeyframeOcr],
) -> list[KeyframeOcr]:
    """Return the chunk's source records in frame order."""
    records: list[KeyframeOcr] = []
    for ordinal in sorted(chunk.keyframe_ordinals):
        record = records_by_ordinal.get((chunk.video_id, ordinal))
        if record is not None:
            records.append(record)
    return records


def clean_chunk(
    chunk: OcrChunk,
    source_records: Sequence[KeyframeOcr],
    persistent_lines: set[str],
    config: RepresentationConfig,
) -> CleanedChunk:
    """Build the cleaned representation of one chunk.

    Expected input:
        - chunk: source chunk (existing schema).
        - source_records: the chunk's ``ok`` OCR records in frame order
          (original per-line boundaries come from here).
        - persistent_lines: video-local persistent normalized line texts.
        - config: transformation thresholds.

    Returns:
        - The :class:`CleanedChunk` with surviving units and removal
          provenance.

    Raises:
        - No exceptions (malformed pieces are skipped with a reason).
    """
    seen: set[str] = set()
    units: list[str] = []
    removed: list[RemovedPiece] = []

    for record in source_records:
        if record.status != "ok":
            continue
        for line in record.ocr_lines:
            raw = " ".join(line.text.split())
            if raw == "":
                continue
            normalized = normalize_ocr_text(raw)
            if normalized == "":
                removed.append(RemovedPiece(text=raw, reason="garbage"))
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized in persistent_lines:
                removed.append(RemovedPiece(text=raw, reason="persistent"))
                continue
            if not _has_letter(raw) and sum(char.isalnum() for char in raw) <= 1:
                removed.append(RemovedPiece(text=raw, reason="garbage"))
                continue
            if len(units) < config.units_per_chunk:
                units.append(raw)
            else:
                removed.append(RemovedPiece(text=raw, reason="cap"))

    return CleanedChunk(
        chunk_id=chunk.chunk_id,
        video_id=chunk.video_id,
        representative_frame_index=chunk.representative_frame_index,
        frame_start=chunk.frame_start,
        frame_end=chunk.frame_end,
        source_count=chunk.source_count,
        text=" ".join(units),
        units=tuple(units),
        removed=tuple(removed),
    )


def build_cleaned_representation(
    chunks: Sequence[OcrChunk],
    records: Sequence[KeyframeOcr],
    config: RepresentationConfig,
) -> tuple[list[CleanedChunk], dict[str, Any]]:
    """Transform every chunk into the cleaned/unit representation.

    Expected input:
        - chunks: all source chunks (existing chunk artifact order).
        - records: ALL merged OCR records (any status; read-only).
        - config: transformation thresholds.

    Returns:
        - ``(cleaned_chunks, diagnostics)`` where ``cleaned_chunks`` keeps
          the input order and ``diagnostics`` aggregates removal statistics
          per reason plus per-video persistent-line counts.

    Raises:
        - ValueError: if duplicate chunk ids exist or config is invalid.
    """
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise RepresentationError("duplicate chunk ids in input")

    records_by_video: dict[str, list[KeyframeOcr]] = {}
    records_by_ordinal: dict[tuple[str, int], KeyframeOcr] = {}
    for record in records:
        records_by_video.setdefault(record.video_id.value, []).append(record)
        records_by_ordinal[(record.video_id.value, record.keyframe_ordinal)] = record

    persistent_by_video = {
        video: collect_video_persistent_lines(
            video_records,
            min_coverage=config.persistent_min_coverage,
            min_records=config.persistent_min_records,
        )
        for video, video_records in records_by_video.items()
    }

    cleaned: list[CleanedChunk] = []
    reason_counts: Counter[str] = Counter()
    persistent_sizes = [len(lines) for lines in persistent_by_video.values()]
    for chunk in chunks:
        cleaned_chunk = clean_chunk(
            chunk,
            _chunk_source_records(chunk, records_by_ordinal),
            persistent_by_video.get(chunk.video_id, set()),
            config,
        )
        for piece in cleaned_chunk.removed:
            reason_counts[piece.reason] += 1
        cleaned.append(cleaned_chunk)

    non_empty = sum(1 for c in cleaned if c.text != "")
    diagnostics: dict[str, Any] = {
        "num_chunks": len(cleaned),
        "num_non_empty_after_cleaning": non_empty,
        "num_empty_after_cleaning": len(cleaned) - non_empty,
        "num_units_total": sum(len(c.units) for c in cleaned),
        "removed_reason_counts": dict(sorted(reason_counts.items())),
        "videos_with_persistent_lines": sum(1 for s in persistent_by_video.values() if s),
        "persistent_lines_per_video": {
            "mean": (sum(persistent_sizes) / len(persistent_sizes)) if persistent_sizes else 0.0,
            "max": max(persistent_sizes) if persistent_sizes else 0,
        },
    }
    logger.info(
        "representation-v2: %d/%d chunks non-empty, %d units",
        non_empty,
        len(cleaned),
        diagnostics["num_units_total"],
    )
    return cleaned, diagnostics


def unit_max_scores(
    query_vector: np.ndarray,
    unit_matrix: np.ndarray,
    unit_norms: np.ndarray,
    chunk_unit_indices: dict[str, Sequence[int]],
    num_chunks: int,
) -> np.ndarray:
    """Score every chunk as the max cosine similarity over its units.

    Expected input:
        - query_vector: L2-normalized query embedding (float, 1-D).
        - unit_matrix: ``(num_unique_units, dim)`` float32 matrix.
        - unit_norms: per-row L2 norms of ``unit_matrix``.
        - chunk_unit_indices: mapping chunk row index -> unique-unit indices.
        - num_chunks: total number of chunks (rows in the output).

    Returns:
        - float32 array of per-chunk UNIT_MAX scores; chunks without units
          get ``-inf``.

    Raises:
        - ValueError: if the query vector dimension mismatches.
    """
    query32 = np.asarray(query_vector, dtype=np.float32)
    _require(query32.ndim == 1, "query vector must be 1-D")
    _require(unit_matrix.shape[1] == query32.shape[0], "dimension mismatch")
    query_norm = float(np.linalg.norm(query32))
    out = np.full(num_chunks, -np.inf, dtype=np.float32)
    if unit_matrix.shape[0] == 0 or query_norm == 0.0:
        return out
    sims = (unit_matrix @ query32) / (unit_norms * np.float32(query_norm))
    for chunk_row, indices in chunk_unit_indices.items():
        if indices:
            out[chunk_row] = float(sims[list(indices)].max())
    return out
