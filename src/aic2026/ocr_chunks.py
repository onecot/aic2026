"""Deterministic temporal chunking of frame-level OCR records.

Groups the records of one completed OCR artifact (see
``aic2026.ocr_generation``) into temporal OCR chunks for retrieval:

    (video_id, ordered OCR frames)
        |
    group within the same video
        |
    temporal windows
        |
    deduplicate repeated OCR text
        |
    OCR chunk

Chunking policy (deterministic):

- records are ordered by ``(video_id, frame_index)``;
- a new window starts when the video changes, when the ``frame_index``
  distance to the previous record exceeds ``max_frame_gap``, or when the
  current window already holds ``max_records`` records. The gap criterion is
  a safety valve for sparse keyframe artifacts; on the current dense
  canonical artifact (largest observed intra-video gap: 352 frames) windows
  are bounded by ``max_records`` alone;
- a window whose aggregated text is empty (all-error or all-empty-text
  records) produces no chunk; its records are counted as skipped in the
  manifest so source coverage stays verifiable:
  ``num_source_records_chunked + num_source_records_skipped ==
  num_source_records``.

Text aggregation per window, in frame order:

- records with ``status != "ok"`` contribute no text;
- per-line texts (reading order) are whitespace-normalized; empty pieces
  are dropped;
- pieces whose :func:`aic2026.ocr_generation.normalize_ocr_text` form was
  already seen in the window are collapsed (case-insensitive dedup); the
  first-seen raw piece is kept;
- retained pieces are joined with single spaces in first-appearance order.
  No spelling correction, translation, or accent stripping is applied.

The representative frame of a chunk is its middle source frame
(``frame_indices[len(frame_indices) // 2]``), a deterministic choice that
always lies inside the chunk's frame range.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from aic2026.ocr_generation import (
    KeyframeOcr,
    normalize_ocr_text,
)

logger = logging.getLogger(__name__)

#: Chunk artifact schema version.
OCR_CHUNKS_SCHEMA_VERSION = "1.0"

#: NDJSON file holding the chunk rows.
CHUNKS_FILENAME = "chunks.ndjson"

#: Manifest file describing the chunk artifact.
MANIFEST_FILENAME = "manifest.json"

#: Default maximum number of source records per chunk.
DEFAULT_MAX_RECORDS = 20

#: Default maximum ``frame_index`` distance between adjacent records of one chunk.
DEFAULT_MAX_FRAME_GAP = 500


class OcrChunkError(RuntimeError):
    """Raised when an OCR chunk artifact cannot be built or is inconsistent."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


@dataclass(frozen=True)
class OcrChunk:
    """One temporal OCR chunk over the keyframes of a single video.

    Fields:
        - chunk_id: deterministic id ``<video_id>-ocr-<seq>`` where ``seq``
          is the 0-based chunk counter within the video.
        - video_id: identity of the video; chunks never span videos.
        - frame_start / frame_end: min/max ``frame_index`` of the sources.
        - representative_frame_index: middle source frame index (deterministic,
          always inside ``[frame_start, frame_end]``).
        - keyframe_ordinals: source keyframe ordinals in frame order.
        - frame_indices: source frame indices in frame order.
        - text: deduplicated OCR text in first-appearance order.
        - source_count: number of source OCR records grouped into the chunk
          (any status; error/empty records count toward provenance but
          contribute no text).
    """

    chunk_id: str
    video_id: str
    frame_start: int
    frame_end: int
    representative_frame_index: int
    keyframe_ordinals: tuple[int, ...]
    frame_indices: tuple[int, ...]
    text: str
    source_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "representative_frame_index": self.representative_frame_index,
            "keyframe_ordinals": list(self.keyframe_ordinals),
            "frame_indices": list(self.frame_indices),
            "text": self.text,
            "source_count": self.source_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrChunk":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        chunk = cls(
            chunk_id=str(data["chunk_id"]),
            video_id=str(data["video_id"]),
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            representative_frame_index=int(data["representative_frame_index"]),
            keyframe_ordinals=tuple(int(v) for v in data["keyframe_ordinals"]),
            frame_indices=tuple(int(v) for v in data["frame_indices"]),
            text=str(data["text"]),
            source_count=int(data["source_count"]),
        )
        _require(chunk.text != "", f"chunk {chunk.chunk_id!r} has empty text")
        return chunk


@dataclass(frozen=True)
class OcrChunksManifest:
    """Versioned metadata describing one OCR chunk artifact.

    Fields:
        - schema_version: chunk artifact schema version
          (``OCR_CHUNKS_SCHEMA_VERSION``).
        - artifact_id: identifier for this artifact.
        - created_at: UTC creation timestamp (ISO-8601).
        - source_ocr_artifact: path of the OCR artifact the chunks were
          built from.
        - max_records / max_frame_gap: chunking parameters used.
        - num_chunks / num_videos: chunk and distinct-video counts.
        - num_source_records: total records in the source OCR artifact.
        - num_source_records_chunked: records grouped into emitted chunks.
        - num_source_records_skipped: records in dropped empty-text windows;
          ``chunked + skipped == num_source_records``.
    """

    schema_version: str
    artifact_id: str
    created_at: str
    source_ocr_artifact: str
    max_records: int
    max_frame_gap: int
    num_chunks: int
    num_videos: int
    num_source_records: int
    num_source_records_chunked: int
    num_source_records_skipped: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "source_ocr_artifact": self.source_ocr_artifact,
            "max_records": self.max_records,
            "max_frame_gap": self.max_frame_gap,
            "num_chunks": self.num_chunks,
            "num_videos": self.num_videos,
            "num_source_records": self.num_source_records,
            "num_source_records_chunked": self.num_source_records_chunked,
            "num_source_records_skipped": self.num_source_records_skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrChunksManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        manifest = cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            created_at=str(data["created_at"]),
            source_ocr_artifact=str(data["source_ocr_artifact"]),
            max_records=int(data["max_records"]),
            max_frame_gap=int(data["max_frame_gap"]),
            num_chunks=int(data["num_chunks"]),
            num_videos=int(data["num_videos"]),
            num_source_records=int(data["num_source_records"]),
            num_source_records_chunked=int(data["num_source_records_chunked"]),
            num_source_records_skipped=int(data["num_source_records_skipped"]),
        )
        _require(
            manifest.schema_version == OCR_CHUNKS_SCHEMA_VERSION,
            f"unsupported chunk artifact schema version: {manifest.schema_version!r}",
        )
        _require(manifest.max_records >= 1, "max_records must be >= 1")
        _require(manifest.max_frame_gap >= 0, "max_frame_gap must be >= 0")
        return manifest


def aggregate_window_text(records: Sequence[KeyframeOcr]) -> str:
    """Return the deduplicated OCR text of one temporal window of records.

    Expected input:
        - records: OCR records of a single window, in frame order.

    Returns:
        - Whitespace-normalized pieces joined with single spaces in
          first-appearance order; error records and empty pieces contribute
          nothing; repeated pieces (compared via
          :func:`aic2026.ocr_generation.normalize_ocr_text`) appear once.

    Raises:
        - No exceptions (malformed pieces are skipped).
    """
    seen: set[str] = set()
    pieces: list[str] = []
    for record in records:
        if record.status != "ok":
            continue
        candidates = [line.text for line in record.ocr_lines]
        if not candidates and record.ocr_text.strip():
            candidates = [record.ocr_text]
        for candidate in candidates:
            piece = " ".join(candidate.split())
            if piece == "":
                continue
            key = normalize_ocr_text(piece)
            if key in seen:
                continue
            seen.add(key)
            pieces.append(piece)
    return " ".join(pieces)


def build_ocr_chunks(
    records: Sequence[KeyframeOcr],
    *,
    artifact_id: str,
    source_ocr_artifact: str,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    created_at: str | None = None,
) -> tuple[OcrChunksManifest, list[OcrChunk]]:
    """Build temporal OCR chunks from frame-level OCR records.

    Expected input:
        - records: OCR records of one completed OCR artifact (row order).
        - artifact_id: identifier recorded in the manifest.
        - source_ocr_artifact: source artifact path recorded in the manifest.
        - max_records: maximum number of source records per chunk (>= 1).
        - max_frame_gap: maximum ``frame_index`` distance between adjacent
          records of one chunk (>= 0).
        - created_at: optional fixed UTC timestamp; defaults to now (mainly
          for tests).

    Returns:
        - ``(manifest, chunks)`` where chunks are ordered by video (first
          appearance in the sorted input) and frame order within each video.

    Raises:
        - ValueError: if ``max_records < 1``, ``max_frame_gap < 0``, or a
          duplicate source key ``(video_id, keyframe_ordinal, frame_index)``
          exists in ``records``.
    """
    _require(isinstance(max_records, int) and max_records >= 1, "max_records must be >= 1")
    _require(isinstance(max_frame_gap, int) and max_frame_gap >= 0, "max_frame_gap must be >= 0")

    ordered = sorted(records, key=lambda r: (r.video_id.value, r.frame_index))
    seen_keys: set[tuple[str, int, int]] = set()
    for record in ordered:
        key = (record.video_id.value, record.keyframe_ordinal, record.frame_index)
        if key in seen_keys:
            raise ValueError(f"duplicate source OCR record: {key}")
        seen_keys.add(key)

    windows = _split_windows(ordered, max_records=max_records, max_frame_gap=max_frame_gap)

    chunks: list[OcrChunk] = []
    skipped = 0
    per_video_seq: dict[str, int] = {}
    for window in windows:
        text = aggregate_window_text(window)
        if text == "":
            skipped += len(window)
            continue
        video_id = window[0].video_id.value
        seq = per_video_seq.get(video_id, 0)
        per_video_seq[video_id] = seq + 1
        frame_indices = tuple(record.frame_index for record in window)
        chunks.append(
            OcrChunk(
                chunk_id=f"{video_id}-ocr-{seq:02d}",
                video_id=video_id,
                frame_start=frame_indices[0],
                frame_end=frame_indices[-1],
                representative_frame_index=frame_indices[len(frame_indices) // 2],
                keyframe_ordinals=tuple(record.keyframe_ordinal for record in window),
                frame_indices=frame_indices,
                text=text,
                source_count=len(window),
            )
        )

    manifest = OcrChunksManifest(
        schema_version=OCR_CHUNKS_SCHEMA_VERSION,
        artifact_id=artifact_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        source_ocr_artifact=source_ocr_artifact,
        max_records=max_records,
        max_frame_gap=max_frame_gap,
        num_chunks=len(chunks),
        num_videos=len(per_video_seq),
        num_source_records=len(records),
        num_source_records_chunked=sum(chunk.source_count for chunk in chunks),
        num_source_records_skipped=skipped,
    )
    logger.info(
        "built %d OCR chunks over %d videos (%d/%d source records chunked)",
        manifest.num_chunks,
        manifest.num_videos,
        manifest.num_source_records_chunked,
        manifest.num_source_records,
    )
    return manifest, chunks


def _split_windows(
    ordered: Sequence[KeyframeOcr],
    *,
    max_records: int,
    max_frame_gap: int,
) -> list[list[KeyframeOcr]]:
    """Split frame-ordered records into temporal windows.

    A new window starts on a video change, on a ``frame_index`` distance
    above ``max_frame_gap`` from the previous record, or when the current
    window already holds ``max_records`` records.
    """
    windows: list[list[KeyframeOcr]] = []
    current: list[KeyframeOcr] = []
    for record in ordered:
        boundary = (
            current
            and (
                record.video_id.value != current[0].video_id.value
                or record.frame_index - current[-1].frame_index > max_frame_gap
                or len(current) >= max_records
            )
        )
        if boundary:
            windows.append(current)
            current = []
        current.append(record)
    if current:
        windows.append(current)
    return windows


def validate_ocr_chunks(manifest: OcrChunksManifest, chunks: Sequence[OcrChunk]) -> list[str]:
    """Validate that a chunk list matches its manifest and internal contracts.

    Expected input:
        - manifest: the artifact's manifest.
        - chunks: chunk rows in artifact order.

    Returns:
        - List of human-readable problems; empty when consistent.

    Raises:
        - No exceptions (problems are accumulated and returned).
    """
    problems: list[str] = []
    if len(chunks) != manifest.num_chunks:
        problems.append(f"chunk rows ({len(chunks)}) != manifest.num_chunks ({manifest.num_chunks})")
    videos = {chunk.video_id for chunk in chunks}
    if len(videos) != manifest.num_videos:
        problems.append(f"distinct videos ({len(videos)}) != manifest.num_videos ({manifest.num_videos})")
    chunked = sum(chunk.source_count for chunk in chunks)
    if chunked != manifest.num_source_records_chunked:
        problems.append(
            f"sum(source_count) ({chunked}) != manifest.num_source_records_chunked "
            f"({manifest.num_source_records_chunked})"
        )
    if chunked + manifest.num_source_records_skipped != manifest.num_source_records:
        problems.append(
            f"chunked ({chunked}) + skipped ({manifest.num_source_records_skipped}) "
            f"!= num_source_records ({manifest.num_source_records})"
        )

    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, int, int]] = set()
    seen_videos: set[str] = set()
    last_video: str | None = None
    last_frame_start: int | None = None
    for chunk in chunks:
        label = chunk.chunk_id or "<missing id>"
        if chunk.chunk_id in seen_ids:
            problems.append(f"duplicate chunk_id: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        if chunk.video_id != last_video:
            if chunk.video_id in seen_videos:
                problems.append(f"{label}: video {chunk.video_id} appears in non-contiguous blocks")
            seen_videos.add(chunk.video_id)
            last_frame_start = None
        if chunk.text == "":
            problems.append(f"{label}: empty text")
        sizes = (len(chunk.keyframe_ordinals), len(chunk.frame_indices), chunk.source_count)
        if len(set(sizes)) != 1 or sizes[0] < 1:
            problems.append(f"{label}: inconsistent provenance sizes {sizes}")
            continue
        if chunk.frame_start > chunk.frame_end:
            problems.append(f"{label}: frame_start {chunk.frame_start} > frame_end {chunk.frame_end}")
        if not chunk.frame_start <= chunk.representative_frame_index <= chunk.frame_end:
            problems.append(
                f"{label}: representative_frame_index {chunk.representative_frame_index} "
                f"outside [{chunk.frame_start}, {chunk.frame_end}]"
            )
        if chunk.representative_frame_index not in chunk.frame_indices:
            problems.append(f"{label}: representative_frame_index is not a source frame")
        if list(chunk.frame_indices) != sorted(chunk.frame_indices):
            problems.append(f"{label}: frame_indices are not ascending")
        if list(chunk.keyframe_ordinals) != sorted(chunk.keyframe_ordinals):
            problems.append(f"{label}: keyframe_ordinals are not ascending")
        for ordinal, frame_index in zip(chunk.keyframe_ordinals, chunk.frame_indices):
            key = (chunk.video_id, ordinal, frame_index)
            if key in seen_keys:
                problems.append(f"{label}: duplicate source key {key}")
            seen_keys.add(key)
        # Determinism structure: chunks of one video are frame-sorted.
        if last_video == chunk.video_id and last_frame_start is not None:
            if chunk.frame_start <= last_frame_start:
                problems.append(f"{label}: chunks of one video are not frame-sorted")
        last_video = chunk.video_id
        last_frame_start = chunk.frame_start
    return problems


def write_ocr_chunks(output_dir: Path | str, manifest: OcrChunksManifest, chunks: Sequence[OcrChunk]) -> Path:
    """Write a chunk artifact (``chunks.ndjson`` + ``manifest.json``) atomically.

    Expected input:
        - output_dir: target directory (created if absent).
        - manifest: manifest as returned by :func:`build_ocr_chunks`.
        - chunks: chunk rows in artifact order.

    Returns:
        - The output directory.

    Raises:
        - OSError: if outputs cannot be written.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chunks_path = output / CHUNKS_FILENAME
    tmp_path = chunks_path.with_name(chunks_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")
    tmp_path.replace(chunks_path)
    _write_json_atomic(output / MANIFEST_FILENAME, manifest.to_dict())
    return output


def read_ocr_chunks(artifact_dir: Path | str) -> tuple[OcrChunksManifest, list[OcrChunk]]:
    """Read a completed OCR chunk artifact.

    Expected input:
        - artifact_dir: directory containing ``manifest.json`` and
          ``chunks.ndjson``.

    Returns:
        - ``(manifest, chunks)`` with chunks in row order.

    Raises:
        - FileNotFoundError: if the manifest or chunks file is missing.
        - ValueError: if the manifest, rows, or their consistency fail
          validation.
    """
    directory = Path(artifact_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing chunk artifact manifest: {manifest_path}")
    chunks_path = directory / CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"missing chunk artifact rows: {chunks_path}")
    manifest = OcrChunksManifest.from_dict(_read_json(manifest_path))
    chunks: list[OcrChunk] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            chunks.append(OcrChunk.from_dict(json.loads(line)))
    problems = validate_ocr_chunks(manifest, chunks)
    if problems:
        raise ValueError("invalid OCR chunk artifact: " + "; ".join(problems))
    return manifest, chunks


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
