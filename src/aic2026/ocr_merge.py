"""Deterministic merge of completed OCR shards into one canonical OCR artifact.

Combines multiple completed shard OCR artifact directories (each produced by
``modal_ocr_distributed.run_ocr_shard``) into a single canonical OCR
artifact::

    merged/
        ocr.ndjson      concatenated KeyframeOcr records (shard order)
        manifest.json   versioned OcrManifest-compatible metadata

The merged ``manifest.json`` conforms to the existing OCR artifact schema
(:class:`OcrManifest`), so the merged directory is readable by
:func:`aic2026.ocr_generation.read_ocr_artifact` exactly like a single-GPU
artifact. ``generate_ocr`` and the OCR record schema are never modified.

Merge guarantees:

- every shard must be completed (a ``manifest.json`` exists) and internally
  consistent (:func:`validate_ocr_artifact` passes);
- all shards must share the same backend, model, languages, device,
  batch_size, and schema versions (mixing is rejected);
- the merge order is deterministic: shards are ordered by the shard index
  parsed from the directory name ``shard_XXX`` (matching the distributed
  runner), and each shard's records are kept in file order;
- no keyframe ``(video_id, keyframe_ordinal, frame_index)`` may appear more
  than once across shards (duplicates are rejected);
- when the canonical ``source_frames`` ``frames.ndjson`` is supplied, shards
  must cover it contiguously: shard indices are exactly ``0..num_shards-1``
  and each shard's record count matches the balanced row split used by the
  runner, so ``sum(shard counts) == canonical rows``;
- when the canonical source and every shard's recorded source frames are
  readable, the shards' keyframe identities must exactly match the canonical
  set and each shard's OCR records must align with its source rows.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.embedding_artifact import iter_frames
from aic2026.ocr_generation import (
    MANIFEST_FILENAME,
    OCR_ARTIFACT_SCHEMA_VERSION,
    OCR_FILENAME,
    KeyframeOcr,
    OcrManifest,
    iter_ocr_records,
    validate_ocr_artifact,
)

logger = logging.getLogger(__name__)

#: Regex matching the distributed runner's shard directory names.
SHARD_DIR_PATTERN = re.compile(r"^shard_(\d+)$")

#: OcrManifest fields that must match across all merged shards.
_IDENTITY_FIELDS = (
    "backend",
    "model",
    "languages",
    "device",
    "batch_size",
    "schema_version",
    "canonical_schema_version",
)


class MergeError(RuntimeError):
    """Raised when shards cannot be merged (incomplete, inconsistent, or overlapping)."""


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


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _shard_id_from_dirname(directory: Path) -> int:
    """Parse the shard index from a ``shard_XXX`` directory name.

    Expected input:
        - directory: a path whose name matches ``shard_<index>``.

    Returns:
        - The shard index as an int.

    Raises:
        - MergeError: if the directory name does not match the runner's
          ``shard_XXX`` naming contract.
    """
    match = SHARD_DIR_PATTERN.fullmatch(directory.name)
    if match is None:
        raise MergeError(
            f"shard directory name {directory.name!r} does not match the "
            f"expected 'shard_XXX' pattern: {directory}"
        )
    return int(match.group(1))


def order_shards(shard_dirs: list[Path | str]) -> list[Path]:
    """Order shard directories deterministically by shard index.

    Expected input:
        - shard_dirs: completed shard directories (each named ``shard_XXX``).

    Returns:
        - The same directories sorted into merge order by shard index.

    Raises:
        - MergeError: on empty input, an unparseable directory name, or
          duplicate shard indices.
    """
    dirs = [Path(d) for d in shard_dirs]
    if not dirs:
        raise MergeError("at least one shard directory is required")
    ordered = sorted(dirs, key=_shard_id_from_dirname)
    ids = [_shard_id_from_dirname(d) for d in ordered]
    if len(set(ids)) != len(ids):
        raise MergeError(f"duplicate shard indices in input: {ids}")
    return ordered


def expected_shard_record_count(total_records: int, num_shards: int, shard_id: int) -> int:
    """Return the balanced record count the distributed runner gives one shard.

    Mirrors ``modal_ocr_distributed.shard_row_bounds`` (``base = ceil(total /
    num_shards)`` rows per shard, last shard possibly shorter) so the merge can
    validate contiguous coverage against the runner's deterministic split.

    Expected input:
        - total_records: total rows in the canonical frames file.
        - num_shards: total number of shards.
        - shard_id: shard index in ``[0, num_shards)``.

    Returns:
        - The number of records ``shard_id`` is expected to contain.

    Raises:
        - ValueError: on invalid inputs.
    """
    _require(isinstance(total_records, int) and total_records >= 0, "total_records must be >= 0")
    _require(isinstance(num_shards, int) and num_shards >= 1, "num_shards must be >= 1")
    _require(isinstance(shard_id, int) and 0 <= shard_id < num_shards, "shard_id out of range")
    if total_records == 0:
        return 0
    base = (total_records + num_shards - 1) // num_shards
    start = shard_id * base
    end = min(start + base, total_records)
    return end - start


def _record_key(record: KeyframeOcr) -> tuple[str, int, int]:
    """Return the unique keyframe key ``(video_id, keyframe_ordinal, frame_index)``."""
    return (record.video_id.value, record.keyframe_ordinal, record.frame_index)


def _read_source_frame_keys(path: Path) -> list[tuple[str, int, int]]:
    """Return keyframe keys ``(video_id, keyframe_ordinal, frame_index)`` from a frames file."""
    if not path.is_file():
        raise FileNotFoundError(f"missing source frames: {path}")
    return [
        (frame.keyframe.video_id.value, frame.keyframe.keyframe_ordinal, frame.frame_index)
        for frame in iter_frames(path)
    ]


def _count_non_empty_lines(path: Path) -> int:
    """Return the number of non-empty lines in a frames file (matches OCR record rows)."""
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def merge_ocr_shards(
    shard_dirs: list[Path | str],
    output_dir: Path | str,
    *,
    source_frames: Path | str | None = None,
    overwrite: bool = False,
) -> OcrManifest:
    """Merge completed OCR shards into a single canonical OCR artifact.

    Expected input:
        - shard_dirs: completed OCR shard directories (each with
          ``manifest.json`` and ``ocr.ndjson``, named ``shard_XXX``).
        - output_dir: target directory for the merged artifact (created if
          absent; refused when non-empty unless ``overwrite``).
        - source_frames: optional canonical ``frames.ndjson`` path. When
          given, enables contiguous-coverage and total-row-count validation
          and is recorded as the merged artifact's ``source_frames``.
        - overwrite: if True, replace an existing merged artifact; otherwise
          raise ``MergeError`` when the output already exists.

    Returns:
        - The merged :class:`OcrManifest` (schema-compatible with
          :func:`aic2026.ocr_generation.read_ocr_artifact`).

    Raises:
        - MergeError: on incomplete/inconsistent shards, mixed identity
          fields, duplicate keyframes, non-contiguous or non-covering shards,
          an unparseable/duplicated shard index, or a pre-existing
          non-overwritten output.
        - FileNotFoundError: if a shard directory or the canonical source is
          missing.
        - OSError: if outputs cannot be written.
    """
    dirs = [Path(d) for d in shard_dirs]
    if not dirs:
        raise MergeError("at least one shard directory is required")
    for directory in dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"shard directory does not exist: {directory}")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise MergeError(
            f"merge output already exists (and is not empty): {output}; "
            "use --overwrite to replace it"
        )

    ordered = order_shards(dirs)

    reference: OcrManifest | None = None
    records_by_shard: list[list[KeyframeOcr]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    for directory in ordered:
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise MergeError(f"shard is not complete (missing {MANIFEST_FILENAME}): {directory}")
        manifest = OcrManifest.from_dict(_read_json(manifest_path))
        problems = validate_ocr_artifact(manifest, directory)
        if problems:
            raise MergeError(
                f"shard artifact is inconsistent: {directory}: " + "; ".join(problems)
            )
        if reference is None:
            reference = manifest
        else:
            for field in _IDENTITY_FIELDS:
                if getattr(manifest, field) != getattr(reference, field):
                    raise MergeError(
                        f"shard {directory} has different {field} than the reference shard: "
                        f"{getattr(manifest, field)!r} != {getattr(reference, field)!r}"
                    )
        records = iter_ocr_records(directory / OCR_FILENAME)
        for record in records:
            key = _record_key(record)
            if key in seen_keys:
                raise MergeError(f"duplicate keyframe across shards: {key}")
            seen_keys.add(key)
        records_by_shard.append(records)

    if reference is None:  # pragma: no cover - dirs is non-empty, so unreachable
        raise MergeError("no shards to merge")

    total_records = sum(len(records) for records in records_by_shard)

    canonical_path: Path | None = None
    if source_frames is not None:
        canonical_path = Path(source_frames)
        if not canonical_path.is_file():
            raise FileNotFoundError(f"missing canonical source frames: {canonical_path}")
        canonical_rows = _count_non_empty_lines(canonical_path)
        num_shards = len(ordered)
        shard_ids = sorted(_shard_id_from_dirname(d) for d in ordered)
        if shard_ids != list(range(num_shards)):
            raise MergeError(f"shard indices are not contiguous 0..{num_shards - 1}: {shard_ids}")
        for directory, records in zip(ordered, records_by_shard):
            shard_id = _shard_id_from_dirname(directory)
            expected = expected_shard_record_count(canonical_rows, num_shards, shard_id)
            if len(records) != expected:
                raise MergeError(
                    f"shard {shard_id} has {len(records)} records, expected {expected} "
                    f"for contiguous coverage of {canonical_rows} rows across {num_shards} shards"
                )
        if total_records != canonical_rows:
            raise MergeError(
                f"merged records {total_records} != canonical rows {canonical_rows}"
            )
        _verify_identity_coverage(ordered, records_by_shard, canonical_path, logger)

    output.mkdir(parents=True, exist_ok=True)
    ocr_out = output / OCR_FILENAME
    tmp_ocr = ocr_out.with_name(ocr_out.name + ".tmp")
    with tmp_ocr.open("w", encoding="utf-8") as handle:
        for records in records_by_shard:
            for record in records:
                handle.write(json.dumps(record.to_dict()) + "\n")
    tmp_ocr.replace(ocr_out)

    merged = OcrManifest(
        schema_version=OCR_ARTIFACT_SCHEMA_VERSION,
        artifact_id=output.name,
        backend=reference.backend,
        model=reference.model,
        languages=reference.languages,
        device=reference.device,
        batch_size=reference.batch_size,
        source_frames=str(canonical_path.resolve()) if canonical_path is not None else reference.source_frames,
        num_records=total_records,
        num_videos=len({record.video_id.value for records in records_by_shard for record in records}),
        num_failed=sum(1 for records in records_by_shard for record in records if record.status == "error"),
        canonical_schema_version=reference.canonical_schema_version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_atomic(output / MANIFEST_FILENAME, merged.to_dict())

    problems = validate_ocr_artifact(merged, output)
    if problems:
        raise MergeError("merged artifact validation failed: " + "; ".join(problems))

    logger.info(
        "merged %d shards -> %d records (backend=%s)", len(ordered), total_records, reference.backend
    )
    return merged


def _verify_identity_coverage(
    ordered: list[Path],
    records_by_shard: list[list[KeyframeOcr]],
    canonical_path: Path,
    logger: logging.Logger,
) -> None:
    """Validate per-shard source alignment and full identity coverage.

    Each shard's manifest records its own ``source_frames`` path. When every
    such file is readable, this verifies that (a) each shard's OCR records
    match its source rows 1:1 and (b) the union of shard source rows equals
    the canonical source rows. Unreadable shard sources (e.g. the Modal
    Volume is not mounted) only log a warning; the bounds-based coverage check
    still applies.

    Expected input:
        - ordered: shard directories in merge order.
        - records_by_shard: OCR records per shard (same order).
        - canonical_path: the canonical ``frames.ndjson`` path.
        - logger: the module logger.

    Raises:
        - MergeError: when a readable shard source does not align with its
          OCR records or the shard sources do not cover the canonical set.
    """
    all_readable = True
    for directory, records in zip(ordered, records_by_shard):
        source = _read_json(directory / MANIFEST_FILENAME)["source_frames"]
        source_path = Path(source)
        if not source_path.is_file():
            all_readable = False
            break
        frame_keys = _read_source_frame_keys(source_path)
        record_keys = [_record_key(record) for record in records]
        if frame_keys != record_keys:
            raise MergeError(
                f"shard {directory} OCR records do not align with its source frames "
                f"({len(record_keys)} records vs {len(frame_keys)} source rows)"
            )
    if not all_readable:
        logger.warning(
            "shard source frames are not all readable; identity coverage "
            "could not be verified (bounds-based coverage already validated)"
        )
        return
    covered = set(
        _record_key(record) for records in records_by_shard for record in records
    )
    canonical_keys = set(_read_source_frame_keys(canonical_path))
    if covered != canonical_keys:
        missing = len(canonical_keys - covered)
        extra = len(covered - canonical_keys)
        raise MergeError(
            f"shard coverage does not match the canonical source "
            f"(missing {missing} keyframes, extra {extra})"
        )