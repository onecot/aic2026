#!/usr/bin/env python3
"""CLI: merge completed OCR shards into a single canonical OCR artifact.

Reads completed OCR shard directories (each with ``manifest.json`` and
``ocr.ndjson``, named ``shard_XXX``), validates that they share the same
backend/model/languages, rejects any duplicate ``(video_id,
keyframe_ordinal, frame_index)`` keyframes, verifies contiguous source
coverage against the canonical ``frames.ndjson`` (when ``--source-frames``
is given), and concatenates them in shard order into a merged
``ocr.ndjson`` and ``manifest.json``.

This is a thin CLI; the merge logic lives in ``aic2026.ocr_merge``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.ocr_merge import merge_ocr_shards


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="merge_ocr_shards",
        description=(
            "Merge completed OCR shards into one canonical OCR artifact, "
            "validating identity consistency, duplicate keyframes, and "
            "contiguous source coverage."
        ),
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        required=True,
        help="one or more completed shard directories (named shard_XXX) to merge",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write merged ocr.ndjson and manifest.json",
    )
    parser.add_argument(
        "--source-frames",
        type=Path,
        default=None,
        help="canonical frames.ndjson to validate contiguous coverage against "
        "(recorded as the merged artifact's source_frames)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing merged artifact instead of failing when present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        manifest = merge_ocr_shards(
            args.shards,
            args.output_dir,
            source_frames=args.source_frames,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Merge complete: {manifest.num_records} records from "
        f"{len(args.shards)} shards (backend={manifest.backend})"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())