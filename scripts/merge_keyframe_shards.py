#!/usr/bin/env python3
"""CLI: merge completed keyframe-extraction shards into one artifact.

Reads completed extraction directories (each with ``manifest.json`` and
``frames.ndjson``), validates that they share the same extractor, scorer,
frame reader, and extraction configuration, rejects any duplicate keyframes or
videos spanning multiple shards, and concatenates them in deterministic order
into a merged ``frames.ndjson``, ``map-keyframes/<VIDEO_ID>.csv``, and
``manifest.json``.

This is a thin CLI; the merge logic lives in
``aic2026.keyframe_extraction_merge``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.keyframe_extraction_merge import merge_shards


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="merge_keyframe_shards",
        description=(
            "Merge completed keyframe-extraction shards into one canonical "
            "artifact, validating config consistency and rejecting duplicates."
        ),
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        required=True,
        help="one or more completed extraction directories to merge",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write merged frames.ndjson, map-keyframes/, manifest.json",
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
        manifest = merge_shards(
            args.shards,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Merge complete: {manifest.num_keyframes} keyframes for "
        f"{manifest.num_videos} videos from {len(manifest.source_shards)} shards"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())