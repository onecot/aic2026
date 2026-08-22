#!/usr/bin/env python3
"""CLI: build temporal OCR chunks from a completed OCR artifact.

Reads one completed OCR artifact (``manifest.json`` + ``ocr.ndjson``),
groups its frame-level records into deterministic temporal chunks per
video (see ``aic2026.ocr_chunks``), deduplicates repeated OCR text, and
writes ``chunks.ndjson`` + ``manifest.json`` into the output directory.
The written artifact is re-read and validated before success is reported.

This is a thin CLI; the chunking logic lives in ``aic2026.ocr_chunks``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.ocr_chunks import (
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_MAX_RECORDS,
    build_ocr_chunks,
    read_ocr_chunks,
    write_ocr_chunks,
)
from aic2026.ocr_generation import read_ocr_artifact


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="build_ocr_chunks",
        description=(
            "Build deterministic temporal OCR chunks from a completed OCR "
            "artifact, preserving per-video temporal locality and "
            "deduplicating repeated OCR text."
        ),
    )
    parser.add_argument(
        "--ocr-artifact",
        type=Path,
        required=True,
        help="completed OCR artifact directory (manifest.json + ocr.ndjson)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write chunks.ndjson and manifest.json",
    )
    parser.add_argument(
        "--artifact-id",
        default="chunks",
        help="artifact id recorded in the chunk manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="maximum source records per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=DEFAULT_MAX_FRAME_GAP,
        help="maximum frame_index distance between adjacent records of one "
        "chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing chunk artifact instead of failing when present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if args.max_records < 1:
            raise ValueError(f"--max-records must be >= 1, got {args.max_records}")
        if args.max_frame_gap < 0:
            raise ValueError(f"--max-frame-gap must be >= 0, got {args.max_frame_gap}")
        if (
            args.output_dir.exists()
            and any(args.output_dir.iterdir())
            and not args.overwrite
        ):
            raise ValueError(
                f"output already exists (and is not empty): {args.output_dir}; "
                "use --overwrite to replace it"
            )

        _, records = read_ocr_artifact(args.ocr_artifact)
        manifest, chunks = build_ocr_chunks(
            records,
            artifact_id=args.artifact_id,
            source_ocr_artifact=str(args.ocr_artifact.resolve()),
            max_records=args.max_records,
            max_frame_gap=args.max_frame_gap,
        )
        write_ocr_chunks(args.output_dir, manifest, chunks)
        read_ocr_chunks(args.output_dir)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Built {manifest.num_chunks} chunks across {manifest.num_videos} videos "
        f"from {manifest.num_source_records} OCR records "
        f"(chunked={manifest.num_source_records_chunked}, "
        f"skipped_empty={manifest.num_source_records_skipped}; "
        f"max_records={args.max_records}, max_frame_gap={args.max_frame_gap})"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
