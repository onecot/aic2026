#!/usr/bin/env python3
"""CLI: deduplicate a canonical frames.ndjson artifact (keep-first).

Reads a canonical artifact ``frames.ndjson`` and writes a deduplicated copy
keeping the first occurrence of each ``(video_id, frame_index)`` key while
preserving the original ordering. The input is never modified. The output and
the optional JSON report are written atomically.

This is a thin CLI; the logic lives in ``aic2026.keyframe_dedup``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.keyframe_dedup import DEDUP_REPORT_FILENAME, deduplicate_keyframes


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="deduplicate_keyframes",
        description=(
            "Create a deduplicated keyframe artifact from a canonical "
            "frames.ndjson (keep first occurrence of each (video_id, frame_index))."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="path to the source canonical frames.ndjson (never modified)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to write the deduplicated frames.ndjson",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            f"path to write the JSON dedup report (default: "
            f"{DEDUP_REPORT_FILENAME} next to --output)"
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="number of duplicate keys to record as examples in the report (default: 5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    report_path = args.report
    if report_path is None:
        from pathlib import Path

        report_path = str(Path(args.output).with_name(DEDUP_REPORT_FILENAME))

    try:
        report = deduplicate_keyframes(
            args.input,
            args.output,
            report_path,
            max_examples=args.max_examples,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Original records:   {report.original_records}")
    print(f"Unique records:     {report.unique_records}")
    print(f"Removed duplicates: {report.removed_duplicates}")
    print(f"Strategy:           {report.strategy}")
    print(f"Output:             {report.output_path}")
    print(f"Report:             {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())