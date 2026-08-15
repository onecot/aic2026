#!/usr/bin/env python3
"""CLI: validate a deterministic video split before distributing workers.

Reads a split directory produced by ``split_keyframe_extraction_shards`` and
verifies it is internally consistent: ``split_manifest.json`` exists, every
worker ``videos.json`` exists and parses, the video/frame totals match the
recorded manifest, and no video is duplicated or missing across workers.

This is a thin CLI; the verification logic lives in
``aic2026.keyframe_extraction_split.verify_video_split``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.keyframe_extraction_split import verify_video_split


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="verify_video_split",
        description=(
            "Validate that a video split directory is consistent: total "
            "videos match, no duplicates, no missing videos, frame counts "
            "match split_manifest.json."
        ),
    )
    parser.add_argument(
        "--split-dir",
        required=True,
        help="directory containing split_manifest.json and worker_*/videos.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code (0 valid, 1 invalid)."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = verify_video_split(args.split_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for worker in report.workers:
        print(f"worker_{worker.worker_id:03d}:")
        print(f"  videos: {worker.num_videos}")
        print(f"  frames: {worker.num_frames}")

    print("Summary:")
    print(f"Total videos: {report.total_videos}")
    print(f"Unique videos: {report.unique_videos}")
    print(f"Duplicates: {report.duplicates}")
    print(f"Total frames: {report.total_frames}")

    if not report.valid:
        print("Validation failed:")
        for issue in report.issues:
            if issue.worker_id is not None:
                print(f"  worker_{issue.worker_id:03d}: {issue.message}")
            else:
                print(f"  {issue.message}")
        return 1

    print("OK: split is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
