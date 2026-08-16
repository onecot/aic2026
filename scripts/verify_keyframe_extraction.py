#!/usr/bin/env python3
"""CLI: verify a completed keyframe extraction artifact before upload.

Reads an extraction output (``outputs/keyframes_worker_XXX/`` or a merged
``outputs/keyframes_full/`` with ``frames.ndjson`` + ``map-keyframes/`` +
``manifest.json``) and verifies it is internally consistent. Worker artifacts
(``source_videos``) and merged artifacts (``source_shards``) are both
supported. When ``--videos`` (a worker's assigned ``videos.json``) is
provided, coverage is checked against it for worker artifacts.

This is a thin CLI; the verification logic lives in
``aic2026.keyframe_extraction_verify.verify_keyframe_extraction``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.keyframe_extraction_verify import verify_keyframe_extraction


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="verify_keyframe_extraction",
        description=(
            "Validate a completed keyframe extraction artifact (worker or "
            "merged): manifest, frame records, deterministic ordering, video "
            "coverage, and per-video map-keyframes files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory containing frames.ndjson, map-keyframes/, and manifest.json",
    )
    parser.add_argument(
        "--videos",
        default=None,
        help="optional videos.json assigned to this worker (checks coverage)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code (0 valid, 1 invalid)."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = verify_keyframe_extraction(args.output_dir, videos_path=args.videos)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    status = "PASS" if report.valid else "FAIL"
    print(f"Keyframe extraction verification: {status}")
    print()
    print(f"Artifact type: {report.artifact_type}")
    print()
    print("Videos:")
    print(f"  expected: {report.expected_videos}")
    print(f"  found: {report.found_videos}")
    print()
    print("Keyframes:")
    print(f"  total: {report.total_keyframes}")
    print(f"  duplicates: {report.duplicates}")
    print()
    print("Manifest:")
    print(f"  extractor: {report.extractor}")
    print(f"  detector: {report.detector}")
    print(f"  sampler: {report.sampler}")

    if not report.valid:
        print()
        print("Validation failed:")
        for issue in report.issues:
            print(f"  [{issue.section}] {issue.message}")
        return 1

    print()
    print("OK: extraction artifact is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
