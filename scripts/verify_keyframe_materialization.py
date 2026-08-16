#!/usr/bin/env python3
"""CLI: verify a completed keyframe-image materialization artifact.

Reads a materialized ``adaptive_sbd_v1`` artifact (``images/`` +
``frames.ndjson`` + ``manifest.json``) and verifies it is internally
consistent: artifact existence, manifest (artifact id, schema version,
counts), canonical frame records (no duplicate ``(video_id,
keyframe_ordinal)``, deterministic ordering), and image coverage (every
record has its ``images/<video_id>/<ordinal:06d>.jpg``, no missing or
unexpected images, total count matches records).

This is a thin CLI; the verification logic lives in
``aic2026.keyframe_materialize_verify.verify_keyframe_materialization``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.keyframe_materialize_verify import verify_keyframe_materialization


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="verify_keyframe_materialization",
        description=(
            "Validate a materialized adaptive_sbd_v1 keyframe image artifact: "
            "manifest, frame records, deterministic ordering, and image "
            "coverage."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory containing images/, frames.ndjson, and manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code (0 valid, 1 invalid)."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = verify_keyframe_materialization(args.output_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    status = "PASS" if report.valid else "FAIL"
    print(f"Keyframe materialization verification: {status}")
    print()
    print("Artifact:")
    print(f"  type: {report.artifact_id}")
    print()
    print("Keyframes:")
    print(f"  records: {report.records}")
    print(f"  images: {report.images}")
    print(f"  missing: {report.missing_images}")
    print(f"  unexpected: {report.unexpected_images}")

    if not report.valid:
        print()
        print("Validation failed:")
        for issue in report.issues:
            print(f"  [{issue.section}] {issue.message}")
        return 1

    print()
    print("OK: materialized keyframe artifact is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
