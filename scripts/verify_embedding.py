#!/usr/bin/env python3
"""CLI: verify a completed embedding artifact.

Reads an embedding artifact (``embeddings.npy`` + ``frames.ndjson`` +
``manifest.json``) and verifies it is internally consistent: artifact
existence, manifest parsing and model metadata, frame record validity, and
the embedding matrix (2-D floating, row count matches frames and manifest,
dimension and dtype match the manifest when recorded, no empty or NaN/Inf
vectors).

This is a thin CLI; the verification logic lives in
``aic2026.embedding_verify.verify_embedding``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.embedding_verify import verify_embedding


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="verify_embedding",
        description=(
            "Validate a completed embedding artifact (embeddings.npy + "
            "frames.ndjson + manifest.json): manifest, model metadata, frame "
            "records, matrix shape/dtype, and vector health."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory containing embeddings.npy, frames.ndjson, and manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code (0 valid, 1 invalid)."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        report = verify_embedding(args.output_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    status = "PASS" if report.valid else "FAIL"
    print(f"Embedding verification: {status}")
    print()
    print("Artifact:")
    print(f"  model: {report.model_id}")
    print(f"  dim: {report.dim if report.dim is not None else 'unavailable'}")
    print(f"  dtype: {report.dtype if report.dtype is not None else 'unavailable'}")
    print()
    print("Rows:")
    print(f"  embeddings: {report.rows}")
    print(f"  frames: {report.frames}")
    print()
    print("Vectors:")
    print(f"  empty: {report.empty_vectors}")
    print(f"  nan/inf: {report.nan_vectors}")

    if not report.valid:
        print()
        print("Validation failed:")
        for issue in report.issues:
            print(f"  [{issue.section}] {issue.message}")
        return 1

    print()
    print("OK: embedding artifact is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())