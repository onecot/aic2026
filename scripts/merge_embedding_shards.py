#!/usr/bin/env python3
"""CLI: merge completed embedding shards into a single embedding artifact.

Reads completed shard directories (each with ``manifest.json``,
``embeddings.npy``, and ``frames.ndjson``), validates that they share the same
model, embedding dimension, and embedding space, rejects any duplicate
keyframes, and concatenates them in deterministic worker order into a merged
``embeddings.npy``, ``frames.ndjson``, and ``manifest.json``.

This is a thin CLI; the merge logic lives in ``aic2026.embedding_merge``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.embedding_merge import merge_shards


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="merge_embedding_shards",
        description=(
            "Merge completed embedding shards into one embedding artifact, "
            "validating model/space consistency and rejecting duplicate frames."
        ),
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        type=Path,
        required=True,
        help="one or more completed shard directories to merge",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write merged embeddings.npy, frames.ndjson, manifest.json",
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
        f"Merge complete: {manifest.total_embeddings} embeddings from "
        f"{len(manifest.source_shards)} shards (model={manifest.model})"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())