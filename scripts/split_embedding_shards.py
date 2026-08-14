#!/usr/bin/env python3
"""CLI: deterministically split a canonical keyframe set into worker inputs.

Reads a canonical artifact ``frames.ndjson`` and partitions it into ``N``
worker input files (``worker_000/frames.ndjson``, ...), keeping each video
intact and balancing the number of frames per worker. The split is
deterministic: the same input and ``--num-workers`` always produce identical
worker files. Writes a ``split_manifest.json`` describing the split.

This is a thin CLI; the splitting logic lives in ``aic2026.embedding_split``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.embedding_split import split_frames


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="split_embedding_shards",
        description=(
            "Partition a canonical frames.ndjson into balanced, deterministic "
            "worker input files, keeping videos intact."
        ),
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="path to a canonical artifact frames.ndjson to split",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="number of worker splits to produce (>= 1)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to write worker_*/frames.ndjson and split_manifest.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing split instead of failing when present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        manifest = split_frames(
            args.frames,
            args.num_workers,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Split complete: {sum(w.num_frames for w in manifest.workers)} frames "
        f"across {manifest.num_workers} workers"
    )
    for worker in manifest.workers:
        print(
            f"  {worker.worker_id:03d}: {worker.num_frames} frames "
            f"({worker.num_videos} videos)"
        )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())