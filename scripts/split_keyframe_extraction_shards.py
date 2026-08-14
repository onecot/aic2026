#!/usr/bin/env python3
"""CLI: deterministically split a video manifest into worker inputs.

Reads a ``videos.json`` manifest and partitions its videos into ``N`` worker
manifests (``worker_000/videos.json``, ...), keeping each video intact and
balancing estimated frame counts per worker (using the optional ``num_frames``
weight). The split is deterministic: the same input and ``--num-workers``
always produce identical worker files. Writes a ``split_manifest.json``
describing the split.

This is a thin CLI; the splitting logic lives in
``aic2026.keyframe_extraction_split``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.keyframe_extraction_split import split_video_manifest


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="split_keyframe_extraction_shards",
        description=(
            "Partition a videos.json manifest into balanced, deterministic "
            "worker manifests, keeping videos intact."
        ),
    )
    parser.add_argument(
        "--videos",
        required=True,
        help="path to a videos.json manifest to split",
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
        help="directory to write worker_*/videos.json and split_manifest.json",
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
        manifest = split_video_manifest(
            args.videos,
            args.num_workers,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Split complete: {sum(w.num_videos for w in manifest.workers)} videos "
        f"across {manifest.num_workers} workers"
    )
    for worker in manifest.workers:
        print(
            f"  {worker.worker_id:03d}: {worker.num_videos} videos "
            f"({worker.estimated_frames} estimated frames)"
        )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())