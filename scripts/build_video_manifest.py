#!/usr/bin/env python3
"""CLI: build a canonical video inventory manifest (``videos.json``).

Discovers ``*.mp4`` files under ``--video-dir``, derives each ``video_id``
from the filename stem, probes per-video metadata with ``ffprobe`` (unless
``--no-metadata``), and writes a deterministic, atomic ``videos.json``
manifest consumed by the keyframe extraction split/execution workflow.

This is a thin CLI; the generation logic lives in
``aic2026.video_manifest``.
"""

from __future__ import annotations

import argparse
import sys

from aic2026.video_manifest import (
    VideoManifestError,
    build_video_manifest,
    save_video_manifest,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="build_video_manifest",
        description=(
            "Discover .mp4 videos and write a canonical videos.json inventory "
            "for keyframe extraction."
        ),
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="directory containing the .mp4 video files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="target videos.json manifest path",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="skip ffprobe metadata probing (fps/duration/frames left unknown)",
    )
    return parser


def _format_bytes(size: int) -> str:
    """Return a compact human-readable byte size."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        manifest = build_video_manifest(args.video_dir, include_metadata=not args.no_metadata)
        save_video_manifest(args.output, manifest)
    except (FileNotFoundError, ValueError, VideoManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total_size = sum(entry.file_size_bytes for entry in manifest.videos)
    fps_available = sum(1 for entry in manifest.videos if entry.fps is not None)
    frames_available = sum(1 for entry in manifest.videos if entry.num_frames is not None)

    print("Video manifest created")
    print()
    print("Videos:")
    print(len(manifest.videos))
    print()
    print("Total size:")
    print(_format_bytes(total_size))
    print()
    print("Metadata:")
    print("fps available:")
    print(fps_available)
    print("frames available:")
    print(frames_available)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())