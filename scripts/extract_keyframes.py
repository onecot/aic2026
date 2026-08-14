#!/usr/bin/env python3
"""CLI: run resumable keyframe extraction over a video manifest.

Reads a ``videos.json`` manifest (``video_id`` + ``path`` entries), runs the
DAKE-inspired extractor over every video, and writes a canonical keyframe
artifact: ``frames.ndjson``, ``map-keyframes/<VIDEO_ID>.csv``,
``manifest.json``, and (during processing) ``checkpoint.json``. Re-running the
same command after an interruption resumes from the checkpoint; a changed
configuration on resume is rejected.

This is a thin CLI; the pipeline lives in ``aic2026.keyframe_extraction``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aic2026.keyframe_extraction import (
    FRAMES_FILENAME,
    ExtractionConfig,
    extract_keyframes,
)
from aic2026.progress import PROGRESS_MODES, create_progress_reporter


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="extract_keyframes",
        description=(
            "Extract keyframes from a video manifest into a canonical "
            "frames.ndjson + map-keyframes artifact (resumable)."
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="path to a videos.json manifest (video_id + path entries)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            f"directory to write {FRAMES_FILENAME}, map-keyframes/, "
            "manifest.json, and checkpoint.json"
        ),
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-keyframes-per-video",
        type=int,
        default=None,
        help="maximum keyframes selected per video (mutually exclusive with --sampling-ratio)",
    )
    target_group.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="fraction of frames selected as keyframes, e.g. 0.1 (mutually exclusive with --target-keyframes-per-video)",
    )
    parser.add_argument(
        "--frame-reader",
        choices=("ffmpeg", "synthetic"),
        default="ffmpeg",
        help="frame source: ffmpeg (real videos) or synthetic (test descriptors) (default: ffmpeg)",
    )
    parser.add_argument(
        "--scorer",
        choices=("frame_difference",),
        default="frame_difference",
        help="change-scoring method (default: frame_difference)",
    )
    parser.add_argument(
        "--min-temporal-distance",
        type=int,
        default=30,
        help="minimum frame gap between selected keyframes (default: 30)",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=8.0,
        help="minimum change score for a candidate peak (default: 8.0)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=64,
        help="longest side (pixels) of downscaled grayscale frames (default: 64)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="override the video frame rate (default: probe from the video)",
    )
    parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="auto",
        help=(
            "progress reporting: auto (interactive on a terminal, periodic "
            "logs otherwise), on (force interactive), or off (default: auto)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="resume from an existing matching checkpoint (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="ignore any existing checkpoint and start fresh (refused if the output is non-empty without --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing artifact instead of failing when present",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = ExtractionConfig(
        scorer=args.scorer,
        frame_reader=args.frame_reader,
        min_temporal_distance=args.min_temporal_distance,
        peak_threshold=args.peak_threshold,
        target_keyframes_per_video=args.target_keyframes_per_video,
        sampling_ratio=args.sampling_ratio,
        max_side=args.max_side,
        fps=args.fps,
    )
    reporter = create_progress_reporter(args.progress, unit="videos")
    reporter.begin()
    try:
        manifest = extract_keyframes(
            args.videos,
            args.output_dir,
            config,
            progress=reporter,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Extraction complete: {manifest.num_keyframes} keyframes across "
        f"{manifest.num_videos} videos (config_hash={manifest.config_hash[:8]})"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())