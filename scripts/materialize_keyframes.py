#!/usr/bin/env python3
"""CLI: materialize adaptive keyframe images from the original videos.

Reads a ``videos.json`` manifest and a canonical artifact ``frames.ndjson``
(produced by extraction or merge), decodes the selected frames from each
original mp4, and writes ``images/<video_id>/<NNN>.jpg`` (the ``adaptive_sbd_v1``
artifact) plus a validated copy of ``frames.ndjson`` and ``manifest.json``.
The output replaces the BTC-provided keyframe image collection as the input
for the embedding step.

This is a thin CLI; the pipeline lives in ``aic2026.keyframe_materialize``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aic2026.keyframe_extraction import FRAMES_FILENAME, MANIFEST_FILENAME
from aic2026.keyframe_materialize import (
    IMAGE_EXTENSION,
    IMAGES_DIRNAME,
    materialize_keyframes,
    verify_materialization,
)
from aic2026.progress import PROGRESS_MODES


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="materialize_keyframes",
        description=(
            "Decode the selected keyframes from the original videos into "
            f"{IMAGES_DIRNAME}/<video_id>/<ordinal>.{IMAGE_EXTENSION} images "
            "for the embedding step."
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="path to a videos.json manifest (video_id + path entries)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help=f"path to a canonical artifact {FRAMES_FILENAME} (extraction or merge output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            f"directory to write {IMAGES_DIRNAME}/, {FRAMES_FILENAME}, and "
            f"{MANIFEST_FILENAME} (the adaptive_sbd_v1 image artifact)"
        ),
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help=(
            "root directory that relative video paths in the manifest are "
            "resolved against (overrides the manifest's source_dir); required "
            "when the manifest is relocatable and videos live outside "
            "source_dir"
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="ffmpeg mjpeg quality -q:v in [1, 31]; lower is better (default: 2)",
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

    try:
        manifest = materialize_keyframes(
            args.videos,
            args.frames,
            args.output_dir,
            video_root=args.video_root,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
            progress=args.progress,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Materialization complete: {manifest.num_keyframes} keyframe images "
        f"across {manifest.num_videos} videos"
    )
    print(f"Output: {args.output_dir}")
    report = verify_materialization(args.output_dir)
    if not report.valid:
        for issue in report.issues:
            print(f"verification issue [{issue.section}]: {issue.message}", file=sys.stderr)
        return 1
    print(
        f"Verification: valid ({report.num_keyframes} images, "
        f"{report.missing_images} missing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
