#!/usr/bin/env python3
"""CLI: profile shot distribution and adaptive keyframe estimates.

Reads a ``videos.json`` manifest, runs the selected shot detector over every
video (reusing the OmniShotCut model for all videos), validates each detected
shot segmentation, and estimates how many keyframes the adaptive sampler
would produce. Writes a structured JSON report plus a human-readable summary.

This is a thin CLI; the analysis logic lives in
``aic2026.keyframe_extraction_analysis``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aic2026.image_embedder import DEVICE_CHOICES
from aic2026.keyframe_extraction import load_video_manifest
from aic2026.keyframe_extraction_analysis import (
    DEFAULT_SAMPLER_CONFIG,
    analyze_shot_distribution,
    format_report,
    progress_unit_for_manifest,
    write_report,
)
from aic2026.progress import PROGRESS_MODES, ProgressReporter, create_progress_reporter
from aic2026.shot_detection import (
    DEFAULT_OMNISHOTCUT_FILENAME,
    SHOT_DETECTOR_NAMES,
    create_shot_detector,
)

DEFAULT_OUTPUT = "data/analysis/keyframes/omnishotcut-profile.json"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_keyframe_extraction",
        description=(
            "Profile OmniShotCut shot distribution and adaptive keyframe "
            "estimates over a video manifest before running full extraction."
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="path to a videos.json manifest (video_id + path entries)",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help=(
            "root directory that relative video paths in the manifest are "
            "resolved against (overrides the manifest's source_dir)"
        ),
    )
    parser.add_argument(
        "--detector",
        choices=SHOT_DETECTOR_NAMES,
        default="omnishotcut",
        help=f"shot detector (default: omnishotcut); choices: {SHOT_DETECTOR_NAMES}",
    )
    parser.add_argument(
        "--sampler",
        choices=("adaptive_v1",),
        default="adaptive_v1",
        help="per-shot sampler used for keyframe estimation (default: adaptive_v1)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="device for the OmniShotCut model (default: auto)",
    )
    parser.add_argument(
        "--omnishotcut-checkpoint",
        type=str,
        default=None,
        help="OmniShotCut checkpoint: local path or HuggingFace repo id (default: uva-cv-lab/OmniShotCut)",
    )
    parser.add_argument(
        "--omnishotcut-filename",
        type=str,
        default=DEFAULT_OMNISHOTCUT_FILENAME,
        help=f"OmniShotCut checkpoint filename inside a repo id (default: {DEFAULT_OMNISHOTCUT_FILENAME})",
    )
    parser.add_argument(
        "--short-shot-seconds",
        type=float,
        default=DEFAULT_SAMPLER_CONFIG["short_shot_seconds"],
        help=f"shots shorter than this (s) get only their middle frame (default: {DEFAULT_SAMPLER_CONFIG['short_shot_seconds']})",
    )
    parser.add_argument(
        "--long-shot-seconds",
        type=float,
        default=DEFAULT_SAMPLER_CONFIG["long_shot_seconds"],
        help=f"shots at least this long (s) use the long-shot policy (default: {DEFAULT_SAMPLER_CONFIG['long_shot_seconds']})",
    )
    parser.add_argument(
        "--uniform-interval-seconds",
        type=float,
        default=DEFAULT_SAMPLER_CONFIG["uniform_interval_seconds"],
        help=f"uniform keyframe spacing (s) for non-short shots (default: {DEFAULT_SAMPLER_CONFIG['uniform_interval_seconds']})",
    )
    parser.add_argument(
        "--min-temporal-distance",
        type=int,
        default=DEFAULT_SAMPLER_CONFIG["min_temporal_distance"],
        help=f"minimum frame gap between high-change keyframes (default: {DEFAULT_SAMPLER_CONFIG['min_temporal_distance']})",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=DEFAULT_SAMPLER_CONFIG["peak_threshold"],
        help=f"minimum change score for a candidate peak (default: {DEFAULT_SAMPLER_CONFIG['peak_threshold']})",
    )
    parser.add_argument(
        "--outlier-std-threshold",
        type=float,
        default=2.0,
        help="flag videos whose shot count/keyframe estimate exceed the mean by this many stddevs (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"JSON report output path (default: {DEFAULT_OUTPUT})",
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

    detector = create_shot_detector(
        args.detector,
        checkpoint=args.omnishotcut_checkpoint,
        filename=args.omnishotcut_filename,
        device=args.device,
    )

    sampler_config = {
        "sampler": args.sampler,
        "short_shot_seconds": args.short_shot_seconds,
        "long_shot_seconds": args.long_shot_seconds,
        "uniform_interval_seconds": args.uniform_interval_seconds,
        "min_temporal_distance": args.min_temporal_distance,
        "peak_threshold": args.peak_threshold,
    }

    manifest = load_video_manifest(args.videos)
    reporter: ProgressReporter = create_progress_reporter(
        args.progress, unit=progress_unit_for_manifest(manifest)
    )
    reporter.begin()

    device_used = "n/a"
    if args.detector == "omnishotcut":
        # Stage: load_model — eager, one-time model + checkpoint load.
        with reporter.stages.time("load_model"):
            device_used = detector.load()
        print(f"OmniShotCut model loaded on device={device_used}")

    try:
        # Stages: detect_shots + estimate_sampling happen inside the analysis.
        with reporter.stages.time("detect_shots"):
            report = analyze_shot_distribution(
                manifest,
                detector,
                sampler_config,
                progress=reporter,
                outlier_std_threshold=args.outlier_std_threshold,
                video_root=args.video_root,
            )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        reporter.finish()
        return 1

    print(format_report(report), end="")
    with reporter.stages.time("write_report"):
        write_report(args.output, report)
    print(f"Shot-distribution report written to: {args.output}")
    reporter.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
