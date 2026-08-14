#!/usr/bin/env python3
"""CLI: analyze BTC-provided keyframe extraction quality for KIS.

Reads an evaluation CSV, a canonical artifact ``frames.ndjson`` (the current
keyframe set), and a ``predict-kis`` prediction artifact, then prints a
human-readable report covering (1) keyframe statistics, (2) GT frame coverage
(exact and relaxed hits), (3) a worst-first per-query error report of misses,
and (4) a CLIP-baseline comparison. Writes the full report as structured JSON.

This is a thin CLI; the analysis logic lives in ``aic2026.keyframe_analysis``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.keyframe_analysis import (
    DEFAULT_PREDICTION_TOP_K,
    DEFAULT_RELAXED_THRESHOLDS,
    format_report,
    run_keyframe_analysis,
    write_report,
)

DEFAULT_EVAL_CSV = "data/eval/eval_set.csv"
DEFAULT_FRAMES = "data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson"
DEFAULT_CLIP_PREDICTIONS = "data/interim/predictions/kis-l14-laion2b-full-en-top1000.json"
DEFAULT_OUTPUT = "data/analysis/keyframes/current_keyframe_analysis.json"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_keyframes",
        description=(
            "Measure how well the current BTC-provided keyframes cover the KIS "
            "ground-truth events, before implementing a new extractor."
        ),
    )
    parser.add_argument(
        "--eval-csv",
        default=DEFAULT_EVAL_CSV,
        help=f"evaluation CSV path (default: {DEFAULT_EVAL_CSV})",
    )
    parser.add_argument(
        "--frames",
        default=DEFAULT_FRAMES,
        help=(
            "path to a canonical artifact frames.ndjson listing the current "
            f"keyframe set (default: {DEFAULT_FRAMES})"
        ),
    )
    parser.add_argument(
        "--clip-predictions",
        default=DEFAULT_CLIP_PREDICTIONS,
        help=(
            "path to a predict-kis prediction artifact for the CLIP-baseline "
            f"comparison (default: {DEFAULT_CLIP_PREDICTIONS})"
        ),
    )
    parser.add_argument(
        "--relaxed-thresholds",
        default="10,30,60,150,300",
        help="comma-separated relaxed-hit distance thresholds (default: 10,30,60,150,300)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_PREDICTION_TOP_K,
        help=f"number of CLIP predictions analyzed per query (default: {DEFAULT_PREDICTION_TOP_K})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"JSON report output path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _parse_thresholds(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated string of positive integer thresholds."""
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--relaxed-thresholds requires at least one positive integer")
    if any(value < 1 for value in values):
        raise ValueError("--relaxed-thresholds must be positive integers")
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        thresholds = _parse_thresholds(args.relaxed_thresholds)
        report = run_keyframe_analysis(
            evaluation_set_path=args.eval_csv,
            frames_path=args.frames,
            clip_predictions_path=args.clip_predictions,
            relaxed_thresholds=thresholds,
            prediction_top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(report), end="")
    try:
        write_report(args.output, report)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Keyframe analysis written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())