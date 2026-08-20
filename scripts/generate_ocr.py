#!/usr/bin/env python3
"""CLI: extract OCR text from canonical keyframes with PaddleOCR.

Reads a canonical artifact ``frames.ndjson`` and its keyframe ``images/``
collection, extracts text from each keyframe with a selected OCR backend
(PaddleOCR PP-OCRv5 multilingual, Vietnamese + English), and writes a
resumable OCR artifact directory containing ``checkpoint.json``,
``ocr.ndjson``, and ``manifest.json``. Re-running the same command after an
interruption resumes from the checkpoint and skips already-processed
records. The source keyframe artifact is never modified.

This is a thin CLI; the pipeline lives in ``aic2026.ocr_generation``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.image_embedder import DEVICE_CHOICES
from aic2026.ocr_generation import (
    DEFAULT_BACKEND,
    DEFAULT_LANGUAGES,
    DEFAULT_OCR_MODEL,
    OCR_BACKENDS,
    OcrGenerationError,
    generate_ocr,
)
from aic2026.progress import PROGRESS_MODES, create_progress_reporter

DEFAULT_KEYFRAMES_DIR = "data/keyframes/adaptive_sbd_v1"
DEFAULT_OUTPUT_DIR = "data/interim/ocr/paddleocr-v5"


def _parse_languages(value: str) -> tuple[str, ...]:
    """Parse a comma-separated language list into a tuple.

    Expected input:
        - value: e.g. ``"vi,en"`` or ``"vi"``.

    Returns:
        - Tuple of non-empty language codes.
    """
    languages = tuple(part.strip() for part in value.split(",") if part.strip())
    if not languages:
        raise ValueError("--languages must contain at least one language")
    return languages


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_ocr",
        description=(
            "Extract OCR text from canonical keyframes with a selected OCR "
            "backend into a resumable OCR artifact on the selected hardware."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=OCR_BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"OCR backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        default=Path(DEFAULT_KEYFRAMES_DIR),
        help=f"canonical keyframe artifact directory (frames.ndjson + images/) (default: {DEFAULT_KEYFRAMES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="directory to write checkpoint.json, ocr.ndjson, manifest.json (default: "
        f"{DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"PaddleOCR model family (default: {DEFAULT_OCR_MODEL})",
    )
    parser.add_argument(
        "--languages",
        type=_parse_languages,
        default=DEFAULT_LANGUAGES,
        help="comma-separated OCR languages (default: vi,en)",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="device selection: auto (CUDA > CPU), cuda, mps (mapped to cpu), or cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="number of keyframe images passed to the OCR engine per call (default: 16)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="only process the first MAX_RECORDS records (default: all records)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="resume an interrupted build when a checkpoint exists (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="restart from scratch, overwriting any partial output",
    )
    parser.add_argument(
        "--upscale-factor",
        type=float,
        default=1.0,
        help="preprocess: upscale factor for small text (1.0 = off, e.g. 2.0)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="preprocess: contrast enhancement factor (1.0 = unchanged, > 1 increases)",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="preprocess: apply a sharpening filter",
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

    reporter = create_progress_reporter(args.progress)
    reporter.begin()
    try:
        with reporter.stages.time("load_engine"):
            manifest = generate_ocr(
                keyframes_dir=args.keyframes_dir,
                output_dir=args.output_dir,
                backend=args.backend,
                model=args.model,
                batch_size=args.batch_size,
                device=args.device,
                resume=args.resume,
                max_records=args.max_records,
                languages=args.languages,
                upscale_factor=args.upscale_factor,
                contrast=args.contrast,
                sharpen=args.sharpen,
                progress=reporter,
            )
    except (OcrGenerationError, FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"OCR complete: {manifest.num_records} records "
        f"for {manifest.num_videos} videos "
        f"({manifest.num_failed} failed) on {manifest.device}"
    )
    print(f"Backend: {manifest.backend} ({manifest.model}) languages={','.join(manifest.languages)}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
