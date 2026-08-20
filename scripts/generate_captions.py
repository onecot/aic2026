#!/usr/bin/env python3
"""CLI: generate captions for canonical keyframes with a caption model.

Reads a canonical artifact ``frames.ndjson`` and its keyframe ``images/``
collection, captions each keyframe with a selected caption backend
(Qwen2.5-VL via Transformers, or Jina VLM MLX on Apple Silicon), and writes a
resumable caption artifact directory containing ``checkpoint.json``,
``captions.ndjson``, and ``manifest.json``. Re-running the same command after
an interruption resumes from the checkpoint and skips already-captioned
records. The source keyframe artifact is never modified.

This is a thin CLI; the pipeline lives in ``aic2026.caption_generation``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.caption_generation import (
    CAPTION_BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_CAPTION_PROMPT,
    CaptionGenerationError,
    generate_captions,
)
from aic2026.image_embedder import DEVICE_CHOICES
from aic2026.progress import PROGRESS_MODES, create_progress_reporter

DEFAULT_KEYFRAMES_DIR = "data/keyframes/adaptive_sbd_v1"
DEFAULT_OUTPUT_DIR = "data/interim/captions/qwen2.5-vl-3b"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_captions",
        description=(
            "Caption canonical keyframes with a selected caption backend into "
            "a resumable caption artifact on the selected hardware."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=CAPTION_BACKENDS,
        default=DEFAULT_BACKEND,
        help=(
            "caption backend: qwen2.5-vl (Transformers, any hardware) or "
            f"jina-vlm-mlx (Apple Silicon only) (default: {DEFAULT_BACKEND})"
        ),
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
        help="directory to write checkpoint.json, captions.ndjson, manifest.json (default: "
        f"{DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "short model name or HuggingFace checkpoint id (default: backend "
            "default, e.g. Qwen2.5-VL-3B-Instruct or jinaai/jina-vlm-mlx)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="device selection: auto (CUDA > MPS > CPU), cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="number of keyframe images captioned per model invocation (default: 4)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="only caption the first MAX_RECORDS records (default: all records)",
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
        "--prompt",
        default=DEFAULT_CAPTION_PROMPT,
        help="caption prompt passed to the model (default: the documented retrieval prompt)",
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
        with reporter.stages.time("load_model"):
            manifest = generate_captions(
                keyframes_dir=args.keyframes_dir,
                output_dir=args.output_dir,
                backend=args.backend,
                model=args.model,
                batch_size=args.batch_size,
                device=args.device,
                resume=args.resume,
                max_records=args.max_records,
                prompt=args.prompt,
                progress=reporter,
            )
    except (CaptionGenerationError, FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Captions complete: {manifest.num_records} records "
        f"for {manifest.num_videos} videos "
        f"({manifest.num_failed} failed) on {manifest.device}"
    )
    print(f"Model: {manifest.model} ({manifest.checkpoint})")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
