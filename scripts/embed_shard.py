#!/usr/bin/env python3
"""CLI: embed a shard of canonical keyframe records into a resumable artifact.

Reads a canonical artifact ``frames.ndjson``, encodes the first ``--max-frames``
records (or all when omitted) through a selected model, and writes a resumable
shard directory containing ``checkpoint.json``, ``embeddings.npy``,
``frames.ndjson``, and ``manifest.json``. Re-running the same command after an
interruption resumes from the checkpoint and skips already-embedded frames.

This is a thin CLI; the pipeline lives in ``aic2026.embedding_shard`` and the
embedder interface in ``aic2026.image_embedder``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.embedding_shard import embed_shard
from aic2026.image_embedder import DEVICE_CHOICES, create_image_embedder
from aic2026.progress import PROGRESS_MODES, create_progress_reporter

DEFAULT_FRAMES = "data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="embed_shard",
        description=(
            "Encode a bounded shard of keyframe records into a resumable "
            "embedding artifact on the selected hardware."
        ),
    )
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help=f"path to a canonical artifact frames.ndjson (e.g. {DEFAULT_FRAMES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write checkpoint.json, embeddings.npy, frames.ndjson, manifest.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model spec '[backend:]model[:pretrained]', e.g. "
            "'openclip:ViT-L-14:laion2b_s32b_b82k' (default: the config's "
            "OpenCLIP model)"
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
        default=32,
        help="number of keyframe images encoded per model invocation (default: 32)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="only embed the first MAX_FRAMES records (default: all records)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="worker machine identifier recorded in the manifest (default: 0)",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        default=None,
        help=(
            "root directory of the keyframe image collection used to resolve "
            "image paths: the materialized adaptive_sbd_v1 images/ layout, "
            "falling back to a legacy keyframes/ collection (default: config "
            "data.raw_dir)"
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="output precision for supported backends (default: fp32)",
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

    config = load_config()
    model = args.model or f"openclip:{config.openclip.model_name}:{config.openclip.pretrained}"
    keyframes_root = args.keyframes_dir or config.data_raw_dir

    reporter = create_progress_reporter(args.progress)
    reporter.begin()
    try:
        with reporter.stages.time("load_model"):
            embedder = create_image_embedder(
                model,
                device=args.device,
                precision=args.precision,
                config=config.openclip,
            )
        print(f"Selected device: {embedder.device}")
        print(
            f"Model: {embedder.model_name} "
            f"(dim={embedder.embedding_dim}, dtype={embedder.embedding_space.output_dtype})"
        )
        manifest = embed_shard(
            frames_path=args.frames,
            output_dir=args.output_dir,
            embedder=embedder,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            worker_id=args.worker_id,
            precision=args.precision,
            keyframes_root=keyframes_root,
            progress=reporter,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Shard complete: {manifest.num_embeddings} embeddings "
        f"for {manifest.num_videos} videos (worker_id={manifest.worker_id})"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())