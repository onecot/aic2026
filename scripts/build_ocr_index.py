#!/usr/bin/env python3
"""CLI: build an OCR-only dense retrieval index from OCR chunks.

Encodes every chunk's ``text`` with the configured Jina CLIP text encoder
and writes ``manifest.json`` + ``embeddings.npy`` + ``chunks.ndjson`` row
metadata. The written artifact is re-read and validated before success is
reported.

This is a thin CLI; the logic lives in ``aic2026.ocr_index``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.ocr_index import build_ocr_index, read_ocr_index, write_ocr_index


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="build_ocr_index",
        description=(
            "Build an exact-cosine OCR chunk index using the configured "
            "Jina CLIP text encoder."
        ),
    )
    parser.add_argument(
        "--chunks-artifact",
        type=Path,
        required=True,
        help="completed OCR chunk artifact directory (manifest.json + chunks.ndjson)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="index output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/jina-clip-v2.yaml"),
        help="experiment config selecting the embedding backend (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default=None,
        help="override the configured encoder device",
    )
    parser.add_argument("--artifact-id", default="ocr-index", help="id recorded in the manifest")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="texts encoded per model call (default: %(default)s)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing index artifact instead of failing when present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        if (
            args.output_dir.exists()
            and any(args.output_dir.iterdir())
            and not args.overwrite
        ):
            raise ValueError(
                f"output already exists (and is not empty): {args.output_dir}; "
                "use --overwrite to replace it"
            )
        config = load_config(args.config)
        if config.jina is None:
            raise ValueError(f"config {args.config} has no models.jina section")

        from aic2026.model_backends.jina import JinaClipEncoder

        jina_config = dataclasses.replace(config.jina, device=args.device) if args.device else config.jina
        encoder = JinaClipEncoder(jina_config)
        manifest, embeddings, rows = build_ocr_index(
            args.chunks_artifact,
            encoder,
            artifact_id=args.artifact_id,
            batch_size=args.batch_size,
        )
        write_ocr_index(args.output_dir, manifest, embeddings, rows)
        read_ocr_index(args.output_dir)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    space = manifest.embedding_space
    print(
        f"Built OCR index: {manifest.num_rows} rows, dim={manifest.embedding_dim}, "
        f"model={space.backend}/{space.model_id}/{space.checkpoint_id}"
    )
    print(f"Space: {space.space_id}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
