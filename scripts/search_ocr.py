#!/usr/bin/env python3
"""CLI: search an OCR chunk index with one text query (manual inspection).

Encodes the query with the configured Jina CLIP text encoder, runs exact
cosine retrieval over an OCR index artifact, and prints a ranked table of
rank / video / frame range / score / OCR text for manual inspection.

This is a thin CLI; the logic lives in ``aic2026.ocr_index``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.ocr_index import read_ocr_index, search_ocr_chunks


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="search_ocr",
        description="Rank OCR chunks for one text query using exact cosine similarity.",
    )
    parser.add_argument("--index-dir", type=Path, required=True, help="OCR index artifact directory")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=10, help="number of hits to print (default: %(default)s)")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if config.jina is None:
            raise ValueError(f"config {args.config} has no models.jina section")

        from aic2026.model_backends.jina import JinaClipEncoder

        jina_config = dataclasses.replace(config.jina, device=args.device) if args.device else config.jina
        encoder = JinaClipEncoder(jina_config)
        results = search_ocr_chunks(
            args.index_dir,
            args.query,
            encoder=encoder,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    handle = read_ocr_index(args.index_dir)
    print(f"query: {args.query}")
    print(f"index: {handle.manifest.num_rows} rows | space {handle.manifest.embedding_space.space_id}")
    for result in results:
        preview = result.text if len(result.text) <= 160 else result.text[:157] + "..."
        print(
            f"#{result.rank:<3} {result.video_id} "
            f"[{result.frame_start}..{result.frame_end}] rep={result.representative_frame_index} "
            f"score={result.score:.4f} {result.chunk_id}\n     {preview}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
