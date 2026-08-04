"""Build and save the searchable CLIP embedding index.

Usage example:
    python -m scripts.build_index --config configs/default.yaml --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from src.data import loader
from src.utils.config import load_config


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for index construction.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed arguments containing config path, video limit, and output override.
    Raises: SystemExit if the supplied CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Build the searchable CLIP index.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of sorted videos to index; 0 indexes all videos.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional index output path overriding index.path in the config.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Build the configured index and report its basic statistics.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``;
    the selected config contains data and index settings.
    Returns: None.
    Raises: FileNotFoundError if configured inputs are missing; ValueError if no videos are
        selected; OSError if the output index cannot be written.
    """
    args = parse_args(arguments)
    cfg = load_config(args.config)

    features_dir = cfg["data"]["features_dir"]
    mapping_dir = cfg["data"]["mapping_dir"]
    normalize = cfg["index"]["normalize"]
    out = args.out or cfg["index"]["path"]

    start_time = perf_counter()
    emb, vids, frame_idx = loader.build_index(
        features_dir,
        mapping_dir,
        normalize=normalize,
        limit=args.limit,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    loader.save_index(out, emb, vids, frame_idx)
    elapsed_seconds = perf_counter() - start_time

    print(f"#videos: {len(np.unique(vids))}")
    print(f"N_total: {emb.shape[0]}")
    print(f"dim: {emb.shape[1]}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print(f"output_path: {out}")


if __name__ == "__main__":
    main()
