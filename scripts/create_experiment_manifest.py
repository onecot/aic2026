#!/usr/bin/env python3
"""CLI: create a lightweight experiment manifest for an embedding/retrieval run.

Writes a deterministic, atomic JSON manifest (``experiment_manifest.json``)
recording the identity, provenance, and free-form parameters of one experiment,
so distinct model runs (OpenCLIP, Jina, SigLIP, MobileCLIP, ...) cannot be
confused.

This is a thin CLI; the manifest logic lives in ``aic2026.experiment_manifest``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from aic2026.experiment_manifest import (
    KNOWN_EXPERIMENT_TYPES,
    new_manifest,
    parse_key_value,
    save_manifest,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="create_experiment_manifest",
        description=(
            "Create an experiment manifest (metadata/documentation only). "
            "Does not run or modify any embedding/retrieval logic."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path of the manifest JSON file to write (e.g. experiments/jina-v1/manifest.json)",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="stable, unique experiment identifier (e.g. jina-v1)",
    )
    parser.add_argument(
        "--type",
        required=True,
        help=(
            "experiment type: one of "
            + ", ".join(KNOWN_EXPERIMENT_TYPES)
            + ", or any custom string"
        ),
    )
    parser.add_argument(
        "--description",
        required=True,
        help="human-readable experiment description",
    )
    parser.add_argument("--model", default=None, help="model identity (e.g. jina-clip-v2)")
    parser.add_argument("--embedding-dim", type=int, default=None, help="vector length")
    parser.add_argument("--frames-source", default=None, help="provenance of the keyframe set")
    parser.add_argument("--split-manifest", default=None, help="path of the split manifest used")
    parser.add_argument("--workers", type=int, default=None, help="number of shard workers")
    parser.add_argument("--hardware", default=None, help="hardware description (e.g. mps, cuda)")
    parser.add_argument("--index-path", default=None, help="path of the index artifact evaluated")
    parser.add_argument("--prediction-artifact", default=None, help="path of the prediction artifact")
    parser.add_argument("--evaluation-set", default=None, help="path of the evaluation set")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="experiment parameter (repeatable); values are typed (bool/int/float/str)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metric key=value (repeatable); values are typed (bool/int/float/str)",
    )
    return parser


def _parse_pairs(raw: list[str], label: str) -> dict[str, Any]:
    """Parse repeated ``key=value`` strings into a typed dict."""
    out: dict[str, Any] = {}
    for item in raw:
        key, value = parse_key_value(item)
        if key in out:
            raise ValueError(f"duplicate {label} key {key!r}")
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        parameters = _parse_pairs(args.parameter, "parameter")
        metrics = _parse_pairs(args.metric, "metric")
        manifest = new_manifest(
            experiment_id=args.experiment_id,
            experiment_type=args.type,
            description=args.description,
            model_name=args.model,
            embedding_dim=args.embedding_dim,
            frames_source=args.frames_source,
            split_manifest=args.split_manifest,
            workers=args.workers,
            hardware=args.hardware,
            index_path=args.index_path,
            prediction_artifact=args.prediction_artifact,
            evaluation_set=args.evaluation_set,
            metrics=metrics or None,
            parameters=parameters,
        )
        save_manifest(args.output, manifest)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Experiment manifest written: {args.output} "
        f"(id={manifest.experiment_id}, type={manifest.experiment_type})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())