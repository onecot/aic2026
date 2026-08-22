#!/usr/bin/env python3
"""CLI: evaluate OCR-only chunk retrieval on canonical evaluation sets.

For each KIS/QA query, retrieves the top-k OCR chunks by exact cosine
similarity (no ground truth is used for retrieval) and reports Hit@k using
the ground-truth video and frame range. Answer generation is not evaluated.

This is a thin CLI; the logic lives in ``aic2026.ocr_eval``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.ocr_eval import evaluate_ocr_retrieval, format_report
from aic2026.ocr_index import read_ocr_index


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_ocr_retrieval",
        description="Report OCR-only retrieval Hit@k on an evaluation set.",
    )
    parser.add_argument("--index-dir", type=Path, required=True, help="OCR index artifact directory")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="canonical evaluation set CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        choices=("kis", "qa"),
        default=None,
        help="evaluate one task; default evaluates both when present",
    )
    parser.add_argument("--top-k", type=int, default=100, help="retrieval cutoff (default: %(default)s)")
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help="count only hits whose representative frame falls in the GT range",
    )
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
        index = read_ocr_index(args.index_dir)
        eval_set = load_eval_set(args.eval_set)

        tasks = [EvaluationTask(args.task)] if args.task else [EvaluationTask.KIS, EvaluationTask.QA]
        reports = []
        for task in tasks:
            if not any(True for _ in eval_set.iter_queries(task)):
                print(f"no {task.value} queries in {args.eval_set}; skipping")
                continue
            report = evaluate_ocr_retrieval(
                eval_set,
                index,
                encoder,
                task=task,
                top_k=args.top_k,
                representative_only=args.representative_only,
            )
            reports.append(report)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for report in reports:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
