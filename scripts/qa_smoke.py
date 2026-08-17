#!/usr/bin/env python3
"""CLI: run the end-to-end QA pipeline smoke test (synthetic data, no VQA).

Builds a small deterministic synthetic index, runs the QA retrieval pipeline
over a synthetic QA eval set, validates the prediction artifact, fills
placeholder answers, and exports the provisional BTC submission CSV. This is
a testing/demo layer: no VQA/answer model is used and the retrieval core /
KIS behavior is not modified. The logic lives in
``aic2026.qa.qa_smoke``; this script is a thin wrapper.

Example:
    python scripts/qa_smoke.py --work-dir data/interim/qa-smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.qa.qa_smoke import (
    QaSmokeConfig,
    format_smoke_report,
    run_qa_smoke,
    write_smoke_report_json,
)

DEFAULT_EVAL_CSV = "data/eval/qa_smoke_eval.csv"
DEFAULT_WORK_DIR = "data/interim/qa-smoke"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="qa_smoke",
        description=(
            "Run the synthetic end-to-end QA pipeline smoke test "
            "(build index, retrieve, validate artifact, export submission)."
        ),
    )
    parser.add_argument(
        "--evaluation-set",
        default=DEFAULT_EVAL_CSV,
        help=f"synthetic QA eval CSV path (default: {DEFAULT_EVAL_CSV})",
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help=f"output directory for regenerated demo artifacts (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="frame retrieval cutoff per query (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the synthetic index (default: 42)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="optional path to write the smoke outcome as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the QA smoke test CLI."""
    args = build_argument_parser().parse_args(argv)
    config = QaSmokeConfig(
        evaluation_set_path=Path(args.evaluation_set),
        work_dir=Path(args.work_dir),
        top_k=args.top_k,
        seed=args.seed,
    )
    try:
        result = run_qa_smoke(config)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_smoke_report(result))
    if args.report_json:
        written = write_smoke_report_json(result, args.report_json)
        print(f"Report written to: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())