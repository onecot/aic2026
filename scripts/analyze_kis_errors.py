#!/usr/bin/env python3
"""CLI: analyze KIS retrieval errors for manual inspection.

Reads an evaluation CSV and a ``predict-kis`` prediction artifact and prints a
human-readable per-query breakdown (HIT / NEAR_VIDEO / FAIL) plus aggregate
statistics. Optional ``--category`` / ``--sort`` flags filter and order the
records, and ``--json-output`` exports the selected records as structured
JSON. This is a thin CLI; the analysis logic lives in ``aic2026.error_analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic2026.error_analysis import (
    analyze_kis_errors,
    format_report,
    select_analyses,
    summarize,
)
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring, read_kis_prediction_artifact

DEFAULT_EVAL_CSV = "data/eval/eval_set.csv"
DEFAULT_PREDICTIONS = "data/interim/predictions/kis-l14-laion2b-full-en.json"


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_kis_errors",
        description="Categorize KIS retrieval failures (HIT / NEAR_VIDEO / FAIL) for manual inspection.",
    )
    parser.add_argument(
        "--eval-csv",
        default=DEFAULT_EVAL_CSV,
        help=f"evaluation CSV path (default: {DEFAULT_EVAL_CSV})",
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTIONS,
        help=f"prediction artifact JSON path (default: {DEFAULT_PREDICTIONS})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="number of predictions to inspect per query (default: 100)",
    )
    parser.add_argument(
        "--category",
        choices=["HIT", "NEAR_VIDEO", "FAIL"],
        default=None,
        help="only report records with this category (default: all categories)",
    )
    parser.add_argument(
        "--sort",
        choices=["distance", "rank", "none"],
        default="none",
        help="order records: 'distance' by closest-frame-distance ascending "
        "(records without a distance sort last); 'rank' by first-GT-video rank "
        "ascending (records without a rank sort last); 'none' keeps query order",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional human-readable report output path; when omitted the report is printed to stdout",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="optional path for structured JSON export of the selected QueryErrorAnalysis records",
    )
    return parser


def write_json_artifact(path: Path | str, payload: object) -> None:
    """Write a JSON array atomically, rejecting NaN/infinity values."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temp_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    eval_set = load_eval_set(args.eval_csv)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    if not queries:
        raise SystemExit("evaluation set contains no KIS queries")

    predictions = load_kis_predictions_for_scoring(args.predictions)
    artifact = read_kis_prediction_artifact(args.predictions)
    scores_by_query = {
        query_result.query_id: {
            (prediction.video_id, prediction.frame_index): float(prediction.score)
            for prediction in query_result.predictions
        }
        for query_result in artifact.query_results
    }

    analyses = analyze_kis_errors(queries, predictions, scores_by_query=scores_by_query, top_k=args.top_k)
    analyses = select_analyses(analyses, category=args.category, sort=args.sort)

    if not analyses:
        message = "No records matched the filter.\n"
        if args.json_output is not None:
            write_json_artifact(args.json_output, [])
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(message, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            print(message, end="")
        return 0

    summary = summarize(analyses)
    report = format_report(analyses, summary)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report, end="")

    if args.json_output is not None:
        write_json_artifact(args.json_output, [analysis.to_dict() for analysis in analyses])
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
