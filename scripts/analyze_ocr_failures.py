#!/usr/bin/env python3
"""CLI: analyze visual retrieval failures for lexical OCR evidence.

For every KIS/QA query whose ground-truth video/frame region is not
retrieved within the visual top-k (existing frame semantics), inspects the
ground-truth video's OCR chunks for query-term overlap, rare terms, and
phrase matches, then classifies the failure and writes machine-readable
records plus a human-readable summary.

Diagnostic only: no ranking, no BM25, no retrieval changes.

This is a thin CLI; the logic lives in ``aic2026.ocr_failure_analysis``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from aic2026.config import load_config
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.ocr_chunks import read_ocr_chunks
from aic2026.ocr_failure_analysis import (
    AnalysisConfig,
    FailureAnalysis,
    run_failure_analysis,
    shortlist,
    summarize,
)
from aic2026.retrieval import IndexRetriever


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_ocr_failures",
        description=(
            "Classify visual retrieval failures by their recoverable OCR "
            "lexical evidence."
        ),
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        nargs="+",
        default=[Path("data/eval/eval_set.csv"), Path("data/eval/eval_set-qa_trake.csv")],
        help="evaluation set CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--index-artifact",
        type=Path,
        default=Path("data/interim/indexes/jina-v2-full"),
        help="visual index artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--ocr-chunks",
        type=Path,
        default=Path("data/interim/ocr/chunks"),
        help="OCR chunk artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--task", choices=("kis", "qa"), default=None, help="analyze one task; default analyzes both"
    )
    parser.add_argument(
        "--visual-top-k", type=int, default=100, help="visual cutoff defining a failure (default: %(default)s)"
    )
    parser.add_argument("--rare-max-df", type=int, default=10, help="rare-term DF threshold (default: %(default)s)")
    parser.add_argument(
        "--min-phrase-tokens", type=int, default=2, help="minimum phrase length (default: %(default)s)"
    )
    parser.add_argument(
        "--gt-near-frames",
        type=int,
        default=250,
        help="chunk-GT gap tolerated as temporally aligned (default: %(default)s)",
    )
    parser.add_argument(
        "--max-print", type=int, default=10, help="promising cases to print per task (default: %(default)s)"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="analysis output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/jina-clip-v2.yaml"),
        help="experiment config selecting the embedding backend (default: %(default)s)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=None, help="override encoder device"
    )
    return parser


def failure_to_dict(analysis: FailureAnalysis) -> dict:
    """Serialize one analysis record to a JSON-safe dict."""
    return {
        "query_id": analysis.query_id,
        "task": analysis.task,
        "query_text_vi": analysis.query_text_vi,
        "gt_video_id": analysis.gt_video_id,
        "gt_frame_start": analysis.gt_frame_start,
        "gt_frame_end": analysis.gt_frame_end,
        "visual_top_k": analysis.visual_top_k,
        "gt_video_in_visual_top_k": analysis.gt_video_in_visual_top_k,
        "gt_best_visual_rank": analysis.gt_best_visual_rank,
        "gt_chunks_total": analysis.gt_chunks_total,
        "gt_chunks_with_text": analysis.gt_chunks_with_text,
        "temporally_near_overlap_chunks": analysis.temporally_near_overlap_chunks,
        "matched_terms": [dataclasses.asdict(term) for term in analysis.matched_terms],
        "matched_phrases": list(analysis.matched_phrases),
        "excerpts": list(analysis.excerpts),
        "classification": analysis.classification,
        "temporal_mismatch": analysis.temporal_mismatch,
        "promising": analysis.promising,
    }


def print_task_report(task_name: str, summary: dict, picks: Sequence[FailureAnalysis]) -> None:
    """Print the human-readable summary and promising examples."""
    pct = summary["percentages"]
    print(f"\n=== {task_name} ===")
    print(f"  total queries              : {summary['total_queries']}")
    print(f"  visual failures            : {summary['visual_failures']} ({pct['visual_failures']}%)")
    print(f"  with GT OCR text           : {summary['failures_with_gt_ocr_text']}")
    print(f"  with any lexical overlap   : {summary['failures_with_lexical_overlap']} ({pct['lexical_overlap']}%)")
    print(f"  rare-term matches          : {summary['failures_rare_term_match']} ({pct['rare_term']}%)")
    print(f"  phrase matches             : {summary['failures_phrase_match']} ({pct['phrase_match']}%)")
    print(f"  common-term-only           : {summary['failures_common_term_only']}")
    print(f"  no lexical overlap         : {summary['failures_no_lexical_overlap']}")
    print(f"  no GT OCR text             : {summary['failures_no_gt_ocr_text']}")
    print(f"  temporal mismatch          : {summary['temporal_mismatch_cases']} ({pct['temporal_mismatch']}%)")
    print(f"  promising (lexical)        : {summary['promising_failures']} ({pct['promising']}%)")
    for pick in picks:
        terms = ", ".join(f"{t.term}(df={t.df})" for t in pick.matched_terms if t.rare)
        print(
            f"\n  [{pick.classification}] {pick.query_id}: {pick.query_text_vi[:80]}\n"
            f"    GT: {pick.gt_video_id} [{pick.gt_frame_start}..{pick.gt_frame_end}] | "
            f"in visual top-k: {pick.gt_video_in_visual_top_k}\n"
            f"    phrases: {list(pick.matched_phrases)} | rare terms: {terms or '-'} | "
            f"near GT: {pick.temporally_near_overlap_chunks} chunks\n"
            f"    excerpt: {pick.excerpts[0]['text'][:120] if pick.excerpts else '-'}"
        )


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
        retriever = IndexRetriever(args.index_artifact)
        _, chunks = read_ocr_chunks(args.ocr_chunks)
        eval_sets = [load_eval_set(path) for path in args.eval_set]
        tasks = [EvaluationTask(args.task)] if args.task else [EvaluationTask.KIS, EvaluationTask.QA]
        analysis_config = AnalysisConfig(
            rare_max_df=args.rare_max_df,
            min_phrase_tokens=args.min_phrase_tokens,
            gt_near_frames=args.gt_near_frames,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        overall: dict[str, dict] = {}
        for task in tasks:
            queries = [q for eval_set in eval_sets for q in eval_set.iter_queries(task)]
            if not queries:
                print(f"no {task.value} queries found; skipping")
                continue

            ranked_by_query: dict[str, list] = {}
            for query in queries:
                embedding = encoder.encode_text(query.query_text_vi)
                ranked_by_query[query.query_id] = retriever.search(
                    embedding, top_k=args.visual_top_k
                )

            failures, total = run_failure_analysis(
                _merged_eval_set(eval_sets),
                task,
                ranked_by_query,
                chunks,
                config=analysis_config,
                visual_top_k=args.visual_top_k,
            )

            summary = summarize(failures, total)
            overall[task.value] = summary
            output_path = args.output_dir / f"failures-{task.value}.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for failure in failures:
                    handle.write(json.dumps(failure_to_dict(failure), ensure_ascii=False) + "\n")
            print_task_report(task.value, summary, shortlist(failures, limit=args.max_print))
            print(f"  records: {output_path}")

        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSummary: {summary_path}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _merged_eval_set(eval_sets):
    """Merge evaluation sets into one (deterministic order)."""
    from aic2026.evaluation import EvaluationSet

    queries = tuple(query for eval_set in eval_sets for query in eval_set.queries)
    return EvaluationSet(queries=queries)


if __name__ == "__main__":
    raise SystemExit(main())
