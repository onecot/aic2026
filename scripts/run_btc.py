"""CLI for BTC-query inference.

Thin interface over :mod:`aic2026.btc_inference`: loads the BTC experimental
query files, builds the configured text encoder, and runs the existing
KIS / QA / TRAKE prediction pipelines (no GT/evaluation metadata) into
standard prediction artifacts compatible with ``scripts/build_submission.py``.

Usage::

    python scripts/run_btc.py \
        --query-dir data/sotuyen/thunghiem \\
        --index-artifact data/indexes/<index-artifact> \\
        --output-dir data/interim/btc \\
        --tasks kis qa trake

Small-scale validation::

    python scripts/run_btc.py ... --limit 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.btc_inference import (
    run_btc_kis_prediction,
    run_btc_qa_prediction,
    run_btc_trake_prediction,
)
from aic2026.btc_queries import load_btc_queries
from aic2026.config import load_config
from aic2026.image_embedder import create_text_encoder
from aic2026.qa.answer_generation import ANSWER_DEVICE_CHOICES, DEFAULT_ANSWER_TOP_N

logger = logging.getLogger(__name__)

TASK_CHOICES = ("kis", "qa", "trake")


def _filter_queries(
    btc_queries: tuple,
    query_names: set[str] | None,
    limit: int | None,
) -> tuple:
    """Apply the small-scale validation filters to loaded queries."""
    filtered = btc_queries
    if query_names:
        filtered = tuple(query for query in filtered if query.info.name in query_names)
        unknown = query_names - {query.info.name for query in btc_queries}
        if unknown:
            raise ValueError(f"unknown query names: {sorted(unknown)}")
    if limit is not None:
        by_task: dict[str, list] = {}
        for query in filtered:
            by_task.setdefault(query.info.task, []).append(query)
        filtered = tuple(
            query
            for task in TASK_CHOICES
            for query in by_task.get(task, [])[:limit]
        )
    if not filtered:
        raise ValueError("query filters selected no queries")
    return filtered


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for BTC-query inference.

    Expected input:
        - argv: optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        - Exit code (0 = success).

    Raises:
        - SystemExit: on argument parsing errors.
    """
    parser = argparse.ArgumentParser(
        description="Run KIS/QA/TRAKE prediction on BTC experimental query files.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query-<name>-<task>.txt files.",
    )
    parser.add_argument(
        "--index-artifact",
        type=Path,
        required=True,
        help="Path to the index artifact directory used for retrieval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the prediction artifacts.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_CHOICES,
        default=list(TASK_CHOICES),
        help="Tasks to run (default: all).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path for the text encoder.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Retrieval cutoff per query/event.")
    parser.add_argument(
        "--window-max-gap",
        type=int,
        default=60,
        help="TRAKE temporal window gap (0 disables windows).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Stable run id prefix override.")
    parser.add_argument(
        "--allow-embedding-space-mismatch",
        action="store_true",
        help="Explicitly allow encoder/index embedding-space mismatch (diagnostics only).",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated query names (e.g. 'p1-1,p1-4') for a small subset run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit per task to the first N queries (small-scale validation).",
    )
    parser.add_argument("--generate-answer", action="store_true", help="Enable QA answer generation.")
    parser.add_argument(
        "--answer-top-n",
        type=int,
        default=DEFAULT_ANSWER_TOP_N,
        help="QA evidence-set size per query.",
    )
    parser.add_argument(
        "--answer-device",
        type=str,
        default="mps",
        choices=sorted(ANSWER_DEVICE_CHOICES),
        help="QA answer-generation device.",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        default=None,
        help="Keyframe artifact root for QA answer generation (default: config value).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    query_names = {name.strip() for name in args.queries.split(",")} if args.queries else None

    try:
        btc_queries = load_btc_queries(args.query_dir)
        btc_queries = _filter_queries(btc_queries, query_names, args.limit)
        counts = {task: 0 for task in TASK_CHOICES}
        for query in btc_queries:
            counts[query.info.task] += 1
        logger.info(
            "Loaded BTC queries from %s: kis=%d qa=%d trake=%d",
            args.query_dir,
            counts["kis"],
            counts["qa"],
            counts["trake"],
        )

        encoder = create_text_encoder(config)
        model_config = config.model_config
        output_dir = args.output_dir
        window_max_gap = args.window_max_gap if args.window_max_gap > 0 else None
        run_id = args.run_id
        results: dict[str, object] = {}

        if "kis" in args.tasks and counts["kis"] > 0:
            results["kis"] = run_btc_kis_prediction(
                btc_queries,
                encoder=encoder,
                index_artifact_dir=args.index_artifact,
                prediction_output_path=output_dir / "kis-predictions.json",
                top_k=args.top_k,
                model_config=model_config,
                allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                run_id=f"{run_id}-kis" if run_id else None,
            )
        if "qa" in args.tasks and counts["qa"] > 0:
            if not args.generate_answer:
                logger.warning(
                    "QA answers will be empty (retrieval evidence only); "
                    "submission export requires --generate-answer"
                )
            results["qa"] = run_btc_qa_prediction(
                btc_queries,
                encoder=encoder,
                index_artifact_dir=args.index_artifact,
                prediction_output_dir=output_dir / "qa-predictions",
                top_k=args.top_k,
                model_config=model_config,
                allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                run_id=f"{run_id}-qa" if run_id else None,
                generate_answer=args.generate_answer,
                answer_top_n=args.answer_top_n,
                answer_device=args.answer_device,
                keyframes_dir=args.keyframes_dir or config.data_keyframes_dir,
            )
        if "trake" in args.tasks and counts["trake"] > 0:
            results["trake"] = run_btc_trake_prediction(
                btc_queries,
                encoder=encoder,
                index_artifact_dir=args.index_artifact,
                prediction_output_dir=output_dir / "trake-predictions",
                top_k=args.top_k,
                window_max_gap=window_max_gap,
                model_config=model_config,
                allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                run_id=f"{run_id}-trake" if run_id else None,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("")
    for task, artifact in results.items():
        manifest = getattr(artifact, "manifest", artifact)
        print(
            f"{task.upper()} artifact: processed "
            f"{manifest.processed_query_count}/{manifest.evaluation_set_query_count} queries"
        )
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
