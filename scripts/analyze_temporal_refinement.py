#!/usr/bin/env python3
"""CLI: temporal/frame-level refinement ceiling analysis (Phase 4A).

Runs the failure decomposition, distance-to-GT statistics, candidate-video
oracle ceiling, and cheap deterministic refinement baselines over the
existing visual index. Diagnostic only: no ranking changes, no new frames,
no models.

The baseline ranking is derived with the exact IndexRetriever numerics and
verified against ``IndexRetriever.search`` for every query before any
analysis is reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from aic2026.config import load_config
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.index_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    NORMS_FILENAME,
    read_manifest,
    validate_index_files,
)
from aic2026.embedding_artifact import iter_frames
from aic2026.kis_scoring import is_kis_correct
from aic2026.retrieval import IndexRetriever
from aic2026.temporal_refinement_analysis import (
    DISTANCE_BUCKETS,
    FRAME_MISS,
    REPORT_K,
    VIDEO_CUTOFFS,
    analyze_query,
    derive_ranking,
    hits_at_k,
    oracle_hit_at_video_cutoffs,
    refine_frame_selection,
    refine_pooling,
    rerank_by_smoothed,
    summarize_outcomes,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze_temporal_refinement",
        description=(
            "Analyze whether frame-level refinement over the existing index "
            "can fix visual retrieval failures."
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
        "--task", choices=("kis", "qa"), default=None, help="analyze one task; default both"
    )
    parser.add_argument(
        "--analysis-pool",
        type=int,
        default=1000,
        help="rows of the full ranking kept for per-query analysis (default: %(default)s)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="distinct candidate videos for refinement methods (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/jina-clip-v2.yaml"),
        help="experiment config selecting the embedding backend (default: %(default)s)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=None, help="override encoder device"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/analysis/temporal-refinement"),
        help="where to write summary.json (default: %(default)s)",
    )
    return parser


def full_scores_for_query(embeddings: np.ndarray, norms: np.ndarray, query_vector) -> np.ndarray:
    """Compute per-row cosine scores exactly like ``IndexRetriever.search``."""
    query32 = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query32))
    dots = embeddings @ query32
    denom = norms.astype(np.float32) * np.float32(query_norm)
    return np.divide(
        dots,
        denom,
        out=np.full_like(dots, fill_value=-np.inf, dtype=np.float32),
        where=denom > 0.0,
    )


def method_rankings(scores: np.ndarray, video_ids, baseline_rows, max_candidates: int):
    """Return all fixed refinement variants for one query."""
    variants: dict[str, list[int]] = {}
    for pool in ("max", "top2-mean", "top3-mean"):
        variants[f"pool-{pool}"] = refine_pooling(
            scores, video_ids, baseline_rows, pool=pool, max_candidates=max_candidates
        ).ranked_rows
    variants["reselect-argmax"] = refine_frame_selection(
        scores, video_ids, baseline_rows, method="argmax", window=0, max_candidates=max_candidates
    ).ranked_rows
    for window in (1, 2, 4):
        variants[f"smooth-w{window}"] = refine_frame_selection(
            scores,
            video_ids,
            baseline_rows,
            method="smooth-argmax",
            window=window,
            max_candidates=max_candidates,
        ).ranked_rows
    variants["local-peak-w1"] = refine_frame_selection(
        scores, video_ids, baseline_rows, method="local-peak", window=1, max_candidates=max_candidates
    ).ranked_rows
    for window in (1, 2):
        variants[f"rerank-smooth-w{window}"] = rerank_by_smoothed(
            scores, video_ids, baseline_rows, window=window
        )
    return variants


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

        manifest = read_manifest(args.index_artifact / "manifest.json")
        problems = validate_index_files(manifest, args.index_artifact)
        if problems:
            raise ValueError(f"invalid index artifact: {problems}")
        retriever = IndexRetriever(args.index_artifact)
        embeddings = np.ascontiguousarray(
            np.load(args.index_artifact / EMBEDDINGS_FILENAME, mmap_mode="r"), dtype=np.float32
        )
        norms = np.load(args.index_artifact / NORMS_FILENAME, mmap_mode="r").astype(np.float32)
        video_ids = [frame.keyframe.video_id.value for frame in iter_frames(args.index_artifact / FRAMES_FILENAME)]
        frame_indices = [frame.frame_index for frame in iter_frames(args.index_artifact / FRAMES_FILENAME)]

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        tasks = [EvaluationTask(args.task)] if args.task else [EvaluationTask.KIS, EvaluationTask.QA]

        report: dict[str, dict] = {}
        for task in tasks:
            queries = [q for eval_set in eval_sets for q in eval_set.iter_queries(task)]
            if not queries:
                print(f"no {task.value} queries found; skipping")
                continue

            analyses = []
            baseline_hits: dict[str, dict[int, bool]] = {}
            method_hits: dict[str, dict[str, dict[int, bool]]] = {}
            mismatches = 0
            for query in queries:
                embedding = encoder.encode_text(query.query_text_vi)
                scores = full_scores_for_query(embeddings, norms, embedding.vector)
                ranked_rows = derive_ranking(scores)[: args.analysis_pool]

                # Baseline reproduction check against the production retriever.
                reference = retriever.search(embedding, top_k=min(100, len(ranked_rows)))
                if [
                    (c.video_id, c.frame_index) for c in reference
                ] != [
                    (video_ids[row], frame_indices[row]) for row in ranked_rows[:100]
                ]:
                    mismatches += 1

                # Cross-check the local predicate against existing semantics
                # (is_kis_correct is defined for KIS only; QA shares the same
                # frame rule by documented mirror).
                if query.task == EvaluationTask.KIS:
                    for candidate in reference[:20]:
                        assert is_kis_correct(query, type(
                            "P", (), {"rank": candidate.rank, "video_id": candidate.video_id, "frame_index": candidate.frame_index}
                        )()) == (
                            candidate.video_id == query.ground_truth_video_id.value
                            and query.ground_truth_frame_range.start
                            <= candidate.frame_index
                            <= query.ground_truth_frame_range.end
                        )

                analysis = analyze_query(
                    query.query_id,
                    query.ground_truth_video_id.value,
                    query.ground_truth_frame_range.start,
                    query.ground_truth_frame_range.end,
                    ranked_rows,
                    video_ids,
                    frame_indices,
                )
                analyses.append(analysis)
                baseline_hits[query.query_id] = hits_at_k(
                    ranked_rows, video_ids, frame_indices,
                    query.ground_truth_video_id.value,
                    query.ground_truth_frame_range.start,
                    query.ground_truth_frame_range.end,
                )
                for name, rows in method_rankings(
                    scores, video_ids, ranked_rows, args.max_candidates
                ).items():
                    method_hits.setdefault(name, {})[query.query_id] = hits_at_k(
                        rows, video_ids, frame_indices,
                        query.ground_truth_video_id.value,
                        query.ground_truth_frame_range.start,
                        query.ground_truth_frame_range.end,
                    )

            print(f"\n=== task={task.value} ({len(queries)} queries) ===")
            print(f"baseline reproduction mismatches vs IndexRetriever: {mismatches} (must be 0)")
            if mismatches:
                raise ValueError(f"baseline derivation mismatched IndexRetriever on {mismatches} queries")

            # Part 1: failure decomposition.
            outcomes_by_k: dict[int, Counter] = {k: Counter() for k in REPORT_K}
            for analysis in analyses:
                for k, outcome in analysis.outcomes.items():
                    outcomes_by_k[k][outcome] += 1
            decomposition = summarize_outcomes(outcomes_by_k)
            print("\nfailure decomposition:")
            for k in REPORT_K:
                d = decomposition[k]
                total = d["total"]
                print(
                    f"  @{k:<4} VIDEO_MISS={d['VIDEO_MISS']:<4}"
                    f" VIDEO_HIT_FRAME_HIT={d['VIDEO_HIT_FRAME_HIT']:<4}"
                    f" VIDEO_HIT_FRAME_MISS={d['VIDEO_HIT_FRAME_MISS']:<4}"
                    f" (hit rate {d['VIDEO_HIT_FRAME_HIT'] / total:.4f})"
                )

            # Part 2: distances for FRAME_MISS@100.
            miss_analyses = [a for a in analyses if a.outcomes[100] == FRAME_MISS]
            distances = sorted(a.min_distance for a in miss_analyses if a.min_distance is not None)
            print(f"\ndistance to GT for VIDEO_HIT_FRAME_MISS@100 (n={len(distances)}):")
            if distances:
                def pct(p: float) -> int:
                    return distances[min(len(distances) - 1, int(p * len(distances)))]
                print(f"  median={pct(0.5)} p75={pct(0.75)} p90={pct(0.9)} p95={pct(0.95)} max={distances[-1]}")
                bucket_counts = Counter(a.distance_bucket for a in miss_analyses)
                for label, _ in DISTANCE_BUCKETS:
                    print(f"  {label:<10}: {bucket_counts.get(label, 0)}")

            # Part 3: oracle ceiling by distinct-video cutoff.
            print("\noracle ceiling (perfect localizer over indexed frames):")
            oracle_report: dict[str, dict] = {}
            for n in VIDEO_CUTOFFS:
                recalled = sum(
                    1 for a in analyses
                    if a.gt_video_candidate_index is not None and a.gt_video_candidate_index <= n
                )
                oracle_all = sum(
                    1 for a in analyses if oracle_hit_at_video_cutoffs(a)[n]
                )
                oracle_fixed = sum(
                    1 for a in analyses if oracle_hit_at_video_cutoffs(a, frame_miss_only=True)[n]
                )
                oracle_report[str(n)] = {
                    "video_recall": recalled,
                    "oracle_hit": oracle_all,
                    "fixable_frame_misses": oracle_fixed,
                }
                print(
                    f"  top-{n:<3} distinct videos: recall={recalled:<4}"
                    f" oracle_hit={oracle_all:<4} fixable_frame_misses={oracle_fixed}"
                )

            # Parts 4-5: refinement baselines vs baseline.
            print(f"\nrefinement baselines (max_candidates={args.max_candidates}):")
            header = "method                 " + "".join(f"  Hit@{k:<4}" for k in REPORT_K) + "   helped/harmed@100"
            print(header)
            methods_report: dict[str, dict] = {}
            examples: dict[str, dict] = {}
            for name, hits_by_query in method_hits.items():
                rates = {
                    k: sum(1 for h in hits_by_query.values() if h[k]) / len(hits_by_query)
                    for k in REPORT_K
                }
                helped = sum(
                    1 for qid in hits_by_query
                    if hits_by_query[qid][100] and not baseline_hits[qid][100]
                )
                harmed = sum(
                    1 for qid in hits_by_query
                    if baseline_hits[qid][100] and not hits_by_query[qid][100]
                )
                deltas = {k: rates[k] - sum(1 for h in baseline_hits.values() if h[k]) / len(baseline_hits)
                          for k in REPORT_K}
                methods_report[name] = {
                    "hit_rates": rates,
                    "deltas": deltas,
                    "helped_at_100": helped,
                    "harmed_at_100": harmed,
                }
                cells = "".join(f"  {rates[k]:>7.4f}" for k in REPORT_K)
                print(f"{name:<23}{cells}   +{helped}/-{harmed}")

                improved = [
                    qid for qid in hits_by_query
                    if hits_by_query[qid][100] and not baseline_hits[qid][100]
                ]
                regressed = [
                    qid for qid in hits_by_query
                    if baseline_hits[qid][100] and not hits_by_query[qid][100]
                ]
                examples[name] = {"improved": improved[:2], "regressed": regressed[:2]}

            report[task.value] = {
                "num_queries": len(queries),
                "decomposition": {str(k): v for k, v in decomposition.items()},
                "distance_stats": {
                    "n": len(distances),
                    "buckets": dict(Counter(a.distance_bucket for a in miss_analyses)),
                },
                "oracle": oracle_report,
                "methods": methods_report,
                "examples": examples,
            }

            # Print concrete examples for the smoothing variant.
            for ex_name in ("smooth-w2", "pool-max"):
                for kind in ("improved", "regressed"):
                    for qid in examples.get(ex_name, {}).get(kind, []):
                        query = next(q for q in queries if q.query_id == qid)
                        print(
                            f"  [{ex_name} {kind}] {qid}: {query.query_text_vi[:70]} "
                            f"GT={query.ground_truth_video_id.value}"
                            f"[{query.ground_truth_frame_range.start}..{query.ground_truth_frame_range.end}]"
                        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "summary.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSummary: {out_path}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
