#!/usr/bin/env python3
"""CLI: evaluate OCR-aware late fusion against the visual-only baseline.

For every evaluation query this script retrieves the visual candidate pool
(existing index retriever) and the OCR chunk list (existing OCR index)
once, then compares an explicit set of fusion variants:

    visual-only              baseline ranking (ablation)
    fixed-ocr                same-video OCR boost, no temporal decay,
                             fixed weight
    temporal-ocr             temporally aligned OCR support, fixed weight
    adaptive-temporal-ocr    temporally aligned OCR support, keyword-based
                             query-adaptive weight

Correctness mirrors the existing frame-level semantics (video match +
inclusive ground-truth frame range) for both KIS and QA retrieval. No
ground-truth information is used for retrieval.

This is a thin CLI; the fusion logic lives in ``aic2026.ocr_fusion``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

from aic2026.config import load_config
from aic2026.evaluation import EvaluationQuery, EvaluationTask, load_eval_set
from aic2026.kis_scoring import KIS_TOP_K
from aic2026.ocr_fusion import (
    FusedCandidate,
    OcrFusionConfig,
    fuse_visual_with_ocr,
    query_ocr_weight,
)
from aic2026.ocr_index import OcrSearchResult, read_ocr_index, search_ocr_chunks
from aic2026.retrieval import IndexRetriever, RetrievalCandidate


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_ocr_fusion",
        description="Compare visual-only retrieval with OCR fusion variants.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        nargs="+",
        default=[Path("data/eval/eval_set.csv"), Path("data/eval/eval_set-qa_trake.csv")],
        help="evaluation set CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--visual-index-dir",
        type=Path,
        default=Path("data/interim/indexes/jina-v2-full"),
        help="visual index artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--ocr-index-dir",
        type=Path,
        default=Path("data/interim/ocr/index"),
        help="OCR chunk index artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--task",
        choices=("kis", "qa"),
        default=None,
        help="evaluate one task; default evaluates every task found",
    )
    parser.add_argument(
        "--visual-top-k", type=int, default=200, help="visual candidate pool size (default: %(default)s)"
    )
    parser.add_argument("--ocr-top-k", type=int, default=50, help="OCR chunks per query (default: %(default)s)")
    parser.add_argument("--normalization", choices=("minmax", "rank"), default="minmax")
    parser.add_argument("--temporal-scale", type=float, default=500.0)
    parser.add_argument("--fixed-weight", type=float, default=0.25)
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.5)
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
        "--show", type=int, default=5, help="ranking-change diagnostic examples to print (default: %(default)s)"
    )
    return parser


def fusion_variants(args: argparse.Namespace) -> dict[str, OcrFusionConfig | None]:
    """Return the explicit, reported fusion configurations."""
    shared = dict(
        normalization=args.normalization,
        temporal_scale=args.temporal_scale,
        fixed_weight=args.fixed_weight,
        base_weight=args.base_weight,
        max_weight=args.max_weight,
    )
    return {
        "visual-only": None,
        "fixed-ocr": OcrFusionConfig(use_temporal_alignment=False, weight_mode="fixed", **shared),
        "temporal-ocr": OcrFusionConfig(use_temporal_alignment=True, weight_mode="fixed", **shared),
        "adaptive-temporal-ocr": OcrFusionConfig(use_temporal_alignment=True, weight_mode="adaptive", **shared),
    }


def frame_hit(query: EvaluationQuery, video_id: str, frame_index: int) -> bool:
    """Return whether a frame prediction satisfies the GT region.

    Mirrors :func:`aic2026.kis_scoring.is_kis_correct` frame semantics
    (video match + inclusive original-frame range) for both KIS and QA
    retrieval; no new ground-truth logic is introduced.
    """
    if video_id != query.ground_truth_video_id.value:
        return False
    gt = query.ground_truth_frame_range
    return gt.start <= frame_index <= gt.end


def first_correct_rank(ranked: list[tuple[str, int]], query: EvaluationQuery) -> int | None:
    """Return the 1-based position of the first correct entry, if any."""
    for position, (video_id, frame_index) in enumerate(ranked, start=1):
        if frame_hit(query, video_id, frame_index):
            return position
    return None


def hits_at_k(first_ranks: list[int | None], cutoffs: Sequence[int]) -> list[float]:
    """Return mean hit indicators at every cutoff."""
    total = len(first_ranks)
    return [sum(1 for r in first_ranks if r is not None and r <= k) / total for k in cutoffs]


def ranked_pairs_visual(candidates: list[RetrievalCandidate]) -> list[tuple[str, int]]:
    """Return ``(video_id, frame_index)`` in candidate order."""
    return [(c.video_id, c.frame_index) for c in candidates]


def ranked_pairs_fused(candidates: list[FusedCandidate]) -> list[tuple[str, int]]:
    """Return ``(video_id, frame_index)`` in fused order."""
    return [(c.video_id, c.frame_index) for c in candidates]


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
        retriever = IndexRetriever(args.visual_index_dir)
        ocr_handle = read_ocr_index(args.ocr_index_dir)
        cutoffs = sorted(KIS_TOP_K)
        variants = fusion_variants(args)

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        tasks = (
            [EvaluationTask(args.task)]
            if args.task
            else [EvaluationTask.KIS, EvaluationTask.QA]
        )

        for task in tasks:
            queries = [q for eval_set in eval_sets for q in eval_set.iter_queries(task)]
            if not queries:
                print(f"no {task.value} queries found; skipping")
                continue

            first_ranks: dict[str, list[int | None]] = {name: [] for name in variants}
            changes: list[dict] = []
            for query in queries:
                embedding = encoder.encode_text(query.query_text_vi)
                visual = retriever.search(embedding, top_k=args.visual_top_k)
                ocr_hits: list[OcrSearchResult] = search_ocr_chunks(
                    ocr_handle, query.query_text_vi, encoder=encoder, top_k=args.ocr_top_k
                )

                baseline_pairs = ranked_pairs_visual(visual)
                first_ranks["visual-only"].append(first_correct_rank(baseline_pairs, query))

                for name, fusion_config in variants.items():
                    if fusion_config is None:
                        continue
                    weight = query_ocr_weight(query.query_text_vi, fusion_config)
                    fused = fuse_visual_with_ocr(visual, ocr_hits, config=fusion_config, ocr_weight=weight)
                    pairs = ranked_pairs_fused(fused)
                    first_ranks[name].append(first_correct_rank(pairs, query))
                    if pairs and baseline_pairs and pairs[0] != baseline_pairs[0]:
                        top = fused[0]
                        changes.append(
                            {
                                "query_id": query.query_id,
                                "text": query.query_text_vi,
                                "variant": name,
                                "baseline_top": baseline_pairs[0],
                                "fused_top": pairs[0],
                                "chunk_id": top.matched_chunk_id,
                                "ocr_support": round(top.ocr_support, 4),
                                "weight": round(top.ocr_weight, 4),
                            }
                        )

            print(f"\n=== task={task.value} ({len(queries)} queries) ===")
            print(
                f"pool: visual_top_k={args.visual_top_k} ocr_top_k={args.ocr_top_k} "
                f"normalization={args.normalization} temporal_scale={args.temporal_scale} "
                f"weights: fixed={args.fixed_weight} adaptive=[{args.base_weight}, {args.max_weight}]"
            )
            header = "variant                 " + "".join(f"  Hit@{k:<4}" for k in cutoffs)
            print(header)
            baseline_scores: list[float] | None = None
            for name in variants:
                scores = hits_at_k(first_ranks[name], cutoffs)
                if name == "visual-only":
                    baseline_scores = scores
                cells = "".join(f"  {score:>7.4f}" for score in scores)
                print(f"{name:<24}{cells}")
            if baseline_scores is not None:
                for name in variants:
                    if name == "visual-only":
                        continue
                    scores = hits_at_k(first_ranks[name], cutoffs)
                    cells = "".join(f"  {score - b:>+7.4f}" for score, b in zip(scores, baseline_scores))
                    print(f"{name + ' Δ':<24}{cells}")

            shown = 0
            for change in changes:
                if shown >= args.show:
                    break
                shown += 1
                print(
                    f"\n[changed] {change['query_id']} ({change['variant']}): "
                    f"{change['text'][:80]}\n"
                    f"  baseline top: {change['baseline_top']} -> fused top: {change['fused_top']}\n"
                    f"  ocr evidence: {change['chunk_id']} support={change['ocr_support']} "
                    f"weight={change['weight']}"
                )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
