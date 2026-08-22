#!/usr/bin/env python3
"""CLI: evaluation reliability + OCR signal audit (Phase 5A).

Reproduces the visual baseline and the Phase 3A OCR fusion variants
deterministically, collects per-query paired results, and reports:

- paired win/tie/loss, permutation test, bootstrap CI, leave-one-out;
- KIS video-recall vs frame-hit decomposition and transition matrix;
- data-driven OCR-evidence strata and per-stratum deltas;
- OCR chunk artifact representation statistics;
- compact informative failure cases.

Writes ``per_query.jsonl``, ``summary.json``, and ``report.md``.

Diagnostic only: no retrieval changes, no new features.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from aic2026.config import load_config
from aic2026.evaluation import EvaluationQuery, EvaluationTask, load_eval_set
from aic2026.kis_scoring import KIS_TOP_K
from aic2026.ocr_chunks import read_ocr_chunks
from aic2026.ocr_evaluation_audit import (
    AUDIT_K,
    assign_stratum,
    bootstrap_ci,
    chunk_artifact_stats,
    first_rank,
    hit_deltas,
    hits_by_k,
    leave_one_out,
    permutation_test,
    stratum_deltas,
    transition_matrix,
    win_tie_loss,
)
from aic2026.ocr_failure_analysis import AnalysisConfig, build_document_frequencies
from aic2026.ocr_fusion import OcrFusionConfig, fuse_visual_with_ocr, query_ocr_weight
from aic2026.ocr_index import read_ocr_index, search_ocr_chunks
from aic2026.retrieval import IndexRetriever


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="audit_ocr_evaluation",
        description="Audit the reliability of OCR fusion conclusions.",
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
        "--ocr-chunks",
        type=Path,
        default=Path("data/interim/ocr/chunks"),
        help="OCR chunk artifact directory (default: %(default)s)",
    )
    parser.add_argument("--visual-top-k", type=int, default=200, help="visual pool size")
    parser.add_argument("--ocr-top-k", type=int, default=50, help="OCR chunks per query")
    parser.add_argument("--seed", type=int, default=20260821, help="deterministic RNG seed")
    parser.add_argument(
        "--permutation-resamples", type=int, default=10_000, help="sign-flip resamples"
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=10_000, help="bootstrap resamples"
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
        default=Path("data/interim/analysis/ocr-evaluation-audit"),
        help="output directory (default: %(default)s)",
    )
    return parser


def fusion_variants() -> dict[str, OcrFusionConfig | None]:
    """Return the exact Phase 3A variant configurations."""
    shared = dict(
        normalization="minmax",
        temporal_scale=500.0,
        fixed_weight=0.25,
        base_weight=0.05,
        max_weight=0.5,
    )
    return {
        "visual-only": None,
        "fixed-ocr": OcrFusionConfig(use_temporal_alignment=False, weight_mode="fixed", **shared),
        "temporal-ocr": OcrFusionConfig(use_temporal_alignment=True, weight_mode="fixed", **shared),
        "adaptive-temporal-ocr": OcrFusionConfig(use_temporal_alignment=True, weight_mode="adaptive", **shared),
    }


def ranked_pairs_visual(candidates) -> list[tuple[str, int]]:
    """Return ``(video_id, frame_index)`` pairs in candidate order."""
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
        _, chunks = read_ocr_chunks(args.ocr_chunks)
        chunks_by_video: dict[str, list] = {}
        for chunk in chunks:
            chunks_by_video.setdefault(chunk.video_id, []).append(chunk)
        dfs = build_document_frequencies(chunks)

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        merged_queries: list[EvaluationQuery] = [
            q for eval_set in eval_sets for q in eval_set.queries
        ]
        variants = fusion_variants()

        # Per-query per-variant observations.
        records: list[dict] = []
        ranks: dict[str, dict[str, dict[str, int | None]]] = {}
        strata_by_query: dict[str, str] = {}
        analysis_config = AnalysisConfig()

        for query in merged_queries:
            gt_video = query.ground_truth_video_id.value
            gt_range = query.ground_truth_frame_range
            embedding = encoder.encode_text(query.query_text_vi)
            visual_candidates = retriever.search(embedding, top_k=args.visual_top_k)
            visual = ranked_pairs_visual(visual_candidates)
            ocr_hits = search_ocr_chunks(
                ocr_handle, query.query_text_vi, encoder=encoder, top_k=args.ocr_top_k
            )

            strata_by_query[query.query_id] = assign_stratum(
                query.query_text_vi, chunks_by_video.get(gt_video, []), dfs, analysis_config
            )

            for name, fusion_config in variants.items():
                if fusion_config is None:
                    pairs = visual
                else:
                    weight = query_ocr_weight(query.query_text_vi, fusion_config)
                    fused = fuse_visual_with_ocr(
                        visual_candidates, ocr_hits, config=fusion_config, ocr_weight=weight
                    )
                    pairs = [(c.video_id, c.frame_index) for c in fused]
                frame_rank = first_rank(pairs, gt_video_id=gt_video, gt_start=gt_range.start, gt_end=gt_range.end)
                video_rank = first_rank(pairs, gt_video_id=gt_video)
                ranks.setdefault(name, {})[query.query_id] = {
                    "frame": frame_rank,
                    "video": video_rank,
                }
                records.append(
                    {
                        "query_id": query.query_id,
                        "task": query.task.value,
                        "query_text": query.query_text_vi,
                        "variant": name,
                        "stratum": strata_by_query[query.query_id],
                        "first_frame_rank": frame_rank,
                        "first_video_rank": video_rank,
                        "frame_hit": hits_by_k(frame_rank),
                        "video_hit": hits_by_k(video_rank),
                    }
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        per_query_path = args.output_dir / "per_query.jsonl"
        with per_query_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: dict[str, dict] = {}
        report_lines: list[str] = ["# OCR Evaluation Audit (Phase 5A)", ""]

        tasks_present = sorted({q.task.value for q in merged_queries})
        for task_name in tasks_present:
            task = EvaluationTask(task_name)
            queries = [q for q in merged_queries if q.task is task]
            n = len(queries)
            low_power = n < 30
            task_summary: dict[str, dict] = {"num_queries": n, "low_sample_size": low_power}
            report_lines.append(f"## {task_name.upper()} (n={n})" + ("  ⚠ low sample size" if low_power else ""))
            report_lines.append("")

            for name, fusion_config in variants.items():
                if fusion_config is None:
                    continue
                base_frame_ranks = [ranks["visual-only"][q.query_id]["frame"] for q in queries]
                var_frame_ranks = [ranks[name][q.query_id]["frame"] for q in queries]
                base_video_ranks = [ranks["visual-only"][q.query_id]["video"] for q in queries]
                var_video_ranks = [ranks[name][q.query_id]["video"] for q in queries]

                wtl_frame = win_tie_loss(base_frame_ranks, var_frame_ranks)
                wtl_video = win_tie_loss(base_video_ranks, var_video_ranks)

                deltas_100 = hit_deltas(
                    [hits_by_k(r)[100] for r in base_frame_ranks],
                    [hits_by_k(r)[100] for r in var_frame_ranks],
                )
                perm = permutation_test(deltas_100, seed=args.seed, n_resamples=args.permutation_resamples)
                boot = bootstrap_ci(deltas_100, seed=args.seed, n_resamples=args.bootstrap_resamples)
                loo = leave_one_out(deltas_100)

                video_deltas_100 = hit_deltas(
                    [hits_by_k(r)[100] for r in base_video_ranks],
                    [hits_by_k(r)[100] for r in var_video_ranks],
                )

                decomposition: dict[str, dict] = {}
                transitions: dict[str, dict] = {}
                for k in AUDIT_K:
                    b_v = [hits_by_k(r)[k] for r in base_video_ranks]
                    v_v = [hits_by_k(r)[k] for r in var_video_ranks]
                    b_f = [hits_by_k(r)[k] for r in base_frame_ranks]
                    v_f = [hits_by_k(r)[k] for r in var_frame_ranks]
                    decomposition[f"k{k}"] = {
                        "baseline_video_recall": sum(b_v) / n,
                        "variant_video_recall": sum(v_v) / n,
                        "baseline_frame_hit": sum(b_f) / n,
                        "variant_frame_hit": sum(v_f) / n,
                    }
                    if k == 100:
                        transitions["k100"] = transition_matrix(b_v, b_f, v_v, v_f)

                strata_map = {q.query_id: strata_by_query[q.query_id] for q in queries}
                strata_report = {
                    "frame_hit@100": stratum_deltas(
                        [strata_map[q.query_id] for q in queries], deltas_100
                    ),
                    "video_hit@100": stratum_deltas(
                        [strata_map[q.query_id] for q in queries], video_deltas_100
                    ),
                }

                task_summary[name] = {
                    "win_tie_loss_frame_rank": wtl_frame,
                    "win_tie_loss_video_rank": wtl_video,
                    "frame_hit@100_delta": {
                        "mean": perm["mean_delta"],
                        "permutation_p": perm["p_value"],
                        "bootstrap_ci": [boot["ci_low"], boot["ci_high"]],
                        "leave_one_out": loo,
                    },
                    "video_hit@100_delta_mean": float(video_deltas_100.mean()),
                    "decomposition": decomposition,
                    "transitions_k100": transitions.get("k100", {}),
                    "strata_deltas": strata_report,
                }

                report_lines.append(f"### {name}")
                report_lines.append(
                    f"- frame-rank W/T/L: {wtl_frame['win']}/{wtl_frame['tie']}/{wtl_frame['loss']} "
                    f"(video-rank W/T/L: {wtl_video['win']}/{wtl_video['tie']}/{wtl_video['loss']})"
                )
                report_lines.append(
                    f"- ΔFrameHit@100 mean={perm['mean_delta']:+.4f}, "
                    f"perm p={perm['p_value']:.3f}, "
                    f"bootstrap95 CI=[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], "
                    f"LOO range=[{loo['loo_min']:+.4f}, {loo['loo_max']:+.4f}] "
                    f"sign_flips={loo['sign_flips']}"
                )
                report_lines.append(
                    f"- ΔVideoHit@100 mean={float(video_deltas_100.mean()):+.4f}"
                )
                report_lines.append("")

            summary[task_name] = task_summary

        artifact_stats = chunk_artifact_stats(chunks)
        summary["chunk_artifact"] = artifact_stats
        report_lines.append("## OCR chunk artifact statistics")
        for key, value in artifact_stats.items():
            report_lines.append(f"- {key}: {value}")

        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_path = args.output_dir / "report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"Wrote {per_query_path}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {report_path}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
