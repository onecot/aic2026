#!/usr/bin/env python3
"""CLI: evaluate OCR representation-v2 against the baseline (Phase 5B).

Frozen protocol (declared before any results were seen):

    BASELINE         existing dense chunk index (data/interim/ocr/index)
    CLEANED_CONCAT   cleaned chunk text, one vector per chunk
    UNIT_MAX         per-unit vectors, chunk score = max unit cosine

Queries: KIS + QA evaluation sets; query text encoded once per query with
the same Jina CLIP text encoder for all variants. Retrieval is exact cosine
over each representation's rows.

Primary metric (pre-declared): GT-video recall@20 on KIS.
Secondary: video recall@{5,10,50,100}, frame hit@{5,20,100} (chunk range
intersects the inclusive GT range - existing semantics), paired win/tie/loss
on first-GT-video-chunk rank, permutation test + bootstrap CI on the
primary delta, leave-one-out sensitivity.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aic2026.config import load_config
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.kis_scoring import KIS_TOP_K
from aic2026.ocr_evaluation_audit import (
    bootstrap_ci,
    leave_one_out,
    permutation_test,
    win_tie_loss,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_ocr_representation",
        description="Compare OCR representation variants under the frozen Phase 5B protocol.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        nargs="+",
        default=[Path("data/eval/eval_set.csv"), Path("data/eval/eval_set-qa_trake.csv")],
    )
    parser.add_argument(
        "--baseline-index-dir",
        type=Path,
        default=Path("data/interim/ocr/index"),
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=Path("data/interim/ocr/representation-v2"),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--permutation-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/jina-clip-v2.yaml")
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=None
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/analysis/representation-v2"),
    )
    return parser


def rank_rows(scores: np.ndarray) -> np.ndarray:
    """Deterministic ranking: score desc, ties by row index."""
    n = scores.shape[0]
    return np.lexsort((np.arange(n, dtype=np.int64), -scores.astype(np.float32)))


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

        # Baseline index rows correspond 1:1 to the chunks-artifact order;
        # representation-v2 preserves that order (validated below).
        baseline_matrix = np.load(args.baseline_index_dir / "embeddings.npy")
        baseline_norms = np.linalg.norm(baseline_matrix, axis=1).astype(np.float32)

        manifest = json.loads((args.v2_dir / "manifest.json").read_text(encoding="utf-8"))
        num_chunks = manifest["num_chunks"]
        if baseline_matrix.shape[0] != num_chunks:
            raise ValueError(
                f"baseline rows ({baseline_matrix.shape[0]}) != v2 chunks ({num_chunks}); "
                "artifacts were built from different chunk sets"
            )
        matrix = np.load(args.v2_dir / "embeddings.npy")
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)

        # Chunk metadata: row i of every variant corresponds to source chunk i.
        v2_chunks = [
            json.loads(line)
            for line in (args.v2_dir / "cleaned-chunks.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunk_video = [c["video_id"] for c in v2_chunks]
        chunk_start = [c["frame_start"] for c in v2_chunks]
        chunk_end = [c["frame_end"] for c in v2_chunks]

        # Map chunk row -> unique-unit embedding rows (offset by num_chunks).
        units = [
            json.loads(line)
            for line in (args.v2_dir / "units.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        chunk_id_to_row = {c["chunk_id"]: i for i, c in enumerate(v2_chunks)}
        unit_chunk_indices: dict[int, list[int]] = {}
        for unit in units:
            chunk_row = chunk_id_to_row[unit["chunk_id"]]
            unit_chunk_indices.setdefault(chunk_row, []).append(
                int(unit["unique_row"]) - num_chunks
            )

        from aic2026.ocr_representation import unit_max_scores

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        queries = [q for eval_set in eval_sets for q in eval_set.queries]

        def search_dense(sub_matrix: np.ndarray, sub_norms: np.ndarray, vector: np.ndarray, top_k: int) -> list[int]:
            q = np.asarray(vector, dtype=np.float32)
            scores = (sub_matrix @ q) / np.maximum(sub_norms, 1e-12)
            return [int(row) for row in rank_rows(scores)[:top_k]]

        def search_unit_max(vector: np.ndarray, top_k: int) -> list[int]:
            scores = unit_max_scores(
                vector, matrix[num_chunks:], norms[num_chunks:], unit_chunk_indices, num_chunks
            )
            return [int(row) for row in rank_rows(scores)[:top_k]]

        summary: dict[str, dict] = {}
        for task in (EvaluationTask.KIS, EvaluationTask.QA):
            task_queries = [q for eval_set in eval_sets for q in eval_set.iter_queries(task)]
            if not task_queries:
                continue
            n = len(task_queries)
            results: dict[str, dict[str, dict[int, bool]]] = {}
            video_results: dict[str, dict[str, dict[int, bool]]] = {}
            first_video_rank: dict[str, dict[str, int | None]] = {}

            for query in task_queries:
                gt_video = query.ground_truth_video_id.value
                gt_start = query.ground_truth_frame_range.start
                gt_end = query.ground_truth_frame_range.end
                embedding = encoder.encode_text(query.query_text_vi)

                def evaluate_hits(rows: Sequence[int]) -> dict[int, bool]:
                    # Frame hit under existing range-intersection semantics.
                    out = {}
                    for k in sorted(set(KIS_TOP_K)):
                        k_eff = min(k, len(rows))
                        hit = False
                        for row in rows[:k_eff]:
                            if chunk_video[row] == gt_video and chunk_start[row] <= gt_end and chunk_end[row] >= gt_start:
                                hit = True
                                break
                        out[k] = hit
                    return out

                def evaluate_video(rows: Sequence[int]) -> dict[int, bool]:
                    # Pure video recall: any GT-video chunk within cutoff.
                    out = {}
                    for k in sorted(set(KIS_TOP_K)):
                        k_eff = min(k, len(rows))
                        out[k] = any(chunk_video[row] == gt_video for row in rows[:k_eff])
                    return out

                def first_video(rows: Sequence[int]) -> int | None:
                    for position, row in enumerate(rows, start=1):
                        if chunk_video[row] == gt_video:
                            return position
                    return None

                # BASELINE: exact cosine over the existing chunk embeddings.
                base_rows = search_dense(baseline_matrix, baseline_norms, embedding.vector, args.top_k)
                clean_rows = search_dense(matrix[:num_chunks], norms[:num_chunks], embedding.vector, args.top_k)
                unit_rows_list = search_unit_max(embedding.vector, args.top_k)

                results.setdefault("BASELINE", {})[query.query_id] = evaluate_hits(base_rows)
                results.setdefault("CLEANED_CONCAT", {})[query.query_id] = evaluate_hits(clean_rows)
                results.setdefault("UNIT_MAX", {})[query.query_id] = evaluate_hits(unit_rows_list)
                video_results.setdefault("BASELINE", {})[query.query_id] = evaluate_video(base_rows)
                video_results.setdefault("CLEANED_CONCAT", {})[query.query_id] = evaluate_video(clean_rows)
                video_results.setdefault("UNIT_MAX", {})[query.query_id] = evaluate_video(unit_rows_list)
                first_video_rank.setdefault("BASELINE", {})[query.query_id] = first_video(base_rows)
                first_video_rank.setdefault("CLEANED_CONCAT", {})[query.query_id] = first_video(clean_rows)
                first_video_rank.setdefault("UNIT_MAX", {})[query.query_id] = first_video(unit_rows_list)

            print(f"\n=== {task.value.upper()} (n={n}) ===")
            ks = sorted(KIS_TOP_K)
            print(f"{'variant':<16}" + "".join(f"  V@{k:<4}" for k in ks) + "".join(f"  F@{k:<4}" for k in ks))
            task_summary: dict[str, dict] = {"num_queries": n}
            for variant in ("BASELINE", "CLEANED_CONCAT", "UNIT_MAX"):
                video_rates = {
                    k: sum(1 for h in video_results[variant].values() if h[k]) / n for k in ks
                }
                frame_rates = {
                    k: sum(1 for h in results[variant].values() if h[k]) / n for k in ks
                }
                print(
                    f"{variant:<16}"
                    + "".join(f"  {video_rates[k]:.3f} " for k in ks)
                    + "".join(f"  {frame_rates[k]:.3f} " for k in ks)
                )
                task_summary[variant] = {"video_recall": video_rates, "frame_hit": frame_rates}

            for variant in ("CLEANED_CONCAT", "UNIT_MAX"):
                deltas = np.array(
                    [
                        float(video_results[variant][q.query_id][20]) - float(video_results["BASELINE"][q.query_id][20])
                        for q in task_queries
                    ]
                )
                wtl = win_tie_loss(
                    [first_video_rank["BASELINE"][q.query_id] for q in task_queries],
                    [first_video_rank[variant][q.query_id] for q in task_queries],
                )
                perm = permutation_test(deltas, seed=args.seed, n_resamples=args.permutation_resamples)
                boot = bootstrap_ci(deltas, seed=args.seed, n_resamples=args.bootstrap_resamples)
                loo = leave_one_out(deltas)
                task_summary[variant].update(
                    {
                        "wtl_first_video_rank": wtl,
                        "video_recall@20_delta": {
                            "mean": perm["mean_delta"],
                            "p": perm["p_value"],
                            "ci95": [boot["ci_low"], boot["ci_high"]],
                            "loo_range": [loo["loo_min"], loo["loo_max"]],
                            "sign_flips": loo["sign_flips"],
                        },
                    }
                )
                print(
                    f"{variant}: dVideoRecall@20 mean={perm['mean_delta']:+.4f} "
                    f"p={perm['p_value']:.3f} CI=[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}] "
                    f"W/T/L(video-rank)={wtl['win']}/{wtl['tie']}/{wtl['loss']} "
                    f"LOO=[{loo['loo_min']:+.4f},{loo['loo_max']:+.4f}] flips={loo['sign_flips']}"
                )
            summary[task.value] = task_summary

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "summary.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSummary: {out_path}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
