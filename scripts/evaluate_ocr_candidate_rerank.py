#!/usr/bin/env python3
"""CLI: evaluate bounded OCR candidate reranking (Phase 5C-A).

Frozen protocol (declared before any results were seen):

    BASELINE  visual-only exact retrieval (existing IndexRetriever over
              data/interim/indexes/jina-v2-full), query = query_text_vi.
    C1        strict frame-pool rerank, pool = top-100 visual frames.
    C2        video-aware rerank, videos derived from top-100 visual frames.
    C3        video-aware rerank, videos derived from top-200 visual frames.

One conservative OCR weight lambda = 0.3; per-query min-max normalization
(aic2026.ocr_fusion.normalize_scores_minmax); OCR evidence = best
CLEANED_CONCAT chunk cosine per candidate VIDEO ONLY (representation-v2);
no temporal decay; ties break by best visual rank; frame order inside a
reranked video stays in the original visual order.

Primary metric: KIS GT Video Recall@20. Secondary: KIS Frame Hit@100.
Frame rule: GT-video match AND original frame index inside the inclusive GT
range (existing repository semantics). Video rule: any GT-video candidate.

This is a thin CLI; all reranking logic lives in aic2026.ocr_candidate_rerank.
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
from aic2026.ocr_candidate_rerank import (
    CandidateRerankConfig,
    VideoOcrEvidence,
    rerank_candidates,
)
from aic2026.ocr_evaluation_audit import (
    bootstrap_ci,
    first_rank,
    hits_by_k,
    leave_one_out,
    permutation_test,
    transition_matrix,
    win_tie_loss,
)
from aic2026.retrieval import IndexRetriever

#: Frozen experiment configurations (name -> config). Declared up front.
FROZEN_CONFIGS: dict[str, CandidateRerankConfig] = {
    "C1": CandidateRerankConfig(policy="strict_frame", pool_size=100, ocr_weight=0.3),
    "C2": CandidateRerankConfig(policy="video_aware", pool_size=100, ocr_weight=0.3),
    "C3": CandidateRerankConfig(policy="video_aware", pool_size=200, ocr_weight=0.3),
}
BASELINE_NAME = "BASELINE"
PRIMARY_K = 20
SECONDARY_FRAME_K = 100


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_ocr_candidate_rerank",
        description="Evaluate bounded OCR candidate reranking under the frozen Phase 5C-A protocol.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        nargs="+",
        default=[Path("data/eval/eval_set.csv"), Path("data/eval/eval_set-qa_trake.csv")],
    )
    parser.add_argument(
        "--visual-index-dir",
        type=Path,
        default=Path("data/interim/indexes/jina-v2-full"),
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=Path("data/interim/ocr/representation-v2"),
    )
    parser.add_argument(
        "--chunks-artifact",
        type=Path,
        default=Path("data/interim/ocr/chunks"),
        help="original chunks artifact used for stratum assignment only",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--permutation-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--retrieval-top-k", type=int, default=200)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/jina-clip-v2.yaml")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/analysis/ocr-candidate-rerank"),
    )
    return parser


def best_video_evidence(
    query_vector: np.ndarray,
    pool_videos: Sequence[str],
    rows_by_video: dict[str, np.ndarray],
    chunk_ids_by_video: dict[str, tuple[str, ...]],
    matrix: np.ndarray,
    norms: np.ndarray,
) -> dict[str, VideoOcrEvidence]:
    """Return best CLEANED_CONCAT evidence for each pool video that has chunks.

    Expected input:
        - query_vector: encoded query text vector.
        - pool_videos: distinct candidate videos of one query's pool.
        - rows_by_video: video id -> array of representation-v2 row indices.
        - chunk_ids_by_video: video id -> chunk ids aligned with the rows.
        - matrix / norms: representation-v2 CLEANED_CONCAT embeddings.

    Returns:
        - Mapping video -> :class:`VideoOcrEvidence` (best raw cosine).
          Videos without chunks are absent.

    Raises:
        - No exceptions beyond numpy errors on malformed arrays.
    """
    evidence: dict[str, VideoOcrEvidence] = {}
    q = np.asarray(query_vector, dtype=np.float32)
    for video in pool_videos:
        rows = rows_by_video.get(video)
        if rows is None or rows.size == 0:
            continue
        sub = matrix[rows]
        scores = (sub @ q) / np.maximum(norms[rows], 1e-12)
        # Deterministic tie handling: argmax returns the first (lowest row).
        best_position = int(np.argmax(scores))
        evidence[video] = VideoOcrEvidence(
            video_id=video,
            chunk_id=chunk_ids_by_video[video][best_position],
            raw_score=float(scores[best_position]),
        )
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if config.jina is None:
            raise ValueError(f"config {args.config} has no models.jina section")

        from aic2026.model_backends.jina import JinaClipEncoder

        jina_config = (
            dataclasses.replace(config.jina, device=args.device) if args.device else config.jina
        )
        encoder = JinaClipEncoder(jina_config)

        retriever = IndexRetriever(args.visual_index_dir)

        manifest = json.loads((args.v2_dir / "manifest.json").read_text(encoding="utf-8"))
        num_chunks = int(manifest["num_chunks"])
        matrix = np.load(args.v2_dir / "embeddings.npy").astype(np.float32)[:num_chunks]
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
        v2_chunks = [
            json.loads(line)
            for line in (args.v2_dir / "cleaned-chunks.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(v2_chunks) != num_chunks:
            raise ValueError("representation-v2 cleaned-chunks rows != manifest num_chunks")
        chunk_meta = {index: chunk for index, chunk in enumerate(v2_chunks)}
        rows_lists: dict[str, list[int]] = {}
        for row_index, chunk in enumerate(v2_chunks):
            rows_lists.setdefault(chunk["video_id"], []).append(row_index)
        rows_by_video = {video: np.array(rows, dtype=np.int64) for video, rows in rows_lists.items()}
        chunk_ids_by_video = {
            video: tuple(v2_chunks[row]["chunk_id"] for row in rows)
            for video, rows in rows_lists.items()
        }

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        variant_names = [BASELINE_NAME, *FROZEN_CONFIGS]

        summary: dict[str, dict] = {"frozen_configs": {
            name: dataclasses.asdict(cfg) for name, cfg in FROZEN_CONFIGS.items()
        }}
        diagnostics_out: dict[str, list[dict]] = {}

        for task in (EvaluationTask.KIS, EvaluationTask.QA):
            task_queries = [q for es in eval_sets for q in es.iter_queries(task)]
            if not task_queries:
                continue
            n = len(task_queries)
            ks = sorted(set(KIS_TOP_K))

            ranks: dict[str, dict[str, dict[str, int | None]]] = {
                name: {} for name in variant_names
            }
            diagnostics: list[dict] = []

            for query in task_queries:
                gt_video = query.ground_truth_video_id.value
                gt_start = query.ground_truth_frame_range.start
                gt_end = query.ground_truth_frame_range.end
                embedding = encoder.encode_text(query.query_text_vi)

                candidates = retriever.search(embedding, top_k=args.retrieval_top_k)

                flat_lists: dict[str, list[tuple[str, int]]] = {
                    BASELINE_NAME: [(c.video_id, c.frame_index) for c in candidates]
                }
                top_candidates: dict[str, tuple[str, int]] = {
                    BASELINE_NAME: flat_lists[BASELINE_NAME][0] if candidates else ("", -1)
                }
                evidence_cache: dict[str, dict[str, VideoOcrEvidence]] = {}

                for name, cfg in FROZEN_CONFIGS.items():
                    pool = candidates[: cfg.pool_size]
                    pool_videos = list(dict.fromkeys(c.video_id for c in pool))
                    evidence = best_video_evidence(
                        embedding.vector,
                        pool_videos,
                        rows_by_video,
                        chunk_ids_by_video,
                        matrix,
                        norms,
                    )
                    evidence_cache[name] = evidence
                    reranked = rerank_candidates(pool, config=cfg, evidence_by_video=evidence)
                    flat_lists[name] = [(c.video_id, c.frame_index) for c in reranked]
                    top_candidates[name] = flat_lists[name][0] if reranked else ("", -1)

                for name in variant_names:
                    pairs = flat_lists[name]
                    ranks[name][query.query_id] = {
                        "frame": first_rank(pairs, gt_video_id=gt_video, gt_start=gt_start, gt_end=gt_end),
                        "video": first_rank(pairs, gt_video_id=gt_video),
                    }

                # Diagnostics for changed primary-metric outcomes (KIS focus).
                base_video_hit20 = (ranks[BASELINE_NAME][query.query_id]["video"] or 10**9) <= PRIMARY_K
                for name in FROZEN_CONFIGS:
                    var_rank = ranks[name][query.query_id]["video"]
                    var_video_hit20 = (var_rank or 10**9) <= PRIMARY_K
                    if base_video_hit20 == var_video_hit20:
                        continue
                    responsible = evidence_cache[name].get(gt_video)
                    diag = {
                        "query_id": query.query_id,
                        "config": name,
                        "query_text_vi": query.query_text_vi,
                        "baseline_gt_video_rank": ranks[BASELINE_NAME][query.query_id]["video"],
                        "reranked_gt_video_rank": var_rank,
                        "transition": f"{'HIT' if base_video_hit20 else 'MISS'}->"
                        f"{'HIT' if var_video_hit20 else 'MISS'}",
                        "baseline_top_candidate": top_candidates[BASELINE_NAME],
                        "reranked_top_candidate": top_candidates[name],
                        "responsible_ocr_chunk_id": (
                            None if responsible is None else responsible.chunk_id
                        ),
                        "responsible_ocr_score": (
                            None if responsible is None else float(responsible.raw_score)
                        ),
                        "cleaned_ocr_text": (
                            None
                            if responsible is None
                            else chunk_meta[
                                next(
                                    i
                                    for i, c in enumerate(v2_chunks)
                                    if c["chunk_id"] == responsible.chunk_id
                                )
                            ]["text"][:400]
                        ),
                    }
                    diagnostics.append(diag)

            print(f"\n=== {task.value.upper()} (n={n}) ===")
            header = f"{'config':<10}" + "".join(f"  V@{k:<4}" for k in ks) + "".join(f"  F@{k:<4}" for k in ks)
            print(header)
            task_summary: dict[str, dict] = {"num_queries": n}
            video_hits: dict[str, dict[str, bool]] = {}
            frame_hits: dict[str, dict[str, bool]] = {}
            for name in variant_names:
                video_hit_rows = {
                    qid: hits_by_k(ranks[name][qid]["video"], ks) for qid in ranks[name]
                }
                frame_hit_rows = {
                    qid: hits_by_k(ranks[name][qid]["frame"], ks) for qid in ranks[name]
                }
                video_rates = {
                    k: sum(1 for h in video_hit_rows.values() if h[k]) / n for k in ks
                }
                frame_rates = {
                    k: sum(1 for h in frame_hit_rows.values() if h[k]) / n for k in ks
                }
                print(
                    f"{name:<10}"
                    + "".join(f"  {video_rates[k]:.3f} " for k in ks)
                    + "".join(f"  {frame_rates[k]:.3f} " for k in ks)
                )
                task_summary[name] = {"video_recall": video_rates, "frame_hit": frame_rates}
                for k in ks:
                    video_hits.setdefault(k, {})[name] = {
                        qid: h[k] for qid, h in video_hit_rows.items()
                    }
                    frame_hits.setdefault(k, {})[name] = {
                        qid: h[k] for qid, h in frame_hit_rows.items()
                    }

            for name in FROZEN_CONFIGS:
                deltas = np.array(
                    [
                        float(video_hits[PRIMARY_K][name][q.query_id])
                        - float(video_hits[PRIMARY_K][BASELINE_NAME][q.query_id])
                        for q in task_queries
                    ]
                )
                wtl_video = win_tie_loss(
                    [ranks[BASELINE_NAME][q.query_id]["video"] for q in task_queries],
                    [ranks[name][q.query_id]["video"] for q in task_queries],
                )
                wtl_frame = win_tie_loss(
                    [ranks[BASELINE_NAME][q.query_id]["frame"] for q in task_queries],
                    [ranks[name][q.query_id]["frame"] for q in task_queries],
                )
                perm = permutation_test(deltas, seed=args.seed, n_resamples=args.permutation_resamples)
                boot = bootstrap_ci(deltas, seed=args.seed, n_resamples=args.bootstrap_resamples)
                loo = leave_one_out(deltas)

                miss_to_hit_v = sum(
                    1
                    for q in task_queries
                    if not video_hits[PRIMARY_K][BASELINE_NAME][q.query_id]
                    and video_hits[PRIMARY_K][name][q.query_id]
                )
                hit_to_miss_v = sum(
                    1
                    for q in task_queries
                    if video_hits[PRIMARY_K][BASELINE_NAME][q.query_id]
                    and not video_hits[PRIMARY_K][name][q.query_id]
                )
                miss_to_hit_f = sum(
                    1
                    for q in task_queries
                    if not frame_hits[SECONDARY_FRAME_K][BASELINE_NAME][q.query_id]
                    and frame_hits[SECONDARY_FRAME_K][name][q.query_id]
                )
                hit_to_miss_f = sum(
                    1
                    for q in task_queries
                    if frame_hits[SECONDARY_FRAME_K][BASELINE_NAME][q.query_id]
                    and not frame_hits[SECONDARY_FRAME_K][name][q.query_id]
                )

                trans = transition_matrix(
                    [video_hits[PRIMARY_K][BASELINE_NAME][q.query_id] for q in task_queries],
                    [frame_hits[SECONDARY_FRAME_K][BASELINE_NAME][q.query_id] for q in task_queries],
                    [video_hits[PRIMARY_K][name][q.query_id] for q in task_queries],
                    [frame_hits[SECONDARY_FRAME_K][name][q.query_id] for q in task_queries],
                )

                secondary_deltas = np.array(
                    [
                        float(frame_hits[SECONDARY_FRAME_K][name][q.query_id])
                        - float(frame_hits[SECONDARY_FRAME_K][BASELINE_NAME][q.query_id])
                        for q in task_queries
                    ]
                )
                secondary_perm = permutation_test(
                    secondary_deltas, seed=args.seed, n_resamples=args.permutation_resamples
                )

                task_summary[name].update(
                    {
                        "primary_delta": {
                            "metric": f"video_recall@{PRIMARY_K}",
                            "mean": perm["mean_delta"],
                            "p": perm["p_value"],
                            "ci95": [boot["ci_low"], boot["ci_high"]],
                            "loo_range": [loo["loo_min"], loo["loo_max"]],
                            "sign_flips": loo["sign_flips"],
                        },
                        "secondary_delta": {
                            "metric": f"frame_hit@{SECONDARY_FRAME_K}",
                            "mean": secondary_perm["mean_delta"],
                            "p": secondary_perm["p_value"],
                        },
                        "wtl_first_video_rank": wtl_video,
                        "wtl_first_frame_rank": wtl_frame,
                        "transitions": {
                            f"video@{PRIMARY_K}": {
                                "miss_to_hit": miss_to_hit_v,
                                "hit_to_miss": hit_to_miss_v,
                            },
                            f"frame@{SECONDARY_FRAME_K}": {
                                "miss_to_hit": miss_to_hit_f,
                                "hit_to_miss": hit_to_miss_f,
                            },
                        },
                        "joint_transition_matrix": trans,
                    }
                )
                print(
                    f"{name}: dVRecall@{PRIMARY_K} mean={perm['mean_delta']:+.4f} p={perm['p_value']:.3f} "
                    f"CI=[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}] "
                    f"LOO=[{loo['loo_min']:+.4f},{loo['loo_max']:+.4f}] flips={loo['sign_flips']} | "
                    f"W/T/L(v-rank)={wtl_video['win']}/{wtl_video['tie']}/{wtl_video['loss']} "
                    f"(f-rank {wtl_frame['win']}/{wtl_frame['tie']}/{wtl_frame['loss']}) | "
                    f"video MISS->HIT={miss_to_hit_v} HIT->MISS={hit_to_miss_v}; "
                    f"frame MISS->HIT={miss_to_hit_f} HIT->MISS={hit_to_miss_f}"
                )

            summary[task.value] = task_summary
            if diagnostics:
                diagnostics_out[task.value] = diagnostics

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if diagnostics_out:
            diag_path = args.output_dir / "diagnostics.json"
            diag_path.write_text(
                json.dumps(diagnostics_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"Diagnostics: {diag_path}")
        print(f"Summary: {summary_path}")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
