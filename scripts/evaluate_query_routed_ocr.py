#!/usr/bin/env python3
"""CLI: evaluate query-routed OCR evidence reranking (Phase 5D).

Frozen protocol (declared before any results were seen):

    BASELINE  visual-only exact retrieval (existing IndexRetriever over
              data/interim/indexes/jina-v2-full), query = query_text_vi,
              pool = top-100 frames, ranking returned unchanged.
    P1        routed OCR evidence + lexical gate G1 (>=2-token normalized
              phrase match within one cleaned chunk).
    P2        routed OCR evidence + lexical gate G2 (G1 OR >=2 overlapping
              content tokens incl. one rare token with df <= 10).

Routing: deterministic keyword router (aic2026.ocr_query_routing). VISUAL
queries never touch OCR (output equals baseline). OCR_TEXT/MIXED queries
examine cleaned chunks (representation-v2) of POOL videos only; the best
per-video lexical evidence decides a parameter-free lexicographic rerank:
key = (-evidence_tier, visual_rank). No dense cosine participates in the
rerank decision.

Primary metric: KIS VideoRecall@20. Frame rule: GT video match AND original
frame index inside the inclusive GT range. This is an offline experiment;
nothing here touches production behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.kis_scoring import KIS_TOP_K
from aic2026.ocr_evidence import (
    RoutedRerankConfig,
    best_video_evidence,
    document_frequencies_from_texts,
    routed_candidate_rerank,
)
from aic2026.ocr_evaluation_audit import (
    bootstrap_ci,
    first_rank,
    hits_by_k,
    leave_one_out,
    permutation_test,
    win_tie_loss,
)
from aic2026.ocr_query_routing import route_query
from aic2026.retrieval import IndexRetriever

PRIMARY_K = 20
SECONDARY_FRAME_KS = (20, 100)

#: Frozen configurations.
P1 = RoutedRerankConfig(pool_size=100, gate="G1")
P2 = RoutedRerankConfig(pool_size=100, gate="G2")
VARIANTS = ("BASELINE", "P1", "P2")


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluate_query_routed_ocr",
        description="Evaluate query-routed lexical-OCR candidate reranking under the frozen Phase 5D protocol.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        nargs="+",
        default=[Path("data/eval/eval_set.csv"), Path("data/eval/eval_set-qa_trake.csv")],
    )
    parser.add_argument("--visual-index-dir", type=Path, default=Path("data/interim/indexes/jina-v2-full"))
    parser.add_argument("--v2-dir", type=Path, default=Path("data/interim/ocr/representation-v2"))
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--permutation-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/jina-clip-v2.yaml")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/analysis/query-routed-ocr")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        from aic2026.config import load_config

        config = load_config(args.config)
        if config.jina is None:
            raise ValueError(f"config {args.config} has no models.jina section")

        from aic2026.model_backends.jina import JinaClipEncoder

        encoder = JinaClipEncoder(config.jina)
        retriever = IndexRetriever(args.visual_index_dir)

        manifest = json.loads((args.v2_dir / "manifest.json").read_text(encoding="utf-8"))
        num_chunks = int(manifest["num_chunks"])
        v2_chunks = [
            json.loads(line)
            for line in (args.v2_dir / "cleaned-chunks.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(v2_chunks) != num_chunks:
            raise ValueError("cleaned-chunks rows != manifest num_chunks")

        dfs = document_frequencies_from_texts(c["text"] for c in v2_chunks)
        from types import SimpleNamespace

        chunk_objects = [
            SimpleNamespace(
                chunk_id=c["chunk_id"], video_id=c["video_id"],
                frame_start=int(c["frame_start"]), frame_end=int(c["frame_end"]),
                text=c["text"],
            )
            for c in v2_chunks
        ]
        chunks_by_video: dict[str, list] = {}
        for chunk_obj in chunk_objects:
            chunks_by_video.setdefault(chunk_obj.video_id, []).append(chunk_obj)

        p1 = RoutedRerankConfig(pool_size=args.pool_size, gate="G1")
        p2 = RoutedRerankConfig(pool_size=args.pool_size, gate="G2")
        configs = {"P1": p1, "P2": p2}

        eval_sets = [load_eval_set(path) for path in args.eval_set]
        summary: dict = {"frozen": {
            "pool_size": args.pool_size,
            "gates": {"P1": "G1", "P2": "G2"},
            "rare_max_df": 10,
        }}
        diagnostics_out: dict[str, list[dict]] = {}

        for task in (EvaluationTask.KIS, EvaluationTask.QA):
            task_queries = [q for es in eval_sets for q in es.iter_queries(task)]
            if not task_queries:
                continue
            n = len(task_queries)
            ks = sorted(set(KIS_TOP_K))

            routes = {q.query_id: route_query(q.query_text_vi) for q in task_queries}
            route_counts = {"VISUAL": 0, "OCR_TEXT": 0, "MIXED": 0}
            for r in routes.values():
                route_counts[r.route] += 1

            ranks: dict[str, dict[str, dict[str, int | None]]] = {v: {} for v in VARIANTS}
            supported_counts: dict[str, int] = {v: 0 for v in VARIANTS}
            queries_with_support: dict[str, int] = {v: 0 for v in VARIANTS}
            diagnostics: list[dict] = []

            for query in task_queries:
                gt_video = query.ground_truth_video_id.value
                gt_start = query.ground_truth_frame_range.start
                gt_end = query.ground_truth_frame_range.end
                embedding = encoder.encode_text(query.query_text_vi)
                candidates = retriever.search(embedding, top_k=args.pool_size)

                flat: dict[str, list[tuple[str, int]]] = {
                    "BASELINE": [(c.video_id, c.frame_index) for c in candidates]
                }
                top_candidate = {
                    "BASELINE": flat["BASELINE"][0] if candidates else ("", -1),
                }
                per_query_support: dict[str, bool] = {}

                if routes[query.query_id].route == "VISUAL":
                    # OCR must not influence VISUAL queries at all.
                    for name in configs:
                        flat[name] = flat["BASELINE"]
                        per_query_support[name] = False
                else:
                    pool_videos = list(dict.fromkeys(c.video_id for c in candidates))
                    for name, rc in configs.items():
                        evidence_map = {
                            video: e
                            for video in pool_videos
                            if (e := best_video_evidence(
                                query.query_text_vi,
                                chunks_by_video.get(video, []),
                                dfs,
                                rc.analysis,
                            ))
                            is not None
                        }
                        reranked = routed_candidate_rerank(candidates, config=rc, evidence_by_video=evidence_map)
                        flat[name] = [(c.video_id, c.frame_index) for c in reranked]
                        top_candidate[name] = flat[name][0] if reranked else ("", -1)
                        supported_videos = {c.video_id for c in reranked if c.ocr_supported}
                        per_query_support[name] = bool(supported_videos)
                        supported_counts[name] += len(supported_videos)
                        if per_query_support[name]:
                            queries_with_support[name] += 1

                for name in VARIANTS:
                    pairs = flat[name]
                    ranks[name][query.query_id] = {
                        "video": first_rank(pairs, gt_video_id=gt_video),
                        "frame": first_rank(pairs, gt_video_id=gt_video, gt_start=gt_start, gt_end=gt_end),
                    }

                # Diagnostics for changed primary-metric outcomes.
                base_v20 = ranks["BASELINE"][query.query_id]["video"]
                for name in configs:
                    var_v20 = ranks[name][query.query_id]["video"]
                    base_hit = (base_v20 or 10**9) <= PRIMARY_K
                    var_hit = (var_v20 or 10**9) <= PRIMARY_K
                    if base_hit == var_hit:
                        continue
                    route = routes[query.query_id]
                    diag = {
                        "query_id": query.query_id,
                        "config": name,
                        "query_text_vi": query.query_text_vi,
                        "route": route.to_dict(),
                        "baseline_gt_video_rank": base_v20,
                        "reranked_gt_video_rank": var_v20,
                        "transition": f"{'HIT' if base_hit else 'MISS'}->{'HIT' if var_hit else 'MISS'}",
                        "baseline_top_candidate": top_candidate["BASELINE"],
                        "reranked_top_candidate": top_candidate[name],
                    }
                    gt_ev = best_video_evidence(query.query_text_vi, chunks_by_video.get(gt_video, []), dfs)
                    diag["gt_video_evidence"] = (
                        None if gt_ev is None else {
                            "chunk_id": gt_ev.chunk_id,
                            "tier": gt_ev.tier_name,
                            "gate": gt_ev.gate,
                            "matched_phrase": gt_ev.matched_phrase,
                            "rare_tokens": list(gt_ev.rare_tokens),
                            "min_rare_df": gt_ev.min_rare_df,
                            "frame_range": [gt_ev.frame_start, gt_ev.frame_end],
                            "text": gt_ev.chunk_text[:300],
                        }
                    )
                    diagnostics.append(diag)

            print(f"\n=== {task.value.upper()} (n={n}) ===")
            print(f"route coverage: VISUAL={route_counts['VISUAL']} "
                  f"OCR_TEXT={route_counts['OCR_TEXT']} MIXED={route_counts['MIXED']} "
                  f"(OCR-routed {n - route_counts['VISUAL']}/{n} = "
                  f"{100.0 * (n - route_counts['VISUAL']) / n:.1f}%)")
            header = f"{'config':<10}" + "".join(f"  V@{k:<4}" for k in ks) + "".join(f"  F@{k:<4}" for k in ks)
            print(header)
            task_summary: dict = {
                "num_queries": n,
                "route_coverage": route_counts,
            }
            video_hits: dict[str, dict[str, dict[str, bool]]] = {}
            frame_hits: dict[str, dict[str, dict[str, bool]]] = {}
            for name in VARIANTS:
                vh = {qid: hits_by_k(ranks[name][qid]["video"], ks) for qid in ranks[name]}
                fh = {qid: hits_by_k(ranks[name][qid]["frame"], ks) for qid in ranks[name]}
                vrates = {k: sum(1 for h in vh.values() if h[k]) / n for k in ks}
                frates = {k: sum(1 for h in fh.values() if h[k]) / n for k in ks}
                print(f"{name:<10}" + "".join(f"  {vrates[k]:.3f} " for k in ks)
                      + "".join(f"  {frates[k]:.3f} " for k in ks))
                task_summary[name] = {
                    "video_recall": vrates,
                    "frame_hit": frates,
                    "supported_videos_total": supported_counts[name],
                    "queries_with_ocr_support": queries_with_support[name],
                }
                for k in ks:
                    video_hits.setdefault(k, {})[name] = {qid: h[k] for qid, h in vh.items()}
                    frame_hits.setdefault(k, {})[name] = {qid: h[k] for qid, h in fh.items()}

            for name in ("P1", "P2"):
                deltas = np.array([
                    float(video_hits[PRIMARY_K][name][q.query_id]) - float(video_hits[PRIMARY_K]["BASELINE"][q.query_id])
                    for q in task_queries
                ])
                wtl_v = win_tie_loss([ranks["BASELINE"][q.query_id]["video"] for q in task_queries],
                                     [ranks[name][q.query_id]["video"] for q in task_queries])
                wtl_f = win_tie_loss([ranks["BASELINE"][q.query_id]["frame"] for q in task_queries],
                                     [ranks[name][q.query_id]["frame"] for q in task_queries])
                perm = permutation_test(deltas, seed=args.seed, n_resamples=args.permutation_resamples)
                boot = bootstrap_ci(deltas, seed=args.seed, n_resamples=args.bootstrap_resamples)
                loo = leave_one_out(deltas)
                m2h = sum(1 for q in task_queries if not video_hits[PRIMARY_K]["BASELINE"][q.query_id]
                          and video_hits[PRIMARY_K][name][q.query_id])
                h2m = sum(1 for q in task_queries if video_hits[PRIMARY_K]["BASELINE"][q.query_id]
                          and not video_hits[PRIMARY_K][name][q.query_id])
                secondary: dict[int, dict[str, float]] = {}
                for fk in SECONDARY_FRAME_KS:
                    fdeltas = np.array([
                        float(frame_hits[fk][name][q.query_id]) - float(frame_hits[fk]["BASELINE"][q.query_id])
                        for q in task_queries
                    ])
                    fperm = permutation_test(fdeltas, seed=args.seed, n_resamples=args.permutation_resamples)
                    secondary[fk] = {"mean_delta": fperm["mean_delta"], "p": fperm["p_value"]}
                task_summary[name].update({
                    "primary_delta": {
                        "metric": f"video_recall@{PRIMARY_K}",
                        "mean": perm["mean_delta"], "p": perm["p_value"],
                        "ci95": [boot["ci_low"], boot["ci_high"]],
                        "loo_range": [loo["loo_min"], loo["loo_max"]],
                        "sign_flips": loo["sign_flips"],
                    },
                    "secondary_frame_deltas": secondary,
                    "wtl_first_video_rank": wtl_v,
                    "wtl_first_frame_rank": wtl_f,
                    "transitions": {"miss_to_hit": m2h, "hit_to_miss": h2m},
                })
                print(
                    f"{name}: dVRecall@{PRIMARY_K} mean={perm['mean_delta']:+.4f} p={perm['p_value']:.3f} "
                    f"CI=[{boot['ci_low']:+.4f},{boot['ci_high']:+.4f}] "
                    f"LOO=[{loo['loo_min']:+.4f},{loo['loo_max']:+.4f}] flips={loo['sign_flips']} | "
                    f"W/T/L(v)={wtl_v['win']}/{wtl_v['tie']}/{wtl_v['loss']} "
                    f"(f {wtl_f['win']}/{wtl_f['tie']}/{wtl_f['loss']}) | "
                    f"MISS->HIT={m2h} HIT->MISS={h2m}"
                )

            summary[task.value] = task_summary
            if diagnostics:
                diagnostics_out[task.value] = diagnostics

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if diagnostics_out:
            diag_path = args.output_dir / "diagnostics.json"
            diag_path.write_text(json.dumps(diagnostics_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Diagnostics: {diag_path}")
        print(f"Summary: {summary_path}")
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
