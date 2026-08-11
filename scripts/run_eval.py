"""Run KIS retrieval over an evaluation JSONL set and report local score metrics.

Usage example:
    python -m scripts.run_eval --eval data/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from eval.score import score_query
from src.data.loader import load_index
from src.model.text_encoder import TextEncoder
from src.retrieve import search
from src.tasks.kis import format_kis
from src.utils.config import load_config


_CUTOFFS = (1, 5, 20, 50, 100)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a KIS evaluation run.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed config, eval, text-field, retrieval, device, and output options.
    Raises: SystemExit if the supplied CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Run KIS retrieval over an evaluation set.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation JSONL path.")
    parser.add_argument(
        "--text-field",
        type=str,
        default="query_text_en",
        help="Evaluation record field used as the encoded query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional retrieval depth overriding retrieve.top_k in the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional model device overriding model.device in the config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional per-query log path with a .json or .csv suffix.",
    )
    return parser.parse_args(arguments)


def load_kis_records(
    eval_path: Path, text_field: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load and validate KIS records while counting skipped task types.

    Expected input: ``eval_path`` is a UTF-8 JSONL eval set from ``parse_eval`` and
    ``text_field`` names a non-empty string field in every KIS record.
    Returns: ``(kis_records, skipped_counts)`` in source order; each KIS record has
        query ID, selected text, and scorer-style ground truth.
    Raises: ValueError for malformed JSONL records, duplicate KIS IDs, or invalid KIS fields;
        json.JSONDecodeError if a non-empty line is invalid JSON.
    """
    kis_records: list[dict[str, Any]] = []
    skipped_counts = {"qa": 0, "trake": 0}
    seen_query_ids: set[str] = set()

    with eval_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Eval line {line_number} must contain a JSON object.")

            task = record.get("task")
            if task in skipped_counts:
                skipped_counts[task] += 1
                continue
            if task != "kis":
                raise ValueError(f"Eval line {line_number} has unsupported task {task!r}.")

            query_id = record.get("query_id")
            text = record.get(text_field)
            video_id = record.get("gt_video_id")
            frame_range = record.get("gt_frame_range")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"Eval line {line_number} has an invalid query_id.")
            if query_id in seen_query_ids:
                raise ValueError(f"Duplicate KIS query_id in eval set: {query_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Eval line {line_number} needs a non-empty string {text_field!r}."
                )
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError(f"Eval line {line_number} has an invalid gt_video_id.")
            if not isinstance(frame_range, list) or len(frame_range) != 2:
                raise ValueError(
                    f"Eval line {line_number} needs gt_frame_range as [start, end]."
                )
            try:
                start, end = (int(value) for value in frame_range)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Eval line {line_number} has non-integer gt_frame_range values."
                ) from error
            if start < 0 or end < 0 or start > end:
                raise ValueError(f"Eval line {line_number} has an invalid gt_frame_range.")

            seen_query_ids.add(query_id)
            kis_records.append(
                {
                    "query_id": query_id,
                    "text": text,
                    "gt": {"video_id": video_id, "s": start, "e": end},
                }
            )

    return kis_records, skipped_counts


def first_hit_rank(answers: Sequence[dict[str, Any]], gt: dict[str, Any]) -> int | None:
    """Find the one-based rank of the first answer inside a KIS ground-truth range.

    Expected input: ``answers`` is a rank-ordered KIS answer list and ``gt`` has string
    ``video_id`` plus inclusive integer ``s`` and ``e`` frame bounds.
    Returns: The first matching one-based rank, or ``None`` when there is no match.
    Raises: KeyError if required answer or ground-truth fields are absent.
    """
    target_video = str(gt["video_id"])
    start_frame = int(gt["s"])
    end_frame = int(gt["e"])
    for rank, answer in enumerate(answers, start=1):
        if (
            str(answer["video_id"]) == target_video
            and start_frame <= int(answer["frame_idx"]) <= end_frame
        ):
            return rank
    return None


def evaluate_kis(
    records: Sequence[dict[str, Any]],
    results_per_query: list[list[dict[str, str | int | float]]],
    top_k: int,
) -> tuple[list[dict[str, int | float | None | str]], dict[int, float], float]:
    """Score KIS retrieval results and aggregate Recall-at-k metrics.

    Expected input: ``records`` has KIS query IDs and scorer-style ground truth, while
    ``results_per_query`` is the matching output of ``search`` and ``top_k`` is non-negative.
    Returns: ``(per_query, recall_means, final_score)`` in source order.
    Raises: ValueError if record/result counts differ or ``top_k`` is negative; KeyError if
        required record fields are absent.
    """
    if len(records) != len(results_per_query):
        raise ValueError("KIS records and retrieval results must have the same length.")

    query_ids = [str(record["query_id"]) for record in records]
    predictions = format_kis(results_per_query, query_ids, max_results=top_k)
    per_query: list[dict[str, int | float | None | str]] = []
    recall_totals = {cutoff: 0 for cutoff in _CUTOFFS}

    for record, prediction in zip(records, predictions, strict=True):
        gt = record["gt"]
        answers = prediction["answers"]
        rank = first_hit_rank(answers, gt)
        score = score_query(answers, gt)
        for cutoff in _CUTOFFS:
            recall_totals[cutoff] += int(rank is not None and rank <= cutoff)
        per_query.append(
            {
                "query_id": str(record["query_id"]),
                "score": score,
                "rank_of_first_hit": rank,
            }
        )

    query_count = len(per_query)
    recall_means = {
        cutoff: recall_totals[cutoff] / query_count if query_count else 0.0
        for cutoff in _CUTOFFS
    }
    final_score = sum(item["score"] for item in per_query) / query_count if query_count else 0.0
    return per_query, recall_means, final_score


def write_eval_log(records: Sequence[dict[str, int | float | None | str]], out_path: Path) -> None:
    """Write per-query evaluation results as JSON or CSV according to the file suffix.

    Expected input: ``records`` contains query IDs, numeric scores, and optional first-hit
    ranks; ``out_path`` has a ``.json`` or ``.csv`` suffix.
    Returns: None.
    Raises: ValueError for unsupported output suffixes; OSError if the output cannot be written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".json":
        with out_path.open("w", encoding="utf-8") as output_file:
            json.dump(records, output_file, ensure_ascii=False, indent=2)
        return
    if out_path.suffix.lower() == ".csv":
        with out_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["query_id", "score", "rank_of_first_hit"],
            )
            writer.writeheader()
            writer.writerows(records)
        return
    raise ValueError("Evaluation log output must have a .json or .csv suffix.")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run configured KIS retrieval on an eval set and print local score metrics.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``;
    config, index, and evaluation JSONL conform to their documented schemas.
    Returns: None.
    Raises: FileNotFoundError if an input is missing; ValueError for invalid eval content or
        retrieval settings; OSError if an optional log cannot be written.
    """
    args = parse_args(arguments)
    start_time = perf_counter()
    config = load_config(args.config)
    emb, vids, frame_idx = load_index(config["index"]["path"])
    records, skipped_counts = load_kis_records(args.eval, args.text_field)
    top_k = args.top_k if args.top_k is not None else config["retrieve"]["top_k"]
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    if records:
        encoder = TextEncoder(
            model_name=config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            device=args.device or config["model"]["device"],
        )
        query_vecs = encoder.encode([str(record["text"]) for record in records])
        results_per_query = search(query_vecs, emb, vids, frame_idx, top_k=top_k)
    else:
        results_per_query = []

    per_query, recall_means, final_score = evaluate_kis(records, results_per_query, top_k)
    print(f"#queries: {len(records)}")
    print(f"Final Score: {final_score:.4f}")
    for cutoff in _CUTOFFS:
        print(f"R@{cutoff}: {recall_means[cutoff]:.4f}")
    for task, count in skipped_counts.items():
        if count:
            print(f"note: skipped {count} {task} record(s); only task='kis' is evaluated.")
    print(f"elapsed_seconds: {perf_counter() - start_time:.2f}")

    if args.out is not None:
        write_eval_log(per_query, args.out)
        print(f"output_path: {args.out}")


if __name__ == "__main__":
    main()
