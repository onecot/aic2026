"""Local scoring for AIC known-item-search submissions.

Ground-truth schema: a CSV or JSONL file with ``query_id``, ``video_id``, ``s``, and
``e`` fields, where ``[s, e]`` is the inclusive valid frame range.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

_CUTOFFS = (1, 5, 20, 50, 100)


def score_query(answers: list[dict[str, Any]], gt: dict[str, Any]) -> float:
    """Compute the organizer's mean R@k score for one ranked answer list.

    Expected input: ``answers`` is rank-ordered and contains ``video_id`` and
    ``frame_idx`` fields; ``gt`` contains one ``video_id`` plus inclusive integer
    frame-range fields ``s`` and ``e``.
    Returns: The mean of R@1, R@5, R@20, R@50, and R@100 as a float in ``[0, 1]``.
    Raises: KeyError if required fields are absent; ValueError if frame values are invalid.
    """
    target_video = str(gt["video_id"])
    start_frame = int(gt["s"])
    end_frame = int(gt["e"])
    if start_frame > end_frame:
        raise ValueError("Ground-truth start frame must not exceed end frame.")

    r_at_k: list[int] = []
    for cutoff in _CUTOFFS:
        # Answers preserve retrieval rank, so this examines exactly the first k results.
        found = any(
            str(answer["video_id"]) == target_video
            and start_frame <= int(answer["frame_idx"]) <= end_frame
            for answer in answers[:cutoff]
        )
        r_at_k.append(int(found))
    return sum(r_at_k) / len(_CUTOFFS)


def score_submission(
    preds: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score a KIS prediction collection against query-level ground truth.

    Expected input: ``preds`` follows ``format_kis`` output with ``query_id`` and
    rank-ordered ``answers``; ``ground_truth`` maps every query ID to video and range data.
    Returns: A dictionary with per-query scores and their arithmetic mean.
    Raises: KeyError if prediction fields are absent; ValueError if duplicate query IDs occur
        in predictions or invalid ground-truth frame ranges are supplied.
    """
    prediction_by_id: dict[str, list[dict[str, Any]]] = {}
    for prediction in preds:
        query_id = str(prediction["query_id"])
        if query_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction query ID: {query_id}")
        prediction_by_id[query_id] = prediction["answers"]

    per_query = {
        query_id: score_query(prediction_by_id.get(query_id, []), ground_truth[query_id])
        for query_id in sorted(ground_truth)
    }
    mean_score = sum(per_query.values()) / len(per_query) if per_query else 0.0
    return {"per_query": per_query, "mean": mean_score}


def load_submission(submission_path: Path) -> list[dict[str, Any]]:
    """Load a CSV or JSON submission into the ``format_kis`` prediction structure.

    Expected input: ``submission_path`` is a CSV with ``query_id``, ``video_id``, and
    ``frame_idx`` columns in rank order, or a JSON file written by ``write_submission``.
    Returns: Prediction dictionaries with answers in source-file rank order.
    Raises: FileNotFoundError if the file is missing; ValueError for unsupported formats or
        malformed JSON submission content.
    """
    suffix = submission_path.suffix.lower()
    if suffix == ".csv":
        grouped_answers: dict[str, list[dict[str, Any]]] = {}
        with submission_path.open("r", encoding="utf-8", newline="") as input_file:
            for row in csv.DictReader(input_file):
                query_id = row["query_id"]
                grouped_answers.setdefault(query_id, []).append(
                    {"video_id": row["video_id"], "frame_idx": int(row["frame_idx"])}
                )
        return [
            {"query_id": query_id, "answers": answers}
            for query_id, answers in grouped_answers.items()
        ]
    if suffix == ".json":
        with submission_path.open("r", encoding="utf-8") as input_file:
            predictions = json.load(input_file)
        if not isinstance(predictions, list):
            raise ValueError("JSON submission must contain a list of predictions.")
        return predictions
    raise ValueError("Submission must be a .csv or .json file.")


def load_ground_truth(ground_truth_path: Path) -> dict[str, dict[str, Any]]:
    """Load organizer-style query ground truth from CSV or JSONL.

    Expected input: ``ground_truth_path`` is a CSV or JSONL file whose records contain
    ``query_id``, ``video_id``, ``s``, and ``e`` fields.
    Returns: A mapping from query ID to its video and inclusive frame range.
    Raises: FileNotFoundError if the file is missing; ValueError for unsupported formats,
        duplicate query IDs, or records missing required fields; json.JSONDecodeError for
        invalid JSONL.
    """
    suffix = ground_truth_path.suffix.lower()
    if suffix == ".csv":
        with ground_truth_path.open("r", encoding="utf-8", newline="") as input_file:
            records = list(csv.DictReader(input_file))
    elif suffix == ".jsonl":
        with ground_truth_path.open("r", encoding="utf-8") as input_file:
            records = [json.loads(line) for line in input_file if line.strip()]
    else:
        raise ValueError("Ground truth must be a .csv or .jsonl file.")

    ground_truth: dict[str, dict[str, Any]] = {}
    required_fields = {"query_id", "video_id", "s", "e"}
    for record in records:
        if not isinstance(record, dict) or not required_fields.issubset(record):
            raise ValueError("Each ground-truth record needs query_id, video_id, s, and e.")
        query_id = str(record["query_id"])
        if query_id in ground_truth:
            raise ValueError(f"Duplicate ground-truth query ID: {query_id}")
        ground_truth[query_id] = record
    return ground_truth


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for local KIS scoring.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed submission and ground-truth paths.
    Raises: SystemExit if CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Score a local KIS submission.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Load a submission and ground truth, then print local KIS scores.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``.
    Returns: None.
    Raises: FileNotFoundError if an input is missing; ValueError if either input violates
        its documented schema; json.JSONDecodeError if JSON content is invalid.
    """
    args = parse_args(arguments)
    scores = score_submission(
        load_submission(args.submission),
        load_ground_truth(args.ground_truth),
    )
    for query_id, query_score in scores["per_query"].items():
        print(f"{query_id}: {query_score:.3f}")
    print(f"mean: {scores['mean']:.3f}")


# TODO(clarify): confirm GT file format from organizer; CSV/JSONL with query_id, video_id, s, e is assumed.


if __name__ == "__main__":
    main()
