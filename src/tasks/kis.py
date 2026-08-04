"""Known-item search result formatting and submission writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def format_kis(
    results_per_query: list[list[dict[str, str | int | float]]],
    query_ids: list[str],
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """Format ranked retrieval results as known-item-search predictions.

    Expected input: ``results_per_query`` contains one descending-ranked retrieval list
    per query, and ``query_ids`` is a parallel list of query identifiers; ``max_results``
    is a non-negative answer limit per query.
    Returns: Prediction dictionaries with contiguous 1-based ranks and deduplicated
        ``(video_id, frame_idx)`` answer pairs.
    Raises: ValueError if input lengths differ or ``max_results`` is negative.
    """
    if len(results_per_query) != len(query_ids):
        raise ValueError("results_per_query and query_ids must have the same length.")
    if max_results < 0:
        raise ValueError("max_results must be non-negative.")

    predictions: list[dict[str, Any]] = []
    for query_id, results in zip(query_ids, results_per_query, strict=True):
        seen: set[tuple[str, int]] = set()
        answers: list[dict[str, str | int]] = []
        if max_results == 0:
            predictions.append({"query_id": query_id, "answers": answers})
            continue
        for result in results:
            answer_key = (str(result["video_id"]), int(result["frame_idx"]))
            if answer_key in seen:
                continue
            seen.add(answer_key)
            answers.append(
                {
                    "rank": len(answers) + 1,
                    "video_id": answer_key[0],
                    "frame_idx": answer_key[1],
                }
            )
            if len(answers) == max_results:
                break
        predictions.append({"query_id": query_id, "answers": answers})
    return predictions


def write_submission(
    preds: list[dict[str, Any]],
    out_path: Path,
    fmt: str = "csv",
) -> None:
    """Write known-item-search predictions to CSV or JSON.

    Expected input: ``preds`` is the output of ``format_kis``; ``out_path`` is the
    destination file; ``fmt`` is either ``csv`` or ``json``.
    Returns: None.
    Raises: ValueError if ``fmt`` is unsupported; OSError if the submission cannot be written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        with out_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["query_id", "video_id", "frame_idx"],
            )
            writer.writeheader()
            for prediction in preds:
                for answer in prediction["answers"]:
                    writer.writerow(
                        {
                            "query_id": prediction["query_id"],
                            "video_id": answer["video_id"],
                            "frame_idx": answer["frame_idx"],
                        }
                    )
        return
    if fmt == "json":
        with out_path.open("w", encoding="utf-8") as output_file:
            json.dump(preds, output_file, ensure_ascii=False, indent=2)
        return
    raise ValueError(f"Unsupported submission format: {fmt}")


# TODO(clarify): confirm exact organizer submission format (columns/header/one file).
