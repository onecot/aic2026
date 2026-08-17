"""BTC QA submission export, isolated in one module.

The AIC 2026 BTC submission format is not yet confirmed for QA, so this
module owns the *entire* assumption about the on-disk submission layout.
Nothing else in the codebase writes or knows this format. When the official
format is published, only this module (and its tests) needs to change.

Current provisional format (``QA_SUBMISSION_FORMAT = "provisional-v1"``) is
one CSV row per ranked evidence candidate:

    query_id,rank,video_id,frame_index,answer

The textual answer is a per-query value repeated on every row of that query.
The formatter refuses to emit a submission when any processed query has an
empty answer or no evidence, because a QA submission must provide a video,
a frame, and a textual answer per query.
"""

from __future__ import annotations

import csv
from pathlib import Path

from aic2026.qa.qa_types import QaPredictionArtifact

QA_SUBMISSION_FORMAT = "provisional-v1"
QA_SUBMISSION_CSV_COLUMNS = ("query_id", "rank", "video_id", "frame_index", "answer")


def build_submission_rows(artifact: QaPredictionArtifact) -> list[dict[str, str]]:
    """Convert one QA artifact to provisional submission CSV rows.

    Expected input:
        - artifact: a validated QA prediction artifact. Every query must
          carry a non-empty answer and at least one evidence prediction.

    Returns:
        - A list of row dictionaries with keys
          :data:`QA_SUBMISSION_CSV_COLUMNS`, one row per ranked prediction.

    Raises:
        - ValueError: if any query has no answer or no evidence, because a
          submission cannot represent it.
    """
    rows: list[dict[str, str]] = []
    for query_result in artifact.query_results:
        if query_result.answer.strip() == "":
            raise ValueError(
                f"query {query_result.query_id!r} has no answer; QA answers are produced "
                "by a later answer-generation stage, not the retrieval foundation"
            )
        if not query_result.predictions:
            raise ValueError(f"query {query_result.query_id!r} has no evidence to submit")
        for prediction in query_result.predictions:
            rows.append(
                {
                    "query_id": query_result.query_id,
                    "rank": str(prediction.rank),
                    "video_id": prediction.video_id,
                    "frame_index": str(prediction.frame_index),
                    "answer": query_result.answer,
                }
            )
    return rows


def write_qa_submission_csv(artifact: QaPredictionArtifact, output_path: Path | str) -> Path:
    """Write the provisional BTC QA submission CSV atomically.

    Expected input:
        - artifact: a validated QA prediction artifact with answers.
        - output_path: target CSV file path.

    Returns:
        - The path that was written.

    Raises:
        - ValueError: if the artifact cannot be represented as a submission.
        - OSError: if writing fails.
    """
    rows = build_submission_rows(artifact)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_SUBMISSION_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(destination)
    return destination