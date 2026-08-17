"""Placeholder for the future QA scoring stage.

QA scoring is intentionally not implemented. A correct QA result requires
the organizer's ground truth (correct video, correct frame, and a matching
textual answer), and the team has not yet defined or validated that scoring
contract. See ``docs/data-contracts.md`` ("QA and TRAKE scoring are
intentionally not implemented").

This module exists so that:

- the QA task's future evaluation contract has a named, importable home;
- no other module accidentally claims to score QA predictions;
- the intended input shape (canonical evaluation set + per-query evidence)
  is documented in code.
"""

from __future__ import annotations

from collections.abc import Mapping

from aic2026.evaluation import EvaluationSet
from aic2026.qa.qa_types import QaPrediction

QA_SCORING_STATUS = "not-implemented"


def evaluate_qa(
    evaluation_set: EvaluationSet,
    predictions_by_query: Mapping[str, object],
) -> object:
    """Score QA predictions (placeholder that always raises).

    Expected input:
        - evaluation_set: a validated evaluation set (future QA ground truth).
        - predictions_by_query: mapping from query id to an iterable of
          :class:`QaPrediction` evidence records.

    Returns:
        - Never returns; raises :class:`NotImplementedError`.

    Raises:
        - TypeError: if the arguments do not match the intended contract.
        - NotImplementedError: always, because QA scoring is not implemented.
    """
    if not isinstance(evaluation_set, EvaluationSet):
        raise TypeError("evaluation_set must be an EvaluationSet")
    if not isinstance(predictions_by_query, Mapping):
        raise TypeError("predictions_by_query must be a Mapping from query id to predictions")
    for query_id, predictions in predictions_by_query.items():
        if not isinstance(query_id, str) or query_id.strip() == "":
            raise ValueError("predictions_by_query keys must be non-empty query ids")
        if not isinstance(predictions, (list, tuple)):
            raise TypeError("predictions_by_query values must be a list or tuple of QaPrediction")
        if any(not isinstance(prediction, QaPrediction) for prediction in predictions):
            raise TypeError("predictions_by_query values must contain only QaPrediction")
    raise NotImplementedError(
        "QA scoring is intentionally not implemented; a scoring contract requires "
        "confirmed organizer ground truth and an explicit correctness definition "
        "(correct video, frame, and matching answer)"
    )