"""QA input contract: selection and text resolution for QA queries.

The QA prediction runner consumes the shared team evaluation set
(:func:`aic2026.evaluation.load_eval_set`) and operates only on rows whose
task is ``qa``. This module keeps that selection and the query-text
resolution in one place so the runner does not re-implement evaluation-set
filtering or text-field handling.

It contains no retrieval, no scoring, and no artifact logic.
"""

from __future__ import annotations

from collections.abc import Sequence

from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask
from aic2026.qa.qa_types import QueryTextField


def iter_qa_queries(evaluation_set: EvaluationSet) -> tuple[EvaluationQuery, ...]:
    """Return the QA queries of an evaluation set, in CSV row order.

    Expected input:
        - evaluation_set: a validated evaluation set.

    Returns:
        - Tuple of QA queries in CSV row order (empty when the set has none).

    Raises:
        - No exceptions.
    """
    return tuple(evaluation_set.iter_queries(EvaluationTask.QA))


def select_qa_queries(
    evaluation_set: EvaluationSet,
    subset_ids: Sequence[str] | None = None,
) -> tuple[EvaluationQuery, ...]:
    """Return ordered QA queries, optionally restricted to explicit ids.

    Expected input:
        - evaluation_set: a validated evaluation set.
        - subset_ids: optional explicit subset of query ids to keep, in
          query order (subset membership only; order follows the CSV).

    Returns:
        - The selected QA queries in CSV row order.

    Raises:
        - ValueError: if a requested id is unknown or no query is selected.
    """
    qa_queries = iter_qa_queries(evaluation_set)
    if subset_ids is None:
        return qa_queries
    expected = {query.query_id for query in qa_queries}
    requested = set(subset_ids)
    unknown = requested - expected
    if unknown:
        raise ValueError(f"query_subset_ids contain unknown query ids: {sorted(unknown)}")
    selected = tuple(query for query in qa_queries if query.query_id in requested)
    if not selected:
        raise ValueError("query_subset_ids selected no QA queries")
    return selected


def resolve_query_text(query: EvaluationQuery, field: QueryTextField) -> str:
    """Resolve and validate one query text field from a canonical query.

    Expected input:
        - query: a canonical evaluation query.
        - field: ``query_text_vi`` or ``query_text_en``.

    Returns:
        - The requested non-empty query text.

    Raises:
        - ValueError: if the requested English field is missing.
    """
    if field == "query_text_vi":
        return query.query_text_vi
    if query.query_text_en is None:
        raise ValueError("query_text_en was requested but is missing for at least one query")
    return query.query_text_en