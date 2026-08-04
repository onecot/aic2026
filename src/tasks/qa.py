"""Question-answering task formatting placeholder."""

from __future__ import annotations

from typing import Any


def format_qa(
    results_per_query: list[list[dict[str, str | int | float]]],
    query_ids: list[str],
) -> list[dict[str, Any]]:
    """Format QA retrieval results for a future visual-language answering stage.

    Expected input: ``results_per_query`` contains ranked keyframes per question and
    ``query_ids`` is a parallel list of question identifiers.
    Returns: Future output will include each query's retrieved results with an ``answer``
        field produced by a visual-language model.
    Raises: NotImplementedError because QA answer generation is not implemented.
    """
    raise NotImplementedError("QA formatting requires a future visual-language model stage.")
