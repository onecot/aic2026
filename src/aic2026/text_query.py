"""Task-neutral query record with resolved text for prediction runners.

Prediction runners may consume queries from different sources (the team
evaluation CSV, BTC experimental query files). This module defines the
source-independent record every prediction core accepts once a source has
resolved each query's identifier and final text. It contains no I/O and no
retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextQuery:
    """One query ready for text-embedding prediction.

    Expected input:
        - query_id: unique stable query identifier within one run.
        - query_text: the final query text after source-specific field
          resolution; never empty.

    Returns:
        - A validated text query.

    Raises:
        - ValueError: if either field is empty or not a string.
    """

    query_id: str
    query_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.query_text, str) or self.query_text.strip() == "":
            raise ValueError("query_text must be a non-empty string")
