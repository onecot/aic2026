"""Deterministic query routing for selective OCR evidence use (Phase 5D).

Classifies each evaluation query into one of three routes:

- ``VISUAL``: no explicit textual intent; OCR must not influence retrieval.
- ``OCR_TEXT``: the query is purely about written/named content (every
  content token is covered by a textual-intent signal).
- ``MIXED``: textual intent plus substantial scene description.

The router is a centralized, explainable keyword heuristic: no LLM, no
learned classifier, no external service. It replaces the Phase 3A fusion
keyword heuristic (:mod:`aic2026.ocr_fusion`) with an explicitly
task-oriented vocabulary and adds named-intent compounds for recipe/dish
queries. Bare ``"món"`` is deliberately excluded so ordinary food-scene
queries stay VISUAL.

All matching is deterministic: lowercase whitespace-normalized text,
single-word signals match whole tokens, multi-word signals match phrases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Route labels.
ROUTES = ("VISUAL", "OCR_TEXT", "MIXED")

#: Textual-intent signal vocabulary: signal -> intent category. Single words
#: must match a whole whitespace token; multi-word entries match as phrases.
SIGNAL_VOCABULARY: tuple[tuple[str, str], ...] = (
    # Written-content intent (Vietnamese)
    ("chữ", "written-text intent"),
    ("dòng chữ", "written-text intent"),
    ("ghi", "written-text intent"),
    ("viết", "written-text intent"),
    ("tiêu đề", "naming intent"),
    ("bảng", "signage/board intent"),
    ("biển", "signage/board intent"),
    ("biển hiệu", "signage/board intent"),
    ("nhãn", "label intent"),
    ("khẩu hiệu", "slogan intent"),
    ("câu thơ", "poem intent"),
    ("địa chỉ", "address intent"),
    ("tên", "naming intent"),
    # Recipe / dish-name intent (compounds only; bare "món" excluded by design)
    ("công thức", "recipe/ingredient intent"),
    ("nguyên liệu", "recipe/ingredient intent"),
    ("tên món", "dish-name intent"),
    ("món ăn", "dish-name intent"),
    ("cách làm", "recipe/ingredient intent"),
    ("hướng dẫn làm", "recipe/ingredient intent"),
    # English equivalents
    ("text", "written-text intent"),
    ("written", "written-text intent"),
    ("says", "written-text intent"),
    ("name", "naming intent"),
    ("title", "naming intent"),
    ("sign", "signage/board intent"),
    ("label", "label intent"),
    ("recipe", "recipe/ingredient intent"),
    ("ingredient", "recipe/ingredient intent"),
    ("poem", "poem intent"),
    ("number", "numeric-content intent"),
    ("address", "address intent"),
)

#: Numeric-content marker handled separately from the tuple above so its
#: Vietnamese form can also match whole tokens only.
_EXTRA_SINGLE_TOKENS: tuple[tuple[str, str], ...] = (
    ("số", "numeric-content intent"),
)


@dataclass(frozen=True)
class QueryRoute:
    """One deterministic routing decision with full provenance.

    Fields:
        - route: one of :data:`ROUTES`.
        - reason: human-readable explanation derived from the categories.
        - signals: matched signals in vocabulary order (deduplicated).
    """

    route: str
    reason: str
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"route": self.route, "reason": self.reason, "signals": list(self.signals)}


def _prepared(query_text: str) -> str:
    """Return the punctuation-stripped, whitespace-normalized query text."""
    from aic2026.ocr_evidence import strip_punctuation

    return strip_punctuation(query_text)


def matched_signals(query_text: str) -> tuple[str, ...]:
    """Return the textual-intent signals present in a query, in vocab order.

    Expected input:
        - query_text: arbitrary query text.

    Returns:
        - Matched signal strings (as written in the vocabulary), deduplicated,
          vocabulary order preserved. Non-string input yields no signals.
        - Matching runs on punctuation-stripped lowercase text so query
          punctuation cannot hide a signal token.

    Raises:
        - No exceptions.
    """
    if not isinstance(query_text, str):
        return ()
    lowered = _prepared(query_text)
    tokens = lowered.split()
    matches: list[str] = []
    seen: set[str] = set()
    for signal, _category in SIGNAL_VOCABULARY + _EXTRA_SINGLE_TOKENS:
        if signal in seen:
            continue
        if " " in signal:
            if f" {signal} " in f" {lowered} ":
                matches.append(signal)
                seen.add(signal)
        elif signal in tokens:
            matches.append(signal)
            seen.add(signal)
    return tuple(matches)


def _covered_tokens(signals: tuple[str, ...]) -> set[str]:
    """Return the normalized token set covered by matched signals."""
    from aic2026.ocr_failure_analysis import normalize_tokens

    covered: set[str] = set()
    for signal in signals:
        covered.update(normalize_tokens(signal))
    return covered


def route_query(query_text: str) -> QueryRoute:
    """Route one query deterministically.

    Expected input:
        - query_text: arbitrary query text.

    Returns:
        - :class:`QueryRoute` where:
          - no matched signals -> ``VISUAL``;
          - every content token of the query is covered by the matched
            signals -> ``OCR_TEXT`` (the information need is textual);
          - otherwise -> ``MIXED`` (textual intent plus scene description).

    Raises:
        - No exceptions.
    """
    from aic2026.ocr_evidence import query_content_tokens

    signals = matched_signals(query_text)
    if not signals:
        return QueryRoute(
            route="VISUAL",
            reason="no textual-intent signal present",
            signals=(),
        )
    residual = set(query_content_tokens(query_text)) - _covered_tokens(signals)
    if not residual:
        return QueryRoute(
            route="OCR_TEXT",
            reason="purely textual information need (" + ", ".join(signals) + ")",
            signals=signals,
        )
    return QueryRoute(
        route="MIXED",
        reason="textual intent plus scene description (" + ", ".join(signals) + ")",
        signals=signals,
    )
