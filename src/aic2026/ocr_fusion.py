"""OCR-aware late fusion for visual retrieval candidates.

Combines independent visual retrieval results with dense OCR-chunk
retrieval results (see ``aic2026.ocr_index``) via candidate-level late
fusion::

    visual candidates (video_id, frame_index, score)
        |
    per-query score normalization
        |
    OCR support = best over same-video OCR chunks of
                  normalized_ocr_score * temporal_alignment
        |
    fused = visual_norm + ocr_weight(query) * ocr_support
        |
    reranked candidates (+ full diagnostics)

Design principles (Phase 3A):

- visual retrieval stays the primary candidate source; OCR never introduces
  new candidates, it only reranks the visual pool;
- OCR chunks are strictly single-video, so cross-video OCR cannot boost a
  candidate;
- OCR usefulness varies by query: the OCR weight is either fixed
  (ablation) or set by a transparent keyword heuristic
  (:func:`query_ocr_weight`);
- visual frames and OCR windows are temporally misaligned; alignment decays
  smoothly with the distance from the chunk's frame range
  (:func:`temporal_alignment`).

All functions are pure and deterministic; no model is trained and no
retrieval is performed here. Callers supply ranked candidates from the
existing retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: OCR-indicative concepts (lowercase). Single tokens must match a whole
#: whitespace token; multi-word entries match as phrases. Centralized here
#: so the heuristic is easy to inspect and extend.
OCR_KEYWORDS: tuple[str, ...] = (
    # Vietnamese
    "tên",
    "gọi là",
    "ghi",
    "chữ",
    "dòng chữ",
    "tiêu đề",
    "bảng",
    "biển",
    "nhãn",
    "khẩu hiệu",
    "số",
    "địa chỉ",
    "công thức",
    "nguyên liệu",
    "câu thơ",
    # English
    "name",
    "called",
    "written",
    "text",
    "title",
    "sign",
    "label",
    "number",
    "address",
    "recipe",
    "ingredient",
    "poem",
)

#: Normalization methods supported by :func:`normalize_scores`.
NORMALIZATION_METHODS = ("minmax", "rank")

#: Weight modes supported by :func:`query_ocr_weight`.
WEIGHT_MODES = ("fixed", "adaptive")


@dataclass(frozen=True)
class OcrFusionConfig:
    """Explicit, inspectable configuration for one fusion variant.

    Fields:
        - normalization: per-query score normalization, ``"minmax"`` or
          ``"rank"``.
        - use_temporal_alignment: when False, same-video OCR support skips
          the temporal decay (video-level boost; ablation variant).
        - temporal_scale: distance decay scale in original-video frames
          (``alignment = 1 / (1 + distance / scale)``).
        - weight_mode: ``"fixed"`` uses ``fixed_weight`` for every query;
          ``"adaptive"`` derives the weight from OCR-indicative keywords.
        - fixed_weight: OCR weight used in fixed mode.
        - base_weight: adaptive-mode weight for queries without OCR cues.
        - max_weight: adaptive-mode weight once ``keyword_saturation``
          distinct keywords are present.
        - keyword_saturation: number of distinct keyword matches at which
          the adaptive weight reaches ``max_weight`` (>= 1).
    """

    normalization: str = "minmax"
    use_temporal_alignment: bool = True
    temporal_scale: float = 500.0
    weight_mode: str = "fixed"
    fixed_weight: float = 0.25
    base_weight: float = 0.05
    max_weight: float = 0.5
    keyword_saturation: int = 2

    def __post_init__(self) -> None:
        if self.normalization not in NORMALIZATION_METHODS:
            raise ValueError(f"normalization must be one of {NORMALIZATION_METHODS}, got {self.normalization!r}")
        if self.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {self.weight_mode!r}")
        if not isinstance(self.temporal_scale, (int, float)) or isinstance(self.temporal_scale, bool) \
                or self.temporal_scale <= 0:
            raise ValueError(f"temporal_scale must be > 0, got {self.temporal_scale!r}")
        for name in ("fixed_weight", "base_weight", "max_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
        if self.base_weight > self.max_weight:
            raise ValueError("base_weight must be <= max_weight")
        if not isinstance(self.keyword_saturation, int) or isinstance(self.keyword_saturation, bool) \
                or self.keyword_saturation < 1:
            raise ValueError(f"keyword_saturation must be >= 1, got {self.keyword_saturation!r}")


@dataclass(frozen=True)
class FusedCandidate:
    """One visual candidate after OCR fusion, with full diagnostics.

    Fields:
        - rank: 1-based rank after fusion.
        - video_id / frame_index: identity of the visual candidate.
        - visual_score: raw cosine score from the visual retriever.
        - visual_score_norm: per-query normalized visual score.
        - matched_chunk_id: id of the strongest aligned OCR chunk, or None.
        - ocr_score: raw cosine score of that chunk (0.0 when unmatched).
        - ocr_score_norm: normalized score of that chunk (0.0 when none).
        - temporal_alignment: alignment of the candidate frame to the chunk
          range in ``[0, 1]``; 1.0 when alignment is disabled and a chunk
          matched; 0.0 when nothing matched.
        - ocr_support: ``ocr_score_norm * temporal_alignment`` (or the best
          normalized score when alignment is disabled).
        - ocr_weight: the OCR weight used for this query.
        - fused_score: ``visual_score_norm + ocr_weight * ocr_support``.
    """

    rank: int
    video_id: str
    frame_index: int
    visual_score: float
    visual_score_norm: float
    matched_chunk_id: str | None
    ocr_score: float
    ocr_score_norm: float
    temporal_alignment: float
    ocr_support: float
    ocr_weight: float
    fused_score: float


def normalize_scores_minmax(scores: Sequence[float]) -> list[float]:
    """Normalize scores per query to ``[0, 1]`` with min-max scaling.

    Expected input:
        - scores: raw scores of one candidate list (order preserved).

    Returns:
        - ``(s - min) / (max - min)`` per element; when every score is
          equal (including single-element lists) the neutral midpoint 0.5
          is returned for all elements because the list carries no
          discriminative signal.

    Raises:
        - No exceptions (empty input returns an empty list).
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.5 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def normalize_scores_rank(scores: Sequence[float]) -> list[float]:
    """Normalize scores per query by descending rank position.

    Expected input:
        - scores: raw scores of one candidate list (order preserved).

    Returns:
        - ``1.0`` for the highest score down to ``0.0`` for the lowest,
          linearly by sorted position. Ties are broken by original order
          (deterministic). Single-element lists return ``[1.0]``.

    Raises:
        - No exceptions (empty input returns an empty list).
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    normalized = [0.0] * n
    for position, index in enumerate(order):
        normalized[index] = 1.0 - position / (n - 1)
    return normalized


def normalize_scores(scores: Sequence[float], method: str) -> list[float]:
    """Normalize scores with the configured per-query method.

    Expected input:
        - scores: raw scores of one candidate list.
        - method: ``"minmax"`` or ``"rank"``.

    Returns:
        - Normalized scores in the same order.

    Raises:
        - ValueError: if ``method`` is unknown.
    """
    if method == "minmax":
        return normalize_scores_minmax(scores)
    if method == "rank":
        return normalize_scores_rank(scores)
    raise ValueError(f"unknown normalization method: {method!r}")


def temporal_alignment(
    frame_index: int,
    frame_start: int,
    frame_end: int,
    scale: float,
) -> float:
    """Return bounded temporal alignment between a frame and a chunk range.

    Expected input:
        - frame_index: original-video frame index of the visual candidate.
        - frame_start / frame_end: inclusive original-video frame range of
          the OCR chunk.
        - scale: positive decay scale in frames.

    Returns:
        - ``1.0`` when the frame lies inside ``[frame_start, frame_end]``;
          otherwise ``1 / (1 + distance / scale)`` where ``distance`` is the
          distance to the nearest range boundary. Bounded in ``(0, 1]``.

    Raises:
        - ValueError: if ``scale <= 0`` or the range is reversed.
    """
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale!r}")
    if frame_end < frame_start:
        raise ValueError(f"invalid chunk range [{frame_start}, {frame_end}]")
    if frame_index < frame_start:
        distance = frame_start - frame_index
    elif frame_index > frame_end:
        distance = frame_index - frame_end
    else:
        return 1.0
    return 1.0 / (1.0 + distance / scale)


def matched_ocr_keywords(
    query_text: str,
    *,
    keywords: tuple[str, ...] = OCR_KEYWORDS,
) -> tuple[str, ...]:
    """Return the OCR-indicative keywords present in a query.

    Expected input:
        - query_text: arbitrary query text.
        - keywords: lowercase keyword vocabulary; single-word entries must
          match a whole whitespace token, multi-word entries match as
          phrases in the whitespace-normalized lowercase text.

    Returns:
        - Matched keywords in vocabulary order (deduplicated).

    Raises:
        - No exceptions (non-string input yields no matches).
    """
    if not isinstance(query_text, str):
        return ()
    lowered = " ".join(query_text.lower().split())
    tokens = set(lowered.split())
    matches: list[str] = []
    for keyword in keywords:
        if " " in keyword:
            if keyword in lowered:
                matches.append(keyword)
        elif keyword in tokens:
            matches.append(keyword)
    return tuple(matches)


def query_ocr_weight(query_text: str, config: OcrFusionConfig) -> float:
    """Return the OCR weight for one query under the configured mode.

    Expected input:
        - query_text: query text (used only in adaptive mode).
        - config: fusion configuration.

    Returns:
        - Fixed mode: ``config.fixed_weight``.
        - Adaptive mode: ``base_weight + (max_weight - base_weight) *
          min(1, distinct_keyword_matches / keyword_saturation)``.

    Raises:
        - ValueError: inherited from config validation.
    """
    if config.weight_mode == "fixed":
        return float(config.fixed_weight)
    matches = matched_ocr_keywords(query_text)
    fraction = min(1.0, len(matches) / config.keyword_saturation)
    return float(config.base_weight + (config.max_weight - config.base_weight) * fraction)


def fuse_visual_with_ocr(
    visual_candidates: Sequence[Any],
    ocr_results: Sequence[Any],
    *,
    config: OcrFusionConfig,
    ocr_weight: float,
) -> list[FusedCandidate]:
    """Rerank visual candidates with temporally aligned OCR evidence.

    Expected input:
        - visual_candidates: ranked objects exposing ``rank``, ``video_id``,
          ``frame_index``, and ``score`` (e.g. ``RetrievalCandidate``).
        - ocr_results: ranked objects exposing ``chunk_id``, ``video_id``,
          ``frame_start``, ``frame_end``, and ``score`` (e.g.
          ``OcrSearchResult``).
        - config: fusion configuration.
        - ocr_weight: the OCR weight for this query (see
          :func:`query_ocr_weight`); must be >= 0.

    Returns:
        - Candidates sorted by ``fused_score`` descending with ties broken
          by the original visual rank; every candidate carries full
          diagnostics. With no same-video OCR evidence the fused score
          reduces to the normalized visual score, preserving the visual
          ordering.

    Raises:
        - ValueError: if ``ocr_weight < 0`` or inputs are malformed.
    """
    if not isinstance(ocr_weight, (int, float)) or isinstance(ocr_weight, bool) or ocr_weight < 0:
        raise ValueError(f"ocr_weight must be >= 0, got {ocr_weight!r}")

    visual_norms = normalize_scores([float(c.score) for c in visual_candidates], config.normalization)
    ocr_norms = normalize_scores([float(r.score) for r in ocr_results], config.normalization)

    drafts: list[dict[str, Any]] = []
    for i, candidate in enumerate(visual_candidates):
        best_value = 0.0
        best: dict[str, Any] | None = None
        for j, result in enumerate(ocr_results):
            if result.video_id != candidate.video_id:
                continue
            alignment = (
                temporal_alignment(
                    candidate.frame_index, result.frame_start, result.frame_end, config.temporal_scale
                )
                if config.use_temporal_alignment
                else 1.0
            )
            value = ocr_norms[j] * alignment
            if best is None or value > best_value:
                best_value = value
                best = {
                    "chunk_id": result.chunk_id,
                    "ocr_score": float(result.score),
                    "ocr_score_norm": ocr_norms[j],
                    "alignment": alignment,
                }
        support = best_value
        drafts.append(
            {
                "video_id": candidate.video_id,
                "frame_index": candidate.frame_index,
                "visual_score": float(candidate.score),
                "visual_score_norm": visual_norms[i],
                "matched_chunk_id": best["chunk_id"] if best else None,
                "ocr_score": best["ocr_score"] if best else 0.0,
                "ocr_score_norm": best["ocr_score_norm"] if best else 0.0,
                "temporal_alignment": best["alignment"] if best else 0.0,
                "ocr_support": support,
                "ocr_weight": float(ocr_weight),
                "fused_score": visual_norms[i] + float(ocr_weight) * support,
                "visual_rank": int(candidate.rank),
            }
        )

    # Deterministic order: fused score descending, ties broken by the
    # original visual rank.
    drafts.sort(key=lambda d: (-d["fused_score"], d["visual_rank"]))
    return [
        FusedCandidate(rank=position, **{k: v for k, v in draft.items() if k != "visual_rank"})
        for position, draft in enumerate(drafts, start=1)
    ]
