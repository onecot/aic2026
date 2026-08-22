"""Bounded OCR candidate reranking over visual retrieval pools (Phase 5C-A).

The visual retriever stays the only candidate source. OCR evidence
(CLEANED_CONCAT representation-v2 chunks) is used exclusively to reorder
candidates that the visual retriever already selected; OCR can never add a
candidate or a video to the pool.

Two pre-declared policies:

- ``strict_frame`` (C1): rerank the top-N visual frames within the pool.
  Every frame inherits its video's best CLEANED_CONCAT cosine (normalized);
  pool membership is unchanged.
- ``video_aware`` (C2/C3): derive candidate videos from the top-N visual
  frames, score each video as ``minmax(max visual) + ocr_weight * ocr_support``
  where ``ocr_support`` is the min-max normalized best CLEANED_CONCAT cosine
  among that video's chunks, rerank videos, and flatten back to frames while
  preserving each video's original visual frame order.

Deterministic conventions (frozen before evaluation):

- score normalization reuses :func:`aic2026.ocr_fusion.normalize_scores_minmax`;
- the OCR normalization domain is the raw best-chunk scores of the pool
  videos that HAVE OCR evidence; videos without evidence get support 0.0;
- ties break by best original visual rank (smaller first);
- no temporal decay and no invented per-frame scores.

All functions are pure; callers supply ranked candidates and a per-video
evidence mapping restricted to pool videos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aic2026.ocr_fusion import normalize_scores_minmax

#: Supported reranking policies.
RERANK_POLICIES = ("strict_frame", "video_aware")


@dataclass(frozen=True)
class CandidateRerankConfig:
    """Explicit, inspectable configuration for one rerank variant.

    Fields:
        - policy: ``"strict_frame"`` or ``"video_aware"``.
        - pool_size: number of top visual frames forming the candidate pool.
        - ocr_weight: non-negative finite OCR contribution weight lambda.

    Raises:
        - ValueError: if any field violates its contract.
    """

    policy: str
    pool_size: int
    ocr_weight: float

    def __post_init__(self) -> None:
        if self.policy not in RERANK_POLICIES:
            raise ValueError(f"policy must be one of {RERANK_POLICIES}, got {self.policy!r}")
        if not isinstance(self.pool_size, int) or isinstance(self.pool_size, bool) or self.pool_size < 1:
            raise ValueError(f"pool_size must be a positive integer, got {self.pool_size!r}")
        if (
            not isinstance(self.ocr_weight, (int, float))
            or isinstance(self.ocr_weight, bool)
            or not math.isfinite(float(self.ocr_weight))
            or self.ocr_weight < 0
        ):
            raise ValueError(f"ocr_weight must be a finite value >= 0, got {self.ocr_weight!r}")


@dataclass(frozen=True)
class VideoOcrEvidence:
    """Best OCR chunk evidence for ONE candidate video.

    Fields:
        - video_id: candidate video the evidence belongs to.
        - chunk_id: identifier of the strongest CLEANED_CONCAT chunk.
        - raw_score: best raw cosine similarity for that video.
    """

    video_id: str
    chunk_id: str
    raw_score: float


@dataclass(frozen=True)
class RerankedCandidate:
    """One visual candidate after bounded OCR reranking, with diagnostics.

    Fields:
        - rank: 1-based position after reranking.
        - video_id / keyframe_ordinal / frame_index / pts_time: candidate
          identity copied from the visual candidate.
        - visual_score: raw visual cosine of the candidate.
        - visual_rank: 1-based rank in the incoming visual pool.
        - combined_score: score used for ordering under the policy.
        - ocr_support: normalized OCR evidence of the candidate's video in
          ``[0, 1]`` (0.0 when the video has no evidence).
        - matched_chunk_id: strongest OCR chunk id, or None.
    """

    rank: int
    video_id: str
    keyframe_ordinal: int
    frame_index: int
    pts_time: float
    visual_score: float
    visual_rank: int
    combined_score: float
    ocr_support: float
    matched_chunk_id: str | None


def _validate_inputs(
    candidates: Sequence[Any],
    evidence_by_video: Mapping[str, VideoOcrEvidence],
) -> None:
    """Validate shared rerank inputs.

    Raises:
        - ValueError: if candidates are empty, unordered, or evidence does
          not match its video.
    """
    if not candidates:
        return
    ranks = [int(c.rank) for c in candidates]
    if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("candidates must be ranked with unique ascending ranks")
    for video_id, evidence in evidence_by_video.items():
        if evidence.video_id != video_id:
            raise ValueError(
                f"evidence keyed {video_id!r} carries video_id {evidence.video_id!r}"
            )


def _ocr_supports(
    pool_videos: Sequence[str],
    evidence_by_video: Mapping[str, VideoOcrEvidence],
) -> dict[str, tuple[float, str | None]]:
    """Return ``(support, chunk_id)`` per pool video.

    The normalization domain is the raw scores of pool videos WITH evidence;
    videos without evidence receive support 0.0. Deterministic via
    :func:`normalize_scores_minmax`.
    """
    bearing = [v for v in dict.fromkeys(pool_videos) if v in evidence_by_video]
    raws = [float(evidence_by_video[v].raw_score) for v in bearing]
    norms = normalize_scores_minmax(raws)
    supports: dict[str, tuple[float, str | None]] = {}
    for video, norm in zip(bearing, norms):
        supports[video] = (float(norm), evidence_by_video[video].chunk_id)
    for video in dict.fromkeys(pool_videos):
        supports.setdefault(video, (0.0, None))
    return supports


def strict_frame_rerank(
    candidates: Sequence[Any],
    *,
    config: CandidateRerankConfig,
    evidence_by_video: Mapping[str, VideoOcrEvidence],
) -> list[RerankedCandidate]:
    """Rerank the visual frame pool without changing membership.

    Expected input:
        - candidates: visual-ranked candidates (e.g. RetrievalCandidate)
          with unique ascending ``rank``.
        - config: policy must be ``strict_frame``.
        - evidence_by_video: best OCR evidence per pool video (videos absent
          from the mapping have no evidence).

    Returns:
        - The same candidates reordered by
          ``minmax_visual + ocr_weight * ocr_support`` descending, ties by
          original visual rank. Membership is exactly the input set.

    Raises:
        - ValueError: on wrong policy or malformed inputs.
    """
    if config.policy != "strict_frame":
        raise ValueError(f"policy must be 'strict_frame', got {config.policy!r}")
    _validate_inputs(candidates, evidence_by_video)
    visual_norms = normalize_scores_minmax([float(c.score) for c in candidates])
    supports = _ocr_supports([c.video_id for c in candidates], evidence_by_video)

    drafts: list[tuple[float, int, dict[str, Any]]] = []
    for candidate, visual_norm in zip(candidates, visual_norms):
        support, chunk_id = supports[candidate.video_id]
        fused = float(visual_norm) + float(config.ocr_weight) * support
        fields = {
            "video_id": candidate.video_id,
            "keyframe_ordinal": int(candidate.keyframe_ordinal),
            "frame_index": int(candidate.frame_index),
            "pts_time": float(candidate.pts_time),
            "visual_score": float(candidate.score),
            "visual_rank": int(candidate.rank),
            "combined_score": fused,
            "ocr_support": float(support),
            "matched_chunk_id": chunk_id,
        }
        drafts.append((-fused, int(candidate.rank), fields))
    drafts.sort(key=lambda item: (item[0], item[1]))
    return [
        RerankedCandidate(rank=position, **fields)
        for position, (_, _, fields) in enumerate(drafts, start=1)
    ]


def video_aware_rerank(
    candidates: Sequence[Any],
    *,
    config: CandidateRerankConfig,
    evidence_by_video: Mapping[str, VideoOcrEvidence],
) -> list[RerankedCandidate]:
    """Rerank candidate VIDEOS derived from the visual pool, then flatten.

    Expected input:
        - candidates: visual-ranked candidates forming the pool.
        - config: policy must be ``video_aware``.
        - evidence_by_video: best OCR evidence per pool video.

    Returns:
        - Flattened candidates: videos ordered by
          ``minmax(video max visual) + ocr_weight * ocr_support`` descending
          (ties by best visual rank), frames of each video in their original
          visual order. No video or frame outside the pool can appear.

    Raises:
        - ValueError: on wrong policy or malformed inputs.
    """
    if config.policy != "video_aware":
        raise ValueError(f"policy must be 'video_aware', got {config.policy!r}")
    _validate_inputs(candidates, evidence_by_video)

    grouped: dict[str, list[Any]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)
    video_order = list(grouped.keys())  # first-appearance order == best-visual-rank order
    best_visual = [float(grouped[v][0].score) for v in video_order]
    visual_norms = normalize_scores_minmax(best_visual)
    supports = _ocr_supports(video_order, evidence_by_video)

    scored: list[tuple[float, int, str]] = []
    for index, video in enumerate(video_order):
        support, _ = supports[video]
        combined = float(visual_norms[index]) + float(config.ocr_weight) * support
        scored.append((-combined, index, video))

    out: list[RerankedCandidate] = []
    position = 0
    for _, video_index, video in sorted(scored):
        support, chunk_id = supports[video]
        video_score = float(visual_norms[video_index]) + float(config.ocr_weight) * support
        for candidate in grouped[video]:  # stored in incoming (visual) order
            position += 1
            out.append(
                RerankedCandidate(
                    rank=position,
                    video_id=candidate.video_id,
                    keyframe_ordinal=int(candidate.keyframe_ordinal),
                    frame_index=int(candidate.frame_index),
                    pts_time=float(candidate.pts_time),
                    visual_score=float(candidate.score),
                    visual_rank=int(candidate.rank),
                    combined_score=video_score,
                    ocr_support=support,
                    matched_chunk_id=chunk_id,
                )
            )
    return out


def rerank_candidates(
    candidates: Sequence[Any],
    *,
    config: CandidateRerankConfig,
    evidence_by_video: Mapping[str, VideoOcrEvidence],
) -> list[RerankedCandidate]:
    """Dispatch to the configured pre-declared rerank policy.

    Expected input:
        - candidates: visual-ranked candidate pool.
        - config: rerank configuration (see :class:`CandidateRerankConfig`).
        - evidence_by_video: best OCR evidence per pool video.

    Returns:
        - Policy-specific reranked candidates (see the policy functions).

    Raises:
        - ValueError: inherited from validation and dispatch.
    """
    if config.policy == "strict_frame":
        return strict_frame_rerank(candidates, config=config, evidence_by_video=evidence_by_video)
    return video_aware_rerank(candidates, config=config, evidence_by_video=evidence_by_video)
