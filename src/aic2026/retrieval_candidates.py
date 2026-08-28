"""Task-independent retrieval foundation for frame, video, and window candidates.

This module sits between the index retriever
(:class:`aic2026.retrieval.IndexRetriever`) and the task-specific prediction
layers (KIS today; QA and TRAKE in the future):

    index
      |
      v
    retrieval core                (aic2026.retrieval.IndexRetriever)
      |
      v
    candidate generation          frame / video / window candidates
      |
      v
    aggregation / ranking / reranking
      |
      v
    task-specific prediction      (KIS, QA, TRAKE)

It provides:

- generic candidate representations (:class:`VideoCandidate`,
  :class:`WindowCandidate`) that are independent of any task CSV contract;
- pure grouping of frame-level candidates by video
  (:func:`group_candidates_by_video`);
- video-level score aggregation (:func:`aggregate_video_score`);
- deterministic (stable) score ranking (:func:`sort_by_score`);
- the composed video candidate generation path
  (:func:`build_video_candidates`);
- temporal window grouping (:func:`group_windows`) for future window-level
  (TRAKE/QA) work.

The module performs no retrieval, no query encoding, and no task-specific
scoring. Every public function is pure: it depends only on the small
candidate contract defined below and never touches the index, the
evaluation set, or the scoring metrics. Task layers compose these
primitives instead of re-implementing grouping or aggregation.

Extension points: to add a video aggregation strategy, extend
:data:`SUPPORTED_VIDEO_AGGREGATIONS` and :func:`aggregate_video_score`. New
candidate generation strategies (top-N pooling variants, weighted pooling,
temporal-window aggregation) should be implemented as task-layer functions
that compose :func:`group_candidates_by_video`, :func:`group_windows`, and
:func:`aggregate_video_score` rather than modifying this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from math import fsum
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from aic2026.retrieval import RetrievalCandidate


@runtime_checkable
class HasVideoId(Protocol):
    """Any candidate carrying a canonical video id."""

    video_id: str


@runtime_checkable
class ScoredCandidate(HasVideoId, Protocol):
    """A candidate with a video id and a numeric similarity score."""

    score: float


@runtime_checkable
class FrameScoredCandidate(ScoredCandidate, Protocol):
    """A scored frame candidate with an original-video frame index."""

    frame_index: int


T = TypeVar("T", bound=HasVideoId)
S = TypeVar("S", bound=ScoredCandidate)


@dataclass(frozen=True)
class VideoCandidate:
    """One video-level retrieval candidate, independent of any task contract.

    Expected input:
        - video_id: canonical video identifier.
        - score: aggregated video-level score (numeric).
        - rank: 1-based rank assigned by the candidate source.
        - frame_count: number of frame candidates that produced this video.
        - top_frame_index: original-video frame index of the highest-scoring
          frame candidate, or ``None`` when unknown.

    Returns:
        - A validated video candidate.

    Raises:
        - ValueError: if any field violates the contract.
    """

    video_id: str
    score: float
    rank: int
    frame_count: int
    top_frame_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or self.video_id == "":
            raise ValueError("video_id must be a non-empty string")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ValueError("score must be numeric")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if (
            not isinstance(self.frame_count, int)
            or isinstance(self.frame_count, bool)
            or self.frame_count < 1
        ):
            raise ValueError("frame_count must be a positive integer")
        if self.top_frame_index is not None and (
            not isinstance(self.top_frame_index, int)
            or isinstance(self.top_frame_index, bool)
            or self.top_frame_index < 0
        ):
            raise ValueError("top_frame_index must be a non-negative integer or None")


@dataclass(frozen=True)
class WindowCandidate:
    """One temporal-window candidate inside a video, for future QA/TRAKE.

    Represents an inclusive original-video frame range
    ``[frame_start, frame_end]`` with a window-level score and rank.

    Expected input:
        - video_id: canonical video identifier.
        - frame_start: 0-based first frame of the window (inclusive).
        - frame_end: 0-based last frame of the window (inclusive).
        - score: window-level score (numeric).
        - rank: 1-based rank assigned by the candidate source.

    Returns:
        - A validated window candidate.

    Raises:
        - ValueError: if any field violates the contract.
    """

    video_id: str
    frame_start: int
    frame_end: int
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or self.video_id == "":
            raise ValueError("video_id must be a non-empty string")
        for name, value in (("frame_start", self.frame_start), ("frame_end", self.frame_end)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ValueError("score must be numeric")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


#: Video-level aggregation methods implemented by :func:`aggregate_video_score`.
SUPPORTED_VIDEO_AGGREGATIONS: frozenset[str] = frozenset({"max", "mean", "mean_top_n"})

VideoAggregationMethod = Literal["max", "mean", "mean_top_n"]


def group_candidates_by_video(candidates: Iterable[T]) -> dict[str, list[T]]:
    """Group frame-level candidates by ``video_id`` in a deterministic order.

    Videos appear in first-appearance order; within each group the input
    order is preserved. All candidates are kept: duplicates (including
    repeated ``(video_id, frame_index)`` records) are never dropped here.

    Expected input:
        - candidates: any iterable of values exposing ``video_id``.

    Returns:
        - Mapping from video id to its candidates, in first-appearance order.

    Raises:
        - No exceptions.
    """
    groups: dict[str, list[T]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.video_id, []).append(candidate)
    return groups


def aggregate_video_score(
    frame_scores: Sequence[float],
    method: str = "max",
    top_n: int | None = None,
) -> float:
    """Aggregate one video's frame candidate scores into a video-level score.

    Supported methods:

    - ``max``: the highest frame score.
    - ``mean``: the arithmetic mean of all frame scores.
    - ``mean_top_n``: the mean of the ``top_n`` highest frame scores,
      averaged over all frames when fewer than ``top_n`` exist.

    Expected input:
        - frame_scores: non-empty retrieval scores of one video's frames.
        - method: aggregation method from
          :data:`SUPPORTED_VIDEO_AGGREGATIONS`.
        - top_n: positive count used only by ``mean_top_n``.

    Returns:
        - The aggregated video-level score.

    Raises:
        - ValueError: if ``frame_scores`` is empty, ``method`` is
          unsupported, or ``top_n`` is invalid for ``mean_top_n``.
    """
    if not frame_scores:
        raise ValueError("video score requires at least one frame score")
    if method not in SUPPORTED_VIDEO_AGGREGATIONS:
        raise ValueError(
            f"unsupported video aggregation {method!r}; "
            f"supported: {sorted(SUPPORTED_VIDEO_AGGREGATIONS)}"
        )
    if method == "max":
        return max(frame_scores)
    if method == "mean":
        return fsum(float(score) for score in frame_scores) / len(frame_scores)
    if top_n is None or not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer for 'mean_top_n'")
    ordered = sorted(float(score) for score in frame_scores)
    ordered.reverse()
    count = min(top_n, len(ordered))
    return fsum(ordered[:count]) / count


def build_video_candidates(
    frame_candidates: Iterable[FrameScoredCandidate],
    method: str = "max",
    top_n: int | None = None,
) -> list[VideoCandidate]:
    """Generate ranked video-level candidates from frame-level candidates.

    Groups ``frame_candidates`` by ``video_id`` in first-appearance order,
    aggregates each video's frame scores with
    :func:`aggregate_video_score`, ranks videos by score in descending
    order (stable for ties), and assigns fresh strictly increasing ranks.
    Each candidate records its ``frame_count`` and the original-video
    ``top_frame_index`` of its highest-scoring frame (first occurrence wins
    ties).

    Expected input:
        - frame_candidates: scored frame candidates carrying ``video_id``,
          ``frame_index``, and ``score`` (for example
          :class:`aic2026.retrieval.RetrievalCandidate`).
        - method / top_n: video-level aggregation options forwarded to
          :func:`aggregate_video_score`.

    Returns:
        - Ranked video candidates in descending score order.

    Raises:
        - ValueError: if the aggregation parameters are invalid.
    """
    groups = group_candidates_by_video(frame_candidates)
    scored_groups: list[tuple[str, list[FrameScoredCandidate], float]] = [
        (
            video_id,
            group,
            aggregate_video_score(
                [float(candidate.score) for candidate in group],
                method=method,
                top_n=top_n,
            ),
        )
        for video_id, group in groups.items()
    ]
    ordered = sorted(scored_groups, key=lambda entry: entry[2], reverse=True)
    return [
        VideoCandidate(
            video_id=video_id,
            score=score,
            rank=rank,
            frame_count=len(group),
            top_frame_index=_top_frame_index(group),
        )
        for rank, (video_id, group, score) in enumerate(ordered, start=1)
    ]


def sort_by_score(candidates: Iterable[S], *, descending: bool = True) -> list[S]:
    """Return candidates stably sorted by ``score``.

    The sort is stable: equal scores preserve input order, which makes the
    result deterministic regardless of the order in which equal-score
    candidates were produced. This is the ranking primitive used by
    reranking and post-processing layers.

    Expected input:
        - candidates: scored candidates.
        - descending: sort highest score first (default) or lowest first.

    Returns:
        - A new list sorted by ``score``.

    Raises:
        - No exceptions.
    """
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=descending)


def group_windows(
    frame_indices: Iterable[int],
    max_gap: int,
) -> list[tuple[int, int]]:
    """Group frame indices into inclusive temporal windows by gap.

    Frames whose indices are within ``max_gap`` frames of the previous frame
    belong to the same window; a larger gap starts a new window. The input
    may arrive in any order; it is sorted ascending internally so the result
    is deterministic.

    Expected input:
        - frame_indices: original-video frame indices (non-negative).
        - max_gap: maximum frame-index distance tolerated inside one window.

    Returns:
        - Inclusive ``(start, end)`` windows in ascending order; ``[]`` for
          empty input; each window has ``end >= start``.

    Raises:
        - ValueError: if ``max_gap`` is not a positive integer or any frame
          index is not a non-negative integer.
    """
    if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 1:
        raise ValueError("max_gap must be a positive integer")
    indices = sorted(frame_indices)
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("frame indices must be non-negative integers")
    if not indices:
        return []
    windows: list[tuple[int, int]] = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index - previous > max_gap:
            windows.append((start, previous))
            start = index
        previous = index
    windows.append((start, previous))
    return windows


def build_window_candidates(
    frames_by_video: dict[str, list[RetrievalCandidate]],
    max_gap: int | None,
) -> tuple[WindowCandidate, ...]:
    """Group each video's retrieved frames into ranked temporal windows.

    One candidate frame set per video is grouped by
    :func:`group_windows`; every resulting window scores as the maximum
    score of its frames and windows are ranked by score (best first,
    stable). This is the single shared implementation used by both QA
    evidence shaping and TRAKE event windows.

    Expected input:
        - frames_by_video: frame candidates grouped by video id (see
          :func:`group_candidates_by_video`).
        - max_gap: maximum frame-index distance inside one window;
          ``None`` disables window generation (empty result).

    Returns:
        - Window candidates ranked by score; empty when disabled or when
          there are no candidates.

    Raises:
        - No exceptions beyond :func:`group_windows` validation.
    """
    if max_gap is None or not frames_by_video:
        return ()
    windows: list[WindowCandidate] = []
    for video_id, frames in frames_by_video.items():
        by_index = sorted(frames, key=lambda candidate: candidate.frame_index)
        indices = [candidate.frame_index for candidate in by_index]
        # Windows are contiguous, non-overlapping ranges over this same
        # sorted list, so a single forward-advancing pointer maps each
        # window to its member candidates in one O(M) pass instead of
        # rescanning the whole per-video list for every window.
        pos = 0
        for start, end in group_windows(indices, max_gap):
            window_start = pos
            while pos < len(by_index) and by_index[pos].frame_index <= end:
                pos += 1
            in_window = by_index[window_start:pos]
            score = max(float(candidate.score) for candidate in in_window)
            windows.append(
                WindowCandidate(
                    video_id=video_id,
                    frame_start=start,
                    frame_end=end,
                    score=score,
                    rank=1,
                )
            )
    ranked = sort_by_score(windows)
    return tuple(replace(window, rank=rank) for rank, window in enumerate(ranked, start=1))


def _top_frame_index(group: Sequence[FrameScoredCandidate]) -> int:
    """Return the frame index of the highest-scoring candidate in ``group``.

    Expected input:
        - group: non-empty frame candidates of one video.

    Returns:
        - Original-video frame index of the candidate with the maximum
          score; ties are resolved by first appearance.

    Raises:
        - No exceptions.
    """
    best = group[0]
    for candidate in group[1:]:
        if candidate.score > best.score:
            best = candidate
    return best.frame_index