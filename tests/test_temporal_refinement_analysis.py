"""Tests for temporal refinement ceiling analysis (Phase 4A)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aic2026.temporal_refinement_analysis import (
    FRAME_HIT,
    FRAME_MISS,
    VIDEO_MISS,
    analyze_query,
    bucket_for_distance,
    derive_ranking,
    distinct_videos_in_order,
    hits_at_k,
    min_distance_to_range,
    oracle_hit_at_video_cutoffs,
    outcome_at_k,
    rerank_by_smoothed,
    refine_frame_selection,
    refine_pooling,
    select_local_peak,
    smoothed_scores,
    summarize_outcomes,
)


def _setup(videos, frames, scores):
    """Return (scores_array, video_ids, frame_indices) in row order."""
    return (
        np.array(scores, dtype=np.float32),
        list(videos),
        list(frames),
    )


def _padded(videos, frames, scores, filler_count=110):
    """Append low-score filler rows so rankings satisfy cutoffs up to 100."""
    v = list(videos)
    f = list(frames)
    s = list(scores)
    base_frame = max(frames, default=0) + 1_000_000
    for i in range(filler_count):
        v.append("VZ_FILLER")
        f.append(base_frame + i)
        s.append(0.001)
    return _setup(v, f, s)


class TestRankingDerivation:
    def test_orders_by_score_desc_with_row_tiebreak(self) -> None:
        scores, _, _ = _setup([], [], [0.1, 0.9, 0.5, 0.9])
        assert derive_ranking(scores).tolist() == [1, 3, 2, 0]

    def test_deterministic(self) -> None:
        scores, _, _ = _setup([], [], [0.3, 0.2, 0.3])
        assert derive_ranking(scores).tolist() == derive_ranking(scores).tolist()


class TestOutcomes:
    VIDEOS = ["VA", "VB", "VA", "VC", "VA"]
    FRAMES = [100, 50, 29300, 70, 400]

    def _ranking(self):
        return np.array([2, 0, 3, 1, 4])  # VA@29300, VA@100, VC, VB, VA@400

    def test_video_hit_frame_hit(self) -> None:
        outcome = outcome_at_k(
            self._ranking(), self.VIDEOS, self.FRAMES, "VA", 29292, 29352, k=1
        )
        assert outcome == FRAME_HIT

    def test_video_hit_frame_miss(self) -> None:
        outcome = outcome_at_k(
            self._ranking(), self.VIDEOS, self.FRAMES, "VA", 30000, 30100, k=5
        )
        assert outcome == FRAME_MISS

    def test_video_miss(self) -> None:
        outcome = outcome_at_k(
            self._ranking(), self.VIDEOS, self.FRAMES, "VZ", 10, 20, k=5
        )
        assert outcome == VIDEO_MISS

    def test_insufficient_rows_rejected(self) -> None:
        with pytest.raises(ValueError, match="ranked rows"):
            outcome_at_k(self._ranking(), self.VIDEOS, self.FRAMES, "VA", 0, 1, k=6)


class TestCandidateVideos:
    def test_first_appearance_order_dedupes(self) -> None:
        rows = [3, 0, 1, 2]
        videos = ["VA", "VC", "VB", "VA"]
        assert distinct_videos_in_order(rows, videos) == ["VA", "VC", "VB"]


class TestDistances:
    def test_inside_is_zero(self) -> None:
        assert min_distance_to_range(150, 100, 200) == 0

    def test_nearest_boundary(self) -> None:
        assert min_distance_to_range(90, 100, 200) == 10
        assert min_distance_to_range(250, 100, 200) == 50

    def test_buckets_fixed_order(self) -> None:
        assert bucket_for_distance(0) == "inside_gt"
        assert bucket_for_distance(50) == "1-50"
        assert bucket_for_distance(51) == "51-100"
        assert bucket_for_distance(999) == "501-1000"
        assert bucket_for_distance(1001) == ">1000"


class TestAnalyzeQuery:
    def test_full_record(self) -> None:
        scores, videos, frames = _padded(
            ["VA", "VB", "VA"], [10, 20, 30], [0.9, 0.8, 0.7]
        )
        ranking = derive_ranking(scores)
        analysis = analyze_query("q1", "VA", 33, 40, ranking, videos, frames)
        # VA is the top row from cutoff 1 onward; no VA frame is in [33, 40].
        assert analysis.outcomes[1] == FRAME_MISS
        assert analysis.outcomes[5] == FRAME_MISS
        assert analysis.gt_video_candidate_index == 1
        # Closest indexed VA frame is 30 (distance 3 below the GT range).
        assert analysis.gt_has_correct_indexed_frame is False
        assert analysis.min_distance == 3
        assert analysis.distance_bucket == "1-50"

    def test_oracle_feasible_when_indexed_frame_exists(self) -> None:
        scores, videos, frames = _padded(
            ["VA", "VB", "VA"], [10, 20, 30], [0.9, 0.8, 0.7]
        )
        ranking = derive_ranking(scores)
        analysis = analyze_query("q1", "VA", 25, 35, ranking, videos, frames)
        assert analysis.gt_has_correct_indexed_frame is True
        oracle = oracle_hit_at_video_cutoffs(analysis)
        assert oracle[1] is True
        assert oracle[100] is True

    def test_oracle_requires_recall(self) -> None:
        scores, videos, frames = _padded(
            ["VB", "VA"], [20, 30], [0.9, 0.8]
        )
        ranking = derive_ranking(scores)
        analysis = analyze_query("q1", "VA", 25, 35, ranking, videos, frames)
        oracle = oracle_hit_at_video_cutoffs(analysis)
        assert oracle[1] is False
        assert oracle[5] is True

    def test_oracle_frame_miss_only_filter(self) -> None:
        scores, videos, frames = _padded(
            ["VA", "VB"], [30, 20], [0.9, 0.8]
        )
        ranking = derive_ranking(scores)
        analysis = analyze_query("q1", "VA", 25, 35, ranking, videos, frames)
        # Baseline already hits at every cutoff, so the filtered oracle is False.
        assert oracle_hit_at_video_cutoffs(analysis, frame_miss_only=True)[1] is False


class TestSmoothing:
    def test_window_mean_with_boundary_clipping(self) -> None:
        scores, _, _ = _setup([], [], [1.0, 2.0, 3.0, 4.0])
        smoothed = smoothed_scores(scores, [0, 1, 2, 3], window=1)
        assert smoothed[0] == pytest.approx(1.5)
        assert smoothed[1] == pytest.approx(2.0)
        assert smoothed[3] == pytest.approx(3.5)

    def test_zero_window_disables_smoothing(self) -> None:
        scores, _, _ = _setup([], [], [1.0, 5.0])
        smoothed = smoothed_scores(scores, [0, 1], window=0)
        assert smoothed[0] == 1.0 and smoothed[1] == 5.0


class TestLocalPeak:
    def test_selects_supported_peak(self) -> None:
        scores, _, _ = _setup([], [], [0.2, 0.9, 0.85])
        assert select_local_peak(scores, [0, 1, 2], window=1) == 1

    def test_plateau_tie_breaks_to_earliest(self) -> None:
        scores, _, _ = _setup([], [], [0.5, 0.5, 0.1])
        assert select_local_peak(scores, [0, 1, 2], window=1) == 0

    def test_invalid_window_rejected(self) -> None:
        scores, _, _ = _setup([], [], [1.0])
        with pytest.raises(ValueError, match="window"):
            select_local_peak(scores, [0], window=0)


class TestRefinements:
    def _world(self):
        # Videos: VA rows {0,1}, VB rows {2,3}, VC row {4}.
        scores, videos, frames = _setup(
            ["VA", "VA", "VB", "VB", "VC"],
            [10, 20, 30, 40, 50],
            [0.80, 0.60, 0.79, 0.10, 0.05],
        )
        baseline = derive_ranking(scores)  # VA@10, VB@30, VA@20, VB@40, VC@50
        return scores, videos, frames, baseline

    def test_pooling_reorders_videos_by_pooled_evidence(self) -> None:
        scores, videos, frames, baseline = self._world()
        result = refine_pooling(
            scores, videos, baseline, pool="top2-mean", max_candidates=2
        )
        # VA pooled top2-mean = 0.70; VB = 0.445 -> VA first, represented by row 0.
        assert result.ranked_rows[0] == 0
        assert result.selected_frame_by_video["VA"] == 0
        assert result.selected_frame_by_video["VB"] == 2

    def test_pooling_max_keeps_order_and_picks_best_frames(self) -> None:
        scores, videos, frames, baseline = self._world()
        result = refine_pooling(scores, videos, baseline, pool="max", max_candidates=2)
        assert result.ranked_rows[:2] == [0, 2]

    def test_unknown_pool_rejected(self) -> None:
        scores, videos, frames, baseline = self._world()
        with pytest.raises(ValueError, match="pooling"):
            refine_pooling(scores, videos, baseline, pool="mean", max_candidates=2)

    def test_reselect_argmax_keeps_video_order(self) -> None:
        scores, videos, frames, baseline = self._world()
        result = refine_frame_selection(
            scores, videos, baseline, method="argmax", window=0, max_candidates=3
        )
        # Video order preserved (VA, VB, VC); each video's best frame chosen.
        assert result.ranked_rows[:3] == [0, 2, 4]

    def test_smooth_tie_breaks_to_earliest_row(self) -> None:
        # VA rows scores [0.80, 0.60]: smoothed w1 -> [0.70, 0.70]; tie breaks
        # toward the earliest row, so selection stays row 0.
        scores, videos, frames, baseline = self._world()
        result = refine_frame_selection(
            scores, videos, baseline, method="smooth-argmax", window=1, max_candidates=1
        )
        assert result.ranked_rows[0] == 0

    def test_non_candidate_rows_appended_after_candidates(self) -> None:
        scores, videos, frames, baseline = self._world()
        result = refine_frame_selection(
            scores, videos, baseline, method="argmax", window=0, max_candidates=1
        )
        assert len(result.ranked_rows) == 5
        assert set(result.ranked_rows) == {0, 1, 2, 3, 4}


class TestRerankBySmoothed:
    def test_preserves_pool_and_reorders(self) -> None:
        # Rows (same video): scores [0.1, 0.9, 0.2]; smoothed w1 ->
        # row0=mean(0.1,0.9)=0.5, row1=mean(0.1,0.9,0.2)=0.4,
        # row2=mean(0.9,0.2)=0.55 -> order [2, 0, 1].
        scores, videos, frames = _setup(["V", "V", "V"], [10, 20, 30], [0.1, 0.9, 0.2])
        pool = [0, 1, 2]
        result = rerank_by_smoothed(scores, videos, pool, window=1)
        assert sorted(result) == sorted(pool)
        assert result == [2, 0, 1]

    def test_never_crosses_video_boundaries(self) -> None:
        # Row 0's neighbors are only within its own video.
        scores, videos, frames = _setup(
            ["VA", "VB", "VA"], [10, 20, 30], [0.0, 0.9, 0.8]
        )
        pool = [0, 1, 2]
        result = rerank_by_smoothed(scores, videos, pool, window=5)
        assert sorted(result) == pool
        assert result[0] == 1  # VB's own high score is untouched by VA


class TestAggregates:
    def test_hits_at_k_and_summary(self) -> None:
        scores, videos, frames = _setup(
            ["VA", "VB"], [100, 50], [0.9, 0.8]
        )
        ranking = derive_ranking(scores)
        hits = hits_at_k(ranking, videos, frames, "VA", 90, 110)
        assert hits == {1: True, 5: True, 20: True, 50: True, 100: True}
        summary = summarize_outcomes({1: Counter({FRAME_HIT: 2, VIDEO_MISS: 1})})
        assert summary[1]["total"] == 3
        assert summary[1][FRAME_HIT] == 2
