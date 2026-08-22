"""Tests for the OCR evaluation reliability audit (Phase 5A)."""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_evaluation_audit import (
    assign_stratum,
    bootstrap_ci,
    chunk_artifact_stats,
    first_rank,
    hit_deltas,
    hits_by_k,
    leave_one_out,
    permutation_test,
    stratum_deltas,
    transition_key,
    transition_matrix,
    win_tie_loss,
)
from aic2026.ocr_failure_analysis import AnalysisConfig


def _chunk(chunk_id: str, video_id: str, start: int, end: int, text: str) -> OcrChunk:
    """Return one OCR chunk with the given identity and text."""
    frames = list(range(start, end + 1))
    return OcrChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        frame_start=start,
        frame_end=end,
        representative_frame_index=frames[len(frames) // 2],
        keyframe_ordinals=tuple(range(1, len(frames) + 1)),
        frame_indices=tuple(frames),
        text=text,
        source_count=len(frames),
    )


class TestFirstRank:
    def test_frame_rule(self) -> None:
        pairs = [("VB", 10), ("VA", 500), ("VA", 120)]
        assert (
            first_rank(pairs, gt_video_id="VA", gt_start=100, gt_end=200) == 3
        )
        assert first_rank(pairs, gt_video_id="VA", gt_start=300, gt_end=400) is None

    def test_video_rule(self) -> None:
        pairs = [("VB", 10), ("VA", 999)]
        assert first_rank(pairs, gt_video_id="VA") == 2
        assert first_rank(pairs, gt_video_id="VZ") is None

    def test_hits_by_k(self) -> None:
        assert hits_by_k(20) == {1: False, 5: False, 20: True, 50: True, 100: True}
        assert hits_by_k(None)[100] is False


class TestWinTieLoss:
    def test_accounting(self) -> None:
        result = win_tie_loss([1, 5, None, 10], [1, 2, 3, None])
        assert result == {"win": 2, "tie": 1, "loss": 1, "n": 4}

    def test_all_ties(self) -> None:
        assert win_tie_loss([1, 1], [1, 1])["tie"] == 2

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            win_tie_loss([1], [1, 2])


class TestPermutationTest:
    def test_deterministic_same_seed(self) -> None:
        deltas = [1.0, 0.0, -1.0, 1.0] * 5
        a = permutation_test(deltas, seed=7, n_resamples=500)
        b = permutation_test(deltas, seed=7, n_resamples=500)
        assert a == b

    def test_all_wins_small_p(self) -> None:
        result = permutation_test([1.0] * 20, seed=1, n_resamples=1000)
        assert result["p_value"] <= 0.001

    def test_all_ties_gives_p_one(self) -> None:
        result = permutation_test([0.0] * 10, seed=1, n_resamples=500)
        assert result["p_value"] == 1.0

    def test_edge_cases(self) -> None:
        empty = permutation_test([])
        assert empty["mean_delta"] is None and empty["n"] == 0
        single = permutation_test([1.0], seed=3, n_resamples=100)
        assert single["p_value"] == 1.0


class TestBootstrapCI:
    def test_deterministic_same_seed(self) -> None:
        deltas = [0.5, -0.5, 0.25, 0.75]
        a = bootstrap_ci(deltas, seed=11, n_resamples=800)
        b = bootstrap_ci(deltas, seed=11, n_resamples=800)
        assert a == b

    def test_ci_brackets_mean_for_symmetric_data(self) -> None:
        result = bootstrap_ci([1.0, -1.0] * 50, seed=5, n_resamples=1000)
        assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]

    def test_edge_cases(self) -> None:
        empty = bootstrap_ci([])
        assert empty["ci_low"] is None and empty["n"] == 0
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci([1.0], confidence=1.5)


class TestLeaveOneOut:
    def test_sensitivity(self) -> None:
        # Deltas [1, 1, 1, -1]: mean 0.5; removing the -1 gives 1.0 (no flip);
        # removing a +1 gives 1/3 > 0 (no flip) -> no sign flips.
        result = leave_one_out([1.0, 1.0, 1.0, -1.0])
        assert result["sign_flips"] is False
        assert result["loo_min"] == pytest.approx(1 / 3)

    def test_removal_to_zero_counts_as_influential(self) -> None:
        # Deltas [1, -1, -1]: mean -1/3 ("hurts"); removing either -1 gives
        # mean 0.0 ("no effect") — the hurt conclusion no longer holds, so
        # those queries are influential.
        result = leave_one_out([1.0, -1.0, -1.0])
        assert result["influential_indices"] == [1, 2]
        assert result["sign_flips"] is True

    def test_single_positive_drives_mean(self) -> None:
        result = leave_one_out([1.0, -0.1, -0.1])
        # Removing the +1 leaves mean -0.1 < 0 -> sign flip.
        assert 0 in result["influential_indices"]
        assert result["sign_flips"] is True

    def test_empty(self) -> None:
        result = leave_one_out([])
        assert result["mean_delta"] is None and result["sign_flips"] is False


class TestTransitions:
    def test_transition_key_labels(self) -> None:
        assert transition_key(False, False, True, True) == "VIDEO_MISS->BOTH"
        assert transition_key(True, False, True, True) == "FRAME_ONLY->BOTH"
        assert transition_key(True, True, True, False) == "BOTH->FRAME_ONLY"
        assert transition_key(True, True, False, False) == "BOTH->VIDEO_MISS"

    def test_matrix_counts(self) -> None:
        matrix = transition_matrix(
            [False, True, True], [False, False, True], [True, True, True], [True, False, True]
        )
        assert matrix == {
            "VIDEO_MISS->BOTH": 1,
            "FRAME_ONLY->FRAME_ONLY": 1,
            "BOTH->BOTH": 1,
        }


class TestStrata:
    CHUNKS = [
        _chunk("c0", "L21_V001", 100, 200, "bánh canh chay ngon"),
        _chunk("c1", "L21_V001", 300, 400, ""),
    ]

    def test_no_gt_ocr_evidence(self) -> None:
        assert (
            assign_stratum("bánh canh", [], {}, AnalysisConfig())
            == "NO_GT_OCR_EVIDENCE"
        )

    def test_gt_ocr_without_overlap(self) -> None:
        dfs = {"bánh": 5, "canh": 5}
        assert (
            assign_stratum("cá heo nhảy", self.CHUNKS, dfs, AnalysisConfig())
            == "GT_OCR_NO_QUERY_OVERLAP"
        )

    def test_strong_lexical_signal_via_rare_term(self) -> None:
        dfs = {"bánh": 3, "canh": 900}
        assert (
            assign_stratum("bánh canh", self.CHUNKS, dfs, AnalysisConfig())
            == "STRONG_LEXICAL_OCR_SIGNAL"
        )

    def test_weak_signal_common_terms_only(self) -> None:
        dfs = {"bánh": 900, "canh": 900}
        # "canh bánh" overlaps both terms but is not contiguous in the chunk.
        assert (
            assign_stratum("canh bánh", self.CHUNKS, dfs, AnalysisConfig(rare_max_df=10))
            == "WEAK_OCR_SIGNAL"
        )


class TestStratumDeltas:
    def test_grouping(self) -> None:
        result = stratum_deltas(["A", "A", "B"], [1.0, -1.0, 0.5])
        assert result == {
            "A": {"n": 2, "mean_delta": 0.0},
            "B": {"n": 1, "mean_delta": 0.5},
        }


class TestChunkArtifactStats:
    def test_basic_distributions(self) -> None:
        chunks = [
            _chunk("a", "V1", 0, 9, "hello world hello world"),
            _chunk("b", "V1", 10, 19, "unique text here"),
        ]
        stats = chunk_artifact_stats(chunks)
        assert stats["num_chunks"] == 2
        assert stats["source_records_per_chunk"]["mean"] == 10.0
        assert stats["frame_span"]["mean"] == 10.0
        assert 0.0 <= stats["fraction_dominated_by_repeated_text"] <= 1.0
        assert stats["vocab_size"] >= 4

    def test_empty(self) -> None:
        assert chunk_artifact_stats([]) == {"num_chunks": 0}

    def test_quantiles_use_sorted_values(self) -> None:
        chunks = [
            _chunk("a", "V1", 0, 9, "aaa"),
            _chunk("b", "V1", 10, 19, "bbb"),
            _chunk("c", "V1", 20, 29, "ccc"),
        ]
        stats = chunk_artifact_stats(chunks)
        # Unsorted lengths would give a nonsense p90; sorted must be monotone.
        assert stats["text_length_chars"]["p50"] <= stats["text_length_chars"]["p90"] <= stats["text_length_chars"]["max"]


class TestHitDeltas:
    def test_values(self) -> None:
        deltas = hit_deltas([True, False, True], [True, True, False])
        assert deltas.tolist() == [0.0, 1.0, -1.0]
