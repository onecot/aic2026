"""Tests for OCR-aware late fusion (Phase 3A)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aic2026.ocr_fusion import (
    FusedCandidate,
    OcrFusionConfig,
    fuse_visual_with_ocr,
    matched_ocr_keywords,
    normalize_scores,
    normalize_scores_minmax,
    normalize_scores_rank,
    query_ocr_weight,
    temporal_alignment,
)


@dataclass(frozen=True)
class _Visual:
    """Minimal visual candidate satisfying the fusion input contract."""

    rank: int
    video_id: str
    frame_index: int
    score: float


@dataclass(frozen=True)
class _Ocr:
    """Minimal OCR result satisfying the fusion input contract."""

    chunk_id: str
    video_id: str
    frame_start: int
    frame_end: int
    score: float
    text: str = ""


class TestTemporalAlignment:
    def test_inside_range_is_maximum(self) -> None:
        assert temporal_alignment(50, 10, 100, scale=500.0) == 1.0
        assert temporal_alignment(10, 10, 100, scale=500.0) == 1.0
        assert temporal_alignment(100, 10, 100, scale=500.0) == 1.0

    def test_decay_outside_range(self) -> None:
        near = temporal_alignment(110, 10, 100, scale=500.0)
        far = temporal_alignment(210, 10, 100, scale=500.0)
        assert 0.0 < far < near < 1.0
        assert near == pytest.approx(1 / (1 + 10 / 500))
        assert far == pytest.approx(1 / (1 + 110 / 500))

    def test_distance_uses_nearest_boundary(self) -> None:
        before = temporal_alignment(0, 10, 100, scale=100.0)
        after = temporal_alignment(110, 10, 100, scale=100.0)
        assert before == pytest.approx(after)

    def test_invalid_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            temporal_alignment(5, 10, 100, scale=0.0)
        with pytest.raises(ValueError, match="range"):
            temporal_alignment(5, 100, 10, scale=1.0)


class TestNormalization:
    def test_minmax_deterministic_and_scaled(self) -> None:
        scores = [3.0, 1.0, 2.0]
        assert normalize_scores_minmax(scores) == [1.0, 0.0, 0.5]
        assert normalize_scores_minmax(scores) == normalize_scores_minmax(scores)

    def test_minmax_degenerate_cases(self) -> None:
        assert normalize_scores_minmax([]) == []
        assert normalize_scores_minmax([2.0]) == [0.5]
        assert normalize_scores_minmax([7.0, 7.0, 7.0]) == [0.5, 0.5, 0.5]

    def test_rank_normalization(self) -> None:
        assert normalize_scores_rank([0.9, 0.1, 0.5]) == [1.0, 0.0, 0.5]
        assert normalize_scores_rank([4.2]) == [1.0]
        assert normalize_scores_rank([]) == []

    def test_dispatcher_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="unknown normalization"):
            normalize_scores([1.0], "zscore")


class TestQueryAdaptiveWeight:
    def test_ocr_heavy_query_gets_higher_weight(self) -> None:
        config = OcrFusionConfig(weight_mode="adaptive")
        visual_query = "một chiếc thuyền chạy trên mặt nước"
        ocr_query = "tên của cửa hàng được ghi trên biển hiệu"
        assert query_ocr_weight(ocr_query, config) > query_ocr_weight(visual_query, config)

    def test_adaptive_saturates_at_max(self) -> None:
        config = OcrFusionConfig(weight_mode="adaptive", base_weight=0.05, max_weight=0.5)
        weight = query_ocr_weight("tên và địa chỉ ghi trên bảng", config)
        assert weight == pytest.approx(0.5)

    def test_fixed_mode_ignores_heuristic(self) -> None:
        config = OcrFusionConfig(weight_mode="fixed", fixed_weight=0.3)
        assert query_ocr_weight("nguyên liệu công thức", config) == 0.3
        assert query_ocr_weight("chiếc thuyền", config) == 0.3

    def test_keyword_matching_is_token_based(self) -> None:
        # "số" must not match inside "nước" or "sống".
        assert matched_ocr_keywords("dòng nước chảy") == ()
        assert "số" in matched_ocr_keywords("ghi số thứ tự")

    def test_multi_word_phrase_matches(self) -> None:
        assert "gọi là" in matched_ocr_keywords("địa danh này được gọi là gì")
        assert matched_ocr_keywords("gọi người lái đò") == ()


class TestFusion:
    def test_no_ocr_preserves_visual_order(self) -> None:
        visual = [_Visual(1, "V1", 10, 0.9), _Visual(2, "V2", 20, 0.8), _Visual(3, "V1", 30, 0.7)]
        fused = fuse_visual_with_ocr(visual, [], config=OcrFusionConfig(), ocr_weight=0.25)
        assert [(f.video_id, f.frame_index) for f in fused] == [("V1", 10), ("V2", 20), ("V1", 30)]
        assert all(f.ocr_support == 0.0 and f.matched_chunk_id is None for f in fused)
        assert [f.fused_score for f in fused] == sorted(
            (f.fused_score for f in fused), reverse=True
        )

    def test_aligned_ocr_boosts_candidate(self) -> None:
        # Three close visual scores give the second candidate a mid-range
        # normalized score that one aligned OCR boost can overtake.
        visual = [
            _Visual(1, "V1", 5000, 0.95),
            _Visual(2, "V1", 200, 0.90),
            _Visual(3, "V1", 6000, 0.89),
        ]
        ocr = [
            _Ocr("V1-ocr-00", "V1", 100, 300, score=0.6),
            _Ocr("V1-ocr-far", "V1", 20000, 30000, score=0.3),
        ]
        config = OcrFusionConfig(weight_mode="fixed", fixed_weight=1.0)
        fused = fuse_visual_with_ocr(visual, ocr, config=config, ocr_weight=1.0)
        assert fused[0].frame_index == 200
        assert fused[0].matched_chunk_id == "V1-ocr-00"
        assert fused[0].temporal_alignment == 1.0
        assert fused[0].ocr_support == pytest.approx(fused[0].ocr_score_norm)

    def test_cross_video_ocr_cannot_boost(self) -> None:
        visual = [_Visual(1, "V1", 150, 0.9), _Visual(2, "V2", 999, 0.1)]
        ocr = [_Ocr("V2-ocr-00", "V2", 900, 1100, score=0.99)]
        fused = fuse_visual_with_ocr(visual, ocr, config=OcrFusionConfig(), ocr_weight=1.0)
        assert fused[0].video_id == "V1"
        assert fused[0].ocr_support == 0.0
        assert fused[1].video_id == "V2"

    def test_strongest_aligned_chunk_wins(self) -> None:
        visual = [_Visual(1, "V1", 250, 0.5)]
        ocr = [
            _Ocr("V1-ocr-far", "V1", 10000, 20000, score=0.90),
            _Ocr("V1-ocr-near", "V1", 200, 300, score=0.85),
            _Ocr("V1-ocr-low", "V1", 200, 300, score=0.40),
        ]
        config = OcrFusionConfig(weight_mode="fixed", temporal_scale=100.0)
        fused = fuse_visual_with_ocr(visual, ocr, config=config, ocr_weight=1.0)
        # The near chunk's slightly lower raw score still yields a much
        # higher normalized-score x alignment product than the far chunk.
        assert fused[0].matched_chunk_id == "V1-ocr-near"

    def test_temporal_disabled_treats_same_video_as_aligned(self) -> None:
        visual = [_Visual(1, "V1", 999999, 0.5)]
        ocr = [_Ocr("V1-ocr-00", "V1", 10, 20, score=0.8)]
        config = OcrFusionConfig(use_temporal_alignment=False, weight_mode="fixed")
        fused = fuse_visual_with_ocr(visual, ocr, config=config, ocr_weight=1.0)
        assert fused[0].temporal_alignment == 1.0
        assert fused[0].ocr_support == pytest.approx(fused[0].ocr_score_norm)

    def test_deterministic_tie_break_by_visual_rank(self) -> None:
        visual = [_Visual(1, "V1", 10, 0.5), _Visual(2, "V2", 10, 0.5)]
        fused_a = fuse_visual_with_ocr(visual, [], config=OcrFusionConfig(), ocr_weight=0.25)
        fused_b = fuse_visual_with_ocr(visual, [], config=OcrFusionConfig(), ocr_weight=0.25)
        assert [(f.rank, f.video_id) for f in fused_a] == [(1, "V1"), (2, "V2")]
        assert [(f.rank, f.video_id) for f in fused_a] == [(f.rank, f.video_id) for f in fused_b]

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="ocr_weight"):
            fuse_visual_with_ocr([_Visual(1, "V1", 0, 1.0)], [], config=OcrFusionConfig(), ocr_weight=-1.0)

    def test_diagnostics_are_complete(self) -> None:
        visual = [_Visual(1, "V1", 150, 0.9)]
        ocr = [_Ocr("V1-ocr-00", "V1", 100, 300, score=0.6)]
        fused = fuse_visual_with_ocr(
            visual, ocr, config=OcrFusionConfig(weight_mode="fixed", fixed_weight=0.25), ocr_weight=0.25
        )
        top = fused[0]
        assert isinstance(top, FusedCandidate)
        assert top.visual_score == 0.9
        # Single-element visual list: min-max degenerate case is the neutral
        # midpoint (documented contract).
        assert top.visual_score_norm == 0.5
        assert top.matched_chunk_id == "V1-ocr-00"
        assert top.ocr_score == 0.6
        # Single-element OCR list: min-max degenerate case is the neutral
        # midpoint (documented contract).
        assert top.ocr_score_norm == 0.5
        assert top.temporal_alignment == 1.0
        assert top.ocr_support == 0.5
        assert top.ocr_weight == 0.25
        assert top.fused_score == pytest.approx(0.5 + 0.25 * 0.5)


class TestConfigValidation:
    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight_mode"):
            OcrFusionConfig(weight_mode="learned")
        with pytest.raises(ValueError, match="temporal_scale"):
            OcrFusionConfig(temporal_scale=-1.0)
        with pytest.raises(ValueError, match="base_weight"):
            OcrFusionConfig(base_weight=0.9, max_weight=0.5)


class TestSyntheticIntegration:
    def test_fusion_changes_top_of_pool(self) -> None:
        visual = [
            _Visual(1, "VA", 5000, 0.95),
            _Visual(2, "VB", 120, 0.90),
            _Visual(3, "VB", 9000, 0.80),
        ]
        ocr = [
            _Ocr("VB-ocr-00", "VB", 50, 200, score=0.95),
            _Ocr("VB-ocr-01", "VB", 8000, 9500, score=0.60),
        ]
        config = OcrFusionConfig(weight_mode="fixed", fixed_weight=0.75)
        fused = fuse_visual_with_ocr(visual, ocr, config=config, ocr_weight=0.75)
        assert fused[0].video_id == "VB"
        assert fused[0].frame_index == 120
        assert fused[0].rank == 1
        # The far VB candidate keeps only a tiny alignment-decayed boost and
        # ranks below the unboosted VA top candidate.
        assert fused[-1].frame_index == 9000
