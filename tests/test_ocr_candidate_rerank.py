"""Focused tests for bounded OCR candidate reranking (Phase 5C-A)."""

from __future__ import annotations

import pytest

from aic2026.ocr_candidate_rerank import (
    CandidateRerankConfig,
    RerankedCandidate,
    VideoOcrEvidence,
    rerank_candidates,
    strict_frame_rerank,
    video_aware_rerank,
)
from aic2026.retrieval import RetrievalCandidate


def candidate(rank: int, video_id: str, frame_index: int, score: float) -> RetrievalCandidate:
    """Build one synthetic visual retrieval candidate."""
    return RetrievalCandidate(
        rank=rank,
        video_id=video_id,
        keyframe_ordinal=frame_index,
        frame_index=frame_index,
        pts_time=float(frame_index),
        score=score,
    )


def evidence(video_id: str, chunk_id: str, raw_score: float) -> VideoOcrEvidence:
    """Build one synthetic per-video OCR evidence entry."""
    return VideoOcrEvidence(video_id=video_id, chunk_id=chunk_id, raw_score=raw_score)


STRICT = CandidateRerankConfig(policy="strict_frame", pool_size=100, ocr_weight=0.3)
VIDEO = CandidateRerankConfig(policy="video_aware", pool_size=100, ocr_weight=0.3)


class TestStrictFrameRerank:
    def test_membership_never_expands_or_shrinks(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.50),
            candidate(2, "L22_V002", 20, 0.45),
            candidate(3, "L21_V001", 30, 0.40),
        ]
        evidence_map = {
            "L21_V001": evidence("L21_V001", "chunk-a", 0.9),
            "L22_V002": evidence("L22_V002", "chunk-b", 0.2),
        }
        out = strict_frame_rerank(candidates, config=STRICT, evidence_by_video=evidence_map)
        assert sorted((c.video_id, c.frame_index) for c in out) == sorted(
            (c.video_id, c.frame_index) for c in candidates
        )
        assert len(out) == len(candidates)
        assert [c.rank for c in out] == [1, 2, 3]

    def test_zero_ocr_weight_preserves_visual_order(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.50),
            candidate(2, "L22_V002", 20, 0.45),
            candidate(3, "L23_V003", 30, 0.10),
        ]
        evidence_map = {
            "L22_V002": evidence("L22_V002", "chunk-b", 0.99),
            "L23_V003": evidence("L23_V003", "chunk-c", 0.01),
        }
        zero = CandidateRerankConfig(policy="strict_frame", pool_size=100, ocr_weight=0.0)
        out = strict_frame_rerank(candidates, config=zero, evidence_by_video=evidence_map)
        assert [(c.video_id, c.frame_index) for c in out] == [
            (c.video_id, c.frame_index) for c in candidates
        ]

    def test_ties_break_by_original_visual_rank(self) -> None:
        # Equal visual scores and equal evidence => equal fused scores.
        candidates = [
            candidate(1, "L22_V002", 20, 0.5),
            candidate(2, "L21_V001", 10, 0.5),
            candidate(3, "L23_V003", 30, 0.5),
        ]
        evidence_map = {
            v: evidence(v, f"chunk-{v}", 0.7)
            for v in ("L21_V001", "L22_V002", "L23_V003")
        }
        out = strict_frame_rerank(candidates, config=STRICT, evidence_by_video=evidence_map)
        assert [c.visual_rank for c in out] == [1, 2, 3]

    def test_no_evidence_videos_get_zero_support(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.4),
            candidate(2, "L22_V002", 20, 0.39),
        ]
        evidence_map = {"L21_V001": evidence("L21_V001", "chunk-a", 0.8)}
        out = strict_frame_rerank(candidates, config=STRICT, evidence_by_video=evidence_map)
        by_video = {c.video_id: c for c in out}
        assert by_video["L21_V001"].ocr_support > 0.0
        assert by_video["L22_V002"].ocr_support == 0.0
        assert by_video["L22_V002"].matched_chunk_id is None


class TestVideoAwareRerank:
    def test_ocr_cannot_introduce_videos_outside_pool(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.5),
            candidate(2, "L22_V002", 20, 0.4),
        ]
        # Evidence for an out-of-pool video with a huge score must be ignored.
        evidence_map = {
            "L99_V099": evidence("L99_V099", "chunk-x", 0.999),
            "L21_V001": evidence("L21_V001", "chunk-a", 0.1),
        }
        out = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        assert set(c.video_id for c in out) == {"L21_V001", "L22_V002"}

    def test_frames_of_one_video_keep_visual_order(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 300, 0.5),
            candidate(2, "L21_V001", 100, 0.48),
            candidate(3, "L21_V001", 200, 0.46),
            candidate(4, "L22_V002", 50, 0.44),
        ]
        evidence_map = {"L21_V001": evidence("L21_V001", "chunk-a", 0.95)}
        out = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        l21_positions = [c.frame_index for c in out if c.video_id == "L21_V001"]
        assert l21_positions == [300, 100, 200]
        assert out[0].video_id == "L21_V001"

    def test_pool_frames_and_videos_are_exactly_preserved(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.5),
            candidate(2, "L22_V002", 20, 0.45),
            candidate(3, "L23_V003", 30, 0.40),
            candidate(4, "L22_V002", 40, 0.35),
        ]
        evidence_map = {"L22_V002": evidence("L22_V002", "chunk-b", 0.8)}
        out = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        assert sorted((c.video_id, c.frame_index) for c in out) == sorted(
            (c.video_id, c.frame_index) for c in candidates
        )

    def test_zero_ocr_weight_reproduces_input_order(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.5),
            candidate(2, "L22_V002", 20, 0.45),
            candidate(3, "L21_V001", 30, 0.40),
            candidate(4, "L23_V003", 40, 0.30),
        ]
        evidence_map = {
            "L23_V003": evidence("L23_V003", "chunk-c", 0.99),
            "L21_V001": evidence("L21_V001", "chunk-a", 0.9),
        }
        zero = CandidateRerankConfig(policy="video_aware", pool_size=100, ocr_weight=0.0)
        out = video_aware_rerank(candidates, config=zero, evidence_by_video=evidence_map)
        # Zero weight: videos ordered by their best visual score (L21, L22,
        # L23) and frames inside each video keep visual order.
        assert [(c.video_id, c.frame_index) for c in out] == [
            ("L21_V001", 10),
            ("L21_V001", 30),
            ("L22_V002", 20),
            ("L23_V003", 40),
        ]

    def test_identical_ocr_scores_are_deterministic(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.5),
            candidate(2, "L22_V002", 20, 0.45),
        ]
        evidence_map = {
            v: evidence(v, f"chunk-{v}", 0.77) for v in ("L21_V001", "L22_V002")
        }
        first = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        second = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        assert [(c.video_id, c.combined_score) for c in first] == [
            (c.video_id, c.combined_score) for c in second
        ]
        # Equal normalized OCR support; visual part decides, ties keep order.
        assert first[0].video_id == "L21_V001"
        assert first[0].ocr_support == pytest.approx(first[1].ocr_support)

    def test_no_evidence_video_still_ranked_by_visual_score(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.9),
            candidate(2, "L22_V002", 20, 0.1),
        ]
        evidence_map = {"L22_V002": evidence("L22_V002", "chunk-b", 0.99)}
        out = video_aware_rerank(candidates, config=VIDEO, evidence_by_video=evidence_map)
        # minmax visual: L21=1.0, L22=0.0; supports: L21=0.0, L22=1.0.
        # fused: L21=1.0 vs L22=0.0 + 0.3*1.0 = 0.3 => L21 stays first.
        assert out[0].video_id == "L21_V001"
        assert out[0].combined_score == pytest.approx(1.0)


class TestNormalizationAndValidation:
    def test_normalization_is_deterministic_minmax(self) -> None:
        candidates = [
            candidate(1, "L21_V001", 10, 0.2),
            candidate(2, "L22_V002", 20, 0.4),
            candidate(3, "L23_V003", 30, 0.6),
        ]
        out = strict_frame_rerank(candidates, config=STRICT, evidence_by_video={})
        scores = {c.video_id: c.combined_score for c in out}
        # minmax of (0.2, 0.4, 0.6) = (0.0, 0.5, 1.0); no OCR anywhere.
        assert scores["L23_V003"] == pytest.approx(1.0)
        assert scores["L22_V002"] == pytest.approx(0.5)
        assert scores["L21_V001"] == pytest.approx(0.0)

    def test_single_candidate_pool_is_valid(self) -> None:
        out_strict = strict_frame_rerank([candidate(1, "L21_V001", 10, 0.5)], config=STRICT, evidence_by_video={})
        out_video = video_aware_rerank([candidate(1, "L21_V001", 10, 0.5)], config=VIDEO, evidence_by_video={})
        assert len(out_strict) == len(out_video) == 1

    def test_invalid_policy_and_weight_rejected(self) -> None:
        with pytest.raises(ValueError):
            CandidateRerankConfig(policy="full_corpus_fusion", pool_size=100, ocr_weight=0.3)
        with pytest.raises(ValueError):
            CandidateRerankConfig(policy="strict_frame", pool_size=100, ocr_weight=-0.1)
        with pytest.raises(ValueError):
            CandidateRerankConfig(policy="strict_frame", pool_size=0, ocr_weight=0.3)

    def test_dispatch_matches_direct_calls(self) -> None:
        candidates = [candidate(1, "L21_V001", 10, 0.5), candidate(2, "L22_V002", 20, 0.4)]
        evidence_map = {"L21_V001": evidence("L21_V001", "chunk-a", 0.5)}
        via_dispatch = rerank_candidates(
            candidates, config=CandidateRerankConfig(policy="video_aware", pool_size=10, ocr_weight=0.3),
            evidence_by_video=evidence_map,
        )
        direct = video_aware_rerank(
            candidates, config=CandidateRerankConfig(policy="video_aware", pool_size=10, ocr_weight=0.3),
            evidence_by_video=evidence_map,
        )
        assert [c == d for c, d in zip(via_dispatch, direct)]

    def test_unordered_candidates_rejected(self) -> None:
        candidates = [candidate(2, "L21_V001", 10, 0.5), candidate(1, "L22_V002", 20, 0.4)]
        with pytest.raises(ValueError):
            strict_frame_rerank(candidates, config=STRICT, evidence_by_video={})

    def test_mismatched_evidence_video_rejected(self) -> None:
        bad = {"L21_V001": evidence("L22_V002", "chunk-a", 0.5)}
        with pytest.raises(ValueError):
            strict_frame_rerank([candidate(1, "L21_V001", 10, 0.5)], config=STRICT, evidence_by_video=bad)


def test_reranked_candidate_fields_present() -> None:
    out = strict_frame_rerank(
        [candidate(1, "L21_V001", 10, 0.5)],
        config=STRICT,
        evidence_by_video={"L21_V001": evidence("L21_V001", "chunk-a", 0.9)},
    )
    item = out[0]
    assert isinstance(item, RerankedCandidate)
    assert item.matched_chunk_id == "chunk-a"
    assert item.visual_rank == 1 and item.rank == 1
    assert item.ocr_support == pytest.approx(0.5)  # single bearing value -> minmax midpoint
