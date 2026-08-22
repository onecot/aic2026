"""Focused tests for query-routed OCR evidence reranking (Phase 5D)."""

from __future__ import annotations

import pytest

from aic2026.ocr_evidence import (
    LexicalEvidence,
    RoutedRerankConfig,
    best_video_evidence,
    document_frequencies_from_texts,
    evaluate_chunk_evidence,
    passes_gate,
    routed_candidate_rerank,
    strip_punctuation,
)
from aic2026.ocr_failure_analysis import AnalysisConfig
from aic2026.ocr_query_routing import matched_signals, route_query
from aic2026.retrieval import RetrievalCandidate


def chunk(chunk_id: str, video_id: str, text: str, start: int = 0, end: int = 100):
    """Build one synthetic cleaned-chunk-like object."""
    from types import SimpleNamespace

    return SimpleNamespace(
        chunk_id=chunk_id, video_id=video_id, frame_start=start, frame_end=end, text=text
    )


def candidate(rank: int, video_id: str, frame_index: int, score: float) -> RetrievalCandidate:
    """Build one synthetic visual retrieval candidate."""
    return RetrievalCandidate(
        rank=rank, video_id=video_id, keyframe_ordinal=frame_index,
        frame_index=frame_index, pts_time=float(frame_index), score=score,
    )


DF = {"mì": 3, "căn": 2, "xốt": 4, "nấm": 5, "cá": 8, "kèo": 1, "chiên": 6, "mè": 2,
      "thanhnien": 40, "htv": 60, "2024": 90, "lion": 12, "dance": 15}
P1 = RoutedRerankConfig(pool_size=100, gate="G1")
P2 = RoutedRerankConfig(pool_size=100, gate="G2")


class TestRouter:
    def test_visual_route_when_no_signals(self) -> None:
        route = route_query("Tìm một video về màn múa lân màu đỏ đang biểu diễn trên các cọc cao")
        assert route.route == "VISUAL"
        assert route.signals == ()
        assert "no textual-intent signal" in route.reason

    def test_mixed_route_for_recipe_scene_query(self) -> None:
        route = route_query("Trong video hướng dẫn làm cua rang xốt tắc, đầu bếp đang thái nguyên liệu màu đỏ")
        assert route.route == "MIXED"
        assert "hướng dẫn làm" in route.signals and "nguyên liệu" in route.signals

    def test_ocr_text_route_for_pure_text_query(self) -> None:
        route = route_query("Chữ trên biển hiệu ghi gì?")
        assert route.route == "OCR_TEXT"

    def test_matched_signals_deterministic_and_punctuation_safe(self) -> None:
        first = matched_signals("Tiêu đề, bảng tên món ăn viết gì?")
        second = matched_signals("tiêu đề bảng tên món ăn viết gì")
        assert first == second
        assert "tiêu đề" in first and "tên món" in first

    def test_bare_mon_does_not_trigger_route(self) -> None:
        assert matched_signals("Tìm video về món phở") == ()

    def test_router_to_dict_shape(self) -> None:
        payload = route_query("công thức nấu ăn").to_dict()
        assert set(payload) == {"route", "reason", "signals"}
        assert payload["signals"] == ["công thức"] or payload["signals"] == ["công thức", "món ăn"]


class TestEvidence:
    def test_exact_phrase_detection_case_and_diacritics(self) -> None:
        c = chunk("c1", "L26_V142", "MÌ CĂN XỐT NẤM #NấuChuẩnĂnLành")
        ev = evaluate_chunk_evidence("đĩa mì căn chay xốt nấm", c, DF)
        assert ev is not None
        assert ev.tier >= 2 and passes_gate(ev, "G1")

    def test_partial_phrase_is_g1_not_full(self) -> None:
        c = chunk("c1", "L26_V158", "CÁ KÈO CHIÊN MÈ #NauChuanAnLanh")
        ev = evaluate_chunk_evidence("cá kèo lăn qua mè vào chảo dầu", c, DF)
        assert ev is not None
        assert ev.tier_name == "PARTIAL_PHRASE" and ev.matched_phrase == "cá kèo"
        assert passes_gate(ev, "G1")

    def test_rare_overlap_only_passes_g2(self) -> None:
        # No contiguous phrase; two overlapping tokens, 'kèo' rare (df=1).
        c = chunk("c1", "L25_V009", "kèo nhà cái hôm nay")
        ev = evaluate_chunk_evidence("cá kèo chiên giòn", c, DF)
        assert ev is None or not passes_gate(ev, "G1")
        if ev is not None:
            assert ev.tier_name == "RARE_OVERLAP" and passes_gate(ev, "G2")

    def test_common_only_overlap_produces_no_evidence(self) -> None:
        dfs = {"video": 5000, "tìm": 4000}
        c = chunk("c1", "L21_V001", "video tìm kiếm thông tin")
        assert evaluate_chunk_evidence("tìm video hay", c, dfs) is None

    def test_single_token_query_never_matches(self) -> None:
        assert evaluate_chunk_evidence("nấm", chunk("c1", "L1_V1", "nấm xốt"), DF) is None

    def test_locality_tokens_across_chunks_do_not_combine(self) -> None:
        # 'mì' only in chunk A, 'căn' only in chunk B; each chunk alone has
        # < 2 overlaps with the query => no evidence from either.
        a = chunk("a", "V", "mì que pate")
        b = chunk("b", "V", "bún căn hộ")
        assert evaluate_chunk_evidence("mì căn", a, DF) is None
        assert evaluate_chunk_evidence("mì căn", b, DF) is None

    def test_best_video_evidence_prefers_strongest_tier(self) -> None:
        weak = chunk("w", "V", "cá hồi áp chảo")
        strong = chunk("s", "V", "CÁ KÈO CHIÊN MÈ ngon")
        best = best_video_evidence("cá kèo chiên mè", [weak, strong], DF)
        assert best is not None and best.chunk_id == "s"

    def test_document_frequencies_from_texts(self) -> None:
        dfs = document_frequencies_from_texts(["mì căn xốt", "mì bò", "bò kho"])
        assert dfs["mì"] == 2 and dfs["bò"] == 2 and dfs["căn"] == 1

    def test_strip_punctuation(self) -> None:
        assert strip_punctuation("Chữ ghi gì? Nói, đi!") == "chữ ghi gì nói đi"


class TestRerankPolicy:
    def evidence(self, tier: int, video_id: str, phrase: str | None = None) -> LexicalEvidence:
        return LexicalEvidence(
            chunk_id=f"{video_id}-c", video_id=video_id, frame_start=10, frame_end=50,
            tier=tier, tier_name={3: "FULL_PHRASE", 2: "PARTIAL_PHRASE", 1: "RARE_OVERLAP"}[tier],
            matched_phrase=phrase, overlap_tokens=("a", "b"), rare_tokens=("b",),
            min_rare_df=2, gate="G1" if tier >= 2 else "G2",
            chunk_text="text",
        )

    def pool(self) -> list[RetrievalCandidate]:
        return [
            candidate(1, "VA", 300, 0.5),   # VA frames ranks 1,3
            candidate(2, "VB", 200, 0.48),
            candidate(3, "VA", 100, 0.46),
            candidate(4, "VC", 50, 0.44),
        ]

    def test_visual_queries_untouched_via_disabled_gate(self) -> None:
        out = routed_candidate_rerank(self.pool(), config=RoutedRerankConfig(pool_size=100, gate=None),
                                      evidence_by_video={"VC": self.evidence(3, "VC")})
        assert [(c.video_id, c.frame_index) for c in out] == [
            (c.video_id, c.frame_index) for c in self.pool()
        ]
        assert all(not c.ocr_supported for c in out)

    def test_out_of_pool_videos_cannot_appear(self) -> None:
        out = routed_candidate_rerank(self.pool(), config=P2,
                                      evidence_by_video={"VD": self.evidence(3, "VD")})
        assert {c.video_id for c in out} <= {"VA", "VB", "VC"}
        assert all(not c.ocr_supported for c in out)

    def test_tier_ordering_with_visual_rank_inside_tiers(self) -> None:
        evidence_map = {
            "VC": self.evidence(3, "VC", "full"),
            "VB": self.evidence(2, "VB", "part"),
        }
        out = routed_candidate_rerank(self.pool(), config=P2, evidence_by_video=evidence_map)
        order = [(c.video_id, c.visual_rank) for c in out]
        assert order[0] == ("VC", 4)
        assert order[1] == ("VB", 2)
        assert order[2:] == [("VA", 1), ("VA", 3)]
        assert [c.frame_index for c in out if c.video_id == "VA"] == [300, 100]

    def test_unsupported_candidates_keep_relative_order(self) -> None:
        out = routed_candidate_rerank(self.pool(), config=P1, evidence_by_video={})
        visual_ranks = [c.visual_rank for c in out]
        assert visual_ranks == sorted(visual_ranks)

    def test_gate_g1_blocks_tier1_but_g2_accepts(self) -> None:
        evidence_map = {"VC": self.evidence(1, "VC")}
        out_p1 = routed_candidate_rerank(self.pool(), config=P1, evidence_by_video=evidence_map)
        out_p2 = routed_candidate_rerank(self.pool(), config=P2, evidence_by_video=evidence_map)
        assert all(not c.ocr_supported for c in out_p1)
        assert all(c.ocr_supported for c in out_p2 if c.video_id == "VC")

    def test_deterministic_repeated_calls(self) -> None:
        evidence_map = {"VB": self.evidence(2, "VB")}
        first = routed_candidate_rerank(self.pool(), config=P2, evidence_by_video=evidence_map)
        second = routed_candidate_rerank(self.pool(), config=P2, evidence_by_video=evidence_map)
        assert [(c.video_id, c.rank, c.frame_index) for c in first] == [
            (c.video_id, c.rank, c.frame_index) for c in second
        ]

    def test_unordered_candidates_rejected(self) -> None:
        with pytest.raises(ValueError):
            routed_candidate_rerank([candidate(2, "VA", 1, 0.5), candidate(1, "VB", 2, 0.4)],
                                    config=P1, evidence_by_video={})

    def test_mismatched_evidence_video_rejected(self) -> None:
        bad = LexicalEvidence(chunk_id="x", video_id="OTHER", frame_start=0, frame_end=1,
                              tier=3, tier_name="FULL_PHRASE", matched_phrase=None,
                              overlap_tokens=(), rare_tokens=(), min_rare_df=None,
                              gate="G1", chunk_text="t")
        with pytest.raises(ValueError):
            routed_candidate_rerank(self.pool(), config=P1, evidence_by_video={"VA": bad})

    def test_invalid_configs_rejected(self) -> None:
        with pytest.raises(ValueError):
            RoutedRerankConfig(pool_size=0, gate="G1")
        with pytest.raises(ValueError):
            RoutedRerankConfig(pool_size=100, gate="G3")

    def test_phase5c_regression_cases_have_no_evidence(self) -> None:
        # Ticker timestamps vs boat query (kis_100 regression source).
        ticker = chunk("t", "L21_V001",
                       "06:50:45 Now Copenhagen attractions promote green actions giám đốc giáo dục")
        assert evaluate_chunk_evidence(
            "thuyền chở nhiều du khách đang di chuyển trên sông", ticker,
            {"now": 30, "actions": 20}) is None
        # Garbage OCR vs lion-dance query (kis_107 regression source).
        garbage = chunk("g", "L24_V044", "my HAUAM 19 10 DAITRUTEN U DAI TRUVÉN H D")
        assert evaluate_chunk_evidence("màn múa lân màu đỏ biểu diễn trên cọc cao", garbage,
                                       {"dai": 7, "truvén": 3}) is None
        # Watermark-only OCR gives no evidence for 'lion dance'.
        watermark = chunk("wm", "L25_V085", "THANHNIEN 2024 HTV")
        assert evaluate_chunk_evidence("lion dance show", watermark,
                                       {"thanhnien": 40, "htv": 60, "2024": 90}) is None


def test_analysis_config_defaults_are_frozen() -> None:
    cfg = AnalysisConfig()
    assert cfg.rare_max_df == 10 and cfg.min_phrase_tokens == 2
