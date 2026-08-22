"""Tests for OCR lexical failure analysis (Phase 3B-lite)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aic2026.evaluation import load_eval_set
from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_failure_analysis import (
    AnalysisConfig,
    build_document_frequencies,
    chunk_distance_to_range,
    content_tokens,
    find_phrases,
    normalize_tokens,
    run_failure_analysis,
    shortlist,
    summarize,
)


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


@dataclass(frozen=True)
class _Visual:
    """Minimal visual candidate satisfying the analysis input contract."""

    rank: int
    video_id: str
    frame_index: int


def _eval_query(
    tmp_path: Path,
    task: str,
    video: str,
    start: int,
    end: int,
    text: str = "một câu truy vấn",
    suffix: str = "",
):
    """Write a one-row evaluation CSV and return its canonical query."""
    header = (
        "query_id,task,query_text_vi,query_text_en,gt_video_id,"
        "gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note\n"
    )
    answer = "thịt bò" if task == "qa" else ""
    row = f"{task}_x{suffix}_{video}_{start},{task},{text},,{video},{start},{end},{answer},,P1,\n"
    path = tmp_path / f"eval-{task}-{video}-{start}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + row, encoding="utf-8")
    return load_eval_set(path).queries[0]


CHUNKS = [
    _chunk("V1-ocr-00", "L21_V001", 100, 200, "VinFast VF3 giá bán"),
    _chunk("V1-ocr-01", "L21_V001", 29200, 29400, "bảng giá VinFast hôm nay"),
    _chunk("V1-ocr-far", "L21_V001", 50000, 51000, "VinFast giới thiệu"),
    _chunk("V1-ocr-empty", "L21_V001", 30000, 30100, ""),
    _chunk("V2-ocr-00", "L22_V007", 22900, 23100, "xe tải chở thùng hàng bị lật"),
]


class TestTokenization:
    def test_normalization_preserves_accents_and_lowercases(self) -> None:
        assert normalize_tokens("ÔN THI VẬT LÍ NĂM 2024") == ["ôn", "thi", "vật", "lí", "năm", "2024"]

    def test_stopwords_removed(self) -> None:
        tokens = content_tokens("một chiếc xe tải trên đường")
        assert tokens == ["chiếc", "xe", "tải", "đường"]

    def test_non_string_input_yields_no_tokens(self) -> None:
        assert normalize_tokens(None) == []  # type: ignore[arg-type]


class TestPhraseMatching:
    def test_contiguous_phrase_found(self) -> None:
        assert find_phrases(["xe", "tải", "lật"], ["trọng", "tải", "xe", "tải", "bị", "lật"], min_len=2) == [
            "xe tải"
        ]

    def test_no_partial_single_token_by_default(self) -> None:
        assert find_phrases(["xe"], ["xe"], min_len=2) == []

    def test_longest_match_reported_first_and_stops(self) -> None:
        phrases = find_phrases(
            ["bảng", "giá", "vinfast"], ["bảng", "giá", "vinfast", "hôm", "nay"], min_len=2
        )
        assert phrases == ["bảng giá vinfast"]


class TestDocumentFrequency:
    def test_df_counts_distinct_chunks(self) -> None:
        dfs = build_document_frequencies(CHUNKS)
        assert dfs["vinfast"] == 3
        assert dfs["xe"] == 1
        # Stopwords never enter the DF table.
        assert "trên" not in dfs

    def test_deterministic_ordering(self) -> None:
        terms = list(build_document_frequencies(CHUNKS))
        assert terms == sorted(terms)


class TestTemporalDistance:
    def test_intersecting_ranges_have_zero_distance(self) -> None:
        assert chunk_distance_to_range(CHUNKS[0], 150, 180) == 0
        assert chunk_distance_to_range(CHUNKS[1], 29250, 29350) == 0

    def test_gap_to_nearest_boundary(self) -> None:
        assert chunk_distance_to_range(CHUNKS[0], 250, 300) == 50
        assert chunk_distance_to_range(CHUNKS[2], 40000, 45000) == 5000


def _run(query, candidates, config=None):
    config = config or AnalysisConfig()
    return run_failure_analysis(
        eval_set=_single_query_set(query),
        task=query.task.value,
        visual_ranked_by_query={query.query_id: candidates},
        chunks=CHUNKS,
        config=config,
        visual_top_k=100,
    )


def _single_query_set(query):
    from aic2026.evaluation import EvaluationSet

    return EvaluationSet(queries=(query,))


class TestClassification:
    def test_rare_term_match(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "vinfast giá")
        failures, total = _run(query, [_Visual(1, "L22_V009", 5)])
        assert total == 1
        failure = failures[0]
        assert failure.classification == "RARE_TERM_MATCH"
        assert failure.promising is True
        rare_terms = {t.term for t in failure.matched_terms if t.rare}
        assert "vinfast" in rare_terms

    def test_common_term_only(self, tmp_path: Path) -> None:
        # "giá" and "hôm" both appear in GT OCR but never contiguously; with
        # rare_max_df=0 nothing counts as rare, leaving common-term overlap.
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "giá hôm")
        failures, _ = _run(query, [_Visual(1, "L22_V009", 5)], config=AnalysisConfig(rare_max_df=0))
        assert failures[0].classification == "COMMON_TERM_ONLY"
        assert failures[0].promising is False

    def test_no_lexical_overlap(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "cá heo nhảy sóng")
        failures, _ = _run(query, [_Visual(1, "L22_V009", 5)])
        assert failures[0].classification == "NO_LEXICAL_OVERLAP"

    def test_no_gt_ocr_text(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L99_V099", 10, 20, "vinfast")
        failures, _ = _run(query, [_Visual(1, "L22_V009", 5)])
        assert failures[0].classification == "NO_GT_OCR_TEXT"

    def test_phrase_match_takes_precedence(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L22_V007", 22989, 23065, "xe tải chở thùng hàng")
        failures, _ = _run(query, [_Visual(1, "L21_V002", 5)])
        failure = failures[0]
        assert failure.classification == "PHRASE_OR_STRONG_MATCH"
        # The longest contiguous match is reported.
        assert failure.matched_phrases[0].startswith("xe tải")

    def test_temporal_mismatch_flagged(self, tmp_path: Path) -> None:
        # "giới thiệu" occurs only in the far chunk (frames 50000+), so the
        # lexical evidence exists in the GT video but nowhere near the GT range.
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "giới thiệu")
        failures, _ = _run(query, [_Visual(1, "L22_V009", 5)], config=AnalysisConfig(gt_near_frames=250))
        failure = failures[0]
        assert failure.promising is True
        assert failure.temporal_mismatch is True
        assert failure.temporally_near_overlap_chunks == 0

    def test_temporally_aligned_evidence_not_mismatch(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "bảng giá vinfast")
        failures, _ = _run(query, [_Visual(1, "L22_V009", 5)])
        failure = failures[0]
        assert failure.classification == "PHRASE_OR_STRONG_MATCH"
        assert failure.temporal_mismatch is False
        assert failure.temporally_near_overlap_chunks >= 1

    def test_hit_is_not_a_failure(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "kis", "L21_V001", 29292, 29352, "vinfast")
        failures, total = _run(query, [_Visual(1, "L21_V001", 29300)])
        assert failures == []
        assert total == 1


class TestDeterminismAndSummary:
    def test_output_order_follows_evaluation_order(self, tmp_path: Path) -> None:
        from aic2026.evaluation import EvaluationSet

        q1 = _eval_query(tmp_path / "a", "kis", "L21_V001", 29292, 29352, "vinfast giá")
        q2 = _eval_query(tmp_path / "b", "kis", "L22_V007", 22989, 23065, "xe tải lật")
        eval_set = EvaluationSet(queries=(q1, q2))
        ranked = {
            q1.query_id: [_Visual(1, "L30_V001", 1)],
            q2.query_id: [_Visual(1, "L30_V001", 1)],
        }
        failures, _ = run_failure_analysis(
            eval_set,
            "kis",
            ranked,
            CHUNKS,
            config=AnalysisConfig(),
            visual_top_k=100,
        )
        assert [f.query_id for f in failures] == [q1.query_id, q2.query_id]

    def test_summary_counts(self, tmp_path: Path) -> None:
        from aic2026.evaluation import EvaluationSet

        q1 = _eval_query(tmp_path / "a", "kis", "L21_V001", 29292, 29352, "vinfast giá")
        q2 = _eval_query(tmp_path / "b", "kis", "L21_V001", 29292, 29352, "hôm nay", suffix="-b")
        q3 = _eval_query(tmp_path / "c", "kis", "L21_V001", 29292, 29352, "cá heo nhảy sóng", suffix="-c")
        eval_set = EvaluationSet(queries=(q1, q2, q3))
        ranked = {q.query_id: [_Visual(1, "L30_V001", 1)] for q in (q1, q2, q3)}
        failures, total = run_failure_analysis(
            eval_set, "kis", ranked, CHUNKS, config=AnalysisConfig(), visual_top_k=100
        )
        summary = summarize(failures, total)
        assert summary["total_queries"] == 3
        assert summary["visual_failures"] == 3
        assert summary["failures_rare_term_match"] == 1
        assert summary["failures_phrase_match"] == 1
        assert summary["failures_no_lexical_overlap"] == 1
        assert summary["promising_failures"] == 2

    def test_shortlist_orders_phrase_before_rare(self, tmp_path: Path) -> None:
        from aic2026.evaluation import EvaluationSet

        q_phrase = _eval_query(tmp_path / "p", "kis", "L22_V007", 22989, 23065, "xe tải chở thùng hàng")
        q_rare = _eval_query(tmp_path / "r", "kis", "L21_V001", 29292, 29352, "vinfast giá")
        eval_set = EvaluationSet(queries=(q_rare, q_phrase))
        ranked = {q.query_id: [_Visual(1, "L30_V001", 1)] for q in (q_rare, q_phrase)}
        failures, _ = run_failure_analysis(
            eval_set, "kis", ranked, CHUNKS, config=AnalysisConfig(), visual_top_k=100
        )
        picks = shortlist(failures)
        assert [p.classification for p in picks] == [
            "PHRASE_OR_STRONG_MATCH",
            "RARE_TERM_MATCH",
        ]


class TestQaIntegration:
    def test_qa_failure_analyzed_with_same_rule(self, tmp_path: Path) -> None:
        query = _eval_query(tmp_path, "qa", "L22_V007", 22989, 23065, "nguyên liệu xe tải")
        failures, total = _run(query, [_Visual(1, "L30_V001", 1)])
        assert total == 1
        assert failures[0].task == "qa"
        assert failures[0].classification == "PHRASE_OR_STRONG_MATCH"


class TestConfigValidation:
    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="rare_max_df"):
            AnalysisConfig(rare_max_df=-1)
        with pytest.raises(ValueError, match="min_phrase_tokens"):
            AnalysisConfig(min_phrase_tokens=0)
