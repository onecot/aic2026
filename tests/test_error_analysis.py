"""Tests for the read-only KIS error analysis utility.

These tests exercise classification, per-query analysis, and the summary
report without running any retrieval or model code.
"""

from __future__ import annotations

from aic2026.canonical import VideoId
from aic2026.error_analysis import (
    CATEGORY_FAIL,
    CATEGORY_HIT,
    CATEGORY_NEAR_VIDEO,
    QueryErrorAnalysis,
    analyze_kis_errors,
    analyze_query,
    classify_query,
    format_report,
    frame_distance_to_interval,
    select_analyses,
    summarize,
)
from aic2026.evaluation import (
    AnnotationMetadata,
    EvaluationQuery,
    EvaluationTask,
    FrameRange,
)
from aic2026.kis_scoring import KisPrediction


def _query(
    query_id: str = "kis_001",
    video_id: str = "L21_V001",
    start: int = 100,
    end: int = 120,
) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        task=EvaluationTask.KIS,
        query_text_vi="T\u00ecm c\u1ea3nh ki\u1ec3m th\u1eed.",
        query_text_en="Find the test scene.",
        ground_truth_video_id=VideoId(video_id),
        ground_truth_frame_range=FrameRange(start, end),
        answer=None,
        event=None,
        annotation=AnnotationMetadata(annotator="P1", note=None),
    )


def _prediction(rank: int, video_id: str, frame_index: int) -> KisPrediction:
    return KisPrediction(rank=rank, video_id=video_id, frame_index=frame_index)


def _scores(*tuples: tuple[str, int, float]) -> dict[tuple[str, int], float]:
    return {(video_id, frame_index): score for video_id, frame_index, score in tuples}


def test_frame_distance_to_interval() -> None:
    assert frame_distance_to_interval(100, 100, 120) == 0
    assert frame_distance_to_interval(120, 100, 120) == 0
    assert frame_distance_to_interval(99, 100, 120) == 1
    assert frame_distance_to_interval(121, 100, 120) == 1
    assert frame_distance_to_interval(50, 100, 120) == 50


def test_classify_query_categories() -> None:
    hit = [_prediction(1, "L21_V001", 105)]
    near = [_prediction(1, "L21_V001", 99), _prediction(2, "L22_V002", 50)]
    fail = [_prediction(1, "L22_V002", 50)]

    assert classify_query(hit, "L21_V001", 100, 120) == CATEGORY_HIT
    assert classify_query(near, "L21_V001", 100, 120) == CATEGORY_NEAR_VIDEO
    assert classify_query(fail, "L21_V001", 100, 120) == CATEGORY_FAIL
    assert classify_query([], "L21_V001", 100, 120) == CATEGORY_FAIL


def test_analyze_query_hit_includes_scores_and_first_gt_rank() -> None:
    predictions = [
        _prediction(1, "L21_V001", 105),
        _prediction(2, "L21_V001", 99),
        _prediction(3, "L22_V002", 50),
    ]
    scores = _scores(("L21_V001", 105, 0.9), ("L21_V001", 99, 0.7), ("L22_V002", 50, 0.6))

    analysis = analyze_query(_query(), predictions, scores=scores, top_k=100)

    assert analysis.category == CATEGORY_HIT
    assert analysis.first_gt_video_rank == 1
    assert analysis.closest_frame_distance == 0
    assert analysis.query_id == "kis_001"
    assert analysis.query_text_en == "Find the test scene."
    assert analysis.gt_video_id == "L21_V001"
    assert analysis.gt_frame_start == 100
    assert analysis.gt_frame_end == 120
    assert analysis.top5[0] == (1, "L21_V001", 105, 0.9)
    assert len(analysis.top5) == 3


def test_analyze_query_near_video_computes_closest_boundary_distance() -> None:
    predictions = [
        _prediction(1, "L22_V002", 50),
        _prediction(2, "L21_V001", 99),
        _prediction(3, "L21_V001", 150),
    ]

    analysis = analyze_query(_query(), predictions)

    assert analysis.category == CATEGORY_NEAR_VIDEO
    assert analysis.first_gt_video_rank == 2
    # frame 99 -> distance 1 to start; frame 150 -> distance 30 to end; min = 1
    assert analysis.closest_frame_distance == 1
    assert len(analysis.top5) == 3


def test_analyze_query_fail_has_no_gt_rank_or_distance() -> None:
    predictions = [_prediction(1, "L22_V002", 50), _prediction(2, "L22_V002", 60)]

    analysis = analyze_query(_query(), predictions)

    assert analysis.category == CATEGORY_FAIL
    assert analysis.first_gt_video_rank is None
    assert analysis.closest_frame_distance is None


def test_top_k_truncation_and_score_lookup() -> None:
    predictions = [
        _prediction(rank, "L22_V002", rank) for rank in range(1, 11)
    ]
    scores = _scores(("L22_V002", 3, 0.42))

    analysis = analyze_query(_query(), predictions, scores=scores, top_k=3)

    assert len(analysis.top5) == 3
    assert [rank for rank, _video, _frame, _score in analysis.top5] == [1, 2, 3]
    assert analysis.top5[2][3] == 0.42
    assert analysis.top5[0][3] != analysis.top5[0][3]  # missing score renders as nan


def test_analyze_kis_errors_preserves_query_order() -> None:
    first = _query("kis_001", video_id="L21_V001")
    second = _query("kis_002", video_id="L22_V002")
    predictions_by_query = {
        "kis_001": [_prediction(1, "L21_V001", 105)],
        "kis_002": [_prediction(1, "L22_V001", 5)],
    }

    analyses = analyze_kis_errors([first, second], predictions_by_query)

    assert [analysis.query_id for analysis in analyses] == ["kis_001", "kis_002"]
    assert analyses[0].category == CATEGORY_HIT
    assert analyses[1].category == CATEGORY_FAIL


def test_analyze_kis_errors_accepts_empty_predictions_mapping() -> None:
    analysis = analyze_kis_errors([_query()], {})[0]

    assert analysis.category == CATEGORY_FAIL
    assert analysis.first_gt_video_rank is None
    assert analysis.top5 == ()


def test_summarize_counts_and_statistics() -> None:
    analyses = [
        QueryErrorAnalysis(
            category=CATEGORY_HIT,
            query_id="kis_001",
            query_text_en="a",
            query_text_vi="b",
            gt_video_id="L21_V001",
            gt_frame_start=100,
            gt_frame_end=120,
            first_gt_video_rank=1,
            closest_frame_distance=0,
            top5=(),
        ),
        QueryErrorAnalysis(
            category=CATEGORY_NEAR_VIDEO,
            query_id="kis_002",
            query_text_en="a",
            query_text_vi="b",
            gt_video_id="L21_V001",
            gt_frame_start=100,
            gt_frame_end=120,
            first_gt_video_rank=7,
            closest_frame_distance=3,
            top5=(),
        ),
        QueryErrorAnalysis(
            category=CATEGORY_FAIL,
            query_id="kis_003",
            query_text_en="a",
            query_text_vi="b",
            gt_video_id="L21_V001",
            gt_frame_start=100,
            gt_frame_end=120,
            first_gt_video_rank=None,
            closest_frame_distance=None,
            top5=(),
        ),
    ]

    summary = summarize(analyses, top_k_values=(5, 100))

    assert summary.total == 3
    assert summary.hits == 1
    assert summary.near_video == 1
    assert summary.fails == 1
    assert dict(summary.gt_video_in_top_k) == {5: 33.33333333333333, 100: 66.66666666666666}
    assert summary.avg_first_gt_video_rank == (1 + 7) / 2
    assert summary.avg_near_frame_distance == 3.0


def test_format_report_contains_query_blocks_and_summary() -> None:
    analysis = analyze_query(
        _query(),
        [_prediction(1, "L21_V001", 105)],
        scores=_scores(("L21_V001", 105, 0.9)),
    )
    report = format_report([analysis], summarize([analysis]))

    assert "[HIT] kis_001" in report
    assert "query_text_en: Find the test scene." in report
    assert "ground_truth: video=L21_V001 frames=[100,120]" in report
    assert "rank=1 video=L21_V001 frame=105 score=0.9000" in report
    assert "HIT: 1 / 1" in report
    assert "NEAR_VIDEO: 0 / 1" in report
    assert "FAIL: 0 / 1" in report
    assert "GT video in top-5:" in report
    assert "avg rank of first GT video occurrence: 1.00" in report


def test_analyze_query_rejects_non_kis_query() -> None:
    non_kis = EvaluationQuery(
        query_id="qa_001",
        task=EvaluationTask.QA,
        query_text_vi="Question?",
        query_text_en=None,
        ground_truth_video_id=VideoId("L21_V001"),
        ground_truth_frame_range=FrameRange(100, 120),
        answer="42",
        event=None,
        annotation=AnnotationMetadata(annotator="P1", note=None),
    )
    try:
        analyze_query(non_kis, [])
    except ValueError as exc:
        assert "kis" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-KIS query")


def _analysis(
    category: str,
    query_id: str,
    first_rank: int | None,
    distance: int | None,
) -> QueryErrorAnalysis:
    return QueryErrorAnalysis(
        category=category,
        query_id=query_id,
        query_text_en="en",
        query_text_vi="vi",
        gt_video_id="L21_V001",
        gt_frame_start=100,
        gt_frame_end=120,
        first_gt_video_rank=first_rank,
        closest_frame_distance=distance,
        top5=(),
    )


def _sample_analyses() -> list[QueryErrorAnalysis]:
    return [
        _analysis(CATEGORY_HIT, "kis_001", first_rank=3, distance=0),
        _analysis(CATEGORY_NEAR_VIDEO, "kis_002", first_rank=5, distance=3),
        _analysis(CATEGORY_NEAR_VIDEO, "kis_003", first_rank=2, distance=1),
        _analysis(CATEGORY_FAIL, "kis_004", first_rank=None, distance=None),
    ]


def test_select_analyses_filters_by_category() -> None:
    selected = select_analyses(_sample_analyses(), category=CATEGORY_NEAR_VIDEO)

    assert [analysis.query_id for analysis in selected] == ["kis_002", "kis_003"]
    assert all(analysis.category == CATEGORY_NEAR_VIDEO for analysis in selected)


def test_select_analyses_sorts_by_distance_ascending_with_missing_last() -> None:
    selected = select_analyses(_sample_analyses(), sort="distance")

    assert [analysis.query_id for analysis in selected] == ["kis_001", "kis_003", "kis_002", "kis_004"]


def test_select_analyses_sorts_by_rank_ascending_with_missing_last_and_stable_ties() -> None:
    analyses = _sample_analyses()
    analyses.append(_analysis(CATEGORY_HIT, "kis_005", first_rank=3, distance=0))

    selected = select_analyses(analyses, sort="rank")

    # kis_003 (rank 2), then kis_001/kis_005 (rank 3, input order), then kis_002 (rank 5), FAIL last
    assert [analysis.query_id for analysis in selected] == ["kis_003", "kis_001", "kis_005", "kis_002", "kis_004"]


def test_select_analyses_none_preserves_original_order() -> None:
    selected = select_analyses(_sample_analyses(), sort="none")

    assert [analysis.query_id for analysis in selected] == ["kis_001", "kis_002", "kis_003", "kis_004"]
    assert select_analyses(_sample_analyses()) == _sample_analyses()


def test_select_analyses_rejects_invalid_sort() -> None:
    try:
        select_analyses(_sample_analyses(), sort="closest")
    except ValueError as exc:
        assert "sort" in str(exc)
    else:
        raise AssertionError("expected ValueError for an invalid sort value")


def test_query_error_analysis_to_dict_round_trips_through_json() -> None:
    import json

    analysis = QueryErrorAnalysis(
        category=CATEGORY_NEAR_VIDEO,
        query_id="kis_007",
        query_text_en="english text",
        query_text_vi="vietnamese text",
        gt_video_id="L21_V001",
        gt_frame_start=100,
        gt_frame_end=120,
        first_gt_video_rank=4,
        closest_frame_distance=7,
        top5=((1, "L21_V001", 99, 0.8), (2, "L22_V002", 50, float("nan"))),
    )

    payload = json.loads(json.dumps(analysis.to_dict()))

    assert payload["category"] == "NEAR_VIDEO"
    assert payload["query_id"] == "kis_007"
    assert payload["ground_truth"] == {"video_id": "L21_V001", "frame_start": 100, "frame_end": 120}
    assert payload["first_gt_video_rank"] == 4
    assert payload["closest_frame_distance"] == 7
    assert payload["top5"] == [
        {"rank": 1, "video_id": "L21_V001", "frame_index": 99, "score": 0.8},
        {"rank": 2, "video_id": "L22_V002", "frame_index": 50, "score": None},
    ]
    assert payload["top5"][1]["score"] is None