"""Focused KIS scoring tests independent of retrieval/model execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.canonical import VideoId
from aic2026.evaluation import (
    AnnotationMetadata,
    EvaluationQuery,
    EvaluationTask,
    FrameRange,
    load_eval_set,
)
from aic2026.kis_scoring import (
    KIS_TOP_K,
    KisPrediction,
    TopKScore,
    calculate_final_score,
    evaluate_kis,
    is_kis_correct,
    score_kis_query,
)


def _query(query_id: str = "kis_001", start: int = 100, end: int = 200) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        task=EvaluationTask.KIS,
        query_text_vi="Tìm cảnh kiểm thử.",
        query_text_en=None,
        ground_truth_video_id=VideoId("L21_V001"),
        ground_truth_frame_range=FrameRange(start, end),
        answer=None,
        event=None,
        annotation=AnnotationMetadata(annotator="test", note=None),
    )


def _prediction(rank: int, video_id: str = "L21_V001", frame_index: int = 150) -> KisPrediction:
    return KisPrediction(rank=rank, video_id=video_id, frame_index=frame_index)


def test_kis_correctness_requires_matching_video_and_inclusive_frame_range() -> None:
    query = _query()

    assert is_kis_correct(query, _prediction(1, frame_index=150))
    assert not is_kis_correct(query, _prediction(1, frame_index=99))
    assert not is_kis_correct(query, _prediction(1, frame_index=201))
    assert not is_kis_correct(query, _prediction(1, video_id="L21_V002", frame_index=150))


@pytest.mark.parametrize("frame_index", [100, 200])
def test_kis_frame_range_boundaries_are_inclusive(frame_index: int) -> None:
    assert is_kis_correct(_query(), _prediction(1, frame_index=frame_index))


def test_per_query_top_k_uses_first_correct_rank_and_all_default_cutoffs() -> None:
    score = score_kis_query(
        _query(),
        [_prediction(1, frame_index=1), _prediction(4, frame_index=150), _prediction(6, frame_index=150)],
    )

    assert score.score_at(1) == 0.0
    assert score.score_at(5) == 1.0
    assert score.score_at(20) == 1.0
    assert tuple(cutoff for cutoff, _ in score.r_at) == KIS_TOP_K


def test_missing_ranks_and_fewer_candidates_do_not_become_correct() -> None:
    query = _query()
    missing_rank_score = score_kis_query(query, [_prediction(2, frame_index=150)], cutoffs=(1, 5))
    fewer_candidate_score = score_kis_query(
        query,
        [_prediction(1, frame_index=1)],
        cutoffs=(1, 5, 20),
    )

    assert missing_rank_score.score_at(1) == 0.0
    assert missing_rank_score.score_at(5) == 1.0
    assert [fewer_candidate_score.score_at(cutoff) for cutoff in (1, 5, 20)] == [0.0, 0.0, 0.0]


def test_evaluate_kis_aggregates_r_at_k_and_final_score_exactly() -> None:
    first = _query("kis_001")
    second = _query("kis_002")
    result = evaluate_kis(
        [first, second],
        {
            "kis_001": [_prediction(1, frame_index=150)],
            "kis_002": [_prediction(6, frame_index=150)],
        },
    )

    assert result.score_at(1) == pytest.approx(0.5)
    assert result.score_at(5) == pytest.approx(0.5)
    assert result.score_at(20) == pytest.approx(1.0)
    assert result.score_at(50) == pytest.approx(1.0)
    assert result.score_at(100) == pytest.approx(1.0)
    assert result.final_score == pytest.approx(0.8)
    assert calculate_final_score(result.top_k_scores) == pytest.approx(0.8)


def test_prediction_mapping_must_cover_exactly_the_scored_queries() -> None:
    query = _query()
    with pytest.raises(ValueError, match="missing predictions"):
        evaluate_kis([query], {})
    with pytest.raises(ValueError, match="unexpected prediction"):
        evaluate_kis([query], {"kis_001": [], "kis_999": []})


def test_invalid_prediction_order_and_final_score_input_fail() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        score_kis_query(_query(), [_prediction(2), _prediction(1)])
    with pytest.raises(ValueError, match="no top-k"):
        calculate_final_score([])
    with pytest.raises(ValueError, match="unique"):
        calculate_final_score([TopKScore(1, 0.0), TopKScore(1, 1.0)])


def test_real_kis_eval_set_accepts_complete_empty_prediction_mapping_when_available() -> None:
    path = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    if not path.is_file():
        pytest.skip("team evaluation CSV not available")
    queries = tuple(load_eval_set(path).iter_queries(EvaluationTask.KIS))

    result = evaluate_kis(queries, {query.query_id: [] for query in queries})

    assert len(result.query_scores) == 101
    assert [result.score_at(cutoff) for cutoff in KIS_TOP_K] == [0.0] * len(KIS_TOP_K)
    assert result.final_score == 0.0
