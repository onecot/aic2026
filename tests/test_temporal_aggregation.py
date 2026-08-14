"""Tests for temporal aggregation as a retrieval post-processing stage.

These tests exercise the aggregation math and the scorer contract without
running any retrieval or model code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.evaluation import (
    EvaluationTask,
    FrameRange,
    load_eval_set,
)
from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    read_kis_prediction_artifact,
)
from aic2026.kis_scoring import KIS_TOP_K, evaluate_kis
from aic2026.temporal_aggregation import (
    TemporalAggregationConfig,
    _mean_window_scores,
    aggregate_artifact_predictions,
    aggregate_prediction_artifact,
    aggregate_query_predictions,
)


def _prediction(
    rank: int,
    video_id: str,
    frame_index: int,
    score: float,
) -> RankedKisPrediction:
    return RankedKisPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=score)


def _query_result(query_id: str, predictions: tuple[RankedKisPrediction, ...]) -> QueryPredictionResult:
    return QueryPredictionResult(query_id=query_id, query_text="test query", predictions=predictions)


def _artifact(query_results: tuple[QueryPredictionResult, ...]) -> KisPredictionArtifact:
    return KisPredictionArtifact(
        schema_version=KIS_PREDICTION_SCHEMA_VERSION,
        run_id="test-run",
        created_at="2026-08-13T00:00:00Z",
        evaluation_set_path="/tmp/eval_set.csv",
        evaluation_set_query_count=len(query_results),
        processed_query_count=len(query_results),
        query_text_field="query_text_en",
        top_k=100,
        embedding_space_compatible=True,
        embedding_space_override_used=False,
        index_artifact_path="/tmp/index",
        index_manifest={"artifact_id": "dummy-index"},
        model_embedding_space={"backend": "dummy"},
        model_config=None,
        selected_query_ids=None,
        query_results=query_results,
    )


def test_sliding_window_mean_raises_isolated_dip_in_consecutive_run() -> None:
    predictions = [
        _prediction(1, "L01_V001", 100, 1.0),
        _prediction(2, "L01_V001", 101, 0.0),
        _prediction(3, "L01_V001", 102, 1.0),
    ]

    output = aggregate_query_predictions(predictions, TemporalAggregationConfig(window_size=1))

    assert [(item.frame_index, item.rank) for item in output] == [(101, 1), (100, 2), (102, 3)]
    assert _mean_window_scores([(100, 1.0), (101, 0.0), (102, 1.0)], 1) == pytest.approx(
        [0.5, 2.0 / 3.0, 0.5]
    )


def test_frames_are_sorted_by_frame_index_before_windowing() -> None:
    shuffled = [
        _prediction(1, "L01_V001", 102, 1.0),
        _prediction(2, "L01_V001", 100, 1.0),
        _prediction(3, "L01_V001", 101, 0.0),
    ]

    output = aggregate_query_predictions(shuffled, TemporalAggregationConfig(window_size=1))

    assert [item.frame_index for item in output] == [101, 100, 102]


def test_aggregation_reassigns_strictly_increasing_ranks_and_preserves_identity() -> None:
    predictions = [
        _prediction(1, "L01_V001", 10, 0.3),
        _prediction(2, "L02_V003", 5, 0.9),
        _prediction(3, "L01_V001", 11, 0.8),
    ]

    output = aggregate_query_predictions(predictions, TemporalAggregationConfig(window_size=1))

    assert [item.rank for item in output] == [1, 2, 3]
    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L02_V003", 5),
        ("L01_V001", 10),
        ("L01_V001", 11),
    ]
    assert len(output) == len(predictions)


def test_no_cross_video_mixing_in_window() -> None:
    predictions = [
        _prediction(1, "L01_V001", 100, 0.0),
        _prediction(2, "L02_V003", 101, 1.0),
    ]

    output = aggregate_query_predictions(predictions, TemporalAggregationConfig(window_size=1))

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L02_V003", 101),
        ("L01_V001", 100),
    ]
    assert _mean_window_scores([(100, 0.0)], 1) == [0.0]
    assert _mean_window_scores([(101, 1.0)], 1) == [1.0]


def test_artifact_level_aggregation_handles_empty_and_multiple_queries() -> None:
    artifact = _artifact(
        (
            _query_result("kis_001", ()),
            _query_result(
                "kis_002",
                (
                    _prediction(1, "L01_V001", 0, 0.9),
                    _prediction(2, "L01_V001", 1, 0.1),
                ),
            ),
        )
    )

    output = aggregate_artifact_predictions(artifact, TemporalAggregationConfig(window_size=1))

    assert set(output) == {"kis_001", "kis_002"}
    assert output["kis_001"] == []
    assert [(item.frame_index, item.rank) for item in output["kis_002"]] == [(0, 1), (1, 2)]


def test_aggregated_output_is_compatible_with_evaluate_kis(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_set.csv"
    csv_path.write_text(
        "\n".join(
            [
                "query_id,task,query_text_vi,query_text_en,gt_video_id,gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note",
                "kis_001,kis,T\u00ecm c\u1ea3nh,mock query,L21_V001,10,20,,,P1,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = _artifact(
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 15, 0.4),
                    _prediction(2, "L21_V001", 14, 0.9),
                    _prediction(3, "L21_V001", 16, 0.9),
                    _prediction(4, "L22_V002", 200, 0.85),
                ),
            ),
        )
    )
    artifact_path = tmp_path / "predictions.json"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    eval_set = load_eval_set(csv_path)
    aggregated = aggregate_prediction_artifact(artifact_path, TemporalAggregationConfig(window_size=1))
    result = evaluate_kis(eval_set, aggregated)

    assert [(item.video_id, item.frame_index) for item in aggregated["kis_001"]] == [
        ("L22_V002", 200),
        ("L21_V001", 15),
        ("L21_V001", 14),
        ("L21_V001", 16),
    ]
    assert result.score_at(1) == 0.0
    assert all(result.score_at(cutoff) == 1.0 for cutoff in (5, 20, 50, 100))
    assert result.final_score == pytest.approx(0.8)


def test_configuration_validation_rejects_invalid_window_and_method() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TemporalAggregationConfig(window_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalAggregationConfig(window_size=-2)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalAggregationConfig(window_size=2.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalAggregationConfig(window_size=True)
    with pytest.raises(ValueError, match="unsupported"):
        TemporalAggregationConfig(method="max")
    with pytest.raises(ValueError, match="unsupported"):
        TemporalAggregationConfig(method="median")


def test_aggregation_rejects_non_ranked_predictions() -> None:
    with pytest.raises(ValueError, match="RankedKisPrediction"):
        aggregate_query_predictions([object()])  # type: ignore[list-item]


def test_aggregate_prediction_artifact_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        aggregate_prediction_artifact(tmp_path / "missing.json")


def test_real_prediction_artifact_aggregates_without_running_an_experiment() -> None:
    path = Path(__file__).resolve().parents[1] / "data/interim/predictions/kis-l14-laion2b-full-en.json"
    if not path.is_file():
        pytest.skip("prediction artifact not available")
    artifact = read_kis_prediction_artifact(path)

    aggregated = aggregate_artifact_predictions(artifact, TemporalAggregationConfig(window_size=1))

    assert len(aggregated) == artifact.processed_query_count
    for predictions in aggregated.values():
        assert len(predictions) == artifact.top_k
        assert [item.rank for item in predictions] == list(range(1, artifact.top_k + 1))
        assert len({item.video_id for item in predictions}) > 1
