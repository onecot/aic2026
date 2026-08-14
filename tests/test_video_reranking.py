"""Tests for video-level reranking as a retrieval post-processing stage.

These tests exercise the reranking math and the scorer contract without running
any retrieval or model code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.evaluation import load_eval_set
from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    read_kis_prediction_artifact,
)
from aic2026.kis_scoring import evaluate_kis
from aic2026.video_reranking import (
    VideoRerankingConfig,
    _video_score,
    rerank_artifact_predictions,
    rerank_prediction_artifact,
    rerank_query_predictions,
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


def test_max_aggregation_groups_frames_by_video() -> None:
    predictions = [
        _prediction(1, "L01_V001", 100, 0.8),
        _prediction(2, "L01_V001", 101, 0.2),
        _prediction(3, "L02_V003", 50, 0.6),
        _prediction(4, "L03_V005", 10, 0.9),
        _prediction(5, "L03_V005", 11, 0.7),
    ]

    output = rerank_query_predictions(predictions, VideoRerankingConfig(aggregation="max"))

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L03_V005", 10),
        ("L03_V005", 11),
        ("L01_V001", 100),
        ("L01_V001", 101),
        ("L02_V003", 50),
    ]


def test_mean_top_n_aggregation_uses_top_frames_and_handles_fewer_than_n() -> None:
    predictions = [
        _prediction(1, "L01_V001", 10, 0.9),
        _prediction(2, "L01_V001", 11, 0.8),
        _prediction(3, "L01_V001", 12, 0.2),
        _prediction(4, "L02_V003", 20, 0.8),
    ]

    assert _video_score([0.9, 0.8, 0.2], VideoRerankingConfig(aggregation="mean_top_n", top_n=2)) == pytest.approx(
        0.85
    )
    assert _video_score([0.8], VideoRerankingConfig(aggregation="mean_top_n", top_n=2)) == pytest.approx(0.8)

    output = rerank_query_predictions(
        predictions,
        VideoRerankingConfig(aggregation="mean_top_n", top_n=2),
    )

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L01_V001", 10),
        ("L01_V001", 11),
        ("L01_V001", 12),
        ("L02_V003", 20),
    ]


def test_no_cross_video_leakage() -> None:
    predictions = [
        _prediction(1, "L01_V001", 100, 0.9),
        _prediction(2, "L01_V001", 101, 0.1),
        _prediction(3, "L02_V003", 200, 0.85),
        _prediction(4, "L02_V003", 201, 0.84),
    ]

    output = rerank_query_predictions(predictions, VideoRerankingConfig(aggregation="max"))

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L01_V001", 100),
        ("L01_V001", 101),
        ("L02_V003", 200),
        ("L02_V003", 201),
    ]
    # L01's weakest frame (0.1) still outranks L02's best (0.85): frames are
    # ordered by their own video's score, never by another video's.
    assert (output[1].video_id, output[1].frame_index) == ("L01_V001", 101)
    assert (output[2].video_id, output[2].frame_index) == ("L02_V003", 200)


def test_deterministic_ranking_with_video_and_frame_ties() -> None:
    predictions = [
        _prediction(1, "L01_V001", 10, 0.5),
        _prediction(2, "L02_V003", 20, 0.5),
        _prediction(3, "L03_V005", 30, 0.5),
        _prediction(4, "L01_V001", 11, 0.5),
    ]
    config = VideoRerankingConfig(aggregation="max")

    first = rerank_query_predictions(predictions, config)
    second = rerank_query_predictions(predictions, config)

    assert [(item.rank, item.video_id, item.frame_index) for item in first] == [
        (item.rank, item.video_id, item.frame_index) for item in second
    ]
    assert [(item.video_id, item.frame_index) for item in first] == [
        ("L01_V001", 10),
        ("L01_V001", 11),
        ("L02_V003", 20),
        ("L03_V005", 30),
    ]


def test_reranked_output_is_compatible_with_evaluate_kis(tmp_path: Path) -> None:
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
                    _prediction(1, "L21_V001", 15, 0.9),
                    _prediction(2, "L21_V001", 5, 0.85),
                    _prediction(3, "L22_V002", 200, 0.8),
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
    reranked = rerank_prediction_artifact(artifact_path, VideoRerankingConfig(aggregation="max"))
    result = evaluate_kis(eval_set, reranked)

    assert [(item.video_id, item.frame_index) for item in reranked["kis_001"]] == [
        ("L21_V001", 15),
        ("L21_V001", 5),
        ("L22_V002", 200),
    ]
    assert result.score_at(1) == 1.0
    assert all(result.score_at(cutoff) == 1.0 for cutoff in (5, 20, 50, 100))
    assert result.final_score == pytest.approx(1.0)


def test_configuration_validation_rejects_invalid_aggregation_and_top_n() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        VideoRerankingConfig(aggregation="median")
    with pytest.raises(ValueError, match="unsupported"):
        VideoRerankingConfig(aggregation="")
    with pytest.raises(ValueError, match="unsupported"):
        VideoRerankingConfig(aggregation="MAX")
    with pytest.raises(ValueError, match="positive integer"):
        VideoRerankingConfig(top_n=0)
    with pytest.raises(ValueError, match="positive integer"):
        VideoRerankingConfig(top_n=-1)
    with pytest.raises(ValueError, match="positive integer"):
        VideoRerankingConfig(top_n=True)
    with pytest.raises(ValueError, match="positive integer"):
        VideoRerankingConfig(top_n=1.5)


def test_artifact_level_reranking_handles_empty_and_multiple_queries() -> None:
    artifact = _artifact(
        (
            _query_result("kis_001", ()),
            _query_result(
                "kis_002",
                (
                    _prediction(1, "L01_V001", 0, 0.9),
                    _prediction(2, "L01_V001", 1, 0.1),
                    _prediction(3, "L02_V003", 5, 0.5),
                ),
            ),
        )
    )

    output = rerank_artifact_predictions(artifact, VideoRerankingConfig(aggregation="max"))

    assert set(output) == {"kis_001", "kis_002"}
    assert output["kis_001"] == []
    assert [(item.video_id, item.frame_index) for item in output["kis_002"]] == [
        ("L01_V001", 0),
        ("L01_V001", 1),
        ("L02_V003", 5),
    ]


def test_reranking_rejects_non_ranked_predictions() -> None:
    with pytest.raises(ValueError, match="RankedKisPrediction"):
        rerank_query_predictions([object()])  # type: ignore[list-item]


def test_empty_predictions_produce_empty_output() -> None:
    output = rerank_query_predictions([])
    assert output == []


def test_real_prediction_artifact_reranks_without_running_an_experiment() -> None:
    path = Path(__file__).resolve().parents[1] / "data/interim/predictions/kis-l14-laion2b-full-en.json"
    if not path.is_file():
        pytest.skip("prediction artifact not available")
    artifact = read_kis_prediction_artifact(path)

    output = rerank_artifact_predictions(artifact, VideoRerankingConfig(aggregation="mean_top_n", top_n=5))

    assert len(output) == artifact.processed_query_count
    for predictions in output.values():
        assert len(predictions) == artifact.top_k
        assert [item.rank for item in predictions] == list(range(1, artifact.top_k + 1))
        assert len({item.video_id for item in predictions}) > 1