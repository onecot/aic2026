"""Tests for temporal window expansion as a retrieval post-processing stage.

These tests exercise the expansion math, frame-existence filtering, the
deterministic ranking, and the scorer contract without running any retrieval
or model code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import FRAMES_FILENAME, write_frames
from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    load_kis_predictions_for_scoring,
    read_kis_prediction_artifact,
    write_kis_prediction_artifact,
)
from aic2026.kis_scoring import KIS_TOP_K, evaluate_kis
from aic2026.temporal_window_expansion import (
    TemporalWindowConfig,
    _window_offsets,
    build_expanded_artifact,
    expand_artifact_predictions,
    expand_prediction_artifact,
    expand_prediction_artifact_to_artifact,
    expand_query_predictions,
    expand_query_predictions_ranked,
    load_existing_frames,
    write_expanded_prediction_artifact,
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


def _timing(video_id: str, ordinal: int, frame_index: int) -> KeyframeTiming:
    return KeyframeTiming(
        keyframe=KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=ordinal),
        pts_time=float(frame_index) / 25.0,
        fps=25.0,
        frame_index=frame_index,
    )


def test_window_offsets_include_zero_and_multiples_of_step() -> None:
    assert _window_offsets(1, 1) == (-1, 0, 1)
    assert _window_offsets(3, 1) == (-3, -2, -1, 0, 1, 2, 3)
    assert _window_offsets(5, 2) == (-4, -2, 0, 2, 4)
    assert _window_offsets(2, 3) == (0,)


def test_expansion_keeps_original_and_propagates_decayed_neighbors() -> None:
    existing = {"L01_V001": {99, 100, 101, 102}}
    predictions = [_prediction(1, "L01_V001", 100, 0.8)]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=1, step=1, decay=0.5),
    )

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L01_V001", 100),
        ("L01_V001", 99),
        ("L01_V001", 101),
    ]
    assert [item.rank for item in output] == [1, 2, 3]


def test_expansion_respects_step_spacing() -> None:
    existing = {"L01_V001": {96, 98, 100, 102, 104}}
    predictions = [_prediction(1, "L01_V001", 100, 0.9)]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=5, step=2, decay=0.8),
    )

    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L01_V001", 100),
        ("L01_V001", 96),
        ("L01_V001", 98),
        ("L01_V001", 102),
        ("L01_V001", 104),
    ]


def test_expansion_does_not_cross_video_boundaries() -> None:
    existing = {
        "L01_V001": {99, 100, 101},
        "L02_V003": {99, 100, 101},
    }
    predictions = [
        _prediction(1, "L01_V001", 100, 0.8),
        _prediction(2, "L02_V003", 100, 0.7),
    ]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=1, step=1, decay=0.5),
    )

    # Each prediction expands only within its own video's frame mapping:
    # L01 originals/propagated and L02 originals/propagated never mix. Ordering
    # is by score: L01 original (0.8), L02 original (0.7), then propagated 0.4
    # and 0.35 groups.
    assert [item.video_id for item in output] == [
        "L01_V001",
        "L02_V003",
        "L01_V001",
        "L01_V001",
        "L02_V003",
        "L02_V003",
    ]
    assert all(item.video_id in ("L01_V001", "L02_V003") for item in output)


def test_invalid_frames_are_removed() -> None:
    existing = {"L01_V001": {0, 1, 2, 3}}
    predictions = [_prediction(1, "L01_V001", 1, 0.8)]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=3, step=1, decay=0.5),
    )

    assert [item.frame_index for item in output] == [1, 0, 2, 3]
    assert all(item.frame_index >= 0 for item in output)
    assert all(item.frame_index in existing["L01_V001"] for item in output)


def test_unknown_video_keeps_original_frame_only() -> None:
    existing = {"L02_V003": {50, 51, 52}}
    predictions = [_prediction(1, "L01_V001", 100, 0.8)]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=2, step=1, decay=0.5),
    )

    assert [(item.video_id, item.frame_index) for item in output] == [("L01_V001", 100)]


def test_ranking_is_deterministic_and_score_ordered() -> None:
    existing = {"L01_V001": {10, 11, 12}}
    predictions = [
        _prediction(1, "L01_V001", 11, 0.9),
        _prediction(2, "L01_V001", 10, 0.2),
    ]
    config = TemporalWindowConfig(window_size=1, step=1, decay=0.5)

    first = expand_query_predictions(predictions, existing, config)
    second = expand_query_predictions(predictions, existing, config)

    assert [(item.rank, item.video_id, item.frame_index) for item in first] == [
        (item.rank, item.video_id, item.frame_index) for item in second
    ]
    # Highest original score first, its propagated neighbors next, then the
    # lower original prediction. Frame 9 is dropped because it does not exist.
    assert [(item.video_id, item.frame_index) for item in first] == [
        ("L01_V001", 11),
        ("L01_V001", 10),
        ("L01_V001", 12),
        ("L01_V001", 10),
        ("L01_V001", 11),
    ]


def test_decay_one_preserves_scores_and_keeps_original_rank_first() -> None:
    existing = {"L01_V001": {100, 101}}
    predictions = [_prediction(1, "L01_V001", 100, 0.8)]

    output = expand_query_predictions(
        predictions,
        existing,
        TemporalWindowConfig(window_size=1, step=1, decay=1),
    )

    # All scores equal (0.8); generation order places the original frame first,
    # and frame 99 is dropped because it does not exist.
    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L01_V001", 100),
        ("L01_V001", 101),
    ]


def test_configuration_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=0, step=1, decay=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=-2, step=1, decay=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=True, step=1, decay=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=1, step=0, decay=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=1, step=-1, decay=0.5)
    with pytest.raises(ValueError, match="positive integer"):
        TemporalWindowConfig(window_size=1, step=True, decay=0.5)
    with pytest.raises(ValueError, match="decay"):
        TemporalWindowConfig(window_size=1, step=1, decay=0)
    with pytest.raises(ValueError, match="decay"):
        TemporalWindowConfig(window_size=1, step=1, decay=1.5)
    with pytest.raises(ValueError, match="decay"):
        TemporalWindowConfig(window_size=1, step=1, decay=True)


def test_expanded_output_is_compatible_with_evaluate_kis(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_set.csv"
    csv_path.write_text(
        "\n".join(
            [
                "query_id,task,query_text_vi,query_text_en,gt_video_id,gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note",
                "kis_001,kis,T\u00ecm c\u1ea3nh,mock query,L21_V001,100,102,,,P1,",
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
                    _prediction(1, "L21_V001", 101, 0.9),
                    _prediction(2, "L22_V002", 200, 0.8),
                ),
            ),
        )
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    write_frames(
        index_dir / FRAMES_FILENAME,
        [
            _timing("L21_V001", 1, 99),
            _timing("L21_V001", 2, 100),
            _timing("L21_V001", 3, 101),
            _timing("L21_V001", 4, 102),
            _timing("L21_V001", 5, 103),
            _timing("L22_V002", 1, 200),
        ],
    )
    artifact_path = tmp_path / "predictions.json"
    artifact_path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    eval_set = load_eval_set(csv_path)
    expanded = expand_prediction_artifact(
        artifact_path,
        TemporalWindowConfig(window_size=2, step=1, decay=0.5),
        index_artifact_dir=index_dir,
    )
    result = evaluate_kis(eval_set, expanded, cutoffs=KIS_TOP_K)

    # Original frame 101 is a hit at rank 1; the hit window therefore covers
    # all cutoffs.
    assert result.score_at(1) == 1.0
    assert all(result.score_at(cutoff) == 1.0 for cutoff in (1, 5, 20, 50, 100))
    assert result.final_score == pytest.approx(1.0)


def test_load_existing_frames_reads_index_artifact(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    write_frames(
        index_dir / FRAMES_FILENAME,
        [
            _timing("L01_V001", 1, 0),
            _timing("L01_V001", 2, 1),
            _timing("L02_V003", 1, 10),
            _timing("L02_V003", 2, 11),
        ],
    )

    mapping = load_existing_frames(index_dir)

    assert mapping == {"L01_V001": {0, 1}, "L02_V003": {10, 11}}


def test_load_existing_frames_missing_file_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="frames.ndjson"):
        load_existing_frames(empty_dir)


def test_artifact_level_expansion_handles_empty_and_multiple_queries() -> None:
    existing = {"L01_V001": {0, 1}}
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

    output = expand_artifact_predictions(
        artifact,
        TemporalWindowConfig(window_size=1, step=1, decay=0.5),
        existing,
    )

    assert set(output) == {"kis_001", "kis_002"}
    assert output["kis_001"] == []
    assert len(output["kis_002"]) == 4
    assert [item.rank for item in output["kis_002"]] == [1, 2, 3, 4]


def test_expansion_rejects_non_ranked_predictions() -> None:
    with pytest.raises(ValueError, match="RankedKisPrediction"):
        expand_query_predictions(
            [object()],  # type: ignore[list-item]
            {"L01_V001": {0}},
            TemporalWindowConfig(window_size=1, step=1, decay=0.5),
        )


def test_empty_predictions_produce_empty_output() -> None:
    output = expand_query_predictions(
        [],
        {},
        TemporalWindowConfig(window_size=1, step=1, decay=0.5),
    )
    assert output == []


def test_real_prediction_artifact_expands_without_running_an_experiment() -> None:
    path = Path(__file__).resolve().parents[1] / "data/interim/predictions/kis-l14-laion2b-full-en.json"
    if not path.is_file():
        pytest.skip("prediction artifact not available")
    artifact = read_kis_prediction_artifact(path)
    index_dir = Path(artifact.index_artifact_path)
    if not (index_dir / FRAMES_FILENAME).is_file():
        pytest.skip("index artifact frames not available")

    existing = load_existing_frames(index_dir)
    output = expand_artifact_predictions(
        artifact,
        TemporalWindowConfig(window_size=3, step=1, decay=0.9),
        existing,
    )

    assert len(output) == artifact.processed_query_count
    for predictions in output.values():
        assert [item.rank for item in predictions] == list(range(1, len(predictions) + 1))
        assert len(predictions) >= artifact.top_k
        # Every expanded frame must exist in the index frame mapping.
        for prediction in predictions:
            assert prediction.frame_index in existing[prediction.video_id]


# ---------------------------------------------------------------------------
# CLI / artifact-level helper tests
# ---------------------------------------------------------------------------


def _write_source_artifact(
    tmp_path: Path,
    index_dir: Path,
    query_results: tuple[QueryPredictionResult, ...],
) -> tuple[Path, Path]:
    """Write a source prediction artifact and a matching index artifact."""
    index_dir.mkdir(parents=True, exist_ok=True)
    write_frames(
        index_dir / FRAMES_FILENAME,
        [
            _timing("L21_V001", 1, 99),
            _timing("L21_V001", 2, 100),
            _timing("L21_V001", 3, 101),
            _timing("L21_V001", 4, 102),
            _timing("L21_V001", 5, 103),
            _timing("L22_V002", 1, 200),
            _timing("L22_V002", 2, 201),
            _timing("L22_V002", 3, 202),
        ],
    )
    artifact = _artifact(query_results)
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, artifact)
    return artifact_path, index_dir


def test_built_artifact_preserves_metadata_and_schema(tmp_path: Path) -> None:
    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 100, 0.9),
                    _prediction(2, "L22_V002", 201, 0.7),
                ),
            ),
        ),
    )
    config = TemporalWindowConfig(window_size=1, step=1, decay=0.5)

    expanded = expand_prediction_artifact_to_artifact(
        artifact_path,
        config,
        index_artifact_dir=index_dir,
    )
    source = read_kis_prediction_artifact(artifact_path)

    assert expanded.schema_version == source.schema_version
    assert expanded.schema_version == KIS_PREDICTION_SCHEMA_VERSION
    assert expanded.evaluation_set_path == source.evaluation_set_path
    assert expanded.evaluation_set_query_count == source.evaluation_set_query_count
    assert expanded.processed_query_count == source.processed_query_count
    assert expanded.query_text_field == source.query_text_field
    assert expanded.top_k == source.top_k
    assert expanded.embedding_space_compatible == source.embedding_space_compatible
    assert expanded.embedding_space_override_used == source.embedding_space_override_used
    assert expanded.index_artifact_path == source.index_artifact_path
    assert expanded.index_manifest == source.index_manifest
    assert expanded.model_embedding_space == source.model_embedding_space
    assert expanded.model_config == source.model_config
    assert expanded.selected_query_ids == source.selected_query_ids
    assert expanded.run_id.startswith("expand-")
    assert expanded.created_at.endswith("Z")


def test_output_artifact_loads_through_load_kis_predictions_for_scoring(
    tmp_path: Path,
) -> None:
    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 101, 0.9),
                    _prediction(2, "L22_V002", 200, 0.8),
                ),
            ),
        ),
    )
    output_path = tmp_path / "expanded.json"
    config = TemporalWindowConfig(window_size=2, step=1, decay=0.5)

    write_expanded_prediction_artifact(
        output_path,
        artifact_path,
        config,
        index_artifact_dir=index_dir,
        run_id="expand-test-run",
    )

    expanded = read_kis_prediction_artifact(output_path)
    assert expanded.run_id == "expand-test-run"
    by_query = load_kis_predictions_for_scoring(output_path)
    assert set(by_query) == {"kis_001"}
    predictions = by_query["kis_001"]
    assert len(predictions) >= 4
    assert [item.rank for item in predictions] == list(range(1, len(predictions) + 1))
    assert all(isinstance(item, object) for item in predictions)


def test_output_artifact_accepted_by_evaluate_kis(tmp_path: Path) -> None:
    csv_path = tmp_path / "eval_set.csv"
    csv_path.write_text(
        "\n".join(
            [
                "query_id,task,query_text_vi,query_text_en,gt_video_id,gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note",
                "kis_001,kis,T\u00ecm c\u1ea3nh,mock query,L21_V001,100,102,,,P1,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 101, 0.9),
                    _prediction(2, "L22_V002", 200, 0.8),
                ),
            ),
        ),
    )
    output_path = tmp_path / "expanded.json"
    config = TemporalWindowConfig(window_size=2, step=1, decay=0.5)
    write_expanded_prediction_artifact(output_path, artifact_path, config, index_artifact_dir=index_dir)

    eval_set = load_eval_set(csv_path)
    by_query = load_kis_predictions_for_scoring(output_path)
    result = evaluate_kis(eval_set, by_query, cutoffs=KIS_TOP_K)

    assert result.score_at(1) == 1.0
    assert all(result.score_at(cutoff) == 1.0 for cutoff in (1, 5, 20, 50, 100))
    assert result.final_score == pytest.approx(1.0)


def test_built_artifact_ranking_is_deterministic(tmp_path: Path) -> None:
    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 101, 0.9),
                    _prediction(2, "L22_V002", 201, 0.7),
                ),
            ),
        ),
    )
    config = TemporalWindowConfig(window_size=2, step=1, decay=0.5)

    first = expand_prediction_artifact_to_artifact(artifact_path, config, index_artifact_dir=index_dir)
    second = expand_prediction_artifact_to_artifact(artifact_path, config, index_artifact_dir=index_dir)

    assert [
        (item.rank, item.video_id, item.frame_index, item.score)
        for item in first.query_results[0].predictions
    ] == [
        (item.rank, item.video_id, item.frame_index, item.score)
        for item in second.query_results[0].predictions
    ]
    assert len(first.query_results[0].predictions) == len(second.query_results[0].predictions)


def test_expand_query_predictions_ranked_keeps_scores() -> None:
    existing = {"L01_V001": {100, 101, 102}}
    predictions = [_prediction(1, "L01_V001", 100, 0.8)]

    ranked = expand_query_predictions_ranked(
        predictions,
        existing,
        TemporalWindowConfig(window_size=1, step=1, decay=0.5),
    )

    assert [(item.rank, item.frame_index, item.score) for item in ranked] == [
        (1, 100, 0.8),
        (2, 101, 0.4),
    ]


def test_cli_temporal_expand_kis_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aic2026.cli import main

    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (
                    _prediction(1, "L21_V001", 101, 0.9),
                    _prediction(2, "L22_V002", 200, 0.8),
                ),
            ),
        ),
    )
    output_path = tmp_path / "cli_expanded.json"

    argv = [
        "temporal-expand-kis",
        "--prediction",
        str(artifact_path),
        "--index-artifact",
        str(index_dir),
        "--window-size",
        "2",
        "--step",
        "1",
        "--decay",
        "0.5",
        "--output",
        str(output_path),
        "--run-id",
        "cli-run-1",
    ]

    exit_code = main(argv)

    assert exit_code == 0
    assert output_path.is_file()
    expanded = read_kis_prediction_artifact(output_path)
    assert expanded.run_id == "cli-run-1"
    by_query = load_kis_predictions_for_scoring(output_path)
    assert set(by_query) == {"kis_001"}
    assert len(by_query["kis_001"]) >= 4
    assert [item.rank for item in by_query["kis_001"]] == list(range(1, len(by_query["kis_001"]) + 1))


def test_cli_temporal_expand_kis_rejects_invalid_config(tmp_path: Path) -> None:
    from aic2026.cli import main

    artifact_path, index_dir = _write_source_artifact(
        tmp_path,
        tmp_path / "index",
        (
            _query_result(
                "kis_001",
                (_prediction(1, "L21_V001", 101, 0.9),),
            ),
        ),
    )

    argv = [
        "temporal-expand-kis",
        "--prediction",
        str(artifact_path),
        "--index-artifact",
        str(index_dir),
        "--window-size",
        "1",
        "--step",
        "0",
        "--decay",
        "0.5",
        "--output",
        str(tmp_path / "out.json"),
    ]

    exit_code = main(argv)

    assert exit_code == 1
    assert not (tmp_path / "out.json").exists()
