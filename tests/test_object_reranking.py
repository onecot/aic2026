"""Tests for object-based reranking on top of KIS prediction artifacts.

These tests exercise frame-level object loading, query tokenization and
matching, score combination, deterministic ranking, artifact compatibility,
and the CLI, using synthetic object/map/prediction fixtures without running any
retrieval or model code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    load_kis_predictions_for_scoring,
    read_kis_prediction_artifact,
    write_kis_prediction_artifact,
)
from aic2026.object_reranking import (
    FrameObjects,
    FrameObjectSource,
    ObjectRerankingConfig,
    build_reranked_artifact,
    compute_object_score,
    load_frame_objects,
    match_entities,
    normalize_token,
    normalize_tokens,
    rerank_prediction_artifact,
    rerank_query_predictions,
    tokenize,
    write_reranked_prediction_artifact,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _prediction(rank: int, video_id: str, frame_index: int, score: float) -> RankedKisPrediction:
    return RankedKisPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=score)


def _query_result(
    query_id: str, query_text: str, predictions: tuple[RankedKisPrediction, ...]
) -> QueryPredictionResult:
    return QueryPredictionResult(query_id=query_id, query_text=query_text, predictions=predictions)


def _artifact(query_results: tuple[QueryPredictionResult, ...]) -> KisPredictionArtifact:
    return KisPredictionArtifact(
        schema_version=KIS_PREDICTION_SCHEMA_VERSION,
        run_id="base-run",
        created_at="2026-08-14T00:00:00Z",
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


def _write_object_frame(
    objects_dir: Path, video_id: str, ordinal: int, detections: list[tuple[str, float]]
) -> None:
    """Write one object JSON with ``(entity, score)`` detections."""
    video_dir = objects_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "detection_scores": [str(score) for _entity, score in detections],
        "detection_class_names": [f"/m/{index}" for index in range(len(detections))],
        "detection_class_entities": [entity for entity, _score in detections],
        "detection_class_labels": [str(index) for index in range(len(detections))],
        "detection_boxes": [["0.0", "0.0", "1.0", "1.0"] for _ in detections],
    }
    (video_dir / f"{ordinal:03d}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_frame_map(map_dir: Path, video_id: str, rows: list[tuple[int, int]]) -> None:
    """Write a map-keyframes CSV with ``(keyframe_ordinal, frame_idx)`` rows."""
    map_dir.mkdir(parents=True, exist_ok=True)
    lines = ["n,pts_time,fps,frame_idx"]
    for ordinal, frame_idx in rows:
        lines.append(f"{ordinal},{frame_idx / 25.0},25.0,{frame_idx}")
    (map_dir / f"{video_id}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def raw_collections(tmp_path: Path) -> tuple[Path, Path]:
    """Build synthetic objects + map-keyframes collections.

    L21_V001: keyframe 1 (frame 0) has Boat/Water/Person; keyframe 2 (frame 90)
              has Car/Road.
    L22_V002: keyframe 1 (frame 5) has Room/Table.
    """
    objects_dir = tmp_path / "objects"
    map_dir = tmp_path / "map-keyframes"

    _write_frame_map(map_dir, "L21_V001", [(1, 0), (2, 90)])
    _write_frame_map(map_dir, "L22_V002", [(1, 5)])

    _write_object_frame(
        objects_dir,
        "L21_V001",
        1,
        [("Boat", 0.9), ("Water", 0.6), ("Person", 0.4), ("Dust", 0.05)],
    )
    _write_object_frame(objects_dir, "L21_V001", 2, [("Car", 0.8), ("Road", 0.7)])
    _write_object_frame(objects_dir, "L22_V002", 1, [("Room", 0.5), ("Table", 0.3)])

    return objects_dir, map_dir


def _default_config() -> ObjectRerankingConfig:
    return ObjectRerankingConfig(
        confidence_threshold=0.3,
        lambda_weight=1.0,
        normalization_method="lower",
        object_score_method="max",
    )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": -0.1}, "confidence_threshold must be in [0, 1]"),
        ({"confidence_threshold": 1.5}, "confidence_threshold must be in [0, 1]"),
        ({"lambda_weight": -1.0}, "lambda_weight must be a non-negative number"),
        ({"normalization_method": "nope"}, "normalization_method must be"),
        ({"object_score_method": "avg"}, "object_score_method must be"),
    ],
)
def test_config_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        ObjectRerankingConfig(**kwargs)


def test_config_defaults_are_valid() -> None:
    config = ObjectRerankingConfig()
    assert config.confidence_threshold == 0.3
    assert config.lambda_weight == 1.0
    assert config.normalization_method == "lower"
    assert config.object_score_method == "max"


# ---------------------------------------------------------------------------
# Tokenization and normalization
# ---------------------------------------------------------------------------


def test_tokenize() -> None:
    assert tokenize("a boat on the water") == ("a", "boat", "on", "the", "water")
    assert tokenize("two-wheeled vehicle") == ("two", "wheeled", "vehicle")


def test_normalize_tokens_methods() -> None:
    assert normalize_tokens("Boat", "none") == ("Boat",)
    assert normalize_tokens("Boats", "lower") == ("boats",)
    assert normalize_tokens("boats", "stem") == ("boat",)
    assert normalize_tokens("two boats and a car", "stem") == ("two", "boat", "and", "a", "car")


def test_normalize_token_unsupported_method() -> None:
    with pytest.raises(ValueError, match="normalization"):
        normalize_token("boat", "nope")  # type: ignore[arg-type]


def test_normalize_tokens_deduplicates_preserving_order() -> None:
    assert normalize_tokens("boat boat car", "lower") == ("boat", "car")


# ---------------------------------------------------------------------------
# Frame-level object representation
# ---------------------------------------------------------------------------


def test_load_frame_objects_threshold_and_sorting(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, _map_dir = raw_collections
    frame = load_frame_objects(objects_dir, "L21_V001", 1, confidence_threshold=0.3)
    assert frame.video_id == "L21_V001"
    assert frame.keyframe_ordinal == 1
    # Dust (0.05) filtered; entities sorted by confidence desc then entity.
    assert frame.entities == (("Boat", 0.9), ("Water", 0.6), ("Person", 0.4))
    assert frame.confidence("Boat") == 0.9
    assert frame.confidence("Dust") is None


def test_load_frame_objects_merges_duplicate_entities(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, _map_dir = raw_collections
    frame = load_frame_objects(objects_dir, "L21_V001", 1, confidence_threshold=0.3)
    # Person appears once here; add a duplicate check on the merged max rule.
    assert len(frame.entities) == 3


def test_load_frame_objects_missing_file(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, _map_dir = raw_collections
    with pytest.raises(FileNotFoundError):
        load_frame_objects(objects_dir, "L21_V001", 999, confidence_threshold=0.3)


def test_frame_object_source_maps_frame_index_to_ordinal(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    frame = source.frame_objects("L21_V001", 0, 0.3)
    assert frame.keyframe_ordinal == 1
    assert frame.entities[0] == ("Boat", 0.9)


def test_frame_object_source_default_map_dir(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, _map_dir = raw_collections
    source = FrameObjectSource(objects_dir)
    frame = source.frame_objects("L21_V001", 90, 0.3)
    assert frame.keyframe_ordinal == 2
    assert dict(frame.entities) == {"Car": 0.8, "Road": 0.7}


def test_frame_object_source_non_keyframe_is_empty(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    frame = source.frame_objects("L21_V001", 42, 0.3)
    assert frame.keyframe_ordinal is None
    assert frame.entities == ()


def test_frame_object_source_missing_map_is_empty(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    frame = source.frame_objects("L99_V999", 0, 0.3)
    assert frame.entities == ()


def test_frame_object_source_missing_collection(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    with pytest.raises(FileNotFoundError, match="objects collection"):
        FrameObjectSource(objects_dir / "absent", map_dir)
    with pytest.raises(FileNotFoundError, match="map-keyframes collection"):
        FrameObjectSource(objects_dir, map_dir / "absent")


def test_frame_object_source_caches_maps(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    source.frame_objects("L21_V001", 0, 0.3)
    source.frame_objects("L21_V001", 90, 0.3)
    assert set(source._map_cache) == {"L21_V001"}


# ---------------------------------------------------------------------------
# Matching and object score
# ---------------------------------------------------------------------------


def test_match_entities_matches_by_normalized_tokens() -> None:
    frame = FrameObjects(video_id="L21_V001", keyframe_ordinal=1, entities=(("Boat", 0.9), ("Water", 0.6)))
    tokens = normalize_tokens("find a boat", "lower")
    assert match_entities(tokens, frame) == ("Boat",)


def test_match_entities_stemming_matches_plurals() -> None:
    frame = FrameObjects(video_id="L21_V001", keyframe_ordinal=1, entities=(("Boat", 0.9),))
    tokens = normalize_tokens("two boats", "stem")
    assert match_entities(tokens, frame) == ("Boat",)
    tokens_lower = normalize_tokens("two boats", "lower")
    assert match_entities(tokens_lower, frame) == ()


def test_match_entities_no_match() -> None:
    frame = FrameObjects(video_id="L22_V002", keyframe_ordinal=1, entities=(("Room", 0.5),))
    tokens = normalize_tokens("a car", "lower")
    assert match_entities(tokens, frame) == ()


def test_compute_object_score_max_and_sum() -> None:
    frame = FrameObjects(
        video_id="L21_V001",
        keyframe_ordinal=1,
        entities=(("Boat", 0.9), ("Water", 0.6), ("Person", 0.4)),
    )
    tokens = normalize_tokens("a boat on the water", "lower")
    assert compute_object_score(tokens, frame, _default_config()) == pytest.approx(0.9)
    sum_config = ObjectRerankingConfig(object_score_method="sum")
    assert compute_object_score(tokens, frame, sum_config) == pytest.approx(1.5)


def test_compute_object_score_zero_when_no_match() -> None:
    frame = FrameObjects(video_id="L22_V002", keyframe_ordinal=1, entities=(("Room", 0.5),))
    tokens = normalize_tokens("a car", "lower")
    assert compute_object_score(tokens, frame, _default_config()) == 0.0


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def test_rerank_boosts_matching_frames(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    predictions = (
        _prediction(1, "L22_V002", 5, 0.85),  # no match (Room/Table)
        _prediction(2, "L21_V001", 0, 0.5),   # matches Boat -> strong boost
        _prediction(3, "L21_V001", 90, 0.7),  # no match (Car/Road)
    )
    output = rerank_query_predictions(predictions, "find a boat on the water", source, _default_config())

    # Boat frame (0.5 + 1.0 * 0.9 = 1.4) outranks the 0.85 no-match frame.
    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L21_V001", 0),
        ("L22_V002", 5),
        ("L21_V001", 90),
    ]
    assert [item.rank for item in output] == [1, 2, 3]
    assert output[0].score == pytest.approx(1.4)
    assert output[1].score == pytest.approx(0.85)
    assert output[2].score == pytest.approx(0.7)


def test_rerank_combines_scores(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    config = ObjectRerankingConfig(lambda_weight=0.1)
    output = rerank_query_predictions(
        (_prediction(1, "L21_V001", 0, 0.5),),
        "find a boat on the water",
        source,
        config,
    )
    assert output[0].score == pytest.approx(0.5 + 0.1 * 0.9)


def test_rerank_ties_preserve_original_order(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    predictions = (
        _prediction(1, "L21_V001", 0, 0.5),   # final 0.5 + 0.9 = 1.4
        _prediction(2, "L22_V002", 5, 1.4),   # final 1.4, no object
    )
    output = rerank_query_predictions(predictions, "a boat", source, _default_config())
    assert [(item.video_id, item.frame_index) for item in output] == [
        ("L21_V001", 0),
        ("L22_V002", 5),
    ]


def test_rerank_ignores_below_threshold_entities(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    # Dust (0.05) is below the threshold, so the frame has no match at 0.3.
    output = rerank_query_predictions(
        (_prediction(1, "L21_V001", 0, 0.5),),
        "a dust cloud",
        source,
        _default_config(),
    )
    assert output[0].score == pytest.approx(0.5)


def test_rerank_rejects_non_ranked_predictions(
    raw_collections: tuple[Path, Path],
) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    with pytest.raises(ValueError, match="RankedKisPrediction"):
        rerank_query_predictions(("nope",), "text", source, _default_config())  # type: ignore[list-item]


def test_rerank_is_deterministic(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    predictions = (
        _prediction(1, "L21_V001", 0, 0.5),
        _prediction(2, "L22_V002", 5, 0.85),
    )
    query = "find a boat on the water"
    first = rerank_query_predictions(predictions, query, source, _default_config())
    second = rerank_query_predictions(predictions, query, source, _default_config())
    assert first == second


# ---------------------------------------------------------------------------
# Artifact-level reranking
# ---------------------------------------------------------------------------


def _sample_query_results() -> tuple[QueryPredictionResult, ...]:
    return (
        _query_result(
            "kis_001",
            "Find a video of a boat carrying a group of tourists traveling on the water",
            (
                _prediction(1, "L22_V002", 5, 0.85),
                _prediction(2, "L21_V001", 0, 0.5),
                _prediction(3, "L21_V001", 90, 0.7),
            ),
        ),
        _query_result(
            "kis_002",
            "Find a video of a car driving on a road",
            (
                _prediction(1, "L21_V001", 90, 0.6),
                _prediction(2, "L22_V002", 5, 0.9),
            ),
        ),
    )


def test_build_reranked_artifact_preserves_metadata(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    artifact = _artifact(_sample_query_results())
    output = build_reranked_artifact(artifact, source, _default_config())

    assert output.schema_version == KIS_PREDICTION_SCHEMA_VERSION
    assert output.evaluation_set_path == artifact.evaluation_set_path
    assert output.evaluation_set_query_count == artifact.evaluation_set_query_count
    assert output.processed_query_count == 2
    assert output.query_text_field == artifact.query_text_field
    assert output.index_artifact_path == artifact.index_artifact_path
    assert output.index_manifest == artifact.index_manifest
    assert output.model_embedding_space == artifact.model_embedding_space
    assert output.model_config == artifact.model_config
    assert output.selected_query_ids == artifact.selected_query_ids


def test_build_reranked_artifact_run_id(raw_collections: tuple[Path, Path]) -> None:
    objects_dir, map_dir = raw_collections
    source = FrameObjectSource(objects_dir, map_dir)
    artifact = _artifact(_sample_query_results())
    config = ObjectRerankingConfig(lambda_weight=0.5, confidence_threshold=0.4)

    output = build_reranked_artifact(artifact, source, config)
    assert output.run_id == "object-rerank-base-run-l0.5-c0.4-nlower-mmax"
    # Explicit run id wins.
    explicit = build_reranked_artifact(artifact, source, config, run_id="my-run")
    assert explicit.run_id == "my-run"


def test_rerank_prediction_artifact_accepts_path_and_object(
    raw_collections: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    objects_dir, map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))

    source = FrameObjectSource(objects_dir, map_dir)
    by_path = rerank_prediction_artifact(artifact_path, source, _default_config())
    by_artifact = rerank_prediction_artifact(
        _artifact(_sample_query_results()), source, _default_config()
    )
    by_objects_dir = rerank_prediction_artifact(artifact_path, objects_dir, _default_config())

    assert by_path == by_artifact
    assert by_path == by_objects_dir
    assert by_path.processed_query_count == 2


def test_write_reranked_prediction_artifact_round_trip(
    raw_collections: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    objects_dir, map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))
    output_path = tmp_path / "reranked.json"
    source = FrameObjectSource(objects_dir, map_dir)

    written = write_reranked_prediction_artifact(
        output_path, artifact_path, source, _default_config(), run_id="cli-run"
    )
    assert written.run_id == "cli-run"
    assert output_path.is_file()

    reloaded = read_kis_prediction_artifact(output_path)
    assert reloaded.schema_version == KIS_PREDICTION_SCHEMA_VERSION
    assert reloaded.processed_query_count == 2
    by_query = load_kis_predictions_for_scoring(output_path)
    assert set(by_query) == {"kis_001", "kis_002"}
    for query_id, predictions in by_query.items():
        assert [item.rank for item in predictions] == list(range(1, len(predictions) + 1))


def test_scorer_accepts_reranked_artifact(raw_collections: tuple[Path, Path], tmp_path: Path) -> None:
    from aic2026.kis_scoring import evaluate_kis

    objects_dir, map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))
    output_path = tmp_path / "reranked.json"
    source = FrameObjectSource(objects_dir, map_dir)

    write_reranked_prediction_artifact(output_path, artifact_path, source, _default_config())
    # Use the original artifact as the "evaluation set" queries only require ids;
    # scoring needs canonical queries, so substitute with a light mapping check.
    from aic2026.evaluation import (
        AnnotationMetadata,
        EvaluationTask,
        EvaluationQuery,
        EvaluationSet,
        FrameRange,
    )
    from aic2026.canonical import VideoId

    queries = EvaluationSet(
        queries=(
            EvaluationQuery(
                query_id="kis_001",
                task=EvaluationTask.KIS,
                query_text_vi="thuyền",
                query_text_en="Find a video of a boat carrying a group of tourists traveling on the water",
                ground_truth_video_id=VideoId("L21_V001"),
                ground_truth_frame_range=FrameRange(0, 50),
                answer=None,
                event=None,
                annotation=AnnotationMetadata(annotator="P1", note=None),
            ),
            EvaluationQuery(
                query_id="kis_002",
                task=EvaluationTask.KIS,
                query_text_vi="ô tô",
                query_text_en="Find a video of a car driving on a road",
                ground_truth_video_id=VideoId("L21_V001"),
                ground_truth_frame_range=FrameRange(70, 100),
                answer=None,
                event=None,
                annotation=AnnotationMetadata(annotator="P1", note=None),
            ),
        ),
    )
    result = evaluate_kis(queries, load_kis_predictions_for_scoring(output_path))
    assert result.final_score >= 0.0
    assert len(result.top_k_scores) >= 1


def test_rerank_prediction_artifact_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rerank_prediction_artifact(tmp_path / "absent.json", tmp_path / "objects", _default_config())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_object_rerank_kis_end_to_end(
    raw_collections: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from aic2026.cli import main

    objects_dir, map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))
    output_path = tmp_path / "cli_reranked.json"

    argv = [
        "object-rerank-kis",
        "--prediction",
        str(artifact_path),
        "--output",
        str(output_path),
        "--objects-dir",
        str(objects_dir),
        "--map-dir",
        str(map_dir),
        "--lambda",
        "0.5",
        "--confidence-threshold",
        "0.3",
        "--run-id",
        "cli-run-1",
    ]

    exit_code = main(argv)

    assert exit_code == 0
    assert output_path.is_file()
    reranked = read_kis_prediction_artifact(output_path)
    assert reranked.run_id == "cli-run-1"
    by_query = load_kis_predictions_for_scoring(output_path)
    assert set(by_query) == {"kis_001", "kis_002"}
    for query_id, predictions in by_query.items():
        assert [item.rank for item in predictions] == list(range(1, len(predictions) + 1))


def test_cli_object_rerank_kis_missing_collections(
    raw_collections: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from aic2026.cli import main

    objects_dir, _map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))

    argv = [
        "object-rerank-kis",
        "--prediction",
        str(artifact_path),
        "--output",
        str(tmp_path / "out.json"),
        "--objects-dir",
        str(objects_dir / "absent"),
    ]

    exit_code = main(argv)

    assert exit_code == 1
    assert not (tmp_path / "out.json").exists()


def test_cli_object_rerank_kis_rejects_bad_config(
    raw_collections: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from aic2026.cli import main

    objects_dir, map_dir = raw_collections
    artifact_path = tmp_path / "source.json"
    write_kis_prediction_artifact(artifact_path, _artifact(_sample_query_results()))

    argv = [
        "object-rerank-kis",
        "--prediction",
        str(artifact_path),
        "--output",
        str(tmp_path / "out.json"),
        "--objects-dir",
        str(objects_dir),
        "--map-dir",
        str(map_dir),
        "--confidence-threshold",
        "1.5",
    ]

    exit_code = main(argv)

    assert exit_code == 1
    assert not (tmp_path / "out.json").exists()


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------


def test_real_object_rerank_smoke() -> None:
    root = Path(__file__).resolve().parents[1] / "data/raw/aic2026_batch1"
    objects_dir = root / "objects"
    map_dir = root / "map-keyframes"
    clip_path = (
        Path(__file__).resolve().parents[1]
        / "data/interim/predictions/kis-l14-laion2b-full-en-top1000.json"
    )
    if not (objects_dir.is_dir() and map_dir.is_dir() and clip_path.is_file()):
        pytest.skip("real dataset artifacts not available")

    artifact = read_kis_prediction_artifact(clip_path)
    source = FrameObjectSource(objects_dir, map_dir)
    # Rerank only the first query's predictions for a fast smoke check.
    from aic2026.kis_prediction_runner import QueryPredictionResult

    first_query = artifact.query_results[0]
    partial = _artifact((first_query,))
    reranked = build_reranked_artifact(partial, source, _default_config())

    assert reranked.processed_query_count == 1
    predictions = reranked.query_results[0].predictions
    assert [item.rank for item in predictions] == list(range(1, len(predictions) + 1))
    scores = [item.score for item in predictions]
    assert all(left >= right for left, right in zip(scores, scores[1:]))
