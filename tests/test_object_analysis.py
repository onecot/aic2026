"""Tests for the object-analysis module.

These tests exercise the vocabulary scan, query-object coverage, GT-frame
analysis, retrieval-potential estimation, report formatting, and the CLI. They
use a small synthetic objects / map-keyframes / evaluation / prediction
fixture so no real dataset is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.object_analysis import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_PREDICTION_TOP_K,
    GtFrameObjectAnalysis,
    ObjectCoverageSummary,
    ObjectVocabularyStats,
    QueryObjectCoverage,
    RetrievalPotentialQuery,
    RetrievalPotentialSummary,
    ThresholdEffect,
    analyze_gt_frames,
    analyze_object_vocabulary,
    analyze_query_object_coverage,
    analyze_retrieval_potential,
    build_entity_token_vocabulary,
    extract_object_words,
    format_report,
    run_object_analysis,
)

_EVAL_HEADER = (
    "query_id,task,query_text_vi,query_text_en,gt_video_id,"
    "gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note"
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_object_frame(objects_dir: Path, video_id: str, ordinal: int, detections: list[tuple[str, float]]) -> None:
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


def _write_eval(path: Path, rows: list[tuple[str, str, str, str, str]]) -> Path:
    """Write a KIS-only eval CSV with ``(id, vi, en, gt_video, frame_range)``."""
    lines = [_EVAL_HEADER]
    for query_id, vi, en, gt_video, frame_range in rows:
        lines.append(f"{query_id},kis,{vi},{en},{gt_video},{frame_range},,,P1,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_predictions(
    path: Path,
    eval_path: Path,
    by_query: dict[str, list[dict[str, object]]],
) -> Path:
    """Write a KIS prediction artifact with explicit per-query predictions.

    Each prediction dict must contain ``rank``, ``video_id``, ``frame_index``
    and may contain ``score``.
    """
    artifact = {
        "schema_version": "1.0",
        "run_id": "test-clip",
        "created_at": "2026-08-14T00:00:00Z",
        "evaluation_set_path": str(eval_path),
        "evaluation_set_query_count": len(by_query),
        "processed_query_count": len(by_query),
        "query_text_field": "query_text_en",
        "top_k": 1000,
        "embedding_space_compatible": True,
        "embedding_space_override_used": False,
        "index_artifact_path": "/tmp/index",
        "index_manifest": {"artifact_id": "dummy"},
        "model_embedding_space": {"backend": "openclip"},
        "model_config": None,
        "selected_query_ids": None,
        "query_results": [
            {
                "query_id": query_id,
                "query_text": "x",
                "predictions": [
                    {"rank": pred.get("rank", index + 1), "score": pred.get("score", 0.5), **pred}
                    for index, pred in enumerate(preds)
                ],
            }
            for query_id, preds in by_query.items()
        ],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


@pytest.fixture
def object_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a small synthetic raw dataset and evaluation inputs.

    Videos:
        - L21_V001: keyframes 1 (frame 0) and 2 (frame 90).
        - L22_V002: keyframe 1 (frame 5).
    """
    root = tmp_path / "raw"
    objects_dir = root / "objects"
    map_dir = root / "map-keyframes"

    _write_frame_map(map_dir, "L21_V001", [(1, 0), (2, 90)])
    _write_frame_map(map_dir, "L22_V002", [(1, 5)])

    # L21_V001 keyframe 1 contains a boat; keyframe 2 contains a car.
    _write_object_frame(
        objects_dir,
        "L21_V001",
        1,
        [("Boat", 0.8), ("Water", 0.6), ("Person", 0.4), ("Noise", 0.05)],
    )
    _write_object_frame(
        objects_dir,
        "L21_V001",
        2,
        [("Car", 0.9), ("Road", 0.7)],
    )
    # L22_V002 contains only generic low-confidence detections.
    _write_object_frame(
        objects_dir,
        "L22_V002",
        1,
        [("Room", 0.35), ("Table", 0.2)],
    )

    eval_path = _write_eval(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "a boat on the water", "L21_V001", "0,50"),
            ("kis_002", "ô tô", "a car driving", "L21_V001", "70,100"),
            ("kis_003", "xe máy", "a motorcycle", "L22_V002", "0,20"),
        ],
    )

    clip_path = _write_predictions(
        tmp_path / "clip.json",
        eval_path,
        {
            # GT frame (L21_V001 frame 0) plus false positives.
            "kis_001": [
                {"rank": 1, "video_id": "L21_V001", "frame_index": 0, "score": 0.9},
                {"rank": 2, "video_id": "L22_V002", "frame_index": 5, "score": 0.8},
                {"rank": 3, "video_id": "L21_V001", "frame_index": 90, "score": 0.7},
            ],
            "kis_002": [
                {"rank": 1, "video_id": "L21_V001", "frame_index": 90, "score": 0.9},
                {"rank": 2, "video_id": "L22_V002", "frame_index": 5, "score": 0.8},
            ],
            "kis_003": [
                {"rank": 1, "video_id": "L22_V002", "frame_index": 5, "score": 0.9},
            ],
        },
    )
    return objects_dir, map_dir, eval_path, clip_path


# ---------------------------------------------------------------------------
# Vocabulary statistics
# ---------------------------------------------------------------------------


def test_vocabulary_counts_frames_detections_and_entities(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, _map_dir, _eval_path, _clip_path = object_fixture
    stats = analyze_object_vocabulary(objects_dir)

    assert stats.total_frames == 3
    assert stats.total_detections == 8
    assert stats.unique_entities == 8
    assert stats.avg_detections_per_frame() == pytest.approx(8 / 3)
    assert stats.confidence_stats["min"] == pytest.approx(0.05)
    assert stats.confidence_stats["max"] == pytest.approx(0.9)


def test_vocabulary_threshold_effects(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, _map_dir, _eval_path, _clip_path = object_fixture
    stats = analyze_object_vocabulary(objects_dir, thresholds=(0.3, 0.7))

    effects = {effect.threshold: effect for effect in stats.threshold_effects}
    assert effects[0.3].kept_detections == 6  # Boat, Water, Person, Car, Road, Room
    assert effects[0.3].kept_unique_entities == 6
    assert effects[0.3].frames_with_detection == 3
    assert effects[0.7].kept_detections == 3  # Boat, Car, Road
    assert effects[0.7].kept_unique_entities == 3
    assert effects[0.7].frames_with_detection == 2


def test_vocabulary_top_classes_and_determinism(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, _map_dir, _eval_path, _clip_path = object_fixture
    stats = analyze_object_vocabulary(objects_dir, top_classes=3)
    top = list(stats.entity_frequency)
    assert len(top) == 3
    assert top[0][0] in {"Boat", "Car", "Road", "Water", "Person", "Room", "Table", "Noise"}

    first = analyze_object_vocabulary(objects_dir).to_dict()
    second = analyze_object_vocabulary(objects_dir).to_dict()
    assert first == second


def test_vocabulary_rejects_bad_inputs(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, _map_dir, _eval_path, _clip_path = object_fixture
    with pytest.raises(ValueError, match="threshold"):
        analyze_object_vocabulary(objects_dir, thresholds=())
    with pytest.raises(ValueError, match="threshold"):
        analyze_object_vocabulary(objects_dir, thresholds=(1.5,))
    with pytest.raises(FileNotFoundError):
        analyze_object_vocabulary(objects_dir / "absent")


def test_vocabulary_json_is_serializable(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, _map_dir, _eval_path, _clip_path = object_fixture
    stats = analyze_object_vocabulary(objects_dir)
    payload = stats.to_dict()
    assert payload["total_detections"] == 8
    assert payload["threshold_effects"][0]["threshold"] == DEFAULT_CONFIDENCE_THRESHOLDS[0]
    json.dumps(payload)


def test_threshold_effect_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ThresholdEffect(threshold=2.0, kept_detections=1, kept_unique_entities=1, frames_with_detection=1, avg_detections_per_frame=1.0)


# ---------------------------------------------------------------------------
# Object-word extraction
# ---------------------------------------------------------------------------


def test_extract_object_words() -> None:
    tokens = {"boat", "car", "water"}
    assert extract_object_words("a boat on the water", tokens) == ("boat", "water")
    assert extract_object_words("A Car.", tokens) == ("car",)
    assert extract_object_words("nothing here", tokens) == ()
    with pytest.raises(ValueError, match="query_text"):
        extract_object_words("", tokens)


def test_build_entity_token_vocabulary() -> None:
    frequency = (("Boat", 10), ("Land vehicle", 5), ("Two-wheeled vehicle", 2))
    tokens = build_entity_token_vocabulary(frequency)
    assert "boat" in tokens
    assert "land" in tokens
    assert "vehicle" in tokens
    assert "two" in tokens


# ---------------------------------------------------------------------------
# Query-object coverage
# ---------------------------------------------------------------------------


def test_coverage_matches_and_no_match(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    eval_set = load_eval_set(eval_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    stats = analyze_object_vocabulary(objects_dir)
    tokens = build_entity_token_vocabulary(stats.entity_frequency)

    records, summary = analyze_query_object_coverage(
        queries, objects_dir, map_dir, tokens, min_confidence=DEFAULT_MIN_CONFIDENCE
    )
    by_id = {record.query_id: record for record in records}

    # kis_001: query has "boat" and GT frame (L21_V001 kf1) contains "Boat".
    assert by_id["kis_001"].status == "MATCH"
    assert "boat" in by_id["kis_001"].object_words
    assert "Boat" in by_id["kis_001"].matched_entities
    # kis_002: query has "car", GT range covers kf2 which contains "Car".
    assert by_id["kis_002"].status == "MATCH"
    # kis_003: query has no word in the vocabulary (motorcycle not detected).
    assert by_id["kis_003"].status == "NO_MATCH"

    assert summary.total == 3
    assert summary.matched_count == 2
    assert summary.no_match_count == 1
    assert len(summary.matched_examples) == 2
    assert len(summary.unmatched_examples) == 1


def test_coverage_example_limit(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    tokens = build_entity_token_vocabulary(analyze_object_vocabulary(objects_dir).entity_frequency)
    _records, summary = analyze_query_object_coverage(
        queries, objects_dir, map_dir, tokens, example_limit=1
    )
    assert len(summary.matched_examples) == 1
    assert len(summary.unmatched_examples) == 1


def test_coverage_uses_query_text_field(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    tokens = build_entity_token_vocabulary(analyze_object_vocabulary(objects_dir).entity_frequency)

    records_vi, _ = analyze_query_object_coverage(
        queries, objects_dir, map_dir, tokens, query_text_field="query_text_vi"
    )
    records_en, _ = analyze_query_object_coverage(
        queries, objects_dir, map_dir, tokens, query_text_field="query_text_en"
    )
    assert records_vi[0].status != records_en[0].status or records_vi[0].object_words != records_en[0].object_words


def test_coverage_rejects_invalid_field(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    tokens = build_entity_token_vocabulary(analyze_object_vocabulary(objects_dir).entity_frequency)
    with pytest.raises(ValueError, match="query_text_field"):
        analyze_query_object_coverage(queries, objects_dir, map_dir, tokens, query_text_field="nope")


def test_coverage_summary_validation() -> None:
    with pytest.raises(ValueError, match="matched"):
        ObjectCoverageSummary(total=3, matched_count=1, no_match_count=1, matched_examples=(), unmatched_examples=())


def test_query_object_coverage_status_validation() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        QueryObjectCoverage(
            query_id="kis_001", query_text_en="x", query_text_vi="y",
            object_words=(), gt_entities=("Boat",), matched_entities=(), status="MATCH",
        )


# ---------------------------------------------------------------------------
# GT-frame analysis
# ---------------------------------------------------------------------------


def test_gt_frame_analysis(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))

    analyses = analyze_gt_frames(queries, objects_dir, map_dir, min_confidence=0.3)
    by_id = {analysis.query_id: analysis for analysis in analyses}

    # kis_001 range 0..50 covers kf1 (frame 0) only.
    assert by_id["kis_001"].gt_keyframe_ordinals == (1,)
    assert dict(by_id["kis_001"].gt_object_list) == {"Boat": 1, "Water": 1, "Person": 1}
    assert by_id["kis_001"].total_detections == 3
    # kis_002 range 70..100 covers kf2 (frame 90).
    assert by_id["kis_002"].gt_keyframe_ordinals == (2,)
    assert dict(by_id["kis_002"].gt_object_list) == {"Car": 1, "Road": 1}
    # kis_003 covers L22_V002 kf1 (frame 5).
    assert by_id["kis_003"].gt_keyframe_ordinals == (1,)
    assert dict(by_id["kis_003"].gt_object_list) == {"Room": 1}


def test_gt_frame_analysis_json(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    analyses = analyze_gt_frames(queries, objects_dir, map_dir)
    payload = analyses[0].to_dict()
    assert payload["query_id"] == "kis_001"
    assert "confidence_stats" in payload
    json.dumps(payload)


def test_gt_frame_analysis_validation() -> None:
    with pytest.raises(ValueError):
        GtFrameObjectAnalysis(
            query_id="", gt_video_id="L21_V001", gt_keyframe_ordinals=(), gt_object_list=(), confidence_stats={}, total_detections=0,
        )


# ---------------------------------------------------------------------------
# Retrieval potential
# ---------------------------------------------------------------------------


def test_retrieval_potential_separates_gt_from_fp(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring

    objects_dir, map_dir, eval_path, clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    predictions = load_kis_predictions_for_scoring(clip_path)

    records, summary = analyze_retrieval_potential(
        queries, predictions, objects_dir, map_dir, top_k=1000, min_confidence=0.3
    )
    by_id = {record.query_id: record for record in records}

    # kis_001: GT frame L21_V001 frame 0 (contains Boat). FPs share no GT entity.
    record = by_id["kis_001"]
    assert record.gt_frame_predictions == 1
    assert record.gt_overlap_count == 1
    assert record.gt_overlap_rate == pytest.approx(1.0)
    assert record.false_positive_overlap_count == 0
    assert record.false_positive_overlap_rate == pytest.approx(0.0)
    assert record.separation == pytest.approx(1.0)

    # kis_002: GT frame L21_V001 frame 90 (Car/Road). FP L22_V002 frame 5 shares nothing.
    record = by_id["kis_002"]
    assert record.gt_overlap_rate == pytest.approx(1.0)
    assert record.false_positive_overlap_rate == pytest.approx(0.0)

    # kis_003: only the GT video frame predicted, no false positives.
    record = by_id["kis_003"]
    assert record.gt_frame_predictions == 1
    assert record.false_positive_overlap_rate is None
    assert record.separation is None

    assert summary.total == 3
    assert summary.queries_with_gt_predictions == 3
    assert summary.queries_with_separation == 2
    assert summary.queries_where_objects_help == 2
    assert summary.avg_separation == pytest.approx(1.0)


def test_retrieval_potential_requires_predictions(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    with pytest.raises(ValueError, match="missing predictions"):
        analyze_retrieval_potential(queries, {}, objects_dir, map_dir)


def test_retrieval_potential_rejects_top_k(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring

    objects_dir, map_dir, eval_path, clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    predictions = load_kis_predictions_for_scoring(clip_path)
    with pytest.raises(ValueError, match="top_k"):
        analyze_retrieval_potential(queries, predictions, objects_dir, map_dir, top_k=0)


def test_retrieval_potential_json(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring

    objects_dir, map_dir, eval_path, clip_path = object_fixture
    queries = tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS))
    predictions = load_kis_predictions_for_scoring(clip_path)
    records, summary = analyze_retrieval_potential(queries, predictions, objects_dir, map_dir)
    json.dumps(records[0].to_dict())
    json.dumps(summary.to_dict())


def test_retrieval_summary_validation() -> None:
    with pytest.raises(ValueError):
        RetrievalPotentialSummary(
            total=-1, queries_with_gt_predictions=0, queries_with_separation=0,
            queries_where_objects_help=0, avg_separation=None,
            mean_gt_overlap_rate=None, mean_false_positive_overlap_rate=None,
        )


# ---------------------------------------------------------------------------
# Report formatting and full pipeline
# ---------------------------------------------------------------------------


def test_format_report_contains_sections(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, clip_path = object_fixture
    report = run_object_analysis(
        objects_dir, map_dir, eval_path, clip_path, min_confidence=0.3
    )
    text = format_report(report)
    assert "1. Object vocabulary" in text
    assert "2. Query-object coverage" in text
    assert "3. Ground-truth frame analysis" in text
    assert "4. Retrieval potential (CLIP top-k)" in text
    assert "queries with object coverage (MATCH): 2/3" in text
    assert "kis_001" in text


def test_run_object_analysis_json(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, clip_path = object_fixture
    report = run_object_analysis(objects_dir, map_dir, eval_path, clip_path)
    payload = report.to_dict()
    assert payload["vocabulary"]["total_frames"] == 3
    assert payload["coverage"]["matched_count"] == 2
    assert len(payload["gt_frames"]) == 3
    assert payload["retrieval_potential"]["summary"]["total"] == 3
    json.dumps(payload)


def test_run_object_analysis_deterministic(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, clip_path = object_fixture
    first = run_object_analysis(objects_dir, map_dir, eval_path, clip_path).to_dict()
    second = run_object_analysis(objects_dir, map_dir, eval_path, clip_path).to_dict()
    assert first == second


def test_run_object_analysis_requires_predictions(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    with pytest.raises(FileNotFoundError):
        run_object_analysis(objects_dir, map_dir, eval_path, "absent.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_objects_analyze_end_to_end(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.cli import main

    objects_dir, map_dir, eval_path, clip_path = object_fixture
    output_path = objects_dir.parent.parent / "analysis.json"
    exit_code = main(
        [
            "objects-analyze",
            "--objects-dir",
            str(objects_dir),
            "--map-dir",
            str(map_dir),
            "--evaluation-set",
            str(eval_path),
            "--clip-predictions",
            str(clip_path),
            "--min-confidence",
            "0.3",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["coverage"]["matched_count"] == 2
    assert payload["vocabulary"]["total_frames"] == 3


def test_cli_objects_analyze_missing_directory(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.cli import main

    _objects_dir, map_dir, eval_path, clip_path = object_fixture
    exit_code = main(
        [
            "objects-analyze",
            "--objects-dir",
            str(object_fixture[0].parent / "absent"),
            "--map-dir",
            str(map_dir),
            "--evaluation-set",
            str(eval_path),
            "--clip-predictions",
            str(clip_path),
        ]
    )
    assert exit_code == 1


def test_cli_objects_analyze_requires_clip(object_fixture: tuple[Path, Path, Path, Path]) -> None:
    from aic2026.cli import main

    objects_dir, map_dir, eval_path, _clip_path = object_fixture
    exit_code = main(
        [
            "objects-analyze",
            "--objects-dir",
            str(objects_dir),
            "--map-dir",
            str(map_dir),
            "--evaluation-set",
            str(eval_path),
            "--clip-predictions",
            str(objects_dir / "absent-clip.json"),
        ]
    )
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------


def test_real_object_analysis_smoke() -> None:
    root = Path(__file__).resolve().parents[1] / "data/raw/aic2026_batch1"
    objects_dir = root / "objects"
    map_dir = root / "map-keyframes"
    eval_csv = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    clip_path = (
        Path(__file__).resolve().parents[1]
        / "data/interim/predictions/kis-l14-laion2b-full-en-top1000.json"
    )
    if not (objects_dir.is_dir() and map_dir.is_dir() and eval_csv.is_file() and clip_path.is_file()):
        pytest.skip("real dataset artifacts not available")
    report = run_object_analysis(
        objects_dir, map_dir, eval_csv, clip_path, max_videos=2, prediction_top_k=50
    )
    assert report.vocabulary.total_frames > 0
    assert report.coverage.total == 101
    assert len(report.gt_analyses) == 101
    assert report.retrieval.total == 101