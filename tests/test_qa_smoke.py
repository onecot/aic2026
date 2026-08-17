"""Integration tests for the end-to-end QA smoke pipeline (synthetic, no VQA).

Covers synthetic QA dataset loading, QA runner execution on a small index,
artifact serialization round-trip, deterministic ranking, structural
validation, and submission generation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.index_artifact import FRAMES_FILENAME, iter_frames
from aic2026.qa.qa_prediction_runner import QaPredictionRunnerConfig, read_qa_artifact, run_qa_prediction
from aic2026.qa.qa_smoke import (
    SMOKE_EMBEDDING_DIM,
    SMOKE_FRAME_STRIDE,
    SMOKE_FRAMES_PER_VIDEO,
    SMOKE_VIDEO_IDS,
    QaPlaceholderEncoder,
    QaSmokeConfig,
    build_synthetic_index,
    load_index_frame_map,
    run_qa_smoke,
    smoke_embedding_space,
    validate_qa_artifact,
    with_placeholder_answers,
)
from aic2026.qa.qa_submission import QA_SUBMISSION_CSV_COLUMNS, build_submission_rows, write_qa_submission_csv
from aic2026.qa.qa_types import QaPrediction, QaPredictionArtifact, QaPredictionManifest, QaQueryResult

SYNTHETIC_EVAL = Path(__file__).resolve().parents[1] / "data/eval/qa_smoke_eval.csv"


def _config(tmp_path: Path, **overrides: object) -> QaSmokeConfig:
    fields = {
        "evaluation_set_path": SYNTHETIC_EVAL,
        "work_dir": tmp_path / "work",
    }
    fields.update(overrides)
    return QaSmokeConfig(**fields)


def _manifest(query_count: int = 1) -> QaPredictionManifest:
    return QaPredictionManifest(
        schema_version="1.0",
        run_id="smoke-test",
        created_at="2026-08-17T00:00:00Z",
        evaluation_set_path=str(SYNTHETIC_EVAL),
        evaluation_set_query_count=query_count,
        processed_query_count=query_count,
        query_text_field="query_text_vi",
        top_k=10,
        video_aggregation=None,
        video_aggregation_top_n=None,
        window_max_gap=None,
        embedding_space_compatible=True,
        embedding_space_override_used=False,
        index_artifact_path="/tmp/index",
        index_manifest={"artifact_id": "smoke-test"},
        model_embedding_space={"space_id": "smoke-test"},
        model_config=None,
        selected_query_ids=None,
    )


def _artifact(predictions: tuple[QaPrediction, ...]) -> QaPredictionArtifact:
    result = QaQueryResult(
        query_id="qa_smoke_001",
        query_text="question",
        answer="",
        predictions=predictions,
        video_candidates=(),
        window_candidates=(),
    )
    return QaPredictionArtifact(manifest=_manifest(1), query_results=(result,))


# --- 1. Synthetic QA dataset loading ---------------------------------------


def test_synthetic_qa_dataset_loads_and_matches_smoke_videos() -> None:
    evaluation_set = load_eval_set(SYNTHETIC_EVAL)
    qa_queries = tuple(evaluation_set.iter_queries(EvaluationTask.QA))
    assert len(qa_queries) == 15
    assert evaluation_set.task_counts() == {EvaluationTask.QA: 15}
    assert {query.query_id for query in qa_queries} == {
        f"qa_smoke_{i:03d}" for i in range(1, 16)
    }
    max_frame = (SMOKE_FRAMES_PER_VIDEO - 1) * SMOKE_FRAME_STRIDE
    for query in qa_queries:
        assert query.query_text_vi.strip() != ""
        assert query.query_text_en and query.query_text_en.strip() != ""
        assert query.answer and query.answer.strip() != ""
        assert str(query.ground_truth_video_id) in SMOKE_VIDEO_IDS
        assert query.ground_truth_frame_range.start <= query.ground_truth_frame_range.end
        assert query.ground_truth_frame_range.end <= max_frame


# --- 2. QA runner execution on a small index --------------------------------


def test_qa_runner_produces_predictions_for_every_query(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index_dir = build_synthetic_index(tmp_path / "index", config)
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), SMOKE_EMBEDDING_DIM)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=config.evaluation_set_path,
            index_artifact_dir=index_dir,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=config.top_k,
            query_text_field="query_text_vi",
            run_id="smoke-test",
        ),
        encoder,
    )

    assert artifact.manifest.processed_query_count == 15
    total_frames = len(SMOKE_VIDEO_IDS) * SMOKE_FRAMES_PER_VIDEO
    expected = min(config.top_k, total_frames)
    for result in artifact.query_results:
        assert result.predictions, f"query {result.query_id} produced no predictions"
        assert len(result.predictions) == expected
        assert [p.rank for p in result.predictions] == list(range(1, expected + 1))


# --- 3. Artifact serialization round-trip -----------------------------------


def test_artifact_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index_dir = build_synthetic_index(tmp_path / "index", config)
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), SMOKE_EMBEDDING_DIM)
    output_dir = tmp_path / "qa_predictions"

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=config.evaluation_set_path,
            index_artifact_dir=index_dir,
            prediction_output_dir=output_dir,
            top_k=10,
            run_id="smoke-test",
        ),
        encoder,
    )
    restored = read_qa_artifact(output_dir)
    assert restored == artifact


def test_placeholder_encoder_is_deterministic() -> None:
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), SMOKE_EMBEDDING_DIM)
    first = encoder.encode_text("Where did the person in red go?")
    second = encoder.encode_text("Where did the person in red go?")
    other = encoder.encode_text("What color was the parked vehicle?")
    assert np.array_equal(first.vector, second.vector)
    assert not np.array_equal(first.vector, other.vector)


# --- 4. Structural validation -----------------------------------------------


def test_validate_qa_artifact_accepts_valid_artifact() -> None:
    frame_map = {"L21_V001": {0, 10, 20}}
    predictions = (
        QaPrediction(rank=1, video_id="L21_V001", frame_index=10, score=0.95),
        QaPrediction(rank=2, video_id="L21_V001", frame_index=20, score=0.80),
    )
    assert validate_qa_artifact(_artifact(predictions), frame_map) == []


def test_validate_qa_artifact_reports_structural_problems() -> None:
    frame_map = {"L21_V001": {0, 10, 20}}
    predictions = (
        QaPrediction(rank=1, video_id="L21_V999", frame_index=10, score=0.95),
        QaPrediction(rank=2, video_id="L21_V001", frame_index=5, score=float("nan")),
    )
    problems = validate_qa_artifact(_artifact(predictions), frame_map)
    joined = "\n".join(problems)
    assert "unknown video 'L21_V999'" in joined
    assert "invalid frame 5 in video 'L21_V001'" in joined
    assert "non-finite score" in joined


def test_validate_qa_artifact_reports_missing_predictions() -> None:
    frame_map = {"L21_V001": {0, 10, 20}}
    problems = validate_qa_artifact(_artifact(()), frame_map)
    assert problems == ["query 'qa_smoke_001' produced no predictions"]


def test_validate_qa_artifact_rejects_infinity_score() -> None:
    frame_map = {"L21_V001": {0, 10, 20}}
    problems = validate_qa_artifact(
        _artifact((QaPrediction(rank=1, video_id="L21_V001", frame_index=10, score=math.inf),)),
        frame_map,
    )
    assert any("non-finite score" in problem for problem in problems)


# --- 5. Submission generation -----------------------------------------------


def test_submission_generation_rows_and_columns(tmp_path: Path) -> None:
    result = run_qa_smoke(_config(tmp_path))
    submission_path = Path(result.submission_path)
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == QA_SUBMISSION_CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == result.prediction_count
    assert len(rows) == result.query_count * 50
    assert all(row["answer"].strip() != "" for row in rows)
    assert {row["query_id"] for row in rows} == {f"qa_smoke_{i:03d}" for i in range(1, 16)}
    for row in rows:
        assert row["video_id"] in SMOKE_VIDEO_IDS
        assert int(row["frame_index"]) % SMOKE_FRAME_STRIDE == 0


def test_submission_rejects_query_without_evidence() -> None:
    result = QaQueryResult(
        query_id="qa_smoke_001",
        query_text="question",
        answer="answer",
        predictions=(),
        video_candidates=(),
        window_candidates=(),
    )
    artifact = QaPredictionArtifact(manifest=_manifest(1), query_results=(result,))
    with pytest.raises(ValueError, match="has no evidence"):
        build_submission_rows(artifact)
    with pytest.raises(ValueError, match="has no evidence"):
        write_qa_submission_csv(artifact, Path("/tmp/unused.csv"))


def test_placeholder_answers_fill_and_fall_back() -> None:
    artifact = _artifact(
        (QaPrediction(rank=1, video_id="L21_V001", frame_index=10, score=0.95),)
    )
    filled = with_placeholder_answers(artifact, {"qa_smoke_001": "expected answer"}, "fallback")
    assert filled.query_results[0].answer == "expected answer"
    missing = with_placeholder_answers(artifact, {}, "fallback")
    assert missing.query_results[0].answer == "fallback"


# --- 6. End-to-end smoke run -------------------------------------------------


def test_full_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    result = run_qa_smoke(_config(tmp_path))

    assert result.query_count == 15
    assert result.prediction_count == 15 * 50
    assert result.determinism_verified is True
    assert result.placeholder_answers_filled == 15
    assert Path(result.artifact_dir).is_dir()
    assert Path(result.submission_path).is_file()

    index_frame_map = load_index_frame_map(result.index_dir)
    assert set(index_frame_map) == set(SMOKE_VIDEO_IDS)
    assert len(index_frame_map["L21_V001"]) == SMOKE_FRAMES_PER_VIDEO
    frame_indices = sorted(index_frame_map["L21_V001"])
    assert frame_indices == list(range(0, (SMOKE_FRAMES_PER_VIDEO - 1) * SMOKE_FRAME_STRIDE + 1, SMOKE_FRAME_STRIDE))

    frame_rows = list(iter_frames(Path(result.index_dir) / FRAMES_FILENAME))
    assert len(frame_rows) == len(SMOKE_VIDEO_IDS) * SMOKE_FRAMES_PER_VIDEO


def test_full_smoke_pipeline_is_reproducible_across_runs(tmp_path: Path) -> None:
    first = run_qa_smoke(_config(tmp_path, work_dir=tmp_path / "run-a"))
    second = run_qa_smoke(_config(tmp_path, work_dir=tmp_path / "run-b"))
    assert first.determinism_verified is True
    assert second.determinism_verified is True
    assert first.prediction_count == second.prediction_count
    first_rows = (Path(first.submission_path).read_text(encoding="utf-8").splitlines())
    second_rows = (Path(second.submission_path).read_text(encoding="utf-8").splitlines())
    assert first_rows == second_rows