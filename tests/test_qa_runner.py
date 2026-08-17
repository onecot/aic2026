"""Tests for the QA prediction runner and artifact contract."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace, legacy_embedding_space
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import IndexManifest, write_frames, write_manifest
from aic2026.qa.qa_prediction_runner import (
    QaPredictionRunnerConfig,
    load_qa_predictions_by_query,
    read_qa_artifact,
    run_qa_prediction,
)
from aic2026.qa.qa_types import QaPrediction, QaQueryResult

_HEADERS = [
    "query_id",
    "task",
    "query_text_vi",
    "query_text_en",
    "gt_video_id",
    "gt_frame_start",
    "gt_frame_end",
    "gt_answer",
    "gt_events",
    "annotator",
    "note",
]


class _FakeEncoder(EmbeddingEncoder):
    def __init__(self, embedding_space: EmbeddingSpace, vectors_by_text: dict[str, np.ndarray] | None = None) -> None:
        self._space = embedding_space
        self._vectors_by_text = vectors_by_text or {}

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_image(self, image: object) -> Embedding:
        raise NotImplementedError("image encoding is not used in QA query prediction tests")

    def encode_text(self, text: str) -> Embedding:
        vector = self._vectors_by_text.get(text)
        if vector is None:
            vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return Embedding(kind="text", vector=vector.astype(np.float32, copy=True), space=self._space)


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _qa_row(query_id: str, query_text_vi: str, query_text_en: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "qa",
        "query_text_vi": query_text_vi,
        "query_text_en": query_text_en,
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "dap an",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }


def _space(space_id: str = "test-space-v1") -> EmbeddingSpace:
    return EmbeddingSpace(
        backend="test-backend",
        model_id="test-model",
        checkpoint_id="test-checkpoint",
        space_id=space_id,
        embedding_dim=3,
        output_dtype="float32",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
    )


def _write_index_artifact(
    path: Path,
    space: EmbeddingSpace,
    vectors: np.ndarray,
    video_ids: list[str],
    frame_indices: list[int],
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(
        schema_version="1.1",
        artifact_id="qa-pred-test-index",
        score_metric="cosine",
        embedding_dim=space.embedding_dim,
        embedding_dtype=space.output_dtype,
        num_rows=vectors.shape[0],
        num_videos=len(set(video_ids)),
        created_at="2026-08-13T00:00:00Z",
        source_dataset_id="synthetic",
        canonical_schema_version="1.0",
        embedding_space=space,
    )
    write_manifest(path / "manifest.json", manifest)
    write_frames(
        path / "frames.ndjson",
        [
            KeyframeTiming(
                keyframe=KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=idx + 1),
                pts_time=float(idx),
                fps=25.0,
                frame_index=frame_indices[idx],
            )
            for idx, video_id in enumerate(video_ids)
        ],
    )
    embedding_dtype = np.dtype(space.output_dtype)
    matrix = vectors.astype(embedding_dtype)
    np.save(path / "embeddings.npy", matrix)
    np.save(path / "norms.npy", np.linalg.norm(matrix.astype(np.float32), axis=1).astype(np.float32))
    return path


def _frame_index_artifact(tmp_path: Path, space: EmbeddingSpace) -> Path:
    """Index with one video and three frames at original frame indices 15/40/99."""
    return _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        video_ids=["L21_V001", "L21_V001", "L21_V001"],
        frame_indices=[15, 40, 99],
    )


def test_one_query_end_to_end_frame_ranking(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_qa_row("qa_001", "cau hoi viet", "english query")],
    )
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)
    output_dir = tmp_path / "qa_predictions"
    encoder = _FakeEncoder(space, {"cau hoi viet": np.array([1.0, 0.0, 0.0], dtype=np.float32)})

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=output_dir,
            top_k=3,
        ),
        encoder,
    )

    assert artifact.manifest.processed_query_count == 1
    assert artifact.manifest.evaluation_set_query_count == 1
    assert artifact.manifest.video_aggregation is None
    result = artifact.query_results[0]
    assert result.query_id == "qa_001"
    assert result.answer == ""
    assert [p.rank for p in result.predictions] == [1, 2, 3]
    assert [p.frame_index for p in result.predictions] == [15, 40, 99]
    assert [p.video_id for p in result.predictions] == ["L21_V001", "L21_V001", "L21_V001"]
    assert [vc.video_id for vc in result.video_candidates] == ["L21_V001"]
    assert result.window_candidates == ()


def test_serialization_round_trip(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)
    output_dir = tmp_path / "qa_predictions"

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=output_dir,
            top_k=3,
        ),
        _FakeEncoder(space),
    )
    restored = read_qa_artifact(output_dir)

    assert restored == artifact
    assert restored.manifest.to_dict() == artifact.manifest.to_dict()
    assert load_qa_predictions_by_query(output_dir) == {
        "qa_001": artifact.query_results[0].predictions
    }


def test_top_k_is_applied(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=2,
        ),
        _FakeEncoder(space),
    )

    assert [p.rank for p in artifact.query_results[0].predictions] == [1, 2]
    assert artifact.manifest.top_k == 2


def test_video_aggregation_ranking_reorders_videos(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    # Video A has one mediocre frame; video B has two strong frames.
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array(
            [[0.5, 0.5, 0.0], [0.95, 0.05, 0.0], [0.93, 0.07, 0.0]],
            dtype=np.float32,
        ),
        video_ids=["L21_V001", "L21_V002", "L21_V002"],
        frame_indices=[10, 20, 25],
    )

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=3,
            video_aggregation="max",
        ),
        _FakeEncoder(space),
    )

    result = artifact.query_results[0]
    assert artifact.manifest.video_aggregation == "max"
    assert [vc.video_id for vc in result.video_candidates] == ["L21_V002", "L21_V001"]
    assert [p.rank for p in result.predictions] == [1, 2, 3]
    assert [p.video_id for p in result.predictions] == ["L21_V002", "L21_V002", "L21_V001"]


def test_mean_top_n_requires_top_n_parameter(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="video_aggregation_top_n must be a positive integer"):
        QaPredictionRunnerConfig(
            evaluation_set_path=Path("x.csv"),
            index_artifact_dir=Path("x"),
            prediction_output_dir=Path("y"),
            video_aggregation="mean_top_n",
        )


def test_window_grouping_produces_window_candidates(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=3,
            window_max_gap=5,
        ),
        _FakeEncoder(space),
    )

    result = artifact.query_results[0]
    assert artifact.manifest.window_max_gap == 5
    assert [w.rank for w in result.window_candidates] == [1, 2, 3]
    assert [w.video_id for w in result.window_candidates] == ["L21_V001", "L21_V001", "L21_V001"]
    windows = [(w.frame_start, w.frame_end) for w in result.window_candidates]
    assert windows == [(15, 15), (40, 40), (99, 99)]


def test_empty_index_preserves_empty_predictions(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "empty-index",
        space,
        vectors=np.zeros((0, 3), dtype=np.float32),
        video_ids=[],
        frame_indices=[],
    )

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
        ),
        _FakeEncoder(space),
    )

    result = artifact.query_results[0]
    assert result.predictions == ()
    assert result.video_candidates == ()
    assert result.window_candidates == ()


def test_no_qa_queries_raises(tmp_path: Path) -> None:
    kis_row = dict(_qa_row("qa_001", "q1", "q1-en"))
    kis_row["task"] = "kis"
    kis_row["gt_answer"] = ""
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [kis_row])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    with pytest.raises(ValueError, match="no QA queries"):
        run_qa_prediction(
            QaPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "qa_predictions",
            ),
            _FakeEncoder(space),
        )


def test_query_subset_selection(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_qa_row("qa_001", "q1", "q1-en"), _qa_row("qa_002", "q2", "q2-en")],
    )
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=1,
            query_subset_ids=("qa_002",),
        ),
        _FakeEncoder(space),
    )

    assert artifact.manifest.evaluation_set_query_count == 2
    assert artifact.manifest.processed_query_count == 1
    assert artifact.manifest.selected_query_ids == ("qa_002",)
    assert artifact.query_results[0].query_id == "qa_002"


def test_unknown_subset_id_raises(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    with pytest.raises(ValueError, match="unknown query ids"):
        run_qa_prediction(
            QaPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "qa_predictions",
                query_subset_ids=("missing",),
            ),
            _FakeEncoder(space),
        )


def test_embedding_space_mismatch_fails_clearly(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    index_space = _space("index-space")
    model_space = _space("model-space")
    index_path = _frame_index_artifact(tmp_path, index_space)

    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        run_qa_prediction(
            QaPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "qa_predictions",
            ),
            _FakeEncoder(model_space),
        )


def test_embedding_space_mismatch_allowed_only_with_explicit_override(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    legacy_space = legacy_embedding_space(embedding_dim=3, output_dtype="float16")
    model_space = _space("openclip-space")
    index_path = _write_index_artifact(
        tmp_path / "legacy-index",
        legacy_space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float16),
        video_ids=["L21_V001"],
        frame_indices=[10],
    )

    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        run_qa_prediction(
            QaPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "qa-predictions-rejected",
            ),
            _FakeEncoder(model_space),
        )

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa-predictions-allowed",
            allow_embedding_space_mismatch=True,
            top_k=1,
        ),
        _FakeEncoder(model_space),
    )

    assert artifact.manifest.embedding_space_compatible is False
    assert artifact.manifest.embedding_space_override_used is True
    assert artifact.manifest.index_manifest["embedding_space"]["backend"] == "legacy"


def test_vietnamese_query_text_selected_by_default(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "van ban tieng viet", "english text")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)
    encoded_texts: list[str] = []

    class _RecordingEncoder(_FakeEncoder):
        def encode_text(self, text: str) -> Embedding:
            encoded_texts.append(text)
            return super().encode_text(text)

    run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
        ),
        _RecordingEncoder(space),
    )

    assert encoded_texts == ["van ban tieng viet"]


def test_manifest_records_experiment_config(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", "q1", "q1-en")])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "qa_predictions",
            top_k=2,
            video_aggregation="mean",
            window_max_gap=4,
            run_id="my-qa-run",
        ),
        _FakeEncoder(space),
    )

    assert artifact.manifest.run_id == "my-qa-run"
    assert artifact.manifest.top_k == 2
    assert artifact.manifest.video_aggregation == "mean"
    assert artifact.manifest.window_max_gap == 4
    assert artifact.manifest.evaluation_set_path == str(eval_path.resolve())


def test_qaprediction_validates_increasing_ranks() -> None:
    with pytest.raises(ValueError, match="predictions ranks must be strictly increasing"):
        QaQueryResult(
            query_id="qa_001",
            query_text="q",
            answer="",
            predictions=(
                QaPrediction(rank=2, video_id="L21_V001", frame_index=10, score=0.9),
                QaPrediction(rank=1, video_id="L21_V001", frame_index=11, score=0.8),
            ),
            video_candidates=(),
            window_candidates=(),
        )