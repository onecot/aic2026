"""KIS prediction runner and scoring contracts.."""

from __future__ import annotations


# ==========================================================================
# Section: test_kis_prediction_runner
# ==========================================================================

import csv
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace, legacy_embedding_space
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.index_artifact import IndexManifest, write_frames, write_manifest
from aic2026.kis_prediction_runner import (
    KisPredictionRunnerConfig,
    read_kis_prediction_artifact,
    run_kis_prediction,
)
from aic2026.kis_scoring import evaluate_kis

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
        raise NotImplementedError("image encoding is not used in KIS query prediction tests")

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


def _eval_row(query_id: str, query_text_vi: str, query_text_en: str, gt_start: int = 10, gt_end: int = 20) -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "kis",
        "query_text_vi": query_text_vi,
        "query_text_en": query_text_en,
        "gt_video_id": "L21_V001",
        "gt_frame_start": str(gt_start),
        "gt_frame_end": str(gt_end),
        "gt_answer": "",
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


def _write_index_artifact(path: Path, space: EmbeddingSpace, vectors: np.ndarray, frame_indices: list[int]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(
        schema_version="1.1",
        artifact_id="kis-pred-test-index",
        score_metric="cosine",
        embedding_dim=space.embedding_dim,
        embedding_dtype=space.output_dtype,
        num_rows=vectors.shape[0],
        num_videos=2,
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
                keyframe=KeyframeKey(video_id=VideoId("L21_V001" if idx < 2 else "L21_V002"), keyframe_ordinal=idx + 1),
                pts_time=float(idx),
                fps=25.0,
                frame_index=frame_indices[idx],
            )
            for idx in range(vectors.shape[0])
        ],
    )
    embedding_dtype = np.dtype(space.output_dtype)
    matrix = vectors.astype(embedding_dtype)
    np.save(path / "embeddings.npy", matrix)
    np.save(path / "norms.npy", np.linalg.norm(matrix.astype(np.float32), axis=1).astype(np.float32))
    return path


def test_one_query_end_to_end_and_rank_preservation(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "cau hoi viet", "english query")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        frame_indices=[15, 40, 99],
    )
    output_path = tmp_path / "predictions.json"
    encoder = _FakeEncoder(space, {"cau hoi viet": np.array([1.0, 0.0, 0.0], dtype=np.float32)})

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=output_path,
            top_k=3,
        ),
        encoder,
    )

    assert artifact.processed_query_count == 1
    assert [prediction.rank for prediction in artifact.query_results[0].predictions] == [1, 2, 3]
    assert [prediction.frame_index for prediction in artifact.query_results[0].predictions] == [15, 40, 99]

    score = evaluate_kis(
        tuple(load_eval_set(eval_path).iter_queries(EvaluationTask.KIS)),
        artifact.to_kis_predictions_by_query(),
    )
    assert score.score_at(1) == 1.0


def test_vietnamese_query_text_selected_by_default(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "van ban tieng viet", "english text")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_indices=[12],
    )
    output_path = tmp_path / "predictions.json"
    encoded_texts: list[str] = []

    class _RecordingEncoder(_FakeEncoder):
        def encode_text(self, text: str) -> Embedding:
            encoded_texts.append(text)
            return super().encode_text(text)

    run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=output_path,
        ),
        _RecordingEncoder(space),
    )

    assert encoded_texts == ["van ban tieng viet"]


def test_top_k_is_applied(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
        frame_indices=[10, 11, 12],
    )

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=tmp_path / "predictions.json",
            top_k=2,
        ),
        _FakeEncoder(space),
    )

    assert len(artifact.query_results[0].predictions) == 2
    assert [prediction.rank for prediction in artifact.query_results[0].predictions] == [1, 2]


def test_prediction_serialization_round_trip(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_indices=[10],
    )
    output_path = tmp_path / "predictions.json"

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=output_path,
        ),
        _FakeEncoder(space),
    )
    restored = read_kis_prediction_artifact(output_path)

    assert restored == artifact
    assert restored.to_kis_predictions_by_query() == artifact.to_kis_predictions_by_query()


def test_embedding_space_mismatch_fails_clearly(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    index_space = _space("index-space")
    model_space = _space("model-space")
    index_path = _write_index_artifact(
        tmp_path / "index",
        index_space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_indices=[10],
    )

    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        run_kis_prediction(
            KisPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_path=tmp_path / "predictions.json",
            ),
            _FakeEncoder(model_space),
        )


def test_embedding_space_mismatch_allowed_only_with_explicit_override(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    legacy_space = legacy_embedding_space(embedding_dim=3, output_dtype="float16")
    model_space = _space("openclip-space")
    index_path = _write_index_artifact(
        tmp_path / "legacy-index",
        legacy_space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float16),
        frame_indices=[10],
    )

    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        run_kis_prediction(
            KisPredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_path=tmp_path / "predictions-rejected.json",
            ),
            _FakeEncoder(model_space),
        )

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=tmp_path / "predictions-allowed.json",
            allow_embedding_space_mismatch=True,
            top_k=1,
        ),
        _FakeEncoder(model_space),
    )

    assert artifact.processed_query_count == 1
    assert artifact.embedding_space_compatible is False
    assert artifact.embedding_space_override_used is True
    assert artifact.index_manifest["embedding_space"]["backend"] == "legacy"
    assert artifact.model_embedding_space["backend"] == "test-backend"


def test_artifact_records_no_override_when_spaces_match(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_indices=[10],
    )

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=tmp_path / "predictions.json",
        ),
        _FakeEncoder(space),
    )

    assert artifact.embedding_space_compatible is True
    assert artifact.embedding_space_override_used is False


def test_complete_101_query_coverage_on_team_eval_set(tmp_path: Path) -> None:
    eval_path = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    if not eval_path.is_file():
        pytest.skip("team evaluation CSV not available")

    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "index",
        space,
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_indices=[10],
    )

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=tmp_path / "predictions.json",
            top_k=1,
        ),
        _FakeEncoder(space),
    )

    assert artifact.evaluation_set_query_count == 101
    assert artifact.processed_query_count == 101
    assert len(artifact.query_results) == 101


def test_empty_retrieval_results_are_preserved(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_eval_row("kis_001", "q1", "q1-en")],
    )
    space = _space()
    index_path = _write_index_artifact(
        tmp_path / "empty-index",
        space,
        vectors=np.zeros((0, 3), dtype=np.float32),
        frame_indices=[],
    )

    artifact = run_kis_prediction(
        KisPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_path=tmp_path / "predictions.json",
        ),
        _FakeEncoder(space),
    )

    assert artifact.query_results[0].predictions == ()
    assert artifact.to_kis_predictions_by_query()["kis_001"] == []


# ==========================================================================
# Section: test_kis_scoring
# ==========================================================================

import json
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
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring


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


def test_evaluate_kis_accepts_evaluationset_from_loader_with_prediction_artifact(tmp_path: Path) -> None:
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

    artifact_path = tmp_path / "predictions.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "test-run",
                "created_at": "2026-08-13T00:00:00Z",
                "evaluation_set_path": str(csv_path),
                "evaluation_set_query_count": 1,
                "processed_query_count": 1,
                "query_text_field": "query_text_vi",
                "top_k": 1,
                "embedding_space_compatible": True,
                "embedding_space_override_used": False,
                "index_artifact_path": "/tmp/index",
                "index_manifest": {"artifact_id": "dummy-index", "embedding_space": {"backend": "legacy"}},
                "model_embedding_space": {"backend": "legacy"},
                "model_config": None,
                "selected_query_ids": None,
                "query_results": [
                    {
                        "query_id": "kis_001",
                        "query_text": "T\u00ecm c\u1ea3nh",
                        "predictions": [
                            {"rank": 1, "video_id": "L21_V001", "frame_index": 15, "score": 0.5},
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    eval_set = load_eval_set(csv_path)
    predictions = load_kis_predictions_for_scoring(artifact_path)
    result = evaluate_kis(eval_set, predictions)

    assert result.score_at(1) == 1.0
    assert result.score_at(5) == 1.0
    assert result.score_at(20) == 1.0
    assert result.score_at(50) == 1.0
    assert result.score_at(100) == 1.0
    assert result.final_score == 1.0
