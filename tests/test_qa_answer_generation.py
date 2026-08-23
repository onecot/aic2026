"""Focused tests for the QA answer-generation stage.

Covers evidence selection, the answer generator receiving the selected
frames, mock-answer flow, retrieval-only mode being unchanged, per-query
failure isolation, and artifact read/write compatibility.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import IndexManifest, write_frames, write_manifest
from aic2026.qa.answer_generation import (
    DEFAULT_ANSWER_MODEL,
    EVIDENCE_MIN_FRAME_GAP,
    JinaVlmAnswerGenerator,
    KeyframeEvidence,
    QA_ANSWER_PROMPT,
    answer_query,
    select_evidence_frames,
)
from aic2026.qa.qa_prediction_runner import (
    QaPredictionRunnerConfig,
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
    def __init__(self, embedding_space: EmbeddingSpace) -> None:
        self._space = embedding_space

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_image(self, image: object) -> Embedding:
        raise NotImplementedError

    def encode_text(self, text: str) -> Embedding:
        return Embedding(
            kind="text",
            vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            space=self._space,
        )


class _RecordingAnswerer:
    """Fake answerer that records the question and images it received."""

    model_name = f"jina-vlm-mlx:{DEFAULT_ANSWER_MODEL}"
    checkpoint = DEFAULT_ANSWER_MODEL
    device = "mps"
    max_new_tokens = 64

    def __init__(self, answer: str = "A blue van parked at the south entrance.", fail: bool = False) -> None:
        self._answer = answer
        self._fail = fail
        self.calls: list[tuple[str, list[object]]] = []

    def generate_answer(self, question: str, images: list[object]) -> str:
        self.calls.append((question, list(images)))
        if self._fail:
            raise RuntimeError("mock VLM failure")
        return self._answer


class _FakeKeyframes:
    """Stand-in for KeyframeEvidence exposing image_path for test frames."""

    def __init__(self, paths: dict[tuple[str, int], Path]) -> None:
        self._paths = paths

    def image_path(self, video_id: str, frame_index: int) -> Path | None:
        return self._paths.get((video_id, frame_index))


def _space() -> EmbeddingSpace:
    return EmbeddingSpace(
        backend="test-backend",
        model_id="test-model",
        checkpoint_id="test-checkpoint",
        space_id="test-space-v1",
        embedding_dim=3,
        output_dtype="float32",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
    )


def _prediction(rank: int, video_id: str, frame_index: int) -> QaPrediction:
    return QaPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=1.0 / rank)


def _query_result(predictions: tuple[QaPrediction, ...], query_text: str = "question") -> QaQueryResult:
    return QaQueryResult(
        query_id="qa_001",
        query_text=query_text,
        answer="",
        predictions=predictions,
        video_candidates=(),
        window_candidates=(),
    )


def _qa_row(query_id: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "qa",
        "query_text_vi": "cau hoi viet",
        "query_text_en": "english query",
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "dap an",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_index_artifact(path: Path, space: EmbeddingSpace) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    vectors = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    video_ids = ["L21_V001", "L21_V001", "L21_V001"]
    frame_indices = [15, 60, 99]
    manifest = IndexManifest(
        schema_version="1.1",
        artifact_id="qa-answer-test-index",
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
    np.save(path / "embeddings.npy", vectors)
    np.save(path / "norms.npy", np.linalg.norm(vectors, axis=1).astype(np.float32))
    return path


def _write_keyframes_dir(path: Path) -> Path:
    """Materialized keyframe artifact matching the synthetic index frames."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    frames = path / "frames.ndjson"
    with frames.open("w", encoding="utf-8") as handle:
        for ordinal, frame_index in enumerate([15, 60, 99], start=1):
            handle.write(
                json.dumps(
                    KeyframeTiming(
                        keyframe=KeyframeKey(video_id=VideoId("L21_V001"), keyframe_ordinal=ordinal),
                        pts_time=float(frame_index),
                        fps=25.0,
                        frame_index=frame_index,
                    ).to_dict()
                )
                + "\n"
            )
    images_dir = path / "images" / "L21_V001"
    images_dir.mkdir(parents=True)
    for ordinal in (1, 2, 3):
        Image.new("RGB", (8, 8), color=(10 * ordinal, 20, 30)).save(
            images_dir / f"{ordinal:06d}.jpg", "JPEG"
        )
    return path


def _diverse_predictions() -> tuple[QaPrediction, ...]:
    return (
        _prediction(1, "L21_V001", 100),
        _prediction(2, "L21_V001", 110),  # near-duplicate of rank 1 (gap < 30)
        _prediction(3, "L21_V001", 200),
        _prediction(4, "L21_V002", 10),
        _prediction(5, "L21_V002", 15),  # near-duplicate of rank 4
        _prediction(6, "L21_V003", 50),
    )


# --- 1. Evidence selection --------------------------------------------------


def test_select_evidence_frames_selects_expected_diverse_count() -> None:
    selected = select_evidence_frames(_diverse_predictions(), top_n=4)
    assert [p.rank for p in selected] == [1, 3, 4, 6]
    assert len(selected) == 4
    assert {p.video_id for p in selected} == {"L21_V001", "L21_V002", "L21_V003"}


def test_select_evidence_frames_skips_near_duplicate_same_video_frames() -> None:
    selected = select_evidence_frames(_diverse_predictions(), top_n=10)
    indices: dict[str, list[int]] = {}
    for prediction in selected:
        indices.setdefault(prediction.video_id, []).append(prediction.frame_index)
    for frames in indices.values():
        if len(frames) > 1:
            assert min(abs(b - a) for a in frames for b in frames if a != b) >= EVIDENCE_MIN_FRAME_GAP


def test_select_evidence_frames_caps_at_top_n_and_validates() -> None:
    predictions = tuple(_prediction(i + 1, "L21_V001", i * 50 + 10) for i in range(20))
    assert len(select_evidence_frames(predictions, top_n=5)) == 5
    with pytest.raises(ValueError, match="positive integer"):
        select_evidence_frames(predictions, top_n=0)


# --- 2. Answer generation receives the selected frames ----------------------


def test_answer_generation_receives_selected_frames(tmp_path: Path) -> None:
    predictions = _diverse_predictions()
    result = _query_result(predictions, query_text="What is parked at the entrance?")
    evidence = select_evidence_frames(predictions, top_n=4)
    paths = {
        (p.video_id, p.frame_index): tmp_path / f"{p.video_id}-{p.frame_index}.jpg"
        for p in evidence
    }
    for path in paths.values():
        path.write_text("fake-image")
    answerer = _RecordingAnswerer()
    keyframes = _FakeKeyframes(paths)
    loader = lambda path: str(path)  # noqa: E731

    answered = answer_query(result, answerer, keyframes, top_n=4, image_loader=loader)

    assert answerer.calls == [(result.query_text, [str(p) for p in paths.values()])]
    assert [p.rank for p in answered.answer_evidence] == [1, 3, 4, 6]


# --- 3. Mock VLM returns an answer ------------------------------------------


def test_mock_vlm_returns_answer(tmp_path: Path) -> None:
    predictions = (
        _prediction(1, "L21_V001", 15),
        _prediction(2, "L21_V001", 60),
    )
    result = _query_result(predictions)
    keyframes = _FakeKeyframes(
        {
            ("L21_V001", 15): tmp_path / "a.jpg",
            ("L21_V001", 60): tmp_path / "b.jpg",
        }
    )
    for path in keyframes._paths.values():
        path.write_text("fake")
    answerer = _RecordingAnswerer(answer="A blue van.")

    answered = answer_query(result, answerer, keyframes, top_n=5, image_loader=lambda p: str(p))

    assert answered.answer == "A blue van."
    assert answered.answer_error is None
    assert [p.rank for p in answered.answer_evidence] == [1, 2]
    assert answered.answer_evidence[0].video_id == "L21_V001"


# --- 4. Retrieval-only mode remains unchanged --------------------------------


def test_retrieval_only_mode_unchanged(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval.csv", [_qa_row("qa_001")])
    space = _space()
    index_path = _write_index_artifact(tmp_path / "index", space)

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "out",
            top_k=3,
        ),
        _FakeEncoder(space),
    )

    result = artifact.query_results[0]
    assert result.answer == ""
    assert result.answer_evidence is None
    assert result.answer_error is None
    assert [p.frame_index for p in result.predictions] == [15, 60, 99]
    assert artifact.manifest.answer_generation is None


# --- 5. Answer-generation failure does not crash the full QA run ------------


def test_answer_failure_does_not_crash_full_qa_run(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval.csv", [_qa_row("qa_001")])
    space = _space()
    index_path = _write_index_artifact(tmp_path / "index", space)
    keyframes_dir = _write_keyframes_dir(tmp_path / "keyframes")
    output_dir = tmp_path / "out"

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=output_dir,
            top_k=3,
            generate_answer=True,
            answer_top_n=2,
            keyframes_dir=keyframes_dir,
        ),
        _FakeEncoder(space),
        answer_generator=_RecordingAnswerer(fail=True),
    )

    result = artifact.query_results[0]
    assert result.answer == ""
    assert result.answer_error is not None
    assert "answer generation failed" in result.answer_error
    assert artifact.manifest.answer_generation["failed_query_count"] == 1
    assert artifact.manifest.answer_generation["evidence_query_count"] == 1
    assert artifact.manifest.answer_generation["prompt"] == QA_ANSWER_PROMPT
    restored = read_qa_artifact(output_dir)
    assert restored.query_results[0].answer_error == result.answer_error


# --- 6. Artifact read/write remains compatible -------------------------------


def test_artifact_round_trip_with_answer_fields(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(tmp_path / "eval.csv", [_qa_row("qa_001")])
    space = _space()
    index_path = _write_index_artifact(tmp_path / "index", space)
    keyframes_dir = _write_keyframes_dir(tmp_path / "keyframes")
    output_dir = tmp_path / "out"

    artifact = run_qa_prediction(
        QaPredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=output_dir,
            top_k=3,
            generate_answer=True,
            answer_top_n=2,
            keyframes_dir=keyframes_dir,
        ),
        _FakeEncoder(space),
        answer_generator=_RecordingAnswerer(answer="A blue van."),
    )

    result = artifact.query_results[0]
    assert result.answer == "A blue van."
    assert result.answer_error is None
    assert [p.rank for p in result.answer_evidence] == [1, 2]

    restored = read_qa_artifact(output_dir)
    assert restored == artifact
    assert restored.query_results[0].answer == "A blue van."
    assert [p.rank for p in restored.query_results[0].answer_evidence] == [1, 2]
    assert restored.manifest.answer_generation == artifact.manifest.answer_generation


def test_keyframe_evidence_resolves_images_from_artifact(tmp_path: Path) -> None:
    keyframes_dir = _write_keyframes_dir(tmp_path / "keyframes")
    keyframes = KeyframeEvidence(keyframes_dir)
    assert keyframes.ordinal_for("L21_V001", 15) == 1
    assert keyframes.image_path("L21_V001", 60) == keyframes_dir / "images" / "L21_V001" / "000002.jpg"
    assert keyframes.image_path("L21_V001", 999) is None
    assert keyframes.image_path("L21_V999", 15) is None


def test_default_answer_model_reuses_existing_jina_vlm() -> None:
    assert DEFAULT_ANSWER_MODEL == "jinaai/jina-vlm-mlx"
    from aic2026.vlm_jina import JinaVlmMlx

    # The answer generator is built on the shared extracted VLM wrapper.
    assert issubclass(JinaVlmAnswerGenerator, JinaVlmMlx)