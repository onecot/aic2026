"""Tests for the BTC-query inference adapter.

Covers query-file loading, the three task adapters (with fake encoders and
synthetic index artifacts), artifact round-trip via the standard readers, and
the end-to-end compatibility contract with ``scripts/build_submission.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_submission import build_submission

from aic2026.btc_inference import (
    run_btc_kis_prediction,
    run_btc_qa_prediction,
    run_btc_trake_prediction,
)
from aic2026.btc_queries import load_btc_queries
from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace
from aic2026.index_artifact import IndexManifest, write_frames, write_manifest
from aic2026.kis_prediction_runner import read_kis_prediction_artifact
from aic2026.qa.answer_generation import DEFAULT_ANSWER_MODEL
from aic2026.qa.qa_prediction_runner import read_qa_artifact
from aic2026.trake import read_trake_artifact


class _FakeEncoder(EmbeddingEncoder):
    def __init__(self, embedding_space: EmbeddingSpace) -> None:
        self._space = embedding_space

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_image(self, image: object) -> Embedding:
        raise NotImplementedError("image encoding is not used in BTC inference tests")

    def encode_text(self, text: str) -> Embedding:
        return Embedding(
            kind="text",
            vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            space=self._space,
        )


class _RecordingAnswerer:
    """Fake answerer satisfying the answer-generation seam contract."""

    model_name = f"jina-vlm-mlx:{DEFAULT_ANSWER_MODEL}"
    checkpoint = DEFAULT_ANSWER_MODEL
    device = "mps"
    max_new_tokens = 16

    def __init__(self, answer: str = "mot chiec thuyen") -> None:
        self._answer = answer
        self.calls: list[tuple[str, list[object]]] = []

    def generate_answer(self, question: str, images: list[object]) -> str:
        self.calls.append((question, list(images)))
        return self._answer


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


def _write_index_artifact(path: Path, space: EmbeddingSpace) -> Path:
    vectors = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    path.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(
        schema_version="1.1",
        artifact_id="btc-inference-test-index",
        score_metric="cosine",
        embedding_dim=space.embedding_dim,
        embedding_dtype=space.output_dtype,
        num_rows=vectors.shape[0],
        num_videos=1,
        created_at="2026-08-21T00:00:00Z",
        source_dataset_id="synthetic",
        canonical_schema_version="1.0",
        embedding_space=space,
    )
    write_manifest(path / "manifest.json", manifest)
    write_frames(
        path / "frames.ndjson",
        [
            KeyframeTiming(
                keyframe=KeyframeKey(video_id=VideoId("L21_V001"), keyframe_ordinal=idx + 1),
                pts_time=float(idx),
                fps=25.0,
                frame_index=frame_index,
            )
            for idx, frame_index in enumerate([15, 40, 99])
        ],
    )
    np.save(path / "embeddings.npy", vectors)
    np.save(path / "norms.npy", np.linalg.norm(vectors, axis=1).astype(np.float32))
    return path


def _write_keyframes_fixture(path: Path, video_id: str, frame_indices: list[int]) -> Path:
    """Minimal keyframe artifact (frames.ndjson + one real image per frame)."""
    from PIL import Image

    path.mkdir(parents=True, exist_ok=True)
    write_frames(
        path / "frames.ndjson",
        [
            KeyframeTiming(
                keyframe=KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=idx + 1),
                pts_time=float(idx),
                fps=25.0,
                frame_index=frame_index,
            )
            for idx, frame_index in enumerate(frame_indices)
        ],
    )
    for idx in range(len(frame_indices)):
        image_dir = path / "images" / video_id
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(idx, 0, 0)).save(image_dir / f"{idx + 1:06d}.jpg")
    return path


def _write_query_files(query_dir: Path) -> dict[str, str]:
    kis_text = "Tìm một chiếc thuyền trên sông"
    qa_text = "Người đàn ông trong video đang làm gì?"
    trake_text = "E1: Lân quay vòng\nE2: 4 chân chạm đất\nE3: Màn kết thúc"
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / "query-p1-1-kis.txt").write_text(kis_text + "\n", encoding="utf-8")
    (query_dir / "query-p1-2-qa.txt").write_text(qa_text + "\n", encoding="utf-8")
    (query_dir / "query-p1-3-trake.txt").write_text(trake_text + "\n", encoding="utf-8")
    return {"kis": kis_text, "qa": qa_text, "trake": trake_text}


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------


class TestLoadBtcQueries:
    def test_loads_all_tasks_with_ids_and_exact_text(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path)
        queries = load_btc_queries(tmp_path)

        assert [q.query_id for q in queries] == ["btc-p1-1-kis", "btc-p1-2-qa", "btc-p1-3-trake"]
        assert [q.info.task for q in queries] == ["kis", "qa", "trake"]
        assert queries[0].query_text == texts["kis"]
        assert queries[1].query_text == texts["qa"]
        assert queries[2].query_text == texts["trake"]

    def test_preserves_multiline_trake_text(self, tmp_path: Path) -> None:
        text = "E1: a\nE2: b"
        (tmp_path / "query-p1-9-trake.txt").write_text(text, encoding="utf-8")
        queries = load_btc_queries(tmp_path)
        assert queries[0].query_text == text

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_btc_queries(tmp_path / "nope")

    def test_no_query_files_raises(self, tmp_path: Path) -> None:
        (tmp_path / "other.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="no 'query-<name>-<task>.txt'"):
            load_btc_queries(tmp_path)

    def test_empty_query_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "query-p1-1-kis.txt").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_btc_queries(tmp_path)


# ---------------------------------------------------------------------------
# Task adapters
# ---------------------------------------------------------------------------


class TestRunBtcKisPrediction:
    def test_writes_artifact_with_exact_query_texts(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path / "queries")
        space = _space()
        artifact = run_btc_kis_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
            prediction_output_path=tmp_path / "out" / "kis.json",
            top_k=3,
        )

        assert artifact.processed_query_count == 1
        result = artifact.query_results[0]
        assert result.query_id == "btc-p1-1-kis"
        assert result.query_text == texts["kis"]
        assert len(result.predictions) == 3

        restored = read_kis_prediction_artifact(tmp_path / "out" / "kis.json")
        assert restored == artifact
        assert restored.query_results[0].query_text == texts["kis"]

    def test_records_btc_provenance(self, tmp_path: Path) -> None:
        _write_query_files(tmp_path / "queries")
        space = _space()
        artifact = run_btc_kis_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
            prediction_output_path=tmp_path / "kis.json",
        )

        assert artifact.evaluation_set_path == str((tmp_path / "queries").resolve())
        assert artifact.evaluation_set_query_count == 1
        assert artifact.run_id.startswith("btc-kis-run-")

    def test_rejects_when_no_kis_queries(self, tmp_path: Path) -> None:
        (tmp_path / "query-p1-2-qa.txt").write_text("q", encoding="utf-8")
        space = _space()
        with pytest.raises(ValueError, match="no BTC kis queries"):
            run_btc_kis_prediction(
                load_btc_queries(tmp_path),
                encoder=_FakeEncoder(space),
                index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
                prediction_output_path=tmp_path / "kis.json",
            )

    def test_accepts_identical_space_id_with_different_model_version(self, tmp_path: Path) -> None:
        """Regression: Jina QA-path semantics ignore build-only model_version.

        The index was built with transformers 4.57.6 and the local encoder
        reports 5.15.1; every comparison-relevant field (including space_id)
        is identical, so prediction must be accepted without any override.
        """
        _write_query_files(tmp_path / "queries")
        index_space = _space()
        index_space_dict = index_space.to_dict() | {"model_version": "4.57.6"}
        index_space = EmbeddingSpace.from_dict(index_space_dict)
        encoder_space = EmbeddingSpace.from_dict(index_space.to_dict() | {"model_version": "5.15.1"})
        assert encoder_space.space_id == index_space.space_id
        assert encoder_space != index_space  # differs only in model_version

        artifact = run_btc_kis_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(encoder_space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", index_space),
            prediction_output_path=tmp_path / "out" / "kis.json",
            top_k=3,
        )

        assert artifact.processed_query_count == 1
        assert artifact.embedding_space_compatible is True
        assert artifact.embedding_space_override_used is False


class TestRunBtcQaPrediction:
    def test_retrieval_only_leaves_answers_empty(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path / "queries")
        space = _space()
        artifact = run_btc_qa_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
            prediction_output_dir=tmp_path / "out" / "qa",
            top_k=3,
        )

        assert artifact.manifest.processed_query_count == 1
        result = artifact.query_results[0]
        assert result.query_text == texts["qa"]
        assert result.answer == ""
        assert len(result.predictions) == 3

        restored = read_qa_artifact(tmp_path / "out" / "qa")
        assert restored.manifest.processed_query_count == 1

    def test_generated_answers_fill_artifact(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path / "queries")
        space = _space()
        keyframes = _write_keyframes_fixture(tmp_path / "keyframes", "L21_V001", [15, 40, 99])
        answerer = _RecordingAnswerer()
        artifact = run_btc_qa_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
            prediction_output_dir=tmp_path / "out" / "qa",
            top_k=3,
            generate_answer=True,
            keyframes_dir=keyframes,
            answer_generator=answerer,
        )

        result = artifact.query_results[0]
        assert result.answer == "mot chiec thuyen"
        assert result.answer_error is None
        assert result.answer_evidence is not None
        # The question presented to the answerer is the exact file text.
        assert answerer.calls[0][0] == texts["qa"]

        restored = read_qa_artifact(tmp_path / "out" / "qa")
        assert restored.query_results[0].answer == "mot chiec thuyen"

    def test_rejects_when_no_qa_queries(self, tmp_path: Path) -> None:
        (tmp_path / "query-p1-1-kis.txt").write_text("q", encoding="utf-8")
        space = _space()
        with pytest.raises(ValueError, match="no BTC qa queries"):
            run_btc_qa_prediction(
                load_btc_queries(tmp_path),
                encoder=_FakeEncoder(space),
                index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
                prediction_output_dir=tmp_path / "qa",
            )


class TestRunBtcTrakePrediction:
    def test_parses_events_from_file_text(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path / "queries")
        space = _space()
        artifact = run_btc_trake_prediction(
            load_btc_queries(tmp_path / "queries"),
            encoder=_FakeEncoder(space),
            index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
            prediction_output_dir=tmp_path / "out" / "trake",
            top_k=3,
        )

        assert artifact.manifest.processed_query_count == 1
        result = artifact.query_results[0]
        assert result.query_text == texts["trake"]
        assert [event.event_id for event in result.events] == ["E1", "E2", "E3"]
        assert all(len(event.predictions) == 3 for event in result.events)

        restored = read_trake_artifact(tmp_path / "out" / "trake")
        assert restored == artifact

    def test_rejects_unparseable_event_list(self, tmp_path: Path) -> None:
        (tmp_path / "query-p1-7-trake.txt").write_text("khong co su kien", encoding="utf-8")
        space = _space()
        with pytest.raises(ValueError, match="no TRAKE events"):
            run_btc_trake_prediction(
                load_btc_queries(tmp_path),
                encoder=_FakeEncoder(space),
                index_artifact_dir=_write_index_artifact(tmp_path / "index", space),
                prediction_output_dir=tmp_path / "trake",
            )


# ---------------------------------------------------------------------------
# Compatibility with the submission exporter
# ---------------------------------------------------------------------------


class TestBuildSubmissionCompatibility:
    def test_btc_artifacts_drive_full_submission_build(self, tmp_path: Path) -> None:
        texts = _write_query_files(tmp_path / "queries")
        query_dir = tmp_path / "queries"
        space = _space()
        index_dir = _write_index_artifact(tmp_path / "index", space)
        btc_queries = load_btc_queries(query_dir)
        encoder = _FakeEncoder(space)

        run_btc_kis_prediction(
            btc_queries,
            encoder=encoder,
            index_artifact_dir=index_dir,
            prediction_output_path=tmp_path / "artifacts" / "kis.json",
            top_k=3,
        )
        run_btc_qa_prediction(
            btc_queries,
            encoder=encoder,
            index_artifact_dir=index_dir,
            prediction_output_dir=tmp_path / "artifacts" / "qa",
            top_k=3,
            generate_answer=True,
            keyframes_dir=_write_keyframes_fixture(tmp_path / "keyframes", "L21_V001", [15, 40, 99]),
            answer_generator=_RecordingAnswerer(),
        )
        run_btc_trake_prediction(
            btc_queries,
            encoder=encoder,
            index_artifact_dir=index_dir,
            prediction_output_dir=tmp_path / "artifacts" / "trake",
            top_k=3,
        )

        result = build_submission(
            query_dir=query_dir,
            output_dir=tmp_path / "submission-out",
            kis_artifact=tmp_path / "artifacts" / "kis.json",
            qa_artifact=tmp_path / "artifacts" / "qa",
            trake_artifact=tmp_path / "artifacts" / "trake",
        )

        assert result["counts"] == {"kis": 1, "qa": 1, "trake": 1}
        assert all(row["method"] == "exact" for row in result["mapping"])
        assert result["zip_info"]["valid"] is True

        mapping_by_file = {row["query_file"]: row for row in result["mapping"]}
        assert mapping_by_file["query-p1-1-kis.txt"]["matched_id"] == "btc-p1-1-kis"
        assert mapping_by_file["query-p1-2-qa.txt"]["matched_id"] == "btc-p1-2-qa"
        assert mapping_by_file["query-p1-3-trake.txt"]["matched_id"] == "btc-p1-3-trake"

        trake_csv = (
            tmp_path / "submission-out" / "submission" / "query-p1-3-trake.csv"
        ).read_text(encoding="utf-8").strip().splitlines()
        assert len(trake_csv[0].split(",")) == 4  # video_id + 3 event frames
