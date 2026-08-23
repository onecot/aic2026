"""Tests for the minimal end-to-end TRAKE prediction pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import IndexManifest, write_frames, write_manifest
from aic2026.trake import (
    TrakeEventParseError,
    TrakePredictionRunnerConfig,
    build_event_windows,
    parse_events,
    read_trake_artifact,
    run_trake_prediction,
)

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
    def __init__(
        self,
        embedding_space: EmbeddingSpace,
        vectors_by_text: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._space = embedding_space
        self._vectors_by_text = vectors_by_text or {}

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_image(self, image: object) -> Embedding:
        raise NotImplementedError("image encoding is not used in TRAKE prediction tests")

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


def _trake_row(
    query_id: str,
    query_text_vi: str,
    query_text_en: str,
    gt_events: str = "E9: ground truth only",
) -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "trake",
        "query_text_vi": query_text_vi,
        "query_text_en": query_text_en,
        "gt_video_id": "L21_V001",
        "gt_frame_start": "",
        "gt_frame_end": "",
        "gt_answer": "",
        "gt_events": gt_events,
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
        artifact_id="trake-pred-test-index",
        score_metric="cosine",
        embedding_dim=space.embedding_dim,
        embedding_dtype=space.output_dtype,
        num_rows=vectors.shape[0],
        num_videos=len(set(video_ids)),
        created_at="2026-08-20T00:00:00Z",
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


def _run(
    tmp_path: Path,
    row: dict[str, str],
    space: EmbeddingSpace,
    encoder: _FakeEncoder,
    top_k: int = 3,
    window_max_gap: int | None = 5,
) -> object:
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [row])
    index_path = _frame_index_artifact(tmp_path, space)
    return run_trake_prediction(
        TrakePredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "trake_predictions",
            top_k=top_k,
            window_max_gap=window_max_gap,
        ),
        encoder,
    )


# ------------------------------------------------------------------ event parsing


def test_parse_events_extracts_ordered_events() -> None:
    query = (
        "E1: first moment flour is added to the bowl\n"
        "E2: first moment the first asparagus piece contacts the oil\n"
        "E3: first moment the first asparagus piece leaves the pan\n"
        "E4: final asparagus piece leaves the pan and is completely on the plate"
    )
    events = parse_events(query)
    assert [event.event_id for event in events] == ["E1", "E2", "E3", "E4"]
    assert events[0].event_text == "first moment flour is added to the bowl"
    assert events[3].event_text == "final asparagus piece leaves the pan and is completely on the plate"


def test_parse_events_preserves_query_text_order() -> None:
    query = "E3: lion starts rotating\nE1: all four feet touch the ground\nE2: performers bow"
    events = parse_events(query)
    assert [event.event_id for event in events] == ["E3", "E1", "E2"]


def test_parse_events_numbered_list_fallback() -> None:
    query = (
        "Find the moments of preparing a breaded fried chicken roll: "
        "(1) washing the chicken, (2) coating it with breadcrumbs, "
        "(3) placing it in hot oil, and (4) slicing the fried chicken."
    )
    events = parse_events(query)
    assert [event.event_id for event in events] == ["E1", "E2", "E3", "E4"]
    assert events[0].event_text == "washing the chicken"
    assert events[2].event_text == "placing it in hot oil, and"


def test_parse_events_raises_without_events() -> None:
    with pytest.raises(TrakeEventParseError, match="no TRAKE events"):
        parse_events("a plain sentence with no event labels")
    with pytest.raises(TrakeEventParseError, match="query text is empty"):
        parse_events("   ")


# ------------------------------------------------------------------ windows


def test_build_event_windows_groups_retrieved_frames() -> None:
    from aic2026.retrieval import RetrievalCandidate

    candidates = [
        RetrievalCandidate(rank=1, video_id="L21_V001", keyframe_ordinal=1, frame_index=10, pts_time=0.4, score=0.9),
        RetrievalCandidate(rank=2, video_id="L21_V001", keyframe_ordinal=2, frame_index=12, pts_time=0.48, score=0.8),
        RetrievalCandidate(rank=3, video_id="L21_V002", keyframe_ordinal=3, frame_index=30, pts_time=1.2, score=0.7),
    ]
    windows = build_event_windows(candidates, max_gap=5)
    assert [(w.video_id, w.frame_start, w.frame_end) for w in windows] == [
        ("L21_V001", 10, 12),
        ("L21_V002", 30, 30),
    ]
    assert [w.rank for w in windows] == [1, 2]
    assert [w.score for w in windows] == [0.9, 0.7]
    assert build_event_windows(candidates, max_gap=None) == ()


# ------------------------------------------------------------------ runner


def test_runner_retrieves_each_event_independently(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row(
        "trake_301",
        "E1: rua mieng thit ga\nE2: phu bot chien xu\nE3: tha ga vao chao dau",
        "E1: washing the chicken\nE2: coating with breadcrumbs\nE3: placing in hot oil",
    )
    encoded_texts: list[str] = []

    class _RecordingEncoder(_FakeEncoder):
        def encode_text(self, text: str) -> Embedding:
            encoded_texts.append(text)
            return super().encode_text(text)

    artifact = _run(tmp_path, row, space, _RecordingEncoder(space))

    result = artifact.query_results[0]
    assert result.query_id == "trake_301"
    assert [event.event_id for event in result.events] == ["E1", "E2", "E3"]
    # The full multi-event query is never embedded; each event is encoded alone.
    assert encoded_texts == ["rua mieng thit ga", "phu bot chien xu", "tha ga vao chao dau"]
    for event_result in result.events:
        assert [p.rank for p in event_result.predictions] == [1, 2, 3]


def test_runner_top_k_is_applied_per_event(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "E1: a\nE2: b", "E1: a\nE2: b")
    artifact = _run(tmp_path, row, space, _FakeEncoder(space), top_k=1)
    result = artifact.query_results[0]
    assert len(result.events) == 2
    for event_result in result.events:
        assert len(event_result.predictions) == 1


def test_prediction_does_not_depend_on_gt_events(tmp_path: Path) -> None:
    space = _space()
    query_text = "E1: washing the chicken\nE2: coating with breadcrumbs"
    # gt_events deliberately contradicts the query text; it must be ignored.
    row = _trake_row("trake_301", query_text, query_text, gt_events="E9: different annotated event")
    artifact = _run(tmp_path, row, space, _FakeEncoder(space))

    result = artifact.query_results[0]
    assert [event.event_id for event in result.events] == ["E1", "E2"]
    assert result.events[0].event_text == "washing the chicken"
    assert result.events[1].event_text == "coating with breadcrumbs"


def test_artifact_round_trip(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "E1: a\nE2: b", "E1: a\nE2: b")
    artifact = _run(tmp_path, row, space, _FakeEncoder(space))

    restored = read_trake_artifact(tmp_path / "trake_predictions")
    assert restored == artifact
    assert restored.manifest.to_dict() == artifact.manifest.to_dict()
    result = restored.query_results[0].to_dict()
    assert result["query_id"] == "trake_301"
    assert result["events"][0]["predictions"][0]["frame_index"] == 15
    assert result["events"][0]["predictions"][0]["score"] == pytest.approx(1.0)
    assert result["events"][0]["windows"][0]["frame_start"] == 15
    assert (tmp_path / "trake_predictions" / "manifest.json").is_file()
    assert (tmp_path / "trake_predictions" / "predictions.jsonl").is_file()


def test_window_generation_in_artifact(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "E1: a\nE2: b", "E1: a\nE2: b")
    artifact = _run(tmp_path, row, space, _FakeEncoder(space))
    event = artifact.query_results[0].events[0]
    # Three isolated frames with a small gap produce one window per frame.
    assert [(w.frame_start, w.frame_end) for w in event.windows] == [(15, 15), (40, 40), (99, 99)]


def test_window_generation_disabled(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "E1: a\nE2: b", "E1: a\nE2: b")
    artifact = _run(tmp_path, row, space, _FakeEncoder(space), window_max_gap=None)
    assert artifact.query_results[0].events[0].windows == ()
    assert artifact.manifest.window_max_gap is None


def test_no_trake_queries_raises(tmp_path: Path) -> None:
    row = dict(_trake_row("trake_301", "E1: a", "E1: a"))
    row["task"] = "kis"
    row["gt_frame_start"] = "10"
    row["gt_frame_end"] = "20"
    row["gt_events"] = ""
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [row])
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    with pytest.raises(ValueError, match="no TRAKE queries"):
        run_trake_prediction(
            TrakePredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "trake_predictions",
            ),
            _FakeEncoder(space),
        )


def test_unparseable_query_fails_clearly(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "no events here at all", "no events here at all")
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [row])
    index_path = _frame_index_artifact(tmp_path, space)

    with pytest.raises(TrakeEventParseError, match="no TRAKE events"):
        run_trake_prediction(
            TrakePredictionRunnerConfig(
                evaluation_set_path=eval_path,
                index_artifact_dir=index_path,
                prediction_output_dir=tmp_path / "trake_predictions",
            ),
            _FakeEncoder(space),
        )


def test_query_subset_selection(tmp_path: Path) -> None:
    eval_path = _write_eval_csv(
        tmp_path / "eval_set.csv",
        [
            _trake_row("trake_301", "E1: a", "E1: a"),
            _trake_row("trake_302", "E1: b", "E1: b"),
        ],
    )
    space = _space()
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_trake_prediction(
        TrakePredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "trake_predictions",
            query_subset_ids=("trake_302",),
        ),
        _FakeEncoder(space),
    )

    assert artifact.manifest.evaluation_set_query_count == 2
    assert artifact.manifest.processed_query_count == 1
    assert artifact.manifest.selected_query_ids == ("trake_302",)
    assert artifact.query_results[0].query_id == "trake_302"


def test_manifest_records_run_metadata(tmp_path: Path) -> None:
    space = _space()
    row = _trake_row("trake_301", "E1: a\nE2: b", "E1: a\nE2: b")
    eval_path = _write_eval_csv(tmp_path / "eval_set.csv", [row])
    index_path = _frame_index_artifact(tmp_path, space)

    artifact = run_trake_prediction(
        TrakePredictionRunnerConfig(
            evaluation_set_path=eval_path,
            index_artifact_dir=index_path,
            prediction_output_dir=tmp_path / "trake_predictions",
            top_k=2,
            window_max_gap=4,
            run_id="my-trake-run",
            query_text_field="query_text_en",
        ),
        _FakeEncoder(space),
    )

    assert artifact.manifest.run_id == "my-trake-run"
    assert artifact.manifest.top_k == 2
    assert artifact.manifest.window_max_gap == 4
    assert artifact.manifest.query_text_field == "query_text_en"
    assert artifact.manifest.evaluation_set_path == eval_path.name  # portable provenance
    assert artifact.manifest.index_manifest["artifact_id"] == "trake-pred-test-index"