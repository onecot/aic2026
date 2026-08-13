"""Tests for the embedding-artifact-backed dataset adapter."""

from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import (
    CANONICAL_SCHEMA_VERSION,
    KeyframeKey,
    KeyframeTiming,
    VideoId,
)
from aic2026.dataset import MissingRecordError
from aic2026.embedding_artifact import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    EmbeddingManifest,
    iter_frames,
    write_frames,
    write_manifest,
)
from aic2026.embedding_dataset import EmbeddingArtifactDataset
from aic2026.embeddings import EmbeddingSpace
from aic2026.index_artifact import (
    iter_frames as iter_index_frames,
    read_manifest as read_index_manifest,
)
from aic2026.index_builder import build_index

#: Synthetic artifact layout: video -> num keyframes, deliberately non-sorted
#: video order so the adapter's artifact ordering (not raw sorting) is tested.
_ARTIFACT_VIDEOS = [("L21_V002", 2), ("L21_V001", 3)]
_DIM = 8


def _space() -> EmbeddingSpace:
    """Return a fixed synthetic embedding space."""
    return EmbeddingSpace(
        backend="test-backend",
        model_id="test-model",
        checkpoint_id="test-checkpoint",
        space_id="test-space-v1",
        embedding_dim=_DIM,
        output_dtype="float32",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
    )


def _make_embedding_artifact(root: Path, space: EmbeddingSpace, artifact_id: str = "emb-1") -> Path:
    """Write a synthetic embedding artifact and return its directory.

    Row ``i`` of ``embeddings.npy`` is a constant vector of value ``i``, so
    row ordering is trivially verifiable.
    """
    frames: list[KeyframeTiming] = []
    vectors: list[np.ndarray] = []
    for video_id, count in _ARTIFACT_VIDEOS:
        for ordinal in range(1, count + 1):
            row_index = len(vectors)
            vectors.append(np.full(_DIM, float(row_index), dtype=np.float32))
            frames.append(
                KeyframeTiming(
                    keyframe=KeyframeKey(VideoId(video_id), ordinal),
                    pts_time=(ordinal - 1) * 0.4,
                    fps=25.0,
                    frame_index=(ordinal - 1) * 10,
                )
            )
    artifact = root / artifact_id
    artifact.mkdir(parents=True, exist_ok=True)
    np.save(artifact / EMBEDDINGS_FILENAME, np.stack(vectors))
    write_frames(artifact / FRAMES_FILENAME, frames)
    write_manifest(
        artifact / "manifest.json",
        EmbeddingManifest(
            schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            artifact_id=artifact_id,
            embedding_space=space,
            embedding_dim=_DIM,
            embedding_dtype="float32",
            num_rows=len(vectors),
            num_videos=len(_ARTIFACT_VIDEOS),
            source_dataset_id="synthetic",
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            created_at="2026-01-01T00:00:00Z",
        ),
    )
    return artifact


def test_adapter_row_ordering_matches_embedding_artifact(tmp_path: Path) -> None:
    """The adapter streams rows in the exact order stored in the artifact."""
    artifact = _make_embedding_artifact(tmp_path, _space())
    dataset = EmbeddingArtifactDataset(artifact)

    matrix = np.load(artifact / EMBEDDINGS_FILENAME)
    frames = list(iter_frames(artifact / FRAMES_FILENAME))

    assert dataset.video_ids() == ["L21_V002", "L21_V001"]

    collected: list[tuple[str, int, np.ndarray]] = []
    for video_id in dataset.iter_video_ids():
        for embedding in dataset.iter_embeddings(video_id):
            collected.append(
                (
                    embedding.keyframe.video_id.value,
                    embedding.keyframe.keyframe_ordinal,
                    embedding.vector,
                )
            )

    assert len(collected) == len(frames)
    for row_index, (video_id, ordinal, vector) in enumerate(collected):
        assert video_id == frames[row_index].keyframe.video_id.value
        assert ordinal == frames[row_index].keyframe.keyframe_ordinal
        np.testing.assert_array_equal(vector, matrix[row_index])
        assert vector.dtype == matrix.dtype
        assert vector.shape == (matrix.shape[1],)

    assert dataset.embedding_dim == matrix.shape[1]
    assert dataset.embedding_dtype == "float32"
    assert dataset.num_rows == len(frames)


def test_timings_preserve_frame_fields(tmp_path: Path) -> None:
    """Per-video timings preserve pts_time and frame_index in ordinal order."""
    artifact = _make_embedding_artifact(tmp_path, _space())
    dataset = EmbeddingArtifactDataset(artifact)

    frames = list(iter_frames(artifact / FRAMES_FILENAME))
    timings = dataset.timings("L21_V002")

    assert [t.keyframe.keyframe_ordinal for t in timings] == [1, 2]
    assert timings == frames[:2]
    assert timings[0].pts_time == 0.0
    assert timings[0].frame_index == 0
    assert timings[1].pts_time == 0.4
    assert timings[1].frame_index == 10


def test_build_index_works_from_synthetic_embedding_artifact(tmp_path: Path) -> None:
    """build_index consumes the adapter and reproduces the artifact exactly."""
    artifact = _make_embedding_artifact(tmp_path, _space())
    dataset = EmbeddingArtifactDataset(artifact)
    index_dir = tmp_path / "index"

    manifest = build_index(
        dataset, index_dir, artifact_id="idx-from-emb", embedding_space=dataset.embedding_space
    )

    assert manifest.num_rows == 5
    assert manifest.num_videos == 2
    stored = read_index_manifest(index_dir / "manifest.json")
    assert stored.num_rows == 5
    assert stored.num_videos == 2
    assert stored.source_dataset_id == str(artifact)

    np.testing.assert_array_equal(
        np.load(index_dir / "embeddings.npy"), np.load(artifact / EMBEDDINGS_FILENAME)
    )
    assert list(iter_index_frames(index_dir / "frames.ndjson")) == list(
        iter_frames(artifact / FRAMES_FILENAME)
    )


def test_embedding_space_metadata_is_preserved(tmp_path: Path) -> None:
    """The index manifest keeps the artifact's embedding-space identity."""
    space = _space()
    artifact = _make_embedding_artifact(tmp_path, space)
    dataset = EmbeddingArtifactDataset(artifact)

    assert dataset.embedding_space == space

    index_dir = tmp_path / "index-space"
    manifest = build_index(
        dataset, index_dir, artifact_id="idx-space", embedding_space=dataset.embedding_space
    )

    assert manifest.embedding_space == space
    assert read_index_manifest(index_dir / "manifest.json").embedding_space == space


def test_discovery_and_missing_video(tmp_path: Path) -> None:
    """Discovery methods work and unknown videos fail explicitly."""
    artifact = _make_embedding_artifact(tmp_path, _space())
    dataset = EmbeddingArtifactDataset(artifact)

    assert len(dataset) == 2
    assert "L21_V001" in dataset
    assert dataset.has_video("L21_V001")
    assert "L99_V999" not in dataset
    assert [v.value for v in dataset] == ["L21_V002", "L21_V001"]
    assert dataset.raw_root == artifact
    assert dataset.artifact_id == "emb-1"

    with pytest.raises(MissingRecordError):
        dataset.timings("L99_V999")
    with pytest.raises(MissingRecordError):
        next(dataset.iter_embeddings("L99_V999"))
    with pytest.raises(NotImplementedError):
        dataset.metadata("L21_V001")
