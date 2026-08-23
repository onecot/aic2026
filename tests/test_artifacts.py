"""tmp."""

from __future__ import annotations


# ==========================================================================
# Section: test_artifacts
# ==========================================================================

# ==========================================================================
# Section: test_index_artifact
# ==========================================================================

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.index_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    NORMS_FILENAME,
    PAYLOAD_FILENAME,
    SUPPORTED_SCORE_METRICS,
    IndexManifest,
    iter_frames,
    read_manifest,
    validate_index_files,
    write_frames,
    write_manifest,
)


def _key(ordinal: int, video_id: str = "L21_V001") -> KeyframeKey:
    return KeyframeKey(video_id=VideoId(video_id), keyframe_ordinal=ordinal)


def _timing(ordinal: int, frame_index: int, video_id: str = "L21_V001") -> KeyframeTiming:
    return KeyframeTiming(
        keyframe=_key(ordinal, video_id),
        pts_time=float(frame_index / 25.0),
        fps=25.0,
        frame_index=frame_index,
    )


def _manifest(**overrides) -> IndexManifest:
    base = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "artifact_id": "run-0001",
        "score_metric": "cosine",
        "embedding_dim": 512,
        "embedding_dtype": "float16",
        "num_rows": 3,
        "num_videos": 1,
        "created_at": "2026-08-13T00:00:00Z",
        "source_dataset_id": "aic2026_batch1",
        "canonical_schema_version": "1.0",
    }
    base.update(overrides)
    return IndexManifest(**base)


# ---------------------------------------------------------------- schema basics


def test_schema_version_constant():
    """The artifact schema version is defined and stable."""
    assert isinstance(INDEX_SCHEMA_VERSION, str)
    assert INDEX_SCHEMA_VERSION == "1.1"
    assert "cosine" in SUPPORTED_SCORE_METRICS


def test_filenames_defined():
    """All artifact filenames are explicit constants."""
    assert MANIFEST_FILENAME.endswith(".json")
    assert FRAMES_FILENAME.endswith(".ndjson")
    assert EMBEDDINGS_FILENAME.endswith(".npy")
    assert NORMS_FILENAME.endswith(".npy")
    assert PAYLOAD_FILENAME == "index.bin"


def test_manifest_valid_construction():
    """A fully-specified manifest constructs successfully."""
    manifest = _manifest()
    assert manifest.num_rows == 3
    assert manifest.embedding_dim == 512


def test_manifest_rejects_unsupported_schema_version():
    """An unknown schema version is rejected while 1.0 remains readable."""
    assert _manifest(schema_version="1.0").embedding_space.backend == "legacy"
    with pytest.raises(ValueError):
        _manifest(schema_version="2.0")
    with pytest.raises(ValueError):
        _manifest(schema_version="")


def test_manifest_rejects_unsupported_score_metric():
    """Unknown score metrics are rejected up front."""
    with pytest.raises(ValueError):
        _manifest(score_metric="hamming")


def test_manifest_rejects_invalid_fields():
    """Invalid required fields are rejected."""
    with pytest.raises(ValueError):
        _manifest(artifact_id="")
    with pytest.raises(ValueError):
        _manifest(embedding_dim=0)
    with pytest.raises(ValueError):
        _manifest(num_rows=-1)
    with pytest.raises(ValueError):
        _manifest(embedding_dtype="")
    with pytest.raises(ValueError):
        _manifest(source_dataset_id="")


def test_manifest_round_trip():
    """Manifest survives to_dict/from_dict and JSON serialization."""
    manifest = _manifest()
    blob = json.dumps(manifest.to_dict())
    restored = IndexManifest.from_dict(json.loads(blob))
    assert restored == manifest


def test_manifest_from_dict_missing_key_raises():
    """A manifest missing a required key raises KeyError."""
    d = _manifest().to_dict()
    d.pop("artifact_id")
    with pytest.raises(KeyError):
        IndexManifest.from_dict(d)


# ---------------------------------------------------------------- file round-trips


def test_write_read_manifest(tmp_path: Path):
    """write_manifest/read_manifest round-trip and validate on load."""
    path = tmp_path / MANIFEST_FILENAME
    write_manifest(path, _manifest())
    restored = read_manifest(path)
    assert restored == _manifest()


def test_read_manifest_missing_file(tmp_path: Path):
    """Reading a nonexistent manifest raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nope.json")


def test_read_manifest_corrupt_json(tmp_path: Path):
    """Malformed manifest JSON is surfaced, not silently ignored."""
    path = tmp_path / MANIFEST_FILENAME
    path.write_text("{not valid")
    with pytest.raises(json.JSONDecodeError):
        read_manifest(path)


def test_write_iter_frames_round_trip(tmp_path: Path):
    """write_frames/iter_frames round-trip canonical timing records."""
    frames = [_timing(1, 0), _timing(2, 10), _timing(3, 20)]
    path = tmp_path / FRAMES_FILENAME
    count = write_frames(path, frames)
    assert count == 3
    restored = list(iter_frames(path))
    assert restored == frames
    # row order and identity preserved
    assert [f.keyframe.keyframe_ordinal for f in restored] == [1, 2, 3]
    assert [f.frame_index for f in restored] == [0, 10, 20]


def test_iter_frames_missing_file(tmp_path: Path):
    """Iterating a nonexistent frames file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nope.ndjson"))


def test_iter_frames_corrupt_line(tmp_path: Path):
    """A malformed line in frames.ndjson is not silently skipped."""
    path = tmp_path / FRAMES_FILENAME
    path.write_text("{not valid\n")
    with pytest.raises(json.JSONDecodeError):
        list(iter_frames(path))


# ---------------------------------------------------------------- file validation


def _write_artifact(manifest: IndexManifest, tmp_path: Path) -> Path:
    """Build a minimal consistent artifact directory for tests."""
    artifact_dir = tmp_path / "artifacts" / manifest.artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(artifact_dir / MANIFEST_FILENAME, manifest)
    frames = [_timing(o, (o - 1) * 10) for o in range(1, manifest.num_rows + 1)]
    write_frames(artifact_dir / FRAMES_FILENAME, frames)
    matrix = np.zeros((manifest.num_rows, manifest.embedding_dim), dtype=np.float16)
    matrix[0] = 1.0
    np.save(artifact_dir / EMBEDDINGS_FILENAME, matrix)
    norms = np.ones(manifest.num_rows, dtype=np.float32)
    np.save(artifact_dir / NORMS_FILENAME, norms)
    return artifact_dir


def test_validate_index_files_consistent(tmp_path: Path):
    """A consistent artifact validates with no problems."""
    manifest = _manifest()
    artifact_dir = _write_artifact(manifest, tmp_path)
    assert validate_index_files(manifest, artifact_dir) == []


def test_validate_index_files_missing_files(tmp_path: Path):
    """Missing artifact files are reported."""
    artifact_dir = tmp_path / "empty"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    problems = validate_index_files(_manifest(), artifact_dir)
    assert any(FRAMES_FILENAME in p for p in problems)
    assert any(EMBEDDINGS_FILENAME in p for p in problems)
    assert any(NORMS_FILENAME in p for p in problems)


def test_validate_index_files_row_mismatch(tmp_path: Path):
    """A frames/embeddings row count mismatch is reported."""
    manifest = _manifest()
    artifact_dir = _write_artifact(manifest, tmp_path)
    # Corrupt the matrix row count to disagree with the manifest.
    matrix = np.zeros((manifest.num_rows + 2, manifest.embedding_dim), dtype=np.float16)
    np.save(artifact_dir / EMBEDDINGS_FILENAME, matrix)
    problems = validate_index_files(manifest, artifact_dir)
    assert any("embeddings.npy" in p and "rows" in p for p in problems)


def test_validate_index_files_reports_layout_independence(tmp_path: Path):
    """Validation works for any directory layout, independent of raw BTC."""
    manifest = _manifest(artifact_id="custom-dir")
    artifact_dir = _write_artifact(manifest, tmp_path)
    assert validate_index_files(manifest, artifact_dir) == []


# ---------------------------------------------------------------- identity mapping


def test_row_identity_maps_back_to_original_frame():
    """Frames records carry both keyframe identity and original frame info.

    A retrieval result at row ``p`` maps to both the keyframe
    (video_id + ordinal) and the original frame (frame_index, pts_time).
    """
    frame = _timing(5, 240)
    assert frame.keyframe.video_id == VideoId("L21_V001")
    assert frame.keyframe.keyframe_ordinal == 5
    assert frame.frame_index == 240
    assert frame.pts_time == 240 / 25.0
    # ordinals and frame indices are distinct concepts.
    assert frame.keyframe.keyframe_ordinal != frame.frame_index


# ==========================================================================
# Section: test_embedding_dataset
# ==========================================================================

import json
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
from aic2026.assets.embedding_merge import MergeManifest
from aic2026.assets.embedding_shard import ShardManifest
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


# --- Shard and merge manifest sources ----------------------------------------


def _write_frames_and_matrix(artifact: Path, space: EmbeddingSpace) -> None:
    """Write ``embeddings.npy`` and ``frames.ndjson`` for ``_ARTIFACT_VIDEOS``."""
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
    np.save(artifact / EMBEDDINGS_FILENAME, np.stack(vectors))
    write_frames(artifact / FRAMES_FILENAME, frames)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_adapter_accepts_shard_manifest(tmp_path: Path) -> None:
    """A completed shard (ShardManifest) is consumable by the adapter."""
    shard = tmp_path / "worker_000"
    shard.mkdir()
    _write_frames_and_matrix(shard, _space())
    _write_json(
        shard / "manifest.json",
        ShardManifest(
            schema_version="1.0",
            worker_id=0,
            model="test-model",
            embedding_dim=_DIM,
            dtype="float32",
            device="cpu",
            num_embeddings=5,
            num_videos=len(_ARTIFACT_VIDEOS),
            source_frames="/tmp/frames.ndjson",
            embedding_space=_space(),
            created_at="2026-01-01T00:00:00Z",
        ).to_dict(),
    )

    dataset = EmbeddingArtifactDataset(shard)
    assert dataset.num_rows == 5
    assert dataset.num_videos == 2
    assert dataset.embedding_dim == _DIM
    assert dataset.embedding_dtype == "float32"
    assert dataset.embedding_space == _space()
    assert dataset.artifact_id == "worker_000"
    assert [v.value for v in dataset] == ["L21_V002", "L21_V001"]


def test_adapter_accepts_merge_manifest(tmp_path: Path) -> None:
    """A merged artifact (MergeManifest) is consumable by the adapter."""
    merged = tmp_path / "merged"
    merged.mkdir()
    _write_frames_and_matrix(merged, _space())
    _write_json(
        merged / "manifest.json",
        MergeManifest(
            schema_version="1.0",
            model="test-model",
            embedding_dim=_DIM,
            total_embeddings=5,
            source_shards=("/tmp/worker_000", "/tmp/worker_001"),
            embedding_space=_space(),
            created_at="2026-01-01T00:00:00Z",
        ).to_dict(),
    )

    dataset = EmbeddingArtifactDataset(merged)
    assert dataset.num_rows == 5
    assert dataset.num_videos == 2
    assert dataset.embedding_space == _space()
    assert dataset.artifact_id == "merged"
    assert dataset.embedding_dtype == "float32"


def test_build_index_works_from_shard_artifact(tmp_path: Path) -> None:
    """build_index consumes a shard-format artifact via the adapter."""
    shard = tmp_path / "worker_000"
    shard.mkdir()
    _write_frames_and_matrix(shard, _space())
    _write_json(
        shard / "manifest.json",
        ShardManifest(
            schema_version="1.0",
            worker_id=0,
            model="test-model",
            embedding_dim=_DIM,
            dtype="float32",
            device="cpu",
            num_embeddings=5,
            num_videos=len(_ARTIFACT_VIDEOS),
            source_frames="/tmp/frames.ndjson",
            embedding_space=_space(),
            created_at="2026-01-01T00:00:00Z",
        ).to_dict(),
    )
    dataset = EmbeddingArtifactDataset(shard)
    index_dir = tmp_path / "index"

    manifest = build_index(
        dataset, index_dir, artifact_id="idx-from-shard", embedding_space=dataset.embedding_space
    )

    stored = read_index_manifest(index_dir / "manifest.json")
    assert stored.num_rows == 5
    assert stored.num_videos == 2
    assert stored.source_dataset_id == str(shard)
    assert stored.embedding_space == _space()
    np.testing.assert_array_equal(
        np.load(index_dir / "embeddings.npy"), np.load(shard / EMBEDDINGS_FILENAME)
    )
    assert list(iter_index_frames(index_dir / "frames.ndjson")) == list(
        iter_frames(shard / FRAMES_FILENAME)
    )


# ==========================================================================
# Section: test_index_builder
# ==========================================================================

import json
from pathlib import Path
import numpy as np
import os

from aic2026.dataset import Dataset
from aic2026.index_builder import build_index
from aic2026.embeddings import EmbeddingSpace
from aic2026.index_artifact import read_manifest, iter_frames


def _write_map_csv(path: Path, n_rows: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("n,pts_time,fps,frame_idx\n")
        for i in range(1, n_rows + 1):
            f.write(f"{i},{(i-1)*0.04},25,{(i-1)*2}\n")


def _write_media_info(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "author": "a",
        "channel_id": "c",
        "channel_url": "u",
        "description": "d",
        "keywords": [],
        "length": 10,
        "publish_date": "2020-01-01",
        "thumbnail_url": "t",
        "title": "title",
        "watch_url": "w",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_clip_npy(path: Path, mat: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mat)


def make_raw_dataset(tmp_path: Path):
    raw = tmp_path / "raw"
    # create collection dirs
    for name in ("clip-features-32", "map-keyframes", "media-info", "keyframes", "objects"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    # two videos
    vids = [("L1_V001", 2), ("L1_V002", 1)]
    dim = 8
    for vid, n in vids:
        _write_map_csv(raw / "map-keyframes" / f"{vid}.csv", n)
        _write_media_info(raw / "media-info" / f"{vid}.json")
        mat = (np.arange(n * dim).reshape(n, dim).astype(np.float16) + 1.0)
        _write_clip_npy(raw / "clip-features-32" / f"{vid}.npy", mat)
    return raw


def test_build_index_basic(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact"
    manifest = build_index(ds, artifact, artifact_id="test-1")
    # check files
    assert (artifact / "frames.ndjson").exists()
    assert (artifact / "embeddings.npy").exists()
    assert (artifact / "norms.npy").exists()
    assert (artifact / "manifest.json").exists()
    # validate manifest
    m = read_manifest(artifact / "manifest.json")
    assert m.num_rows == 3
    assert m.num_videos == 2
    emb = np.load(artifact / "embeddings.npy")
    norms = np.load(artifact / "norms.npy")
    assert emb.shape[0] == 3
    assert norms.shape[0] == 3
    # check alignment
    frames = list(iter_frames(artifact / "frames.ndjson"))
    assert len(frames) == 3
    # check that manifest embedding dim matches file
    assert emb.shape[1] == m.embedding_dim


def test_new_build_accepts_explicit_embedding_space(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    artifact = tmp_path / "model_identified_artifact"
    embedding_space = EmbeddingSpace(
        backend="test-backend",
        model_id="test-model",
        checkpoint_id="test-checkpoint",
        space_id="test-space-v1",
        embedding_dim=8,
        output_dtype="float16",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
    )

    manifest = build_index(
        Dataset(raw),
        artifact,
        artifact_id="model-identified-test",
        embedding_space=embedding_space,
    )

    assert manifest.embedding_space == embedding_space
    assert read_manifest(artifact / "manifest.json").embedding_space == embedding_space


def test_resume_behavior(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact_resume"
    # first run: stop after 1 row
    try:
        build_index(ds, artifact, artifact_id="test-resume", stop_after_rows=1)
    except RuntimeError:
        pass
    # manifest should not exist
    assert not (artifact / "manifest.json").exists()
    # checkpoint should exist
    assert (artifact / ".checkpoint.json").exists()
    # now resume full build
    manifest = build_index(ds, artifact, artifact_id="test-resume")
    assert (artifact / "manifest.json").exists()
    m = read_manifest(artifact / "manifest.json")
    assert m.num_rows == 3
    emb = np.load(artifact / "embeddings.npy")
    norms = np.load(artifact / "norms.npy")
    assert emb.shape[0] == 3
    assert norms.shape[0] == 3
