"""Tests for the indexing-stage artifact contract.

These verify the schema (manifest), serialization round-trips,
file-layout consistency validation, and that the contract is independent
of the raw BTC directory layout. They do not build any index.
"""

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