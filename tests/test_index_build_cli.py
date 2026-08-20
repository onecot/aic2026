"""Tests for the ``build-index`` CLI entrypoint.

Covers the thin command-line wrapper that connects
``EmbeddingArtifactDataset`` to ``build_index`` and reuses the existing
index artifact validators. The underlying builder/adapter logic is already
tested in ``test_embedding_dataset.py`` and ``test_index_builder.py``; these
tests only exercise the new entrypoint behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aic2026.canonical import (
    CANONICAL_SCHEMA_VERSION,
    KeyframeKey,
    KeyframeTiming,
    VideoId,
)
from aic2026.embedding_artifact import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    EmbeddingManifest,
    iter_frames as iter_embedding_frames,
    write_frames,
    write_manifest,
)
from aic2026.embedding_merge import MergeManifest
from aic2026.embedding_shard import ShardManifest
from aic2026.embeddings import EmbeddingSpace
from aic2026.index_artifact import (
    iter_frames as iter_index_frames,
    read_manifest,
    validate_index_files,
)

_DIM = 8
_ARTIFACT_VIDEOS = [("L21_V002", 2), ("L21_V001", 3)]


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


def _make_embedding_artifact(root: Path, artifact_id: str = "emb-1") -> Path:
    """Write a synthetic embedding artifact and return its directory."""
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
            embedding_space=_space(),
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


def _build_argv(source: Path, output: Path, **overrides: object) -> list[str]:
    fields: dict[str, object] = {
        "--source": str(source),
        "--output": str(output),
        "--artifact-id": "cli-index",
    }
    for flag, value in overrides.items():
        flag_text = f"--{flag.replace('_', '-')}"
        if value is True:
            fields[flag_text] = ""
        else:
            fields[flag_text] = str(value)
    argv: list[str] = []
    for key in ("--source", "--output", "--artifact-id"):
        argv.append(key)
        argv.append(str(fields[key]))
    for flag, value in fields.items():
        if flag in ("--source", "--output", "--artifact-id"):
            continue
        if value == "":
            argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    return ["build-index", *argv]


def test_cli_build_index_end_to_end(tmp_path: Path) -> None:
    from aic2026.cli import main

    source = _make_embedding_artifact(tmp_path)
    output = tmp_path / "index"

    exit_code = main(_build_argv(source, output))

    assert exit_code == 0
    assert (output / "manifest.json").is_file()
    manifest = read_manifest(output / "manifest.json")
    assert manifest.artifact_id == "cli-index"
    assert manifest.num_rows == 5
    assert manifest.num_videos == 2
    assert manifest.source_dataset_id == str(source)
    assert manifest.embedding_space == _space()
    assert validate_index_files(manifest, output) == []
    np.testing.assert_array_equal(
        np.load(output / "embeddings.npy"), np.load(source / EMBEDDINGS_FILENAME)
    )
    assert list(iter_index_frames(output / "frames.ndjson")) == list(
        iter_embedding_frames(source / FRAMES_FILENAME)
    )


def test_cli_build_index_with_verify(tmp_path: Path, capsys) -> None:
    from aic2026.cli import main

    source = _make_embedding_artifact(tmp_path)
    output = tmp_path / "index"

    exit_code = main(_build_argv(source, output, verify=True))

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Verification: OK" in captured.out
    assert validate_index_files(read_manifest(output / "manifest.json"), output) == []


def test_cli_build_index_missing_source_dir_fails(tmp_path: Path, capsys) -> None:
    from aic2026.cli import main

    output = tmp_path / "index"
    exit_code = main(_build_argv(tmp_path / "does-not-exist", output))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not output.exists()


def test_cli_build_index_invalid_source_artifact_fails(tmp_path: Path, capsys) -> None:
    from aic2026.cli import main

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output = tmp_path / "index"

    exit_code = main(_build_argv(empty_dir, output))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert not output.exists()


def _write_frames_and_matrix(artifact: Path, space: EmbeddingSpace) -> None:
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


def _make_shard_artifact(root: Path, artifact_id: str = "worker_000") -> Path:
    shard = root / artifact_id
    shard.mkdir(parents=True, exist_ok=True)
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
    return shard


def _make_merge_artifact(root: Path, artifact_id: str = "merged") -> Path:
    merged = root / artifact_id
    merged.mkdir(parents=True, exist_ok=True)
    _write_frames_and_matrix(merged, _space())
    _write_json(
        merged / "manifest.json",
        MergeManifest(
            schema_version="1.0",
            model="test-model",
            embedding_dim=_DIM,
            total_embeddings=5,
            source_shards=("/tmp/worker_000",),
            embedding_space=_space(),
            created_at="2026-01-01T00:00:00Z",
        ).to_dict(),
    )
    return merged


def _assert_index_matches_source(output: Path, source: Path) -> None:
    manifest = read_manifest(output / "manifest.json")
    assert manifest.num_rows == 5
    assert manifest.num_videos == 2
    assert manifest.embedding_space == _space()
    assert validate_index_files(manifest, output) == []
    np.testing.assert_array_equal(
        np.load(output / "embeddings.npy"), np.load(source / EMBEDDINGS_FILENAME)
    )
    assert list(iter_index_frames(output / "frames.ndjson")) == list(
        iter_embedding_frames(source / FRAMES_FILENAME)
    )


def test_cli_build_index_from_shard_source(tmp_path: Path) -> None:
    from aic2026.cli import main

    source = _make_shard_artifact(tmp_path)
    output = tmp_path / "index"

    exit_code = main(_build_argv(source, output, artifact_id="cli-shard-index"))

    assert exit_code == 0
    _assert_index_matches_source(output, source)
    assert read_manifest(output / "manifest.json").source_dataset_id == str(source)


def test_cli_build_index_from_merge_source(tmp_path: Path) -> None:
    from aic2026.cli import main

    source = _make_merge_artifact(tmp_path)
    output = tmp_path / "index"

    exit_code = main(_build_argv(source, output, verify=True))

    assert exit_code == 0
    _assert_index_matches_source(output, source)