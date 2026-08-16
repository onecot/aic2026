"""Tests for embedding-artifact verification and its CLI.

Builds a synthetic embedding artifact directly (``embeddings.npy``,
``frames.ndjson``, ``manifest.json``) without running a model, verifies it,
and mutates it to exercise failure modes. Fast, deterministic, and
cross-platform. Both manifest schemas (runner ``EmbeddingManifest`` and merge
``MergeManifest``) are exercised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding_artifact import EmbeddingManifest, write_manifest
from aic2026.embedding_merge import MERGE_MANIFEST_SCHEMA_VERSION, MergeManifest
from aic2026.embedding_shard import SHARD_MANIFEST_SCHEMA_VERSION, ShardManifest
from aic2026.embedding_verify import (
    EmbeddingVerificationReport,
    MODEL_ID_UNKNOWN,
    verify_embedding,
)
from aic2026.embeddings import EmbeddingSpace

DIM = 4


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "jina-clip-v2",
        "checkpoint_id": "v2.1",
        "space_id": "test-space-v1",
        "embedding_dim": DIM,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


def _write_frames(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ordinal in range(1, count + 1):
            record = {
                "keyframe": {"video_id": {"value": "L21_V001"}, "keyframe_ordinal": ordinal},
                "pts_time": (ordinal - 1) * 0.4,
                "fps": 25.0,
                "frame_index": (ordinal - 1) * 10,
            }
            handle.write(json.dumps(record) + "\n")


def _make_embedding_manifest(
    output_dir: Path,
    *,
    num_rows: int,
    dim: int = DIM,
    dtype: str = "float32",
    **overrides: object,
) -> None:
    manifest = EmbeddingManifest(
        schema_version="1.0",
        artifact_id="emb-test",
        embedding_space=_space(embedding_dim=dim, output_dtype=dtype),
        embedding_dim=dim,
        embedding_dtype=dtype,
        num_rows=num_rows,
        num_videos=1,
        source_dataset_id="aic2026_batch1",
        canonical_schema_version="1.0",
        created_at="2026-01-01T00:00:00+00:00",
    )
    write_manifest(output_dir / "manifest.json", manifest)


def _make_merge_manifest(
    output_dir: Path,
    *,
    num_rows: int,
    dim: int = DIM,
    **overrides: object,
) -> None:
    payload: dict = {
        "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
        "model": "jina-clip-v2",
        "embedding_dim": dim,
        "total_embeddings": num_rows,
        "source_shards": ["shard_000", "shard_001"],
        "embedding_space": _space(embedding_dim=dim).to_dict(),
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_shard_manifest(
    output_dir: Path,
    *,
    num_rows: int,
    dim: int = DIM,
    dtype: str = "float32",
    model: str = "jina-clip:jina-clip-v2:jinaai/jina-clip-v2",
    **overrides: object,
) -> None:
    manifest = ShardManifest(
        schema_version=SHARD_MANIFEST_SCHEMA_VERSION,
        worker_id=0,
        model=model,
        embedding_dim=dim,
        dtype=dtype,
        device="cuda",
        num_embeddings=num_rows,
        num_videos=1,
        source_frames="frames.ndjson",
        embedding_space=_space(embedding_dim=dim, output_dtype=dtype),
        created_at="2026-01-01T00:00:00+00:00",
    )
    payload = manifest.to_dict()
    payload.update(overrides)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_matrix(output_dir: Path, matrix: np.ndarray) -> None:
    np.save(output_dir / "embeddings.npy", matrix, allow_pickle=False)


def _make_artifact(
    tmp_path: Path,
    *,
    rows: int = 5,
    name: str = "artifact",
    merged: bool = False,
    shard: bool = False,
    matrix: np.ndarray | None = None,
    dtype: str = "float32",
) -> Path:
    out = tmp_path / name
    out.mkdir(parents=True)
    if matrix is None:
        matrix = np.arange(rows * DIM, dtype=np.dtype(dtype)).reshape(rows, DIM) / 10.0
    _write_matrix(out, matrix)
    _write_frames(out / "frames.ndjson", rows)
    if merged:
        _make_merge_manifest(out, num_rows=rows)
    elif shard:
        _make_shard_manifest(out, num_rows=rows, dtype=dtype)
    else:
        _make_embedding_manifest(out, num_rows=rows, dtype=dtype)
    return out


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify_embedding.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script()), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestVerifyEmbedding:
    def test_valid_artifact(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        report = verify_embedding(out)

        assert isinstance(report, EmbeddingVerificationReport)
        assert report.valid is True
        assert report.issues == ()
        assert report.rows == 5
        assert report.frames == 5
        assert report.dim == DIM
        assert report.dtype == "float32"
        assert report.empty_vectors == 0
        assert report.nan_vectors == 0
        assert report.model_id == "jina-clip-v2"

    def test_valid_merged_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, merged=True)
        report = verify_embedding(out)

        assert report.valid is True
        assert report.issues == ()
        assert report.rows == 5
        assert report.frames == 5
        assert report.dim == DIM
        assert report.model_id == "jina-clip-v2"

    def test_missing_output_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_embedding(tmp_path / "does-not-exist")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "manifest.json").unlink()

        report = verify_embedding(out)

        assert report.valid is False
        assert report.model_id == MODEL_ID_UNKNOWN
        assert any("manifest.json missing" in issue.message for issue in report.issues)

    def test_missing_frames(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "frames.ndjson").unlink()

        report = verify_embedding(out)

        assert report.valid is False
        assert report.frames == -1
        assert any("frames.ndjson missing" in issue.message for issue in report.issues)

    def test_missing_embeddings(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "embeddings.npy").unlink()

        report = verify_embedding(out)

        assert report.valid is False
        assert report.rows == -1
        assert any("embeddings.npy missing" in issue.message for issue in report.issues)

    def test_malformed_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "manifest.json").write_text("{not json\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert report.model_id == MODEL_ID_UNKNOWN
        assert any("invalid manifest.json" in issue.message for issue in report.issues)

    def test_manifest_missing_model_metadata(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["embedding_space"]["model_id"] = "unknown"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert report.model_id == MODEL_ID_UNKNOWN
        assert any("model metadata missing" in issue.message for issue in report.issues)

    def test_rows_mismatch_with_frames(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        _write_frames(out / "frames.ndjson", 4)

        report = verify_embedding(out)

        assert report.valid is False
        assert any("rows (5) != frames.ndjson records (4)" in issue.message for issue in report.issues)

    def test_rows_mismatch_with_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["num_rows"] = 999
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert any("rows (5) != manifest num_rows (999)" in issue.message for issue in report.issues)

    def test_dim_mismatch_with_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["embedding_dim"] = 8
        payload["embedding_space"]["embedding_dim"] = 8
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert any("dim (4) != manifest embedding_dim (8)" in issue.message for issue in report.issues)

    def test_dtype_mismatch_with_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, dtype="float64")
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["embedding_dtype"] = "float32"
        payload["embedding_space"]["output_dtype"] = "float32"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert any("dtype (float64) != manifest dtype (float32)" in issue.message for issue in report.issues)

    def test_merged_manifest_dtype_matches_embedding_space(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, merged=True)
        report = verify_embedding(out)

        assert report.valid is True

    def test_merged_manifest_dtype_mismatch_caught(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, merged=True, dtype="float64")
        report = verify_embedding(out)

        assert report.valid is False
        assert any("dtype (float64) != manifest dtype (float32)" in issue.message for issue in report.issues)

    def test_valid_shard_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, shard=True)
        report = verify_embedding(out)

        assert report.valid is True
        assert report.issues == ()
        assert report.rows == 5
        assert report.frames == 5
        assert report.dim == DIM
        assert report.dtype == "float32"
        assert report.model_id == "jina-clip:jina-clip-v2:jinaai/jina-clip-v2"

    def test_shard_manifest_dtype_mismatch_caught(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, shard=True, dtype="float64")
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["dtype"] = "float32"
        payload["embedding_space"]["output_dtype"] = "float32"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        report = verify_embedding(out)

        assert report.valid is False
        assert any("dtype (float64) != manifest dtype (float32)" in issue.message for issue in report.issues)

    def test_shard_manifest_rows_mismatch_caught(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, shard=True)
        _write_frames(out / "frames.ndjson", 4)
        report = verify_embedding(out)

        assert report.valid is False
        assert any("frames.ndjson records (4)" in issue.message for issue in report.issues)

    def test_shard_manifest_model_metadata_required(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5, shard=True)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["model"] = "unknown"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert report.model_id == MODEL_ID_UNKNOWN
        assert any("model metadata missing" in issue.message for issue in report.issues)

    def test_shard_manifest_recognized_regardless_of_key_order(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=3, shard=True)
        report = verify_embedding(out)

        assert report.valid is True
        assert report.model_id == "jina-clip:jina-clip-v2:jinaai/jina-clip-v2"

    def test_empty_vectors(self, tmp_path: Path) -> None:
        matrix = np.ones((5, DIM), dtype=np.float32)
        matrix[0] = 0.0
        matrix[2] = 0.0
        out = _make_artifact(tmp_path, rows=5, matrix=matrix)

        report = verify_embedding(out)

        assert report.valid is False
        assert report.empty_vectors == 2
        assert any("2 embedding rows are all-zero" in issue.message for issue in report.issues)

    def test_nan_vectors(self, tmp_path: Path) -> None:
        matrix = np.arange(5 * DIM, dtype=np.float32).reshape(5, DIM) / 10.0
        matrix[1, 0] = np.nan
        out = _make_artifact(tmp_path, rows=5, matrix=matrix)

        report = verify_embedding(out)

        assert report.valid is False
        assert report.nan_vectors == 1
        assert any("1 embedding rows contain NaN/Inf" in issue.message for issue in report.issues)

    def test_inf_vectors(self, tmp_path: Path) -> None:
        matrix = np.arange(5 * DIM, dtype=np.float32).reshape(5, DIM) / 10.0
        matrix[3, 2] = np.inf
        out = _make_artifact(tmp_path, rows=5, matrix=matrix)

        report = verify_embedding(out)

        assert report.valid is False
        assert report.nan_vectors == 1

    def test_not_2d(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        np.save(out / "embeddings.npy", np.zeros(DIM, dtype=np.float32), allow_pickle=False)

        report = verify_embedding(out)

        assert report.valid is False
        assert report.rows == -1
        assert any("must be 2-D" in issue.message for issue in report.issues)

    def test_invalid_manifest_schema(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = "9.9"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert report.model_id == MODEL_ID_UNKNOWN
        assert any("invalid manifest.json" in issue.message for issue in report.issues)

    def test_malformed_frame_record(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        frames_path = out / "frames.ndjson"
        first = json.loads(frames_path.read_text(encoding="utf-8").strip().splitlines()[0])
        del first["keyframe"]["keyframe_ordinal"]
        frames_path.write_text(json.dumps(first) + "\n", encoding="utf-8")

        report = verify_embedding(out)

        assert report.valid is False
        assert report.frames == 0
        assert any("invalid frame record on line 1" in issue.message for issue in report.issues)

    def test_report_round_trip(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        payload = verify_embedding(out).to_dict()

        assert payload["valid"] is True
        assert payload["rows"] == 5
        assert payload["frames"] == 5
        assert payload["dim"] == DIM
        assert payload["dtype"] == "float32"
        assert payload["empty_vectors"] == 0
        assert payload["nan_vectors"] == 0
        assert payload["model_id"] == "jina-clip-v2"


class TestVerifyEmbeddingCli:
    def test_help(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--output-dir" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = _run_cli()
        assert result.returncode == 2

    def test_valid_artifact_exits_0(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        result = _run_cli("--output-dir", str(out))

        assert result.returncode == 0, result.stderr
        assert "Embedding verification: PASS" in result.stdout
        assert "model: jina-clip-v2" in result.stdout
        assert "dim: 4" in result.stdout
        assert "dtype: float32" in result.stdout
        assert "embeddings: 5" in result.stdout
        assert "frames: 5" in result.stdout
        assert "empty: 0" in result.stdout
        assert "nan/inf: 0" in result.stdout
        assert "OK: embedding artifact is valid" in result.stdout

    def test_invalid_artifact_exits_1(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path, rows=5)
        _write_frames(out / "frames.ndjson", 4)
        result = _run_cli("--output-dir", str(out))

        assert result.returncode == 1
        assert "Embedding verification: FAIL" in result.stdout
        assert "Validation failed:" in result.stdout
        assert "[embeddings] embeddings.npy rows (5) != frames.ndjson records (4)" in result.stdout

    def test_missing_dir_exits_1(self, tmp_path: Path) -> None:
        result = _run_cli("--output-dir", str(tmp_path / "does-not-exist"))
        assert result.returncode == 1
        assert "error:" in result.stderr
