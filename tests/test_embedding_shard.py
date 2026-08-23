"""Tests for the resumable shard embedding pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aic2026.canonical import KeyframeTiming
from aic2026.embeddings import EmbeddingSpace
from aic2026.embedding_shard import (
    CHECKPOINT_FILENAME,
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    ShardEmbeddingError,
    ShardManifest,
    embed_shard,
    read_frames,
    select_frames,
)
from aic2026.frame_images import find_keyframe_image as keyframe_image_path

#: video_id -> (ordinals) in the synthetic shard input order.
LAYOUT = [("L21_V001", [1, 2, 3]), ("L21_V002", [1, 2]), ("L22_V001", [1])]
VIDEO_INDEX = {vid: idx for idx, (vid, _) in enumerate(LAYOUT, start=1)}


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "test-model",
        "checkpoint_id": "test-checkpoint",
        "space_id": "test-shard-space-v1",
        "embedding_dim": 4,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


def _rows() -> list[tuple[str, int, float, float, int]]:
    """Return frame rows in file order: (video, ordinal, pts, fps, frame_idx)."""
    out: list[tuple[str, int, float, float, int]] = []
    for vid, ordinals in LAYOUT:
        for ordinal in ordinals:
            out.append((vid, ordinal, (ordinal - 1) * 0.4, 25.0, (ordinal - 1) * 10))
    return out


def _make_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    keyframes = raw / "keyframes"
    for vid, ordinals in LAYOUT:
        vid_dir = keyframes / vid
        vid_dir.mkdir(parents=True)
        for ordinal in ordinals:
            (vid_dir / f"{ordinal:03d}.jpg").write_bytes(bytes([VIDEO_INDEX[vid], ordinal, 0]))
    return raw


def _write_frames(path: Path, rows: list[tuple[str, int, float, float, int]] | None = None) -> None:
    rows = rows if rows is not None else _rows()
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinal, pts, fps, frame_idx in rows:
            record = {
                "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                "pts_time": pts,
                "fps": fps,
                "frame_index": frame_idx,
            }
            handle.write(json.dumps(record) + "\n")


class _StubImageEmbedder:
    """Deterministic embedder deriving vectors from keyframe image bytes."""

    def __init__(
        self,
        space: EmbeddingSpace,
        *,
        device: str = "cpu",
        model: str = "stub:model",
    ) -> None:
        self._space = space
        self._device = device
        self._model = model
        self.encode_calls: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._space.embedding_dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_images(self, batch_images: list[object]) -> np.ndarray:
        self.encode_calls.append(len(batch_images))
        out = np.zeros((len(batch_images), self._space.embedding_dim), dtype=self._space.output_dtype)
        for i, image in enumerate(batch_images):
            data = bytes(image)  # type: ignore[arg-type]
            out[i, 0] = float(data[0])
            out[i, 1] = float(data[1])
        return out


def _image_loader(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    return _make_raw(tmp_path)


@pytest.fixture
def frames_file(tmp_path: Path) -> Path:
    path = tmp_path / "frames.ndjson"
    _write_frames(path)
    return path


def _shard(
    frames_file: Path,
    out_dir: Path,
    raw: Path,
    **kwargs: object,
) -> tuple[_StubImageEmbedder, ShardManifest]:
    embedder = _StubImageEmbedder(_space())
    manifest = embed_shard(
        frames_file,
        out_dir,
        embedder,
        keyframes_root=raw,
        image_loader=_image_loader,
        **kwargs,
    )
    return embedder, manifest


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_shard_creates_valid_artifact(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard"
    _, manifest = _shard(frames_file, out, raw)

    for name in (CHECKPOINT_FILENAME, EMBEDDINGS_FILENAME, FRAMES_FILENAME, MANIFEST_FILENAME):
        assert (out / name).is_file()

    assert manifest.num_embeddings == 6
    assert manifest.num_videos == 3
    assert manifest.embedding_dim == 4
    assert manifest.dtype == "float32"
    assert manifest.model == "stub:model"
    assert manifest.device == "cpu"
    assert manifest.created_at != ""

    embeddings = np.load(out / EMBEDDINGS_FILENAME)
    assert embeddings.shape == (6, 4)
    assert len(_load_rows(out / FRAMES_FILENAME)) == 6


def test_manifest_records_required_fields(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-fields"
    _, _ = _shard(frames_file, out, raw)
    payload = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for key in ("model", "embedding_dim", "dtype", "device", "num_embeddings", "created_at"):
        assert key in payload
    assert payload["schema_version"] == "1.0"
    assert payload["embedding_space"]["embedding_dim"] == 4


def test_manifest_round_trips_embedding_space(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-space"
    embedder, _ = _shard(frames_file, out, raw)
    manifest = ShardManifest.from_dict(json.loads((out / MANIFEST_FILENAME).read_text()))
    assert manifest.embedding_space == embedder.embedding_space
    assert manifest.embedding_space == _space()


def test_embeddings_align_with_frames(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-align"
    _, _ = _shard(frames_file, out, raw)
    embeddings = np.load(out / EMBEDDINGS_FILENAME)
    rows = _load_rows(out / FRAMES_FILENAME)

    assert embeddings.shape[0] == len(rows)
    for i, frame in enumerate(rows):
        vid = frame["keyframe"]["video_id"]["value"]
        ordinal = frame["keyframe"]["keyframe_ordinal"]
        assert embeddings[i][0] == float(VIDEO_INDEX[vid])
        assert embeddings[i][1] == float(ordinal)
        assert frame["frame_index"] == (ordinal - 1) * 10


def test_max_frames_limits_shard(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-limited"
    _, manifest = _shard(frames_file, out, raw, max_frames=3)

    assert manifest.num_embeddings == 3
    assert manifest.num_videos == 1
    embeddings = np.load(out / EMBEDDINGS_FILENAME)
    assert embeddings.shape == (3, 4)
    rows = _load_rows(out / FRAMES_FILENAME)
    assert len(rows) == 3
    assert rows[0]["keyframe"]["video_id"]["value"] == "L21_V001"
    assert rows[2]["keyframe"]["video_id"]["value"] == "L21_V001"


def test_max_frames_zero_produces_empty_shard(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-empty"
    _, manifest = _shard(frames_file, out, raw, max_frames=0)
    assert manifest.num_embeddings == 0
    assert (out / MANIFEST_FILENAME).is_file()
    assert np.load(out / EMBEDDINGS_FILENAME).shape == (0, 4)


def test_interrupt_leaves_no_manifest(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-interrupt"
    embedder = _StubImageEmbedder(_space())
    with pytest.raises(ShardEmbeddingError, match="interrupted shard"):
        embed_shard(
            frames_file, out, embedder, keyframes_root=raw, image_loader=_image_loader,
            batch_size=1, stop_after_frames=2,
        )
    assert not (out / MANIFEST_FILENAME).exists()
    assert (out / CHECKPOINT_FILENAME).is_file()
    assert (out / EMBEDDINGS_FILENAME).is_file()
    checkpoint = json.loads((out / CHECKPOINT_FILENAME).read_text())
    assert checkpoint["completed_frame_count"] == 2
    assert set(checkpoint.keys()) >= {
        "completed_frame_count", "completed_video_ids", "model", "device", "timestamp",
    }


def test_resume_skips_embedded_frames(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-resume"
    embedder = _StubImageEmbedder(_space())
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, embedder, keyframes_root=raw, image_loader=_image_loader,
            batch_size=1, stop_after_frames=2,
        )
    encoded_before_resume = sum(embedder.encode_calls)

    manifest = embed_shard(
        frames_file, out, embedder, keyframes_root=raw, image_loader=_image_loader,
        batch_size=1,
    )
    assert manifest.num_embeddings == 6
    assert (out / MANIFEST_FILENAME).is_file()
    assert sum(embedder.encode_calls) - encoded_before_resume == 4


def test_resume_matches_clean_run(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    clean_out = tmp_path / "clean"
    _, clean_manifest = _shard(frames_file, clean_out, raw)

    resumed_out = tmp_path / "resumed"
    embedder = _StubImageEmbedder(_space())
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, resumed_out, embedder, keyframes_root=raw, image_loader=_image_loader,
            batch_size=2, stop_after_frames=4,
        )
    resume_manifest = embed_shard(
        frames_file, resumed_out, embedder, keyframes_root=raw, image_loader=_image_loader,
        batch_size=2,
    )

    assert clean_manifest.num_embeddings == resume_manifest.num_embeddings == 6
    clean_emb = np.load(clean_out / EMBEDDINGS_FILENAME)
    resumed_emb = np.load(resumed_out / EMBEDDINGS_FILENAME)
    np.testing.assert_array_equal(clean_emb, resumed_emb)
    assert _load_rows(clean_out / FRAMES_FILENAME) == _load_rows(resumed_out / FRAMES_FILENAME)


def test_deterministic_output(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _, _ = _shard(frames_file, out_a, raw)
    _, _ = _shard(frames_file, out_b, raw)
    np.testing.assert_array_equal(
        np.load(out_a / EMBEDDINGS_FILENAME), np.load(out_b / EMBEDDINGS_FILENAME)
    )


def test_resume_rejects_model_mismatch(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-model"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, stop_after_frames=2,
        )
    other = _StubImageEmbedder(_space(), model="stub:other")
    with pytest.raises(ValueError, match="model"):
        embed_shard(frames_file, out, other, keyframes_root=raw, image_loader=_image_loader)


def test_resume_rejects_max_frames_change(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-maxchange"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, max_frames=5, stop_after_frames=2,
        )
    with pytest.raises(ValueError, match="total_frames"):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, max_frames=3,
        )


def test_resume_rejects_source_change(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-source"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, stop_after_frames=2,
        )
    other_source = tmp_path / "other.ndjson"
    _write_frames(other_source)
    with pytest.raises(ValueError, match="source_frames"):
        embed_shard(
            other_source, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader,
        )


def test_complete_shard_returns_without_rebuild(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-idempotent"
    embedder, first = _shard(frames_file, out, raw)
    calls_before = list(embedder.encode_calls)
    second = embed_shard(
        frames_file, out, embedder, keyframes_root=raw, image_loader=_image_loader,
    )
    assert second == first
    assert embedder.encode_calls == calls_before


def test_batch_encoding_preserves_order(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-batch"
    embedder, _ = _shard(frames_file, out, raw, batch_size=2)
    assert embedder.encode_calls == [2, 2, 2]


def test_embeddings_are_mmap_readable(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-mmap"
    _, _ = _shard(frames_file, out, raw)
    matrix = np.load(out / EMBEDDINGS_FILENAME, mmap_mode="r")
    assert matrix.shape == (6, 4)
    assert matrix[0][0] == 1.0
    assert matrix[5][1] == 1.0


def test_shard_embeds_materialized_layout(frames_file: Path, tmp_path: Path) -> None:
    images_root = tmp_path / "materialized"
    for vid, ordinals in LAYOUT:
        vid_dir = images_root / "images" / vid
        vid_dir.mkdir(parents=True)
        for ordinal in ordinals:
            (vid_dir / f"{ordinal:06d}.jpg").write_bytes(
                bytes([VIDEO_INDEX[vid], ordinal, 0])
            )
    out = tmp_path / "shard-materialized"
    _, manifest = _shard(frames_file, out, images_root)
    assert manifest.num_embeddings == 6
    embeddings = np.load(out / EMBEDDINGS_FILENAME)
    assert embeddings[0][0] == 1.0


def test_missing_keyframe_image_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "keyframes" / "L21_V001").mkdir(parents=True)
    (raw / "keyframes" / "L21_V001" / "001.jpg").write_bytes(bytes([1, 1, 0]))
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, [("L21_V001", 1, 0.0, 25.0, 0), ("L99_V999", 1, 0.0, 25.0, 0)])
    out = tmp_path / "shard-missing"
    with pytest.raises(FileNotFoundError, match="L99_V999"):
        embed_shard(
            frames, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader,
        )


def test_keyframe_image_path_resolution(raw: Path) -> None:
    assert keyframe_image_path(raw, "L21_V001", 1) == raw / "keyframes" / "L21_V001" / "001.jpg"
    with pytest.raises(FileNotFoundError):
        keyframe_image_path(raw, "L21_V001", 99)


def test_keyframe_image_path_prefers_materialized_layout(tmp_path: Path) -> None:
    root = tmp_path / "keyframes-root"
    images = root / "images" / "L21_V001"
    images.mkdir(parents=True)
    (images / "000001.jpg").write_bytes(bytes([1, 1, 0]))
    (images / "000002.jpg").write_bytes(bytes([2, 2, 0]))
    assert keyframe_image_path(root, "L21_V001", 1) == images / "000001.jpg"
    assert keyframe_image_path(root, "L21_V001", 2) == images / "000002.jpg"


def test_keyframe_image_path_materialized_layout_missing_image(tmp_path: Path) -> None:
    root = tmp_path / "keyframes-root"
    (root / "images" / "L21_V001").mkdir(parents=True)
    (root / "images" / "L21_V001" / "000001.jpg").write_bytes(bytes([1, 1, 0]))
    with pytest.raises(FileNotFoundError, match="ordinal 99"):
        keyframe_image_path(root, "L21_V001", 99)


def test_keyframe_image_path_legacy_fallback_without_images(tmp_path: Path) -> None:
    root = tmp_path / "keyframes-root"
    legacy = root / "keyframes" / "L21_V001"
    legacy.mkdir(parents=True)
    (legacy / "001.jpg").write_bytes(bytes([1, 1, 0]))
    assert keyframe_image_path(root, "L21_V001", 1) == legacy / "001.jpg"


def test_read_frames_respects_limit(frames_file: Path) -> None:
    assert len(read_frames(frames_file)) == 6
    assert len(read_frames(frames_file, limit=3)) == 3


def test_select_frames_validates_max_frames(frames_file: Path) -> None:
    frames = read_frames(frames_file)
    with pytest.raises(ValueError, match="max_frames"):
        select_frames(frames, -1)
    assert len(select_frames(frames, None)) == 6


def test_shard_frames_are_valid_canonical(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-canonical"
    _, _ = _shard(frames_file, out, raw)
    rows = _load_rows(out / FRAMES_FILENAME)
    for row in rows:
        timing = KeyframeTiming.from_dict(row)
        assert timing.keyframe.keyframe_ordinal >= 1


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "embed_shard.py"


def test_cli_help_lists_device_choices() -> None:
    result = subprocess.run(
        [sys.executable, str(_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "auto" in result.stdout and "mps" in result.stdout and "cuda" in result.stdout
    assert "--max-frames" in result.stdout
    assert "--worker-id" in result.stdout


def test_cli_missing_required_args_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--frames" in result.stderr
    assert "--output-dir" in result.stderr