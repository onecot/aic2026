"""Tests for deterministic shard splitting and merging."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aic2026.embedding_merge import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MERGE_MANIFEST_FILENAME,
    MERGE_MANIFEST_SCHEMA_VERSION,
    MergeError,
    MergeManifest,
    merge_shards,
    order_shards,
)
from aic2026.embedding_shard import (
    CHECKPOINT_FILENAME,
    ShardEmbeddingError,
    ShardManifest,
    embed_shard,
    read_frames,
)
from aic2026.embedding_split import (
    SPLIT_FRAMES_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SplitError,
    SplitManifest,
    assign_frames,
    duplicate_frame_keys,
    group_by_video,
    split_frames,
    validate_no_duplicate_frames,
    worker_dir_name,
)
from aic2026.embeddings import EmbeddingSpace


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "test-model",
        "checkpoint_id": "test-checkpoint",
        "space_id": "test-space-v1",
        "embedding_dim": 4,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


def _write_frames(
    path: Path,
    videos: list[tuple[str, list[int]]],
) -> None:
    """Write a synthetic canonical frames.ndjson for the given video layout."""
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinals in videos:
            for ordinal in ordinals:
                record = {
                    "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                    "pts_time": (ordinal - 1) * 0.4,
                    "fps": 25.0,
                    "frame_index": (ordinal - 1) * 10,
                }
                handle.write(json.dumps(record) + "\n")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _frame_keys(rows: list[dict]) -> list[tuple[str, int]]:
    return [(r["keyframe"]["video_id"]["value"], int(r["keyframe"]["keyframe_ordinal"])) for r in rows]


def _write_raw_frames(path: Path, records: list[tuple[str, int, int]]) -> None:
    """Write frames with explicit ``(video_id, keyframe_ordinal, frame_index)``."""
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinal, frame_index in records:
            record = {
                "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                "pts_time": float(frame_index) / 25.0,
                "fps": 25.0,
                "frame_index": frame_index,
            }
            handle.write(json.dumps(record) + "\n")


def _make_keyframes(raw: Path, videos: list[tuple[str, list[int]]]) -> Path:
    for idx, (vid, ordinals) in enumerate(videos, start=1):
        vid_dir = raw / "keyframes" / vid
        vid_dir.mkdir(parents=True)
        for ordinal in ordinals:
            (vid_dir / f"{ordinal:03d}.jpg").write_bytes(bytes([idx, ordinal, 0]))
    return raw


class _StubImageEmbedder:
    """Deterministic embedder deriving vectors from keyframe image bytes."""

    def __init__(self, space: EmbeddingSpace, *, device: str = "cpu", model: str = "stub:model") -> None:
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


#: Videos with deliberately uneven frame counts to exercise balancing.
LAYOUT = [
    ("L21_V001", [1, 2, 3]),
    ("L21_V002", [1, 2]),
    ("L21_V003", [1]),
    ("L22_V001", [1, 2, 3, 4]),
    ("L22_V002", [1]),
]


@pytest.fixture
def frames_file(tmp_path: Path) -> Path:
    path = tmp_path / "frames.ndjson"
    _write_frames(path, LAYOUT)
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    return _make_keyframes(tmp_path / "raw", LAYOUT)


def _make_shard(
    shard_dir: Path,
    raw: Path,
    frames: list[tuple[str, list[int]]],
    *,
    worker_id: int = 0,
    model: str = "stub:model",
    device: str = "cpu",
    space: EmbeddingSpace | None = None,
) -> ShardManifest:
    """Produce a completed shard directory from a video layout."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    frames_path = shard_dir / f"frames_source_{worker_id}.ndjson"
    _write_frames(frames_path, frames)
    return embed_shard(
        frames_path,
        shard_dir,
        _StubImageEmbedder(space or _space(), device=device, model=model),
        worker_id=worker_id,
        keyframes_root=raw,
        image_loader=_image_loader,
    )


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_worker_dir_name() -> None:
    assert worker_dir_name(0) == "worker_000"
    assert worker_dir_name(5) == "worker_005"
    assert worker_dir_name(42) == "worker_042"
    with pytest.raises(ValueError):
        worker_dir_name(-1)


def test_group_by_video_preserves_order(frames_file: Path) -> None:
    frames = read_frames(frames_file)
    grouped = group_by_video(frames)
    assert [v for v, _ in grouped] == ["L21_V001", "L21_V002", "L21_V003", "L22_V001", "L22_V002"]
    assert len(grouped[0][1]) == 3
    assert len(grouped[3][1]) == 4


def test_split_deterministic(frames_file: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "split-a"
    out_b = tmp_path / "split-b"
    split_frames(frames_file, 3, out_a)
    split_frames(frames_file, 3, out_b)

    for worker in range(3):
        a = _rows(out_a / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME)
        b = _rows(out_b / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME)
        assert a == b


def test_split_no_missing_or_duplicate_frames(frames_file: Path, tmp_path: Path) -> None:
    original = _frame_keys(_rows(frames_file))
    out = tmp_path / "split"
    split_frames(frames_file, 4, out)

    collected: list[tuple[str, int]] = []
    for worker in range(4):
        collected.extend(_frame_keys(_rows(out / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME)))

    assert len(collected) == len(original)
    assert len(set(collected)) == len(original)
    assert sorted(collected) == sorted(original)


def test_split_keeps_videos_intact(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    split_frames(frames_file, 3, out)

    seen_videos: list[list[str]] = []
    for worker in range(3):
        videos = {r["keyframe"]["video_id"]["value"] for r in _rows(out / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME)}
        seen_videos.append(sorted(videos))
    # A video may only appear in one worker.
    combined = [v for group in seen_videos for v in group]
    assert len(combined) == len(set(combined))
    assert set(combined) == {"L21_V001", "L21_V002", "L21_V003", "L22_V001", "L22_V002"}


def test_split_balanced_frame_counts(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    manifest = split_frames(frames_file, 3, out)

    counts = [w.num_frames for w in manifest.workers]
    # 11 frames across 3 workers: greedy balancing should keep max-min small.
    assert max(counts) - min(counts) <= 2
    assert sum(counts) == 11


def test_split_manifest_contract(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    manifest = split_frames(frames_file, 3, out)

    assert manifest.schema_version == SPLIT_MANIFEST_SCHEMA_VERSION
    assert manifest.num_workers == 3
    assert [w.worker_id for w in manifest.workers] == [0, 1, 2]
    assert sum(w.num_frames for w in manifest.workers) == 11

    payload = json.loads((out / SPLIT_MANIFEST_FILENAME).read_text())
    for key in ("source_frames", "num_workers", "workers"):
        assert key in payload
    assert payload["workers"][0]["video_ids"] == list(manifest.workers[0].video_ids)
    assert SplitManifest.from_dict(payload) == manifest


def test_split_manifest_round_trip(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    manifest = split_frames(frames_file, 2, out)
    assert SplitManifest.from_dict(manifest.to_dict()) == manifest


def test_split_preserves_frame_order(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    split_frames(frames_file, 2, out)

    for worker in range(2):
        keys = _frame_keys(_rows(out / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME))
        # Frame order within the original file must be preserved per video.
        by_video: dict[str, list[int]] = {}
        for vid, ordinal in keys:
            by_video.setdefault(vid, []).append(ordinal)
        for vid, ordinals in by_video.items():
            assert ordinals == sorted(ordinals)


def test_split_refuses_existing_output(frames_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "split"
    split_frames(frames_file, 2, out)
    with pytest.raises(SplitError, match="already exists"):
        split_frames(frames_file, 2, out)
    # overwrite replaces the split deterministically.
    split_frames(frames_file, 2, out, overwrite=True)


def test_assign_frames_validates_num_workers() -> None:
    with pytest.raises(ValueError):
        assign_frames([], 0)
    assert assign_frames([], 3) == [[], [], []]


def test_split_invalid_num_workers(frames_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_workers"):
        split_frames(frames_file, 0, tmp_path / "x")


# ---------------------------------------------------------------------------
# Duplicate keyframe detection
# ---------------------------------------------------------------------------


def test_duplicate_frame_keys_helper(tmp_path: Path) -> None:
    path = tmp_path / "dup.ndjson"
    _write_raw_frames(
        path,
        [
            ("L21_V006", 1, 0),
            ("L21_V006", 2, 0),  # duplicate (L21_V006, 0)
            ("L21_V007", 1, 0),
            ("L21_V007", 2, 5),  # distinct frame_index: not a duplicate
            ("L21_V008", 1, 0),
        ],
    )
    frames = read_frames(path)
    assert duplicate_frame_keys(frames) == [("L21_V006", 0)]


def test_duplicate_frame_keys_empty_when_unique(frames_file: Path) -> None:
    assert duplicate_frame_keys(read_frames(frames_file)) == []


def test_validate_no_duplicate_frames_ok(frames_file: Path) -> None:
    # Must not raise for a clean input.
    validate_no_duplicate_frames(read_frames(frames_file), "input frames file x")


def test_split_rejects_duplicate_keyframes(tmp_path: Path) -> None:
    path = tmp_path / "dup.ndjson"
    _write_raw_frames(
        path,
        [
            ("L21_V006", 1, 0),
            ("L21_V006", 2, 0),
            ("L21_V007", 1, 0),
            ("L21_V007", 2, 0),
            ("L21_V008", 1, 0),
        ],
    )
    with pytest.raises(SplitError) as excinfo:
        split_frames(path, 2, tmp_path / "splits")
    message = str(excinfo.value)
    assert "duplicate keyframes" in message
    assert "2 duplicate keyframes" in message  # total duplicate count
    # No split output should be produced on failure.
    assert not (tmp_path / "splits" / "worker_000").exists()


def test_split_error_message_contains_example_keys(tmp_path: Path) -> None:
    path = tmp_path / "dup.ndjson"
    _write_raw_frames(
        path,
        [
            ("L21_V006", 1, 0),
            ("L21_V006", 2, 0),
            ("L21_V007", 1, 0),
            ("L21_V007", 2, 0),
            ("L21_V008", 1, 0),
            ("L21_V008", 2, 0),
        ],
    )
    with pytest.raises(SplitError) as excinfo:
        split_frames(path, 3, tmp_path / "splits")
    message = str(excinfo.value)
    assert repr(("L21_V006", 0)) in message
    assert repr(("L21_V007", 0)) in message
    assert repr(("L21_V008", 0)) in message
    assert "3 duplicate keyframes" in message


def test_split_accepts_unique_frame_indexes(tmp_path: Path) -> None:
    # Same video with distinct frame_index values is valid, even when two
    # videos both start at frame_index 0 (they are distinct videos).
    path = tmp_path / "ok.ndjson"
    _write_raw_frames(
        path,
        [
            ("L21_V006", 1, 0),
            ("L21_V006", 2, 1),
            ("L21_V007", 1, 0),
            ("L21_V007", 2, 1),
        ],
    )
    manifest = split_frames(path, 2, tmp_path / "splits")
    assert manifest.num_workers == 2
    assert sum(w.num_frames for w in manifest.workers) == 4


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------


@pytest.fixture
def shards(raw: Path, tmp_path: Path) -> list[Path]:
    """Create three completed shards with distinct videos (no overlap)."""
    layouts = {
        0: [("L21_V001", [1, 2, 3])],
        1: [("L21_V002", [1, 2]), ("L21_V003", [1])],
        2: [("L22_V001", [1, 2, 3, 4])],
    }
    dirs: list[Path] = []
    for worker_id in (0, 1, 2):
        shard_dir = tmp_path / f"worker_{worker_id:03d}"
        _make_shard(shard_dir, raw, layouts[worker_id], worker_id=worker_id)
        dirs.append(shard_dir)
    return dirs


def _merge(shards: list[Path], tmp_path: Path, **kwargs: object) -> MergeManifest:
    return merge_shards(shards, tmp_path / "merged", **kwargs)


def test_merge_correct_concatenation(shards: list[Path], tmp_path: Path) -> None:
    manifest = _merge(shards, tmp_path)

    assert manifest.total_embeddings == 3 + 3 + 4
    assert manifest.embedding_dim == 4
    assert manifest.model == "stub:model"
    assert manifest.schema_version == MERGE_MANIFEST_SCHEMA_VERSION
    assert len(manifest.source_shards) == 3

    merged_emb = np.load(tmp_path / "merged" / EMBEDDINGS_FILENAME)
    rows = _rows(tmp_path / "merged" / FRAMES_FILENAME)
    assert merged_emb.shape == (10, 4)
    assert len(rows) == 10


def test_merge_aligns_embeddings_with_frames(shards: list[Path], tmp_path: Path) -> None:
    _merge(shards, tmp_path)
    merged_emb = np.load(tmp_path / "merged" / EMBEDDINGS_FILENAME)
    rows = _rows(tmp_path / "merged" / FRAMES_FILENAME)

    for i, row in enumerate(rows):
        # Stub embeddings store [video_index, ordinal, 0, 0].
        vid = row["keyframe"]["video_id"]["value"]
        ordinal = row["keyframe"]["keyframe_ordinal"]
        expected_video = {"L21_V001": 1.0, "L21_V002": 2.0, "L21_V003": 3.0, "L22_V001": 4.0}[vid]
        assert merged_emb[i][0] == expected_video
        assert merged_emb[i][1] == float(ordinal)


def test_merge_deterministic_order(raw: Path, tmp_path: Path) -> None:
    # Shards passed out of order must merge in worker_id order.
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1, 2])], worker_id=1)
    _make_shard(shard_b, raw, [("L21_V002", [1, 2])], worker_id=0)

    manifest = merge_shards([shard_a, shard_b], tmp_path / "merged")
    rows = _rows(tmp_path / "merged" / FRAMES_FILENAME)
    assert rows[0]["keyframe"]["video_id"]["value"] == "L21_V002"  # worker 0 first
    assert rows[2]["keyframe"]["video_id"]["value"] == "L21_V001"
    assert manifest.source_shards == (str(shard_b), str(shard_a))


def test_order_shards_uses_worker_id(raw: Path, tmp_path: Path) -> None:
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1])], worker_id=3)
    _make_shard(shard_b, raw, [("L21_V002", [1])], worker_id=1)
    ordered = order_shards([shard_a, shard_b])
    assert ordered == [shard_b, shard_a]


def test_merge_rejects_missing_manifest(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(MergeError, match="not complete"):
        merge_shards([incomplete], tmp_path / "merged")


def test_merge_rejects_mixed_models(raw: Path, tmp_path: Path) -> None:
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1])], worker_id=0, model="stub:model")
    _make_shard(shard_b, raw, [("L21_V002", [1])], worker_id=1, model="stub:other")
    with pytest.raises(MergeError, match="model"):
        merge_shards([shard_a, shard_b], tmp_path / "merged")


def test_merge_rejects_mixed_dimensions(raw: Path, tmp_path: Path) -> None:
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1])], worker_id=0, space=_space(embedding_dim=4))
    _make_shard(shard_b, raw, [("L21_V002", [1])], worker_id=1, space=_space(embedding_dim=8))
    with pytest.raises(MergeError, match="embedding_dim"):
        merge_shards([shard_a, shard_b], tmp_path / "merged")


def test_merge_rejects_mixed_spaces(raw: Path, tmp_path: Path) -> None:
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1])], worker_id=0, space=_space(space_id="space-v1"))
    _make_shard(shard_b, raw, [("L21_V002", [1])], worker_id=1, space=_space(space_id="space-v2"))
    with pytest.raises(MergeError, match="embedding_space"):
        merge_shards([shard_a, shard_b], tmp_path / "merged")


def test_merge_rejects_duplicate_frames(raw: Path, tmp_path: Path) -> None:
    shard_a = tmp_path / "shard-a"
    shard_b = tmp_path / "shard-b"
    _make_shard(shard_a, raw, [("L21_V001", [1, 2])], worker_id=0)
    _make_shard(shard_b, raw, [("L21_V001", [2, 3])], worker_id=1)  # overlaps on ordinal 2
    with pytest.raises(MergeError, match="duplicate keyframe"):
        merge_shards([shard_a, shard_b], tmp_path / "merged")


def test_merge_rejects_empty_input() -> None:
    with pytest.raises(MergeError, match="at least one"):
        merge_shards([], Path("x"))  # type: ignore[list-item]


def test_merge_manifest_round_trip(shards: list[Path], tmp_path: Path) -> None:
    manifest = _merge(shards, tmp_path)
    payload = json.loads((tmp_path / "merged" / MERGE_MANIFEST_FILENAME).read_text())
    assert MergeManifest.from_dict(payload) == manifest
    for key in ("model", "embedding_dim", "total_embeddings", "source_shards", "embedding_space", "created_at"):
        assert key in payload


def test_merge_refuses_existing_output(shards: list[Path], tmp_path: Path) -> None:
    _merge(shards, tmp_path)
    with pytest.raises(MergeError, match="already exists"):
        merge_shards(shards, tmp_path / "merged")
    _merge(shards, tmp_path, overwrite=True)


def test_merge_missing_shard_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        merge_shards([tmp_path / "nope"], tmp_path / "merged")


# ---------------------------------------------------------------------------
# Resume behavior
# ---------------------------------------------------------------------------


def test_resume_after_device_change_allowed(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-device"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space(), device="cpu"), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, stop_after_frames=2,
        )
    # Same model, different device: resume must NOT raise.
    manifest = embed_shard(
        frames_file, out, _StubImageEmbedder(_space(), device="mps"), keyframes_root=raw,
        image_loader=_image_loader, batch_size=1,
    )
    assert manifest.num_embeddings == 11


def test_resume_after_precision_change_allowed(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-precision"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, precision="fp32", stop_after_frames=2,
        )
    manifest = embed_shard(
        frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
        image_loader=_image_loader, batch_size=1, precision="fp16",
    )
    assert manifest.num_embeddings == 11


def test_resume_rejects_model_change(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-model"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, stop_after_frames=2,
        )
    other = _StubImageEmbedder(_space(), model="stub:other")
    with pytest.raises(ValueError, match="model"):
        embed_shard(frames_file, out, other, keyframes_root=raw, image_loader=_image_loader)


def test_resume_rejects_embedding_space_change(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-space"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, stop_after_frames=2,
        )
    other = _StubImageEmbedder(_space(space_id="different-space"))
    with pytest.raises(ValueError, match="embedding_space"):
        embed_shard(frames_file, out, other, keyframes_root=raw, image_loader=_image_loader)


def test_checkpoint_records_precision_and_space(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "shard-checkpoint-fields"
    with pytest.raises(ShardEmbeddingError):
        embed_shard(
            frames_file, out, _StubImageEmbedder(_space()), keyframes_root=raw,
            image_loader=_image_loader, batch_size=1, precision="fp32", stop_after_frames=2,
        )
    checkpoint = json.loads((out / CHECKPOINT_FILENAME).read_text())
    assert checkpoint["precision"] == "fp32"
    assert checkpoint["embedding_space"]["space_id"] == "test-space-v1"


# ---------------------------------------------------------------------------
# CLI validation (no model load)
# ---------------------------------------------------------------------------


def _split_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "split_embedding_shards.py"


def _merge_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "merge_embedding_shards.py"


def test_split_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_split_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--num-workers" in result.stdout
    assert "--output-dir" in result.stdout


def test_split_cli_missing_args_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_split_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--num-workers" in result.stderr


def test_merge_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_merge_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--shards" in result.stdout
    assert "--output-dir" in result.stdout


def test_merge_cli_missing_args_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_merge_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--shards" in result.stderr


def test_split_cli_missing_source_frames(frames_file: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_split_script()),
            "--frames",
            str(tmp_path / "missing.ndjson"),
            "--num-workers",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_split_merge_cli_end_to_end(frames_file: Path, raw: Path, tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    result = subprocess.run(
        [
            sys.executable,
            str(_split_script()),
            "--frames",
            str(frames_file),
            "--num-workers",
            "3",
            "--output-dir",
            str(splits),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Split complete: 11 frames" in result.stdout

    # Embed each worker split into a shard (synthetic raw images).
    shards: list[Path] = []
    for worker in range(3):
        worker_frames = splits / worker_dir_name(worker) / SPLIT_FRAMES_FILENAME
        shard = tmp_path / f"worker_{worker:03d}"
        embed_shard(
            worker_frames,
            shard,
            _StubImageEmbedder(_space()),
            worker_id=worker,
            keyframes_root=raw,
            image_loader=_image_loader,
        )
        shards.append(shard)

    merged = tmp_path / "merged"
    result = subprocess.run(
        [
            sys.executable,
            str(_merge_script()),
            "--shards",
            *[str(s) for s in shards],
            "--output-dir",
            str(merged),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Merge complete: 11 embeddings" in result.stdout
    assert np.load(merged / EMBEDDINGS_FILENAME).shape == (11, 4)
    assert len(_rows(merged / FRAMES_FILENAME)) == 11