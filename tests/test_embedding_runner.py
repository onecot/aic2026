"""Tests for the reusable image-embedding runner and its artifact contract."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

import aic2026.embedding_runner as embedding_runner
from aic2026.canonical import VideoId
from aic2026.dataset import Dataset
from aic2026.embedding_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    EmbeddingManifest,
    read_manifest,
    validate_embedding_artifact,
)
from aic2026.embedding_runner import (
    EmbeddingBuildError,
    build_embeddings,
    default_image_loader,
)
from aic2026.embeddings import Embedding, EmbeddingSpace

COLLECTIONS = ("clip-features-32", "keyframes", "map-keyframes", "media-info", "objects")

#: video_id -> num keyframes; synthetic images carry (video_index, ordinal).
LAYOUT = [("L21_V001", 3), ("L21_V002", 2), ("L22_V001", 1)]


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "test-model",
        "checkpoint_id": "test-checkpoint",
        "space_id": "test-image-space-v1",
        "embedding_dim": 4,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "test-preprocess-v1",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


class _RecordingSyntheticEncoder:
    """Deterministic image encoder that also records every encode call."""

    def __init__(self, space: EmbeddingSpace, *, batch: bool = True) -> None:
        self.space = space
        self.batch = batch
        self.encode_calls: list[list[int]] = []

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self.space

    def encode_image_batch(self, images: list[object]) -> list[Embedding]:
        self.encode_calls.append([self._row_index(image) for image in images])
        return self._embed_many(images)

    def encode_image(self, image: object) -> Embedding:
        self.encode_calls.append([self._row_index(image)])
        return self._embed_many([image])[0]

    def encode_text(self, text: str) -> Embedding:
        raise NotImplementedError("synthetic encoder is image-only")

    @staticmethod
    def _row_index(image: object) -> int:
        data = bytes(image)  # type: ignore[arg-type]
        return data[0] * 1000 + data[1]

    def _embed_many(self, images: list[object]) -> list[Embedding]:
        out: list[Embedding] = []
        for image in images:
            data = bytes(image)  # type: ignore[arg-type]
            vector = np.zeros(self.space.embedding_dim, dtype=np.float32)
            vector[0] = float(data[0])
            vector[1] = float(data[1])
            out.append(Embedding(kind="image", vector=vector, space=self.space))
        return out


def _image_loader(path: Path) -> bytes:
    return path.read_bytes()


def make_raw_dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for collection in COLLECTIONS:
        (raw / collection).mkdir(parents=True, exist_ok=True)
    for video_index, (vid, n) in enumerate(LAYOUT, start=1):
        map_path = raw / "map-keyframes" / f"{vid}.csv"
        with map_path.open("w", encoding="utf-8") as handle:
            handle.write("n,pts_time,fps,frame_idx\n")
            for ordinal in range(1, n + 1):
                handle.write(f"{ordinal},{(ordinal - 1) * 0.4},25,{(ordinal - 1) * 10}\n")
        keyframes_dir = raw / "keyframes" / vid
        keyframes_dir.mkdir()
        for ordinal in range(1, n + 1):
            (keyframes_dir / f"{ordinal:03d}.jpg").write_bytes(
                bytes([video_index, ordinal, 0])
            )
    return raw


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    return make_raw_dataset(tmp_path)


def _build(
    raw: Path,
    artifact_dir: Path,
    artifact_id: str = "emb-test",
    **kwargs: object,
) -> tuple[Dataset, _RecordingSyntheticEncoder, EmbeddingManifest]:
    dataset = Dataset(raw)
    encoder = _RecordingSyntheticEncoder(_space())
    manifest = build_embeddings(
        dataset, encoder, artifact_dir, artifact_id, image_loader=_image_loader, **kwargs
    )
    return dataset, encoder, manifest


def test_build_creates_valid_artifact(raw_dataset: Path, tmp_path: Path) -> None:
    """A build writes the three artifact files and a valid manifest."""
    artifact = tmp_path / "artifact"
    _, _, manifest = _build(raw_dataset, artifact)
    assert (artifact / "manifest.json").is_file()
    assert (artifact / EMBEDDINGS_FILENAME).is_file()
    assert (artifact / FRAMES_FILENAME).is_file()
    assert (artifact / ".checkpoint.json").exists() is False
    assert (artifact / ".build_meta.json").exists() is False

    restored = read_manifest(artifact / "manifest.json")
    assert restored == manifest
    assert manifest.num_rows == 6
    assert manifest.num_videos == 3
    assert manifest.embedding_dim == 4
    assert manifest.embedding_dtype == "float32"
    assert validate_embedding_artifact(manifest, artifact) == []


def test_manifest_records_embedding_space(raw_dataset: Path, tmp_path: Path) -> None:
    """The manifest preserves the encoder's full embedding-space identity."""
    space = _space()
    artifact = tmp_path / "space-artifact"
    dataset = Dataset(raw_dataset)
    encoder = _RecordingSyntheticEncoder(space)
    build_embeddings(dataset, encoder, artifact, "emb-space", image_loader=_image_loader)
    manifest = read_manifest(artifact / "manifest.json")
    assert manifest.embedding_space == space
    payload = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert payload["embedding_space"]["space_id"] == space.space_id
    assert payload["embedding_space"]["output_dtype"] == "float32"


def test_deterministic_row_ordering(raw_dataset: Path, tmp_path: Path) -> None:
    """Rows follow dataset video-id order, then canonical keyframe order."""
    expected = [
        ("L21_V001", 1), ("L21_V001", 2), ("L21_V001", 3),
        ("L21_V002", 1), ("L21_V002", 2),
        ("L22_V001", 1),
    ]
    artifact = tmp_path / "order-artifact"
    _, _, _ = _build(raw_dataset, artifact)
    rows = []
    with (artifact / FRAMES_FILENAME).open(encoding="utf-8") as handle:
        for line in handle:
            timing = json.loads(line)
            rows.append(
                (timing["keyframe"]["video_id"]["value"], timing["keyframe"]["keyframe_ordinal"])
            )
    assert rows == expected


def test_embedding_frame_alignment(raw_dataset: Path, tmp_path: Path) -> None:
    """embeddings[i] corresponds exactly to frames[i]."""
    artifact = tmp_path / "align-artifact"
    _, _, _ = _build(raw_dataset, artifact)
    embeddings = np.load(artifact / EMBEDDINGS_FILENAME)
    with (artifact / FRAMES_FILENAME).open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle]

    assert embeddings.shape == (6, 4)
    expected_video_index = {"L21_V001": 1.0, "L21_V002": 2.0, "L22_V001": 3.0}
    for i, frame in enumerate(frames):
        vid = frame["keyframe"]["video_id"]["value"]
        ordinal = frame["keyframe"]["keyframe_ordinal"]
        assert embeddings[i][0] == expected_video_index[vid]
        assert embeddings[i][1] == float(ordinal)
        assert frame["frame_index"] == (ordinal - 1) * 10
        assert frame["pts_time"] == (ordinal - 1) * 0.4
        assert frame["fps"] == 25.0


def test_embedding_space_consistency_is_enforced(raw_dataset: Path, tmp_path: Path) -> None:
    """An encoder returning embeddings in a different space is rejected."""
    artifact = tmp_path / "space-mismatch"
    dataset = Dataset(raw_dataset)

    class _WrongSpaceEncoder(_RecordingSyntheticEncoder):
        def _embed_many(self, images: list[object]) -> list[Embedding]:
            other = _space(space_id="different-space")
            out = []
            for image in images:
                data = bytes(image)  # type: ignore[arg-type]
                vector = np.zeros(4, dtype=np.float32)
                vector[0] = float(data[0])
                vector[1] = float(data[1])
                out.append(Embedding(kind="image", vector=vector, space=other))
            return out

    with pytest.raises(ValueError, match="embedding space mismatch"):
        build_embeddings(
            dataset, _WrongSpaceEncoder(_space()), artifact, "emb-space-mismatch",
            image_loader=_image_loader,
        )


def test_artifact_validator_reports_bad_matrix(raw_dataset: Path, tmp_path: Path) -> None:
    """Dimension/dtype mismatches are caught by the artifact validator."""
    artifact = tmp_path / "bad-matrix"
    _, _, manifest = _build(raw_dataset, artifact)
    np.save(
        artifact / EMBEDDINGS_FILENAME,
        np.zeros((6, 2), dtype=np.float16),
    )
    problems = validate_embedding_artifact(manifest, artifact)
    assert any("dim" in problem for problem in problems)
    assert any("dtype" in problem for problem in problems)


def test_artifact_validator_reports_missing_file(raw_dataset: Path, tmp_path: Path) -> None:
    """A missing artifact file is reported by the validator."""
    artifact = tmp_path / "missing-file"
    _, _, manifest = _build(raw_dataset, artifact)
    (artifact / FRAMES_FILENAME).unlink()
    problems = validate_embedding_artifact(manifest, artifact)
    assert any(FRAMES_FILENAME in problem for problem in problems)


def test_no_manifest_for_incomplete_build(raw_dataset: Path, tmp_path: Path) -> None:
    """An interrupted build never exposes a final manifest."""
    artifact = tmp_path / "interrupted"
    dataset = Dataset(raw_dataset)
    encoder = _RecordingSyntheticEncoder(_space())
    with pytest.raises(EmbeddingBuildError, match="interrupted build"):
        build_embeddings(
            dataset, encoder, artifact, "emb-interrupted",
            stop_after_rows=2, batch_size=1, image_loader=_image_loader,
        )
    assert not (artifact / "manifest.json").exists()
    assert (artifact / ".checkpoint.json").exists()
    checkpoint = json.loads((artifact / ".checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["next_row"] == 2


def test_resume_completes_without_recompute(raw_dataset: Path, tmp_path: Path) -> None:
    """Resume finishes the build and never re-encodes completed rows."""
    artifact = tmp_path / "resume"
    dataset = Dataset(raw_dataset)
    encoder = _RecordingSyntheticEncoder(_space())
    with pytest.raises(EmbeddingBuildError):
        build_embeddings(
            dataset, encoder, artifact, "emb-resume",
            stop_after_rows=2, batch_size=1, image_loader=_image_loader,
        )
    calls_after_interrupt = len(encoder.encode_calls)

    manifest = build_embeddings(
        dataset, encoder, artifact, "emb-resume", batch_size=1, image_loader=_image_loader,
    )
    assert manifest.num_rows == 6
    assert (artifact / "manifest.json").is_file()
    # exactly the 4 remaining rows were encoded, not the full 6 again
    assert len(encoder.encode_calls) - calls_after_interrupt == 4

    embeddings = np.load(artifact / EMBEDDINGS_FILENAME)
    assert embeddings.shape == (6, 4)
    with (artifact / FRAMES_FILENAME).open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle]
    assert len(frames) == 6
    expected_video_index = {"L21_V001": 1.0, "L21_V002": 2.0, "L22_V001": 3.0}
    for i, frame in enumerate(frames):
        assert embeddings[i][0] == expected_video_index[frame["keyframe"]["video_id"]["value"]]
        assert embeddings[i][1] == float(frame["keyframe"]["keyframe_ordinal"])
    assert validate_embedding_artifact(manifest, artifact) == []


def test_resume_rejects_different_embedding_space(raw_dataset: Path, tmp_path: Path) -> None:
    """A resumed build must keep the original embedding space."""
    artifact = tmp_path / "resume-space"
    dataset = Dataset(raw_dataset)
    with pytest.raises(EmbeddingBuildError):
        build_embeddings(
            dataset, _RecordingSyntheticEncoder(_space()), artifact, "emb-resume-space",
            stop_after_rows=1, batch_size=1, image_loader=_image_loader,
        )
    other = _RecordingSyntheticEncoder(_space(space_id="other-space"))
    with pytest.raises(ValueError, match="embedding space"):
        build_embeddings(
            dataset, other, artifact, "emb-resume-space", image_loader=_image_loader,
        )


def test_resume_rejects_different_artifact_id(raw_dataset: Path, tmp_path: Path) -> None:
    """A resumed build must keep the original artifact id."""
    artifact = tmp_path / "resume-id"
    dataset = Dataset(raw_dataset)
    with pytest.raises(EmbeddingBuildError):
        build_embeddings(
            dataset, _RecordingSyntheticEncoder(_space()), artifact, "emb-resume-id",
            stop_after_rows=1, batch_size=1, image_loader=_image_loader,
        )
    with pytest.raises(ValueError, match="artifact_id"):
        build_embeddings(
            dataset, _RecordingSyntheticEncoder(_space()), artifact, "different-id",
            image_loader=_image_loader,
        )


def test_subset_build(raw_dataset: Path, tmp_path: Path) -> None:
    """An explicit video-id subset builds without materializing the dataset."""
    artifact = tmp_path / "subset"
    _, encoder, manifest = _build(
        raw_dataset, artifact, video_ids=["L21_V001", "L22_V001"]
    )
    assert manifest.num_rows == 4
    assert manifest.num_videos == 2
    with (artifact / FRAMES_FILENAME).open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle]
    assert [f["keyframe"]["video_id"]["value"] for f in frames] == [
        "L21_V001", "L21_V001", "L21_V001", "L22_V001"
    ]
    # subset is processed in the dataset's deterministic order even when
    # the caller passes the ids in a different order
    assert encoder.encode_calls == [[1001, 1002, 1003, 3001]]


def test_subset_rejects_unknown_video(raw_dataset: Path, tmp_path: Path) -> None:
    """An unknown video id fails before any encoding work happens."""
    artifact = tmp_path / "subset-unknown"
    dataset = Dataset(raw_dataset)
    with pytest.raises(ValueError, match="not found in dataset"):
        build_embeddings(
            dataset, _RecordingSyntheticEncoder(_space()), artifact, "emb-subset-unknown",
            video_ids=["L99_V999"], image_loader=_image_loader,
        )


def test_embeddings_are_mmap_readable(raw_dataset: Path, tmp_path: Path) -> None:
    """The final matrix loads efficiently with mmap_mode='r'."""
    artifact = tmp_path / "mmap"
    _, _, _ = _build(raw_dataset, artifact)
    matrix = np.load(artifact / EMBEDDINGS_FILENAME, mmap_mode="r")
    assert matrix.shape == (6, 4)
    assert matrix.dtype == np.dtype("float32")
    assert matrix[0][0] == 1.0
    assert matrix[5][1] == 1.0


def test_batch_encoding_preserves_order(raw_dataset: Path, tmp_path: Path) -> None:
    """Batches are encoded with the encoder's batch method in canonical order."""
    artifact = tmp_path / "batch"
    dataset = Dataset(raw_dataset)
    encoder = _RecordingSyntheticEncoder(_space())
    build_embeddings(
        dataset, encoder, artifact, "emb-batch", batch_size=2, image_loader=_image_loader,
    )
    assert len(encoder.encode_calls) == 3
    assert encoder.encode_calls == [[1001, 1002], [1003, 2001], [2002, 3001]]


def test_encoder_without_batch_method_falls_back(raw_dataset: Path, tmp_path: Path) -> None:
    """An encoder exposing only single-image encode still works."""
    artifact = tmp_path / "fallback"
    dataset = Dataset(raw_dataset)

    class _SingleImageEncoder(_RecordingSyntheticEncoder):
        encode_image_batch = None  # type: ignore[assignment]

    encoder = _SingleImageEncoder(_space(), batch=False)
    build_embeddings(
        dataset, encoder, artifact, "emb-fallback", batch_size=2, image_loader=_image_loader,
    )
    assert len(encoder.encode_calls) == 6
    assert encoder.encode_calls == [[1001], [1002], [1003], [2001], [2002], [3001]]


def test_complete_artifact_is_returned_without_rebuild(raw_dataset: Path, tmp_path: Path) -> None:
    """Building over a valid complete artifact resumes idempotently."""
    artifact = tmp_path / "idempotent"
    _, encoder, manifest = _build(raw_dataset, artifact)
    calls_before = len(encoder.encode_calls)

    again = build_embeddings(
        Dataset(raw_dataset), encoder, artifact, "emb-test", image_loader=_image_loader,
    )
    assert again == manifest
    assert len(encoder.encode_calls) == calls_before

    with pytest.raises(ValueError, match="embedding space does not match"):
        build_embeddings(
            Dataset(raw_dataset),
            _RecordingSyntheticEncoder(_space(space_id="other")),
            artifact,
            "emb-test",
            image_loader=_image_loader,
            resume=True,
        )


def test_default_image_loader_loads_a_real_image(tmp_path: Path) -> None:
    """The default Pillow-based loader decodes a genuine image file."""
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(_tiny_png())
    image = default_image_loader()(image_path)
    assert image.mode == "RGB"


def _tiny_png() -> bytes:
    """Return the bytes of a 1x1 RGB PNG (hand-written, dependency-free)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"\x00" + b"\xff\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class _FakeTqdm:
    """Records construction and updates without rendering a real bar."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.updates = 0
        self.postfix: dict[str, object] | None = None
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.updates += int(n)

    def set_postfix(self, **kwargs: object) -> None:
        self.postfix = kwargs

    def close(self) -> None:
        self.closed = True


def test_progress_logging_fallback_reports_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The logging fallback reports totals and completed rows at batch ends."""
    progress_logger = logging.getLogger("aic2026.embedding_runner.test-progress")
    with caplog.at_level(logging.INFO, logger=progress_logger.name):
        reporter = embedding_runner._LoggingProgress(progress_logger, 6, 3, 0)
        reporter.on_build_start()
        reporter.on_rows_written(2, "L21_V001", 2, 2)
        reporter.on_rows_written(4, "L21_V002", 2, 2)
        reporter.complete()
    messages = [record.getMessage() for record in caplog.records]
    assert any("6 rows across 3 videos" in message for message in messages)
    assert any("2/6 rows" in message for message in messages)
    assert any("4/6 rows" in message for message in messages)
    assert any("complete" in message for message in messages)


def test_progress_tqdm_bar_starts_at_resume_row_and_tracks_batch() -> None:
    """The tqdm bar initializes from the resume row and reports batch progress."""
    fake = _FakeTqdm()
    created: dict[str, object] = {}

    def factory(**kwargs: object) -> _FakeTqdm:
        created.update(kwargs)
        return fake

    reporter = embedding_runner._TqdmProgress(
        factory, logging.getLogger("tqdm-test"), 6, 3, 2
    )
    assert created["total"] == 6
    assert created["initial"] == 2
    reporter.on_build_start()
    reporter.on_rows_written(3, "L21_V001", 1, 2)
    assert fake.updates == 1
    assert fake.postfix == {"video": "L21_V001", "batch": "1/2"}
    reporter.complete()
    assert fake.closed


def test_progress_create_falls_back_to_logging_without_tqdm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When tqdm cannot be imported, the runner falls back to logging."""
    monkeypatch.setitem(__import__("sys").modules, "tqdm", None)
    reporter = embedding_runner._create_progress(
        logging.getLogger("tqdm-fallback"), 6, 3, 0
    )
    assert isinstance(reporter, embedding_runner._LoggingProgress)


def test_progress_resume_initializes_from_checkpoint(
    raw_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed build starts the progress reporter at the checkpoint row."""
    artifact = tmp_path / "progress-resume"
    dataset = Dataset(raw_dataset)
    captured: list[tuple[int, int, int]] = []

    class _SpyReporter:
        def __init__(
            self, logger: object, total_rows: int, total_videos: int, start_row: int
        ) -> None:
            captured.append((total_rows, total_videos, start_row))

        def on_build_start(self) -> None:
            pass

        def on_rows_written(self, *args: object, **kwargs: object) -> None:
            pass

        def complete(self) -> None:
            pass

        def abort(self, completed: int) -> None:
            pass

    monkeypatch.setattr(embedding_runner, "_create_progress", _SpyReporter)

    with pytest.raises(EmbeddingBuildError):
        build_embeddings(
            dataset, _RecordingSyntheticEncoder(_space()), artifact, "emb-progress-resume",
            stop_after_rows=2, batch_size=1, image_loader=_image_loader,
        )
    assert captured[-1] == (6, 3, 0)

    build_embeddings(
        dataset, _RecordingSyntheticEncoder(_space()), artifact, "emb-progress-resume",
        batch_size=1, image_loader=_image_loader,
    )
    assert captured[-1] == (6, 3, 2)
