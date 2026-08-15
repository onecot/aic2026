"""Tests for runtime progress reporting in the embedding shard pipeline."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from aic2026.embeddings import EmbeddingSpace
from aic2026.embedding_shard import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    embed_shard,
)
from aic2026.progress import (
    PROGRESS_MODES,
    ProgressReporter,
    StageTimer,
    compute_speed,
    create_progress_reporter,
    estimate_eta,
    format_duration,
    format_eta,
    format_progress_line,
)


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


def _write_frames(path: Path, videos: list[tuple[str, list[int]]]) -> None:
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


def _make_keyframes(raw: Path, videos: list[tuple[str, list[int]]]) -> None:
    for vid, ordinals in videos:
        vid_dir = raw / "keyframes" / vid
        vid_dir.mkdir(parents=True)
        for ordinal in ordinals:
            (vid_dir / f"{ordinal:03d}.jpg").write_bytes(bytes([1, ordinal, 0]))


LAYOUT = [("L21_V001", [1, 2, 3]), ("L21_V002", [1, 2]), ("L21_V003", [1])]


class _StubImageEmbedder:
    def __init__(self, space: EmbeddingSpace, *, device: str = "cpu") -> None:
        self._space = space
        self._device = device

    @property
    def model_name(self) -> str:
        return "stub:model"

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
        out = np.zeros((len(batch_images), self._space.embedding_dim), dtype=self._space.output_dtype)
        for i, image in enumerate(batch_images):
            data = bytes(image)  # type: ignore[arg-type]
            out[i, 0] = float(data[0])
            out[i, 1] = float(data[1])
        return out


def _image_loader(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    _make_keyframes(raw, LAYOUT)
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, LAYOUT)
    return frames, raw


def _shard(env: tuple[Path, Path], out: Path, *, progress: object = "off") -> None:
    frames, raw = env
    embed_shard(
        frames,
        out,
        _StubImageEmbedder(_space()),
        keyframes_root=raw,
        image_loader=_image_loader,
        progress=progress,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# formatting / ETA units
# ---------------------------------------------------------------------------


def test_format_duration() -> None:
    assert format_duration(0.0) == "0.0s"
    assert format_duration(8.4) == "8.4s"
    assert format_duration(59.9) == "59.9s"
    assert format_duration(60) == "00:01:00"
    assert format_duration(95) == "00:01:35"
    assert format_duration(5720) == "01:35:20"
    assert format_duration(7960) == "02:12:40"
    assert format_duration(-5.0) == "0.0s"


def test_compute_speed() -> None:
    assert compute_speed(100, 50.0) == pytest.approx(2.0)
    assert compute_speed(100, 0.0) == 0.0
    assert compute_speed(0, 50.0) == 0.0


def test_estimate_eta() -> None:
    # 75200 done of 177321 in 5720s -> remaining ~= (177321-75200)*5720/75200
    eta = estimate_eta(75200, 177321, 5720.0)
    expected = (177321 - 75200) * 5720.0 / 75200.0
    assert eta == pytest.approx(expected)
    assert eta > 0

    assert estimate_eta(0, 177321, 5720.0) == 0.0
    assert estimate_eta(75200, 75200, 5720.0) == 0.0
    assert estimate_eta(100, 200, 0.0) == 0.0
    assert estimate_eta(200, 100, 50.0) == 0.0  # clamp negative


def test_format_progress_line() -> None:
    line = format_progress_line(100, 873, 300.0, unit="videos")
    assert "42.4%" not in line
    assert "11.5%" in line
    assert "100/873 videos" in line
    assert "20.0 videos/min" in line
    assert "ETA " in line
    assert "[" in line and "]" in line  # bar present


def test_format_progress_line_without_bar_and_note() -> None:
    line = format_progress_line(3, 6, 1.0, bar=False, note="video=L21_V001")
    assert "video=L21_V001" in line
    assert "[" not in line


def test_format_eta() -> None:
    assert format_eta(0) == "0s"
    assert format_eta(42) == "42s"
    assert format_eta(720) == "12m"
    assert format_eta(9660) == "2h41m"


def test_resumed_eta_uses_only_current_run_pace() -> None:
    # Fresh run: 100 videos done in 300s of a 873-video job.
    fresh = estimate_eta(100, 873, 300.0)
    # Resumed run: 100 videos were already done before this run began, and this
    # run has done 2 more in 300s. The ETA must be based on this run's pace
    # (2 videos / 300s), not on the 100 from the previous session.
    resumed = estimate_eta(102, 873, 300.0, initial_completed=100)
    assert fresh == pytest.approx((873 - 100) * 300.0 / 100.0)
    assert resumed == pytest.approx((873 - 102) * 300.0 / 2.0)
    assert resumed > fresh

    # No progress in this run yet -> no measurable ETA.
    assert estimate_eta(100, 873, 300.0, initial_completed=100) == 0.0


def test_resumed_format_progress_line_speed_and_eta() -> None:
    line = format_progress_line(
        102,
        873,
        300.0,
        bar=False,
        unit="videos",
        initial_completed=100,
        keyframes=500,
    )
    # 2 videos in 300s = 0.4 videos/min; 500 keyframes in 300s = 100 keyframes/min.
    assert "102/873 videos" in line
    assert "0.4 videos/min" in line
    assert "100.0 keyframes/min" in line
    assert "ETA " in line
    assert "32h7m" in line


def test_reporter_resume_offset_uses_initial_completed() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        "interactive", interval_seconds=0.0, stream=stream, unit="videos"
    )
    reporter.begin()
    reporter.set_total(873, completed=100)
    reporter.update(102, note="video=L21_V003")
    reporter.finish()

    text = stream.getvalue()
    assert "102/873 videos" in text
    # The reporter was set with a resume offset of 100, so speed reflects only
    # the work done after begin(): 2 videos over this run's elapsed time.
    assert "videos/min" in text
    assert "ETA " in text


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------


def test_stage_timer_accumulates() -> None:
    timer = StageTimer()
    with timer.time("encode"):
        time.sleep(0.01)
    with timer.time("encode"):
        time.sleep(0.01)
    assert timer.elapsed()["encode"] >= 0.02
    assert "load_frames" not in timer.elapsed()

    report = timer.format()
    assert "Stage timing:" in report
    assert "encode:" in report


def test_stage_timer_nested_is_idempotent() -> None:
    timer = StageTimer()
    timer.start("a")
    timer.start("a")
    timer.stop("a")
    assert "a" in timer.elapsed()


# ---------------------------------------------------------------------------
# reporter behavior
# ---------------------------------------------------------------------------


def test_reporter_begin_is_idempotent() -> None:
    reporter = ProgressReporter("periodic", interval_seconds=0.0, logger_name="aic2026.progress.test")
    reporter.begin()
    reporter.begin()
    assert reporter.enabled


def test_reporter_off_is_noop() -> None:
    reporter = ProgressReporter("off")
    reporter.begin()
    reporter.set_total(10)
    reporter.update(5)
    reporter.finish()
    assert not reporter.enabled


def test_reporter_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        ProgressReporter("bogus")
    with pytest.raises(ValueError, match="mode"):
        create_progress_reporter("bogus")


def test_progress_modes_constants() -> None:
    assert PROGRESS_MODES == ("auto", "on", "off")


def test_create_progress_reporter_on_off() -> None:
    assert create_progress_reporter("off").mode == "off"
    assert create_progress_reporter("on").mode == "interactive"


# ---------------------------------------------------------------------------
# integration with embed_shard
# ---------------------------------------------------------------------------


def test_progress_disabled_does_not_affect_output(env: tuple[Path, Path], tmp_path: Path) -> None:
    out_off = tmp_path / "off"
    out_on = tmp_path / "on"
    _shard(env, out_off, progress="off")
    _shard(env, out_on, progress="on")

    np.testing.assert_array_equal(
        np.load(out_off / EMBEDDINGS_FILENAME), np.load(out_on / EMBEDDINGS_FILENAME)
    )
    assert (out_off / FRAMES_FILENAME).read_bytes() == (out_on / FRAMES_FILENAME).read_bytes()
    off_manifest = json.loads((out_off / MANIFEST_FILENAME).read_text())
    on_manifest = json.loads((out_on / MANIFEST_FILENAME).read_text())
    for key in ("model", "embedding_dim", "num_embeddings", "num_videos", "source_frames"):
        assert off_manifest[key] == on_manifest[key]


def test_progress_disabled_matches_default(env: tuple[Path, Path], tmp_path: Path) -> None:
    out_default = tmp_path / "default"
    out_off = tmp_path / "off"
    _shard(env, out_default, progress="auto")
    _shard(env, out_off, progress="off")
    np.testing.assert_array_equal(
        np.load(out_default / EMBEDDINGS_FILENAME), np.load(out_off / EMBEDDINGS_FILENAME)
    )


def test_interactive_reporter_writes_to_stream(env: tuple[Path, Path], tmp_path: Path) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter("interactive", interval_seconds=0.0, stream=stream)
    out = tmp_path / "shard"
    _shard(env, out, progress=reporter)

    text = stream.getvalue()
    assert "% |" in text
    assert "frames" in text
    assert "ETA " in text
    assert "Stage timing:" in text
    assert "load_frames:" in text
    assert "encode:" in text
    assert "write_outputs:" in text


def test_periodic_reporter_logs(caplog: pytest.LogCaptureFixture, env: tuple[Path, Path], tmp_path: Path) -> None:
    reporter = ProgressReporter(
        "periodic", interval_seconds=0.0, logger_name="aic2026.progress.test"
    )
    out = tmp_path / "shard"
    with caplog.at_level(logging.INFO, logger="aic2026.progress.test"):
        _shard(env, out, progress=reporter)

    messages = [r.getMessage() for r in caplog.records]
    assert any("embedding progress" in m and "% |" in m and "frames" in m for m in messages)
    assert any("Stage timing:" in m for m in messages)
    assert any("load_frames:" in m for m in messages)


def test_periodic_reporter_throttles(
    caplog: pytest.LogCaptureFixture, env: tuple[Path, Path], tmp_path: Path
) -> None:
    # A huge interval means intermediate updates are suppressed; with batch_size=1
    # six batches run, but only the final 100% line and stage timing are emitted.
    reporter = ProgressReporter(
        "periodic", interval_seconds=1e9, logger_name="aic2026.progress.test"
    )
    out = tmp_path / "shard"
    with caplog.at_level(logging.INFO, logger="aic2026.progress.test"):
        embed_shard(
            env[0],
            out,
            _StubImageEmbedder(_space()),
            keyframes_root=env[1],
            image_loader=_image_loader,
            batch_size=1,
            progress=reporter,
        )

    progress_lines = [r.getMessage() for r in caplog.records if "embedding progress" in r.getMessage()]
    assert len(progress_lines) == 1
    assert "100.0%" in progress_lines[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "embed_shard.py"


def test_cli_progress_choices_listed() -> None:
    result = subprocess.run(
        [sys.executable, str(_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--progress" in result.stdout
    assert "auto" in result.stdout and "on" in result.stdout and "off" in result.stdout


def test_cli_invalid_progress_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_script()), "--progress", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--progress" in result.stderr