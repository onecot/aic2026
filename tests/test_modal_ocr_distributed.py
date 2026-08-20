"""Tests for the distributed Modal OCR runner wiring.

The ``modal_ocr_distributed`` module shards the canonical keyframe collection
into deterministic row ranges and runs one independent, resumable OCR artifact
per shard on a Modal GPU worker. These tests cover the pure, importable parts
(shard bounds, filtered frames writing, keyframes-dir symlink wiring, heartbeat
formatting) and do not require a Modal account or network access. The whole
module is skipped when the optional ``modal`` package is unavailable.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")

import modal_ocr_distributed as mod  # noqa: E402

FRAME_RECORD = {
    "keyframe": {"video_id": {"value": "L21_V001"}, "keyframe_ordinal": 1},
    "pts_time": 0.0,
    "fps": 25.0,
    "frame_index": 0,
}


def _write_canonical(path: Path, n: int) -> None:
    """Write an n-row canonical frames file with distinct row content."""
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            record = {
                **FRAME_RECORD,
                "frame_index": i * 10,
                "keyframe": {"video_id": {"value": f"L{i:02d}_V001"}, "keyframe_ordinal": 1},
            }
            handle.write(json.dumps(record) + "\n")


def test_shard_row_bounds_balanced_and_total():
    num_shards = 8
    bounds = [mod.shard_row_bounds(148322, num_shards, i) for i in range(num_shards)]
    assert bounds[0] == (0, 18541)
    starts = [lo for lo, _ in bounds]
    assert starts == sorted(starts)
    assert bounds[0][0] == 0 and bounds[-1][1] == 148322
    sizes = [hi - lo for lo, hi in bounds]
    assert max(sizes) - min(sizes) < num_shards  # deterministic, near-balanced
    assert sum(sizes) == 148322


def test_shard_row_bounds_deterministic_and_disjoint():
    bounds = [mod.shard_row_bounds(100, 7, i) for i in range(7)]
    prev_end = 0
    for lo, hi in bounds:
        assert lo == prev_end
        prev_end = hi
    assert prev_end == 100
    assert mod.shard_row_bounds(100, 7, 0) == mod.shard_row_bounds(100, 7, 0)


def test_shard_row_bounds_validates_inputs():
    with pytest.raises(ValueError):
        mod.shard_row_bounds(10, 0, 0)
    with pytest.raises(ValueError):
        mod.shard_row_bounds(10, 4, 4)
    with pytest.raises(ValueError):
        mod.shard_row_bounds(-1, 4, 0)
    with pytest.raises(ValueError):
        mod.shard_row_bounds(10, 4, -1)


def test_write_shard_frames_writes_exact_subset(tmp_path: Path):
    canonical = tmp_path / "frames.ndjson"
    _write_canonical(canonical, 10)
    target = tmp_path / "shard_frames.ndjson"
    count = mod.write_shard_frames(canonical, target, 2, 6)
    assert count == 4
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["frame_index"] == 20
    assert json.loads(lines[-1])["frame_index"] == 50


def test_write_shard_frames_clamps_end_to_file_length(tmp_path: Path):
    canonical = tmp_path / "frames.ndjson"
    _write_canonical(canonical, 10)
    target = tmp_path / "shard_frames.ndjson"
    count = mod.write_shard_frames(canonical, target, 8, 100)
    assert count == 2


def test_write_shard_frames_limit_caps_rows(tmp_path: Path):
    canonical = tmp_path / "frames.ndjson"
    _write_canonical(canonical, 10)
    target = tmp_path / "shard_frames.ndjson"
    count = mod.write_shard_frames(canonical, target, 0, 10, limit=3)
    assert count == 3
    assert target.read_text(encoding="utf-8").strip().splitlines()[0]


def test_write_shard_frames_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        mod.write_shard_frames(tmp_path / "missing.ndjson", tmp_path / "out.ndjson", 0, 1)


def test_build_shard_keyframes_dir_symlinks(tmp_path: Path):
    shared_images = tmp_path / "shared_images"
    (shared_images / "L00_V001").mkdir(parents=True)
    (shared_images / "L00_V001" / "000001.jpg").write_bytes(b"\xff\xd8\xff")
    shard_frames = tmp_path / "shard_frames.ndjson"
    shard_frames.write_text(json.dumps(FRAME_RECORD) + "\n", encoding="utf-8")

    view = tmp_path / "view"
    mod.build_shard_keyframes_dir(view, shard_frames, shared_images)

    assert view.is_dir()
    assert view.is_symlink() is False
    frames_link = view / "frames.ndjson"
    images_link = view / "images"
    assert frames_link.is_symlink()
    assert images_link.is_symlink()
    assert frames_link.resolve() == shard_frames.resolve()
    assert images_link.resolve() == shared_images.resolve()
    assert (images_link / "L00_V001" / "000001.jpg").is_file()


def test_format_heartbeat_line_contains_required_fields():
    line = mod.format_heartbeat_line("shard_000", 100, 200, 18540, 123.4, "L4-15GB")
    assert line.startswith("[shard_000]")
    assert "processed=100/current=200/total=18540" in line
    assert "frames/min" in line
    assert "elapsed=" in line
    assert "ETA" in line
    assert "gpu=L4-15GB" in line


def test_format_heartbeat_line_computes_rate_and_eta():
    line = mod.format_heartbeat_line("shard_000", 600, 600, 1200, 60.0, "L4-15GB")
    assert "600.0 frames/min" in line
    assert "ETA 1m" in line or "ETA 60s" in line


def test_format_heartbeat_line_validates_inputs():
    with pytest.raises(ValueError):
        mod.format_heartbeat_line("", 0, 0, 1, 0.0, "L4")
    with pytest.raises(ValueError):
        mod.format_heartbeat_line("shard_000", -1, 0, 1, 0.0, "L4")
    with pytest.raises(ValueError):
        mod.format_heartbeat_line("shard_000", 0, 5, 1, 0.0, "L4")
    with pytest.raises(ValueError):
        mod.format_heartbeat_line("shard_000", 0, 0, 1, -1.0, "L4")


def test_heartbeat_reporter_prints_flush_on_each_update():
    stream = io.StringIO()
    reporter = mod.ShardHeartbeatReporter("shard_000", "L4-15GB", stream=stream)
    reporter.begin()
    reporter.set_total(18540, completed=0)
    reporter.update(100)
    reporter.update(200)
    reporter.finish()
    lines = stream.getvalue().splitlines()
    assert len(lines) >= 3
    assert lines[0].startswith("[shard_000]")
    assert "processed=100/current=100/total=18540" in lines[0]
    assert "processed=200/current=200/total=18540" in lines[1]


def test_heartbeat_reporter_includes_resume_offset():
    stream = io.StringIO()
    reporter = mod.ShardHeartbeatReporter("shard_003", "A10G-24GB", stream=stream)
    reporter.begin()
    reporter.set_total(18540, completed=5000)
    reporter.update(5025)
    reporter.finish()
    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("[shard_003]")
    assert "processed=25/current=5025/total=18540" in lines[0]