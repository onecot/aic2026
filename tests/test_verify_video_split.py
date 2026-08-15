"""Tests for split-directory verification and its CLI.

Uses synthetic video entries only (the verifier reads JSON manifests, so no
real media files are needed). Tests are fast, deterministic, and cross-platform.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.keyframe_extraction import (
    VIDEOS_MANIFEST_FILENAME,
    VideoEntry,
    VideoManifest,
    save_video_manifest,
)
from aic2026.keyframe_extraction_split import (
    SPLIT_MANIFEST_FILENAME,
    SplitError,
    SplitVerificationReport,
    split_video_manifest,
    verify_video_split,
    worker_dir_name,
)


def _entries(count: int, *, frames: int = 200, start: int = 1) -> list[VideoEntry]:
    return [
        VideoEntry(video_id=f"L21_V{i:03d}", path=f"/tmp/{i:03d}.mp4", num_frames=frames)
        for i in range(start, start + count)
    ]


def _manifest(videos: list[VideoEntry]) -> VideoManifest:
    return VideoManifest(schema_version="1.0", videos=tuple(videos))


def _write_worker_manifest(
    split_dir: Path, worker_id: int, videos: list[VideoEntry]
) -> Path:
    path = split_dir / worker_dir_name(worker_id) / VIDEOS_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video_manifest(path, _manifest(videos))
    return path


def _split(tmp_path: Path, *, num_workers: int = 2, videos: int = 4) -> Path:
    source = tmp_path / "source.json"
    save_video_manifest(source, _manifest(_entries(videos)))
    split_dir = tmp_path / "splits"
    split_video_manifest(source, num_workers, split_dir)
    return split_dir


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify_video_split.py"


def _run_cli(split_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script()), "--split-dir", str(split_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestVerifyVideoSplit:
    def test_valid_split(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        report = verify_video_split(split_dir)

        assert isinstance(report, SplitVerificationReport)
        assert report.valid is True
        assert report.issues == ()
        assert report.total_videos == 4
        assert report.unique_videos == 4
        assert report.duplicates == 0
        assert report.total_frames == 800
        assert len(report.workers) == 2
        assert all(w.manifest_found for w in report.workers)
        assert sum(w.num_videos for w in report.workers) == 4

    def test_missing_split_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="split manifest not found"):
            verify_video_split(tmp_path)

    def test_duplicate_video_across_workers(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=2)
        # Worker 1 keeps its own V002 and gains a copy of V001 (worker 0's).
        _write_worker_manifest(split_dir, 1, _entries(1, start=2) + _entries(1, start=1))

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert report.duplicates == 1
        assert any("appears in" in i.message for i in report.issues)
        assert report.unique_videos == 2

    def test_missing_worker_file(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        (split_dir / worker_dir_name(1) / VIDEOS_MANIFEST_FILENAME).unlink()

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert any("worker manifest missing" in i.message for i in report.issues)
        assert report.total_videos == 2  # only worker 0 counted

    def test_count_mismatch(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        # Worker 0 records 2 videos but only gets 1.
        _write_worker_manifest(split_dir, 0, _entries(1, start=1))

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert any("videos do not match split_manifest.json" in i.message for i in report.issues)
        assert any("has 1 videos; split_manifest.json records 2" in i.message for i in report.issues)

    def test_frame_count_mismatch(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        # Worker 1 keeps both video ids but one frame count is wrong.
        wrong = [_entries(1, start=3, frames=200)[0]]
        wrong.append(VideoEntry(video_id="L21_V004", path="/tmp/004.mp4", num_frames=999))
        _write_worker_manifest(split_dir, 1, wrong)

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert any("has 1199 frames; split_manifest.json records" in i.message for i in report.issues)

    def test_malformed_worker_json(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        path = split_dir / worker_dir_name(1) / VIDEOS_MANIFEST_FILENAME
        path.write_text("{not valid json", encoding="utf-8")

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert any("invalid worker manifest" in i.message for i in report.issues)
        assert any("manifest_found" in w.to_dict() for w in report.workers)

    def test_malformed_split_manifest(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        (split_dir / SPLIT_MANIFEST_FILENAME).write_text("{{{", encoding="utf-8")

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert report.total_videos == 0
        assert any("invalid split_manifest.json" in i.message for i in report.issues)

    def test_missing_video_recorded_in_manifest(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=1, videos=3)
        # Drop one video entirely; manifest still records it.
        kept = _entries(2, start=1)
        _write_worker_manifest(split_dir, 0, kept)

        report = verify_video_split(split_dir)

        assert report.valid is False
        assert any("missing from manifest" in i.message for i in report.issues)

    def test_report_round_trip(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        payload = verify_video_split(split_dir).to_dict()
        assert payload["valid"] is True
        assert payload["total_videos"] == 4
        assert payload["workers"][0]["num_frames"] == 400

    def test_absolute_paths_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=2)
        with caplog.at_level(logging.WARNING, logger="aic2026.keyframe_extraction_split"):
            verify_video_split(split_dir)
        messages = [r.getMessage() for r in caplog.records]
        assert any("video manifest contains absolute paths" in m for m in messages)

    def test_relative_paths_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A worker manifest with genuinely relative paths must not warn.
        source = tmp_path / "source.json"
        entries = [
            VideoEntry(video_id=f"L21_V{i:03d}", path=f"videos/{i:03d}.mp4", num_frames=200)
            for i in range(1, 3)
        ]
        save_video_manifest(source, _manifest(entries))
        split_dir = tmp_path / "splits"
        split_manifest = split_video_manifest(source, 2, split_dir)
        # Rewrite each worker manifest with its assigned videos (relative paths).
        by_id = {entry.video_id: entry for entry in entries}
        for worker in split_manifest.workers:
            _write_worker_manifest(
                split_dir,
                worker.worker_id,
                [by_id[vid] for vid in worker.video_ids],
            )
        with caplog.at_level(logging.WARNING, logger="aic2026.keyframe_extraction_split"):
            verify_video_split(split_dir)
        messages = [r.getMessage() for r in caplog.records]
        assert not any("video manifest contains absolute paths" in m for m in messages)


class TestVerifyVideoSplitCli:
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_script()), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--split-dir" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_script())],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2

    def test_valid_split_exits_0(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        result = _run_cli(split_dir)

        assert result.returncode == 0, result.stderr
        assert "worker_000:" in result.stdout
        assert "videos: 2" in result.stdout
        assert "Total videos: 4" in result.stdout
        assert "Unique videos: 4" in result.stdout
        assert "Duplicates: 0" in result.stdout
        assert "OK: split is valid" in result.stdout

    def test_invalid_split_exits_1(self, tmp_path: Path) -> None:
        split_dir = _split(tmp_path, num_workers=2, videos=4)
        (split_dir / worker_dir_name(1) / VIDEOS_MANIFEST_FILENAME).unlink()
        result = _run_cli(split_dir)

        assert result.returncode == 1
        assert "worker_001: worker manifest missing" in result.stdout
        assert "Validation failed:" in result.stdout

    def test_missing_split_dir_exits_1(self, tmp_path: Path) -> None:
        result = _run_cli(tmp_path / "does-not-exist")

        assert result.returncode == 1
        assert "error:" in result.stderr


def test_split_manifest_missing_worker_dir_named(tmp_path: Path) -> None:
    """Sanity check: worker dir names used by the verifier match the splitter."""
    split_dir = _split(tmp_path, num_workers=3, videos=3)
    report = verify_video_split(split_dir)
    assert report.valid is True
    assert [w.worker_id for w in report.workers] == [0, 1, 2]
