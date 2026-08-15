"""Tests for the canonical video inventory manifest generator.

Uses small synthetic ``.mp4`` files (dummy bytes) only; metadata probing via
``ffprobe`` is mocked. Never scans a real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.video_manifest import (
    VIDEO_MANIFEST_SCHEMA_VERSION,
    VideoEntry,
    VideoManifest,
    VideoManifestError,
    build_video_manifest,
    load_video_manifest,
    save_video_manifest,
)


def _make_video(dir_path: Path, video_id: str, *, size: int = 1024) -> Path:
    path = dir_path / f"{video_id}.mp4"
    path.write_bytes(b"\x00" * size)
    return path


def _probe_result(*, fps: float = 25.0, duration: float = 8.0, nb_frames: str = "200") -> str:
    streams = [{"avg_frame_rate": f"{int(fps)}/1", "duration": str(duration)}]
    if nb_frames is not None:
        streams[0]["nb_frames"] = nb_frames
    payload = {"streams": streams, "format": {"duration": str(duration)}}
    return json.dumps(payload)


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    """A directory with a handful of synthetic .mp4 files."""
    d = tmp_path / "videos"
    d.mkdir()
    for vid in ("L21_V003", "L21_V001", "L21_V002"):
        _make_video(d, vid)
    return d


@pytest.fixture
def mock_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock subprocess.run so ffprobe returns deterministic metadata."""
    import subprocess

    def fake_run(command, **kwargs):
        path = command[-1]
        if path.endswith("L21_ERROR.mp4"):
            import subprocess as sp

            return sp.CompletedProcess(command, 1, stdout="", stderr="boom")
        return __import__("subprocess").CompletedProcess(
            command, 0, stdout=_probe_result(), stderr=""
        )

    monkeypatch.setattr("aic2026.video_manifest.subprocess.run", fake_run)


# ---------------------------------------------------------------- VideoEntry


class TestVideoEntry:
    def test_valid(self) -> None:
        entry = VideoEntry(
            video_id="L21_V001",
            path="/v/L21_V001.mp4",
            file_size_bytes=2048,
            duration_seconds=10.0,
            fps=25.0,
            num_frames=250,
        )
        assert entry.video_id == "L21_V001"

    def test_to_dict_round_trip(self) -> None:
        entry = VideoEntry(video_id="L21_V001", path="/v/a.mp4", file_size_bytes=100)
        assert VideoEntry.from_dict(entry.to_dict()) == entry

    def test_legacy_entry_loads_with_defaults(self) -> None:
        entry = VideoEntry.from_dict({"video_id": "L21_V001", "path": "/v/a.mp4"})
        assert entry.file_size_bytes == 0
        assert entry.duration_seconds is None
        assert entry.fps is None
        assert entry.num_frames is None

    def test_invalid_video_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoEntry(video_id="nope", path="/v/a.mp4", file_size_bytes=1)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoEntry(video_id="L21_V001", path="/v/a.mp4", file_size_bytes=-1)


# ---------------------------------------------------------------- VideoManifest


class TestVideoManifest:
    def test_duplicate_ids_rejected(self) -> None:
        a = VideoEntry(video_id="L21_V001", path="/v/a.mp4", file_size_bytes=1)
        b = VideoEntry(video_id="L21_V001", path="/v/b.mp4", file_size_bytes=1)
        with pytest.raises(ValueError):
            VideoManifest(
                schema_version=VIDEO_MANIFEST_SCHEMA_VERSION,
                source_dir="/v",
                videos=(a, b),
                created_at="2026-01-01T00:00:00+00:00",
            )

    def test_to_dict_round_trip(self) -> None:
        entry = VideoEntry(
            video_id="L21_V001", path="/v/a.mp4", file_size_bytes=10, num_frames=100
        )
        manifest = VideoManifest(
            schema_version=VIDEO_MANIFEST_SCHEMA_VERSION,
            source_dir="/v",
            videos=(entry,),
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert VideoManifest.from_dict(manifest.to_dict()) == manifest

    def test_legacy_manifest_loads(self) -> None:
        data = {
            "schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
            "videos": [{"video_id": "L21_V001", "path": "/v/a.mp4"}],
        }
        manifest = VideoManifest.from_dict(data)
        assert manifest.source_dir == ""
        assert manifest.created_at == ""

    def test_unsupported_schema_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoManifest.from_dict(
                {
                    "schema_version": "9.9",
                    "source_dir": "/v",
                    "created_at": "x",
                    "videos": [],
                }
            )

    def test_by_id(self) -> None:
        a = VideoEntry(video_id="L21_V001", path="/v/a.mp4", file_size_bytes=1)
        b = VideoEntry(video_id="L21_V002", path="/v/b.mp4", file_size_bytes=1)
        manifest = VideoManifest(
            schema_version=VIDEO_MANIFEST_SCHEMA_VERSION,
            source_dir="/v",
            videos=(a, b),
            created_at="x",
        )
        assert set(manifest.by_id()) == {"L21_V001", "L21_V002"}


# ---------------------------------------------------------------- build_video_manifest


class TestBuildVideoManifest:
    def test_video_discovery_and_ordering(self, video_dir: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        assert [e.video_id for e in manifest.videos] == [
            "L21_V001",
            "L21_V002",
            "L21_V003",
        ]

    def test_invalid_extension_ignored(self, video_dir: Path) -> None:
        (video_dir / "L21_V004.txt").write_text("not a video")
        (video_dir / "notes.json").write_text("{}")
        manifest = build_video_manifest(video_dir, include_metadata=False)
        assert [e.video_id for e in manifest.videos] == [
            "L21_V001",
            "L21_V002",
            "L21_V003",
        ]

    def test_file_sizes_recorded(self, video_dir: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        assert {e.file_size_bytes for e in manifest.videos} == {1024}

    def test_paths_relative_to_source_dir(self, video_dir: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        for entry in manifest.videos:
            assert not Path(entry.path).is_absolute()
            assert Path(manifest.source_dir, entry.path).resolve() == video_dir.joinpath(entry.path).resolve()
            assert entry.path == f"{entry.video_id}.mp4"

    def test_relative_paths_resolve_against_source_dir(self, video_dir: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        for entry in manifest.videos:
            resolved = Path(manifest.source_dir, entry.path).resolve()
            assert resolved.is_file()
            assert resolved == (video_dir / f"{entry.video_id}.mp4").resolve()

    def test_source_dir_and_created_at(self, video_dir: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        assert Path(manifest.source_dir).resolve() == video_dir.resolve()
        assert manifest.created_at != ""

    def test_empty_file_rejected(self, video_dir: Path) -> None:
        _make_video(video_dir, "L21_V004", size=0)
        with pytest.raises(VideoManifestError, match="empty"):
            build_video_manifest(video_dir, include_metadata=False)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_video_manifest(tmp_path / "nope", include_metadata=False)

    def test_non_directory_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "file"
        f.write_text("x")
        with pytest.raises(ValueError):
            build_video_manifest(f, include_metadata=False)

    def test_no_videos_yields_empty_manifest(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        manifest = build_video_manifest(d, include_metadata=False)
        assert manifest.videos == ()

    def test_metadata_attached(self, video_dir: Path, mock_ffprobe: None) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=True)
        for entry in manifest.videos:
            assert entry.fps == 25.0
            assert entry.duration_seconds == 8.0
            assert entry.num_frames == 200

    def test_metadata_failure_raises(
        self, video_dir: Path, mock_ffprobe: None
    ) -> None:
        _make_video(video_dir, "L21_ERROR")
        with pytest.raises(VideoManifestError, match="ffprobe failed"):
            build_video_manifest(video_dir, include_metadata=True)

    def test_ffprobe_missing_raises(
        self, video_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def missing(command, **kwargs):
            raise FileNotFoundError("ffprobe")

        monkeypatch.setattr("aic2026.video_manifest.subprocess.run", missing)
        with pytest.raises(VideoManifestError, match="not found"):
            build_video_manifest(video_dir, include_metadata=True)

    def test_no_video_stream_raises(
        self, video_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def no_stream(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout='{"streams": []}', stderr="")

        monkeypatch.setattr("aic2026.video_manifest.subprocess.run", no_stream)
        with pytest.raises(VideoManifestError, match="no video stream"):
            build_video_manifest(video_dir, include_metadata=True)

    def test_bad_frame_rate_raises(
        self, video_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def bad_rate(command, **kwargs):
            stdout = json.dumps(
                {"streams": [{"avg_frame_rate": "0/0", "duration": "8.0"}]}
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        monkeypatch.setattr("aic2026.video_manifest.subprocess.run", bad_rate)
        with pytest.raises(VideoManifestError, match="frame rate"):
            build_video_manifest(video_dir, include_metadata=True)

    def test_num_frames_estimated_when_missing(
        self, video_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def no_frames(command, **kwargs):
            stdout = json.dumps(
                {
                    "streams": [{"avg_frame_rate": "25/1", "duration": "8.0"}],
                    "format": {"duration": "8.0"},
                }
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        monkeypatch.setattr("aic2026.video_manifest.subprocess.run", no_frames)
        manifest = build_video_manifest(video_dir, include_metadata=True)
        assert all(e.num_frames == 200 for e in manifest.videos)


# ---------------------------------------------------------------- save / load


class TestSaveLoad:
    def test_save_load_round_trip(self, video_dir: Path, tmp_path: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        out = tmp_path / "videos.json"
        save_video_manifest(out, manifest)
        assert load_video_manifest(out) == manifest

    def test_output_is_valid_json(self, video_dir: Path, tmp_path: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        out = tmp_path / "videos.json"
        save_video_manifest(out, manifest)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == VIDEO_MANIFEST_SCHEMA_VERSION
        assert [v["video_id"] for v in data["videos"]] == [
            "L21_V001",
            "L21_V002",
            "L21_V003",
        ]

    def test_no_temp_left_behind(self, video_dir: Path, tmp_path: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        out = tmp_path / "videos.json"
        save_video_manifest(out, manifest)
        assert not list(tmp_path.glob("*.tmp"))

    def test_atomic_write_failure_raises(self, video_dir: Path, tmp_path: Path) -> None:
        manifest = build_video_manifest(video_dir, include_metadata=False)
        # Make the parent directory read-only so the temp file cannot be created.
        parent = tmp_path / "ro"
        parent.mkdir()
        out = parent / "videos.json"
        parent.chmod(0o444)
        try:
            with pytest.raises(OSError):
                save_video_manifest(out, manifest)
        finally:
            parent.chmod(0o755)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_video_manifest(tmp_path / "missing.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "videos.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_video_manifest(p)

    def test_legacy_manifest_still_loads(self, tmp_path: Path) -> None:
        p = tmp_path / "videos.json"
        p.write_text(
            json.dumps(
                {
                    "schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
                    "videos": [{"video_id": "L21_V001", "path": "/v/a.mp4"}],
                }
            ),
            encoding="utf-8",
        )
        manifest = load_video_manifest(p)
        assert manifest.videos[0].video_id == "L21_V001"


# ---------------------------------------------------------------- compatibility


class TestDownstreamCompatibility:
    def test_consumed_by_keyframe_extraction_loader(
        self, video_dir: Path, tmp_path: Path, mock_ffprobe: None
    ) -> None:
        from aic2026.keyframe_extraction import load_video_manifest as legacy_load

        manifest = build_video_manifest(video_dir, include_metadata=True)
        out = tmp_path / "videos.json"
        save_video_manifest(out, manifest)

        legacy = legacy_load(out)
        assert [v.video_id for v in legacy.videos] == [
            "L21_V001",
            "L21_V002",
            "L21_V003",
        ]
        assert all(v.num_frames == 200 for v in legacy.videos)

    def test_split_accepts_generated_manifest(
        self, video_dir: Path, tmp_path: Path, mock_ffprobe: None
    ) -> None:
        from aic2026.keyframe_extraction_split import split_video_manifest

        manifest = build_video_manifest(video_dir, include_metadata=True)
        out = tmp_path / "videos.json"
        save_video_manifest(out, manifest)

        split_dir = tmp_path / "splits"
        split_manifest = split_video_manifest(out, 2, split_dir)
        assert split_manifest.num_workers == 2
        assert sum(w.num_videos for w in split_manifest.workers) == 3