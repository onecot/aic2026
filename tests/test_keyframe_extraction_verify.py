"""Tests for keyframe extraction verification and its CLI.

Builds a real extraction artifact with the synthetic detector + frame reader
(no ffmpeg, no model), then verifies it and mutates it to exercise failure
modes. Tests are fast, deterministic, and cross-platform.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.keyframe_extraction import (
    ExtractionConfig,
    VideoEntry,
    VideoManifest,
    extract_keyframes,
    save_video_manifest,
)
from aic2026.keyframe_extraction_merge import merge_shards
from aic2026.keyframe_extractor import write_synthetic_video
from aic2026.keyframe_extraction_verify import (
    KeyframeExtractionVerificationReport,
    verify_keyframe_extraction,
)


def _adaptive_config(**overrides: object) -> ExtractionConfig:
    kwargs: dict[str, object] = {
        "extractor": "omnishotcut_adaptive",
        "frame_reader": "synthetic",
        "detector": "synthetic",
        "sampler": "adaptive_v1",
    }
    kwargs.update(overrides)
    return ExtractionConfig(**kwargs)


def _write_video(tmp_path: Path, name: str, num_frames: int) -> Path:
    values = [float((i // 40) * 100.0) for i in range(num_frames)]
    return write_synthetic_video(
        tmp_path / f"{name}.json",
        values,
        shots=[(0, num_frames - 1)],
    )


def _make_artifact(tmp_path: Path, *, videos: list[str], output_name: str = "out") -> tuple[Path, Path]:
    """Build a complete extraction artifact and the assigned videos.json."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    entries: list[VideoEntry] = []
    for vid in videos:
        path = _write_video(video_dir, vid, num_frames=500)
        entries.append(VideoEntry(video_id=vid, path=str(path), num_frames=500))
    videos_path = tmp_path / "videos.json"
    save_video_manifest(videos_path, VideoManifest(schema_version="1.0", videos=tuple(entries)))
    output_dir = tmp_path / output_name
    extract_keyframes(videos_path, output_dir, _adaptive_config())
    return output_dir, videos_path


def _make_merged_artifact(tmp_path: Path, *, shards: list[list[str]]) -> Path:
    """Build a complete merged artifact from per-shard video lists."""
    shard_dirs: list[Path] = []
    for idx, videos in enumerate(shards):
        shard_dir, _ = _make_artifact(tmp_path, videos=videos, output_name=f"shard_{idx}")
        shard_dirs.append(shard_dir)
    output_dir = tmp_path / "merged"
    merge_shards(shard_dirs, output_dir)
    return output_dir


def _records(output_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (output_dir / "frames.ndjson").open(encoding="utf-8")]


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify_keyframe_extraction.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script()), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestVerifyKeyframeExtraction:
    def test_valid_artifact(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert isinstance(report, KeyframeExtractionVerificationReport)
        assert report.valid is True
        assert report.issues == ()
        assert report.expected_videos == 2
        assert report.found_videos == 2
        assert report.total_keyframes > 0
        assert report.duplicates == 0
        assert report.extractor == "omnishotcut_adaptive"
        assert report.detector == "synthetic"
        assert report.sampler == "adaptive_v1"

    def test_valid_without_videos_path(self, tmp_path: Path) -> None:
        output_dir, _ = _make_artifact(tmp_path, videos=["L21_V001"])
        report = verify_keyframe_extraction(output_dir)

        assert report.valid is True
        assert report.expected_videos == 1
        assert report.found_videos == 1

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        output_dir, _ = _make_artifact(tmp_path, videos=["L21_V001"])
        (output_dir / "manifest.json").unlink()
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            verify_keyframe_extraction(output_dir)

    def test_missing_output_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_keyframe_extraction(tmp_path / "does-not-exist")

    def test_duplicate_frame(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        frames_path = output_dir / "frames.ndjson"
        lines = frames_path.read_text(encoding="utf-8").strip().splitlines()
        lines.append(lines[0])
        frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert report.duplicates == 1
        assert any("duplicated keyframe key" in issue.message for issue in report.issues)

    def test_missing_video_coverage(self, tmp_path: Path) -> None:
        output_dir, _ = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        # The assigned videos.json now lists a video the output never produced.
        entries = [
            VideoEntry(video_id="L21_V001", path="/tmp/001.mp4", num_frames=500),
            VideoEntry(video_id="L21_V002", path="/tmp/002.mp4", num_frames=500),
            VideoEntry(video_id="L21_V003", path="/tmp/003.mp4", num_frames=500),
        ]
        videos_path = tmp_path / "videos_expected.json"
        save_video_manifest(
            videos_path,
            VideoManifest(schema_version="1.0", videos=tuple(entries)),
        )

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert report.expected_videos == 3
        assert report.found_videos == 2
        assert any("missing from frames.ndjson" in issue.message for issue in report.issues)

    def test_unexpected_video(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        # Assign only one video; the output contains both.
        entries = [VideoEntry(video_id="L21_V001", path="/tmp/001.mp4", num_frames=500)]
        videos_expected_path = tmp_path / "videos_expected.json"
        save_video_manifest(
            videos_expected_path,
            VideoManifest(schema_version="1.0", videos=tuple(entries)),
        )

        report = verify_keyframe_extraction(output_dir, videos_path=videos_expected_path)

        assert report.valid is False
        assert any("not in the input videos.json" in issue.message for issue in report.issues)

    def test_malformed_ndjson(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001"])
        frames_path = output_dir / "frames.ndjson"
        frames_path.write_text("{not valid json\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert any("invalid frame record" in issue.message for issue in report.issues)

    def test_record_missing_required_field(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001"])
        frames_path = output_dir / "frames.ndjson"
        record = json.loads(frames_path.read_text(encoding="utf-8").strip().splitlines()[0])
        del record["keyframe"]["keyframe_ordinal"]
        frames_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert any("invalid frame record" in issue.message for issue in report.issues)

    def test_count_mismatch(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["num_keyframes"] = manifest["num_keyframes"] + 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert any("num_keyframes" in issue.message for issue in report.issues)

    def test_empty_frames(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001"])
        (output_dir / "frames.ndjson").write_text("", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert any("no frame records" in issue.message for issue in report.issues)

    def test_missing_map_file(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        (output_dir / "map-keyframes" / "L21_V002.csv").unlink()

        report = verify_keyframe_extraction(output_dir, videos_path=videos_path)

        assert report.valid is False
        assert any("map-keyframes file missing" in issue.message for issue in report.issues)

    def test_report_round_trip(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001"])
        payload = verify_keyframe_extraction(output_dir, videos_path=videos_path).to_dict()
        assert payload["valid"] is True
        assert payload["found_videos"] == 1
        assert payload["total_keyframes"] > 0

    def test_worker_artifact_type(self, tmp_path: Path) -> None:
        output_dir, _ = _make_artifact(tmp_path, videos=["L21_V001"])
        report = verify_keyframe_extraction(output_dir)
        assert report.artifact_type == "worker"

    def test_valid_merged_artifact(self, tmp_path: Path) -> None:
        output_dir = _make_merged_artifact(
            tmp_path, shards=[["L21_V001", "L21_V002"], ["L21_V003"]]
        )
        report = verify_keyframe_extraction(output_dir)

        assert isinstance(report, KeyframeExtractionVerificationReport)
        assert report.artifact_type == "merged"
        assert report.valid is True
        assert report.issues == ()
        assert report.expected_videos == 3
        assert report.found_videos == 3
        assert report.total_keyframes > 0
        assert report.duplicates == 0
        assert report.extractor == "omnishotcut_adaptive"
        assert report.detector == "synthetic"
        assert report.sampler == "adaptive_v1"

    def test_merged_artifact_missing_source_shards(self, tmp_path: Path) -> None:
        output_dir = _make_merged_artifact(
            tmp_path, shards=[["L21_V001", "L21_V002"], ["L21_V003"]]
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["source_shards"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir)

        assert report.valid is False
        assert any("source_shards" in issue.message for issue in report.issues)

    def test_merged_artifact_empty_source_shards(self, tmp_path: Path) -> None:
        output_dir = _make_merged_artifact(tmp_path, shards=[["L21_V001"]])
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_shards"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir)

        assert report.valid is False
        assert any("source_shards" in issue.message for issue in report.issues)

    def test_merged_artifact_count_mismatch(self, tmp_path: Path) -> None:
        output_dir = _make_merged_artifact(
            tmp_path, shards=[["L21_V001", "L21_V002"], ["L21_V003"]]
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["num_keyframes"] = manifest["num_keyframes"] + 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_extraction(output_dir)

        assert report.valid is False
        assert any("num_keyframes" in issue.message for issue in report.issues)


class TestVerifyKeyframeExtractionCli:
    def test_help(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--output-dir" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = _run_cli()
        assert result.returncode == 2

    def test_valid_artifact_exits_0(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        result = _run_cli("--output-dir", str(output_dir), "--videos", str(videos_path))

        assert result.returncode == 0, result.stderr
        assert "Keyframe extraction verification: PASS" in result.stdout
        assert "expected: 2" in result.stdout
        assert "found: 2" in result.stdout
        assert "extractor: omnishotcut_adaptive" in result.stdout
        assert "detector: synthetic" in result.stdout
        assert "sampler: adaptive_v1" in result.stdout
        assert "OK: extraction artifact is valid" in result.stdout

    def test_invalid_artifact_exits_1(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001", "L21_V002"])
        (output_dir / "map-keyframes" / "L21_V002.csv").unlink()
        result = _run_cli("--output-dir", str(output_dir), "--videos", str(videos_path))

        assert result.returncode == 1
        assert "Keyframe extraction verification: FAIL" in result.stdout
        assert "Validation failed:" in result.stdout
        assert "[maps] map-keyframes file missing" in result.stdout

    def test_missing_dir_exits_1(self, tmp_path: Path) -> None:
        result = _run_cli("--output-dir", str(tmp_path / "does-not-exist"))
        assert result.returncode == 1
        assert "error:" in result.stderr

    def test_worker_artifact_type_exits_0(self, tmp_path: Path) -> None:
        output_dir, videos_path = _make_artifact(tmp_path, videos=["L21_V001"])
        result = _run_cli("--output-dir", str(output_dir), "--videos", str(videos_path))

        assert result.returncode == 0, result.stderr
        assert "Artifact type: worker" in result.stdout

    def test_merged_artifact_type_exits_0(self, tmp_path: Path) -> None:
        output_dir = _make_merged_artifact(
            tmp_path, shards=[["L21_V001", "L21_V002"], ["L21_V003"]]
        )
        result = _run_cli("--output-dir", str(output_dir))

        assert result.returncode == 0, result.stderr
        assert "Artifact type: merged" in result.stdout
        assert "expected: 3" in result.stdout
        assert "found: 3" in result.stdout
        assert "OK: extraction artifact is valid" in result.stdout


def test_detector_fields_required_for_adaptive(tmp_path: Path) -> None:
    """An adaptive manifest whose sampler is n/a must be flagged."""
    output_dir, _ = _make_artifact(tmp_path, videos=["L21_V001"])
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampler"] = "n/a"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = verify_keyframe_extraction(output_dir)

    assert report.valid is False
    assert any("requires field 'sampler'" in issue.message for issue in report.issues)
