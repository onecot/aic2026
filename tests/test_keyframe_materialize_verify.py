"""Tests for keyframe-image materialization verification and its CLI.

Builds a synthetic ``adaptive_sbd_v1`` artifact directly (no ffmpeg, no
videos): an ``images/`` tree, ``frames.ndjson``, and ``manifest.json``, then
verifies it and mutates it to exercise failure modes. Fast, deterministic,
and cross-platform.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.keyframe_materialize import (
    ARTIFACT_ID,
    IMAGES_DIRNAME,
    MATERIALIZE_MANIFEST_SCHEMA_VERSION,
)
from aic2026.keyframe_materialize_verify import (
    ARTIFACT_ID_UNKNOWN,
    KeyframeMaterializationVerificationReport,
    verify_keyframe_materialization,
)

#: video_id -> ordinals present in the synthetic artifact.
LAYOUT = [("L21_V001", [1, 2, 3]), ("L21_V002", [1, 2])]


def _manifest_payload(num_keyframes: int = 5, **overrides: object) -> dict:
    payload: dict = {
        "schema_version": MATERIALIZE_MANIFEST_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "source_frames": "/frames.ndjson",
        "source_videos": "/videos.json",
        "num_videos": 2,
        "num_keyframes": num_keyframes,
        "image_format": "jpg",
        "jpeg_quality": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _write_frames(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for video_id, ordinal in rows:
            record = {
                "keyframe": {"video_id": {"value": video_id}, "keyframe_ordinal": ordinal},
                "pts_time": (ordinal - 1) * 0.4,
                "fps": 25.0,
                "frame_index": (ordinal - 1) * 10,
            }
            handle.write(json.dumps(record) + "\n")


def _make_artifact(
    tmp_path: Path,
    *,
    rows: list[tuple[str, int]] | None = None,
    name: str = "artifact",
) -> Path:
    rows = rows if rows is not None else [row for vid, ords in LAYOUT for row in [(vid, o) for o in ords]]
    out = tmp_path / name
    (out / IMAGES_DIRNAME).mkdir(parents=True)
    for video_id, ordinal in rows:
        video_dir = out / IMAGES_DIRNAME / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / f"{ordinal:06d}.jpg").write_bytes(bytes([1, ordinal, 0]))
    _write_frames(out / "frames.ndjson", rows)
    (out / "manifest.json").write_text(
        json.dumps(_manifest_payload(len(rows)), indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "verify_keyframe_materialization.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script()), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestVerifyKeyframeMaterialization:
    def test_valid_artifact(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        report = verify_keyframe_materialization(out)

        assert isinstance(report, KeyframeMaterializationVerificationReport)
        assert report.valid is True
        assert report.issues == ()
        assert report.artifact_id == ARTIFACT_ID
        assert report.records == 5
        assert report.images == 5
        assert report.missing_images == 0
        assert report.unexpected_images == 0
        assert report.duplicates == 0
        assert report.num_videos == 2

    def test_missing_output_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_keyframe_materialization(tmp_path / "does-not-exist")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "manifest.json").unlink()

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.artifact_id == ARTIFACT_ID_UNKNOWN
        assert any("manifest.json missing" in issue.message for issue in report.issues)

    def test_missing_frames(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "frames.ndjson").unlink()

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.records == 0
        assert any("frames.ndjson missing" in issue.message for issue in report.issues)

    def test_empty_frames(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "frames.ndjson").write_text("", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("no frame records" in issue.message for issue in report.issues)

    def test_missing_images_directory(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        import shutil

        shutil.rmtree(out / IMAGES_DIRNAME)

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.images == 0
        assert report.missing_images == 0
        assert any("images directory missing" in issue.message for issue in report.issues)

    def test_missing_image(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / IMAGES_DIRNAME / "L21_V001" / "000002.jpg").unlink()

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.missing_images == 1
        assert report.images == 4
        assert any("missing image for L21_V001 ordinal 2" in issue.message for issue in report.issues)

    def test_extra_image(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        extra_dir = out / IMAGES_DIRNAME / "L21_V001"
        (extra_dir / "000099.jpg").write_bytes(bytes([1, 1, 0]))

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.unexpected_images == 1
        assert report.images == 6
        assert any("unexpected image for L21_V001 ordinal 99" in issue.message for issue in report.issues)

    def test_misnamed_image_is_unexpected(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        video_dir = out / IMAGES_DIRNAME / "L21_V001"
        (video_dir / "1.jpg").write_bytes(bytes([1, 1, 0]))

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.unexpected_images == 1
        assert any("unexpected image file" in issue.message for issue in report.issues)

    def test_malformed_frames(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "frames.ndjson").write_text("{not valid json\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.records == 0
        assert any("invalid frame record" in issue.message for issue in report.issues)

    def test_record_missing_required_field(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        frames_path = out / "frames.ndjson"
        first = json.loads(frames_path.read_text(encoding="utf-8").strip().splitlines()[0])
        del first["keyframe"]["keyframe_ordinal"]
        frames_path.write_text(json.dumps(first) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("invalid frame record" in issue.message for issue in report.issues)

    def test_duplicate_ordinal(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        frames_path = out / "frames.ndjson"
        lines = frames_path.read_text(encoding="utf-8").strip().splitlines()
        duplicate = json.loads(lines[0])
        duplicate["keyframe"]["keyframe_ordinal"] = 2
        lines.append(json.dumps(duplicate))
        frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.duplicates == 1
        assert any("duplicated keyframe key" in issue.message for issue in report.issues)

    def test_disordered_records(self, tmp_path: Path) -> None:
        rows = [("L21_V001", 2), ("L21_V001", 1)]
        out = _make_artifact(tmp_path, rows=rows)

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("records not ordered" in issue.message for issue in report.issues)

    def test_manifest_count_mismatch(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["num_keyframes"] = 999
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("num_keyframes" in issue.message for issue in report.issues)

    def test_manifest_num_images_mismatch(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["num_images"] = 42
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("num_images" in issue.message for issue in report.issues)

    def test_manifest_num_images_matches(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["num_images"] = 5
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is True

    def test_wrong_artifact_id(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["artifact_id"] = "something-else"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("artifact_id" in issue.message for issue in report.issues)

    def test_wrong_schema_version(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        manifest_path = out / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["schema_version"] = "9.9"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert any("schema_version" in issue.message for issue in report.issues)

    def test_invalid_manifest_json(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / "manifest.json").write_text("{not json\n", encoding="utf-8")

        report = verify_keyframe_materialization(out)

        assert report.valid is False
        assert report.artifact_id == ARTIFACT_ID_UNKNOWN
        assert any("invalid manifest.json" in issue.message for issue in report.issues)

    def test_report_round_trip(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        payload = verify_keyframe_materialization(out).to_dict()

        assert payload["valid"] is True
        assert payload["artifact_id"] == ARTIFACT_ID
        assert payload["records"] == 5
        assert payload["images"] == 5
        assert payload["missing_images"] == 0
        assert payload["unexpected_images"] == 0


class TestVerifyKeyframeMaterializationCli:
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
        assert "Keyframe materialization verification: PASS" in result.stdout
        assert "type: adaptive_sbd_v1" in result.stdout
        assert "records: 5" in result.stdout
        assert "images: 5" in result.stdout
        assert "missing: 0" in result.stdout
        assert "unexpected: 0" in result.stdout
        assert "OK: materialized keyframe artifact is valid" in result.stdout

    def test_invalid_artifact_exits_1(self, tmp_path: Path) -> None:
        out = _make_artifact(tmp_path)
        (out / IMAGES_DIRNAME / "L21_V001" / "000002.jpg").unlink()
        result = _run_cli("--output-dir", str(out))

        assert result.returncode == 1
        assert "Keyframe materialization verification: FAIL" in result.stdout
        assert "Validation failed:" in result.stdout
        assert "[images] missing image for L21_V001 ordinal 2" in result.stdout

    def test_missing_dir_exits_1(self, tmp_path: Path) -> None:
        result = _run_cli("--output-dir", str(tmp_path / "does-not-exist"))
        assert result.returncode == 1
        assert "error:" in result.stderr
