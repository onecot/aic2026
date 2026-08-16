"""Tests for the keyframe-image materialization pipeline.

Materialization decodes frames from real videos with ffmpeg, so these tests
require an ``ffmpeg`` on PATH and skip when it is unavailable (keeping the
default suite cross-platform).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.keyframe_extraction import (
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    VideoEntry,
    VideoManifest,
    save_video_manifest,
)
from aic2026.keyframe_materialize import (
    ARTIFACT_ID,
    IMAGES_DIRNAME,
    MATERIALIZE_MANIFEST_SCHEMA_VERSION,
    MaterializeError,
    MaterializeManifest,
    image_filename,
    image_relpath,
    materialized_image_path,
    materialize_keyframes,
    verify_materialization,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="materialization requires ffmpeg on PATH",
)

#: 50-frame 320x240 test pattern; 4s at 25fps keeps the fixture fast.
_FFMPEG_ARGS = [
    "ffmpeg",
    "-v",
    "error",
    "-f",
    "lavfi",
    "-i",
    "testsrc=size=320x240:rate=25:duration=4",
    "-pix_fmt",
    "yuv420p",
    "-y",
]


def _make_video(
    directory: Path,
    name: str = "video.mp4",
    *,
    duration: int = 4,
    size: str = "320x240",
) -> Path:
    path = directory / name
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate=25:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _write_frames(path: Path, rows: list[tuple[str, int, int]]) -> None:
    """Write frames.ndjson rows: (video_id, ordinal, frame_index)."""
    with path.open("w", encoding="utf-8") as handle:
        for video_id, ordinal, frame_index in rows:
            record = {
                "keyframe": {"video_id": {"value": video_id}, "keyframe_ordinal": ordinal},
                "pts_time": frame_index / 25.0,
                "fps": 25.0,
                "frame_index": frame_index,
            }
            handle.write(json.dumps(record) + "\n")


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return _make_video(tmp_path)


@pytest.fixture
def artifact_inputs(tmp_path: Path, video_path: Path) -> tuple[Path, Path]:
    """Return (videos.json, frames.ndjson) for a single 3-keyframe video."""
    videos = tmp_path / "videos.json"
    save_video_manifest(
        videos,
        VideoManifest(
            schema_version="1.0",
            videos=(VideoEntry(video_id="L21_V001", path=str(video_path), num_frames=50),),
        ),
    )
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, [("L21_V001", 1, 0), ("L21_V001", 2, 13), ("L21_V001", 3, 42)])
    return videos, frames


# ------------------------------------------------------------------- helpers


def test_image_filename_validates() -> None:
    assert image_filename(1) == "000001.jpg"
    assert image_filename(99) == "000099.jpg"
    with pytest.raises(ValueError, match="positive integer"):
        image_filename(0)
    with pytest.raises(ValueError, match="positive integer"):
        image_filename("1")  # type: ignore[arg-type]


def test_image_relpath_and_materialized_path() -> None:
    assert image_relpath("L21_V001", 1) == "images/L21_V001/000001.jpg"
    root = Path("/tmp/root")
    assert materialized_image_path(root, "L21_V001", 1) == root / "images" / "L21_V001" / "000001.jpg"


def test_manifest_validates_schema() -> None:
    base: dict = {
        "schema_version": MATERIALIZE_MANIFEST_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "source_frames": "/frames.ndjson",
        "source_videos": "/videos.json",
        "num_videos": 1,
        "num_keyframes": 3,
        "image_format": "jpg",
        "jpeg_quality": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    manifest = MaterializeManifest.from_dict(base)
    assert manifest.to_dict()["artifact_id"] == ARTIFACT_ID
    with pytest.raises(ValueError, match="unsupported materialize schema"):
        MaterializeManifest.from_dict({**base, "schema_version": "9.9"})
    with pytest.raises(ValueError, match="jpeg_quality"):
        MaterializeManifest.from_dict({**base, "jpeg_quality": 0})
    with pytest.raises(ValueError, match="jpeg_quality"):
        MaterializeManifest.from_dict({**base, "jpeg_quality": 32})


# ----------------------------------------------------------------- pipeline


def test_materialize_writes_valid_artifact(
    tmp_path: Path, artifact_inputs: tuple[Path, Path]
) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    manifest = materialize_keyframes(videos, frames, out, progress="off")

    assert manifest.artifact_id == ARTIFACT_ID
    assert manifest.num_videos == 1
    assert manifest.num_keyframes == 3
    assert manifest.image_format == "jpg"
    assert manifest.schema_version == MATERIALIZE_MANIFEST_SCHEMA_VERSION

    assert (out / FRAMES_FILENAME).is_file()
    assert (out / MANIFEST_FILENAME).is_file()
    for ordinal in (1, 2, 3):
        assert (out / IMAGES_DIRNAME / "L21_V001" / f"{ordinal:06d}.jpg").is_file()
    assert (out / "images" / "L21_V001" / "000004.jpg").is_file() is False

    report = verify_materialization(out)
    assert report.valid
    assert report.num_videos == 1
    assert report.num_keyframes == 3
    assert report.images_present == 3
    assert report.missing_images == 0


def test_materialize_is_idempotent(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    again = materialize_keyframes(videos, frames, out, progress="off")
    assert again.num_keyframes == 3
    assert verify_materialization(out).valid


def test_materialize_refuses_nonempty_output(
    tmp_path: Path, artifact_inputs: tuple[Path, Path]
) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    out.mkdir()
    (out / "leftover.txt").write_text("x")
    with pytest.raises(MaterializeError, match="already exists"):
        materialize_keyframes(videos, frames, out, progress="off")


def test_materialize_overwrite_replaces(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    (out / "leftover.txt").write_text("x")
    materialize_keyframes(videos, frames, out, overwrite=True, progress="off")
    assert (out / "leftover.txt").exists() is False
    assert verify_materialization(out).valid


def _make_manifest(videos_path: Path, video: Path, video_id: str = "L21_V001") -> Path:
    save_video_manifest(
        videos_path,
        VideoManifest(
            schema_version="1.0",
            videos=(VideoEntry(video_id=video_id, path=str(video), num_frames=25 * 21),),
        ),
    )
    return videos_path


def _decode_single_frame(video: Path, frame_index: int, dest: Path) -> bytes:
    """Decode one frame with ffmpeg and return the JPEG bytes (reference)."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select='eq(n,{frame_index})'",
            "-fps_mode",
            "passthrough",
            "-vsync",
            "vfr",
            "-f",
            "image2",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(dest / "%d.jpg"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return (dest / "0.jpg").read_bytes()


def test_materialize_video_with_more_than_500_frames(tmp_path: Path) -> None:
    video = _make_video(tmp_path, "long.mp4", duration=21, size="160x120")
    videos = _make_manifest(tmp_path / "videos.json", video)
    frames = tmp_path / "frames.ndjson"
    rows = [("L21_V001", ordinal, ordinal - 1) for ordinal in range(1, 511)]
    _write_frames(frames, rows)
    out = tmp_path / "artifact"

    manifest = materialize_keyframes(videos, frames, out, progress="off")
    assert manifest.num_keyframes == 510
    report = verify_materialization(out)
    assert report.valid
    assert report.images_present == 510
    assert report.missing_images == 0
    images = sorted(p.name for p in (out / IMAGES_DIRNAME / "L21_V001").iterdir())
    assert images[0] == "000001.jpg"
    assert images[-1] == "000510.jpg"
    assert len(images) == 510


def test_materialize_preserves_ordinal_order_across_batches(tmp_path: Path) -> None:
    video = _make_video(tmp_path, "ord.mp4", duration=6, size="160x120")
    videos = _make_manifest(tmp_path / "videos.json", video)
    frames = tmp_path / "frames.ndjson"
    rows = [
        ("L21_V001", 1, 100),
        ("L21_V001", 2, 5),
        ("L21_V001", 3, 149),
        ("L21_V001", 4, 0),
        ("L21_V001", 5, 77),
        ("L21_V001", 6, 50),
    ]
    _write_frames(frames, rows)
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")

    for ordinal, frame_index in [(1, 100), (2, 5), (3, 149), (4, 0), (5, 77), (6, 50)]:
        got = (out / IMAGES_DIRNAME / "L21_V001" / f"{ordinal:06d}.jpg").read_bytes()
        ref = _decode_single_frame(video, frame_index, tmp_path / f"ref_{ordinal}")
        assert got == ref, f"ordinal {ordinal} (frame {frame_index}) decoded to the wrong frame"


def test_materialize_uses_small_batches_when_forced(tmp_path: Path) -> None:
    video = _make_video(tmp_path, "batched.mp4", duration=6, size="160x120")
    videos = _make_manifest(tmp_path / "videos.json", video)
    frames = tmp_path / "frames.ndjson"
    rows = [("L21_V001", ordinal, ordinal - 1) for ordinal in range(1, 11)]
    _write_frames(frames, rows)
    out = tmp_path / "artifact"

    manifest = materialize_keyframes(
        videos, frames, out, max_frames_per_batch=3, progress="off"
    )
    assert manifest.num_keyframes == 10
    assert verify_materialization(out).valid


def test_materialize_raises_on_partial_batch_failure(tmp_path: Path) -> None:
    video = _make_video(tmp_path, "fail.mp4", duration=6, size="160x120")
    videos = _make_manifest(tmp_path / "videos.json", video)
    frames = tmp_path / "frames.ndjson"
    rows = [
        ("L21_V001", 1, 0),
        ("L21_V001", 2, 1),
        ("L21_V001", 3, 2),
        ("L21_V001", 4, 3),
        ("L21_V001", 5, 4),
        ("L21_V001", 6, 9999),
    ]
    _write_frames(frames, rows)
    out = tmp_path / "artifact"

    with pytest.raises(MaterializeError, match="decoded|ffmpeg failed"):
        materialize_keyframes(videos, frames, out, max_frames_per_batch=2, progress="off")
    assert out.exists() is False


def test_materialize_batch_size_validates(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    with pytest.raises(ValueError, match="max_frames_per_batch"):
        materialize_keyframes(videos, frames, tmp_path / "artifact", max_frames_per_batch=0, progress="off")


def test_materialize_rejects_unknown_video(tmp_path: Path, video_path: Path) -> None:
    videos = tmp_path / "videos.json"
    save_video_manifest(
        videos,
        VideoManifest(
            schema_version="1.0",
            videos=(VideoEntry(video_id="L21_V001", path=str(video_path), num_frames=50),),
        ),
    )
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, [("L99_V999", 1, 0)])
    with pytest.raises(MaterializeError, match="not in the manifest"):
        materialize_keyframes(videos, frames, tmp_path / "artifact", progress="off")


def test_materialize_raises_on_out_of_range_frame_index(
    tmp_path: Path, video_path: Path
) -> None:
    videos = tmp_path / "videos.json"
    save_video_manifest(
        videos,
        VideoManifest(
            schema_version="1.0",
            videos=(VideoEntry(video_id="L21_V001", path=str(video_path), num_frames=50),),
        ),
    )
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, [("L21_V001", 1, 9999)])
    with pytest.raises(MaterializeError, match="decoded|ffmpeg failed"):
        materialize_keyframes(videos, frames, tmp_path / "artifact", progress="off")
    assert (tmp_path / "artifact").exists() is False


def test_materialize_uses_video_root_for_relative_paths(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video = _make_video(videos_dir, "rel.mp4")
    videos = tmp_path / "videos.json"
    save_video_manifest(
        videos,
        VideoManifest(
            schema_version="1.0",
            videos=(VideoEntry(video_id="L21_V001", path="rel.mp4", num_frames=50),),
        ),
    )
    frames = tmp_path / "frames.ndjson"
    _write_frames(frames, [("L21_V001", 1, 0)])
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, video_root=videos_dir, progress="off")
    assert verify_materialization(out).valid
    assert video.is_file()


# ------------------------------------------------------------------ verification


def test_verify_detects_missing_images(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    (out / IMAGES_DIRNAME / "L21_V001" / "000002.jpg").unlink()

    report = verify_materialization(out)
    assert not report.valid
    assert report.missing_images == 1
    assert report.images_present == 2
    assert any("000002.jpg" in issue.message for issue in report.issues)


def test_verify_detects_extra_images(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    (out / IMAGES_DIRNAME / "L21_V001" / "000099.jpg").write_bytes(b"\xff\xd8\xff")

    report = verify_materialization(out)
    assert not report.valid
    assert any("records" in issue.message for issue in report.issues)


def test_verify_detects_missing_manifest(tmp_path: Path, artifact_inputs: tuple[Path, Path]) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    (out / MANIFEST_FILENAME).unlink()

    report = verify_materialization(out)
    assert not report.valid
    assert any(issue.section == "artifact" for issue in report.issues)


def test_verify_detects_manifest_count_mismatch(
    tmp_path: Path, artifact_inputs: tuple[Path, Path]
) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "artifact"
    materialize_keyframes(videos, frames, out, progress="off")
    payload = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["num_keyframes"] = 999
    (out / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    report = verify_materialization(out)
    assert not report.valid
    assert any("num_keyframes" in issue.message for issue in report.issues)


# ------------------------------------------------------------------------- CLI


def test_materialize_cli(artifact_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    videos, frames = artifact_inputs
    out = tmp_path / "cli-artifact"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_keyframes.py",
            "--videos",
            str(videos),
            "--frames",
            str(frames),
            "--output-dir",
            str(out),
            "--progress",
            "off",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert "Materialization complete: 3 keyframe images" in result.stdout
    assert "Verification: valid" in result.stdout
    assert verify_materialization(out).valid
