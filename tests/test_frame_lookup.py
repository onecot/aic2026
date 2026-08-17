"""Tests for the general-purpose frame lookup tool.

Covers canonical artifact loading, reference parsing, keyframe lookup,
GT frame range derivation (including negative-start clamping), deterministic
output, and tolerance of optional timing metadata across artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.frame_lookup import (
    FRAMES_FILENAME,
    IMAGES_DIRNAME,
    FrameLookupError,
    FrameLookupRecord,
    KeyframeArtifact,
    compute_gt_range,
    format_record,
    image_relpath,
    load_frame_lookup_config,
    parse_frame_record,
    parse_keyframe_reference,
)


def _record_dict(
    video_id: str,
    ordinal: int,
    frame_index: int,
    pts_time: float | None = None,
    fps: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "keyframe": {"video_id": {"value": video_id}, "keyframe_ordinal": ordinal},
        "frame_index": frame_index,
    }
    if pts_time is not None:
        record["pts_time"] = pts_time
    if fps is not None:
        record["fps"] = fps
    return record


def _write_artifact(
    dir_path: Path,
    records: list[dict[str, object]],
    manifest: dict[str, object] | None = None,
    images: dict[tuple[str, int], bool] | None = None,
) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    with (dir_path / FRAMES_FILENAME).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    if manifest is not None:
        (dir_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for (video_id, ordinal), present in (images or {}).items():
        if present:
            image = dir_path / IMAGES_DIRNAME / video_id / f"{ordinal:06d}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"\xff\xd8\xff\xe0")
    return dir_path


ADAPTIVE_RECORDS = [
    _record_dict("L25_V085", 123, 456, pts_time=4.92, fps=25.0),
    _record_dict("L25_V085", 124, 482, pts_time=5.24, fps=25.0),
    _record_dict("L21_V001", 1, 0, pts_time=0.0, fps=25.0),
]

ADAPTIVE_MANIFEST = {
    "schema_version": "1.0",
    "artifact_id": "adaptive_sbd_v1",
    "extractor": "adaptive_sbd",
}


# --- 1. Artifact loading across different extractor layouts -----------------


def test_artifact_loads_from_adaptive_sbd_layout(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(
        tmp_path / "adaptive_sbd_v1", ADAPTIVE_RECORDS, ADAPTIVE_MANIFEST
    )
    artifact = KeyframeArtifact(artifact_dir)
    assert artifact.num_keyframes == 3
    assert artifact.schema_version == "1.0"
    assert artifact.manifest["artifact_id"] == "adaptive_sbd_v1"


def test_artifact_loads_from_different_extractor_layout(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "2.0",
        "artifact_id": "extractor_v2",
        "extractor": "btc_custom_2026",
    }
    artifact_dir = _write_artifact(tmp_path / "extractor_v2", ADAPTIVE_RECORDS, manifest)
    artifact = KeyframeArtifact(artifact_dir)
    assert artifact.schema_version == "2.0"
    assert artifact.manifest["artifact_id"] == "extractor_v2"
    assert artifact.lookup("L25_V085", 123).frame_index == 456


def test_artifact_without_manifest_and_without_timing_fields(tmp_path: Path) -> None:
    records = [
        _record_dict("L25_V085", 123, 456),
        _record_dict("L25_V085", 124, 482),
    ]
    artifact_dir = _write_artifact(tmp_path / "minimal", records, manifest=None)
    artifact = KeyframeArtifact(artifact_dir)
    assert artifact.schema_version is None
    assert artifact.manifest is None
    record = artifact.lookup("L25_V085", 123)
    assert record.pts_time is None
    assert record.fps is None


# --- 2. Keyframe lookup ------------------------------------------------------


def test_lookup_resolves_known_keyframe(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "a", ADAPTIVE_RECORDS, ADAPTIVE_MANIFEST)
    artifact = KeyframeArtifact(artifact_dir)
    record = artifact.lookup("L25_V085", 124)
    assert record.video_id == "L25_V085"
    assert record.keyframe_ordinal == 124
    assert record.frame_index == 482
    assert record.pts_time == 5.24
    assert record.fps == 25.0


def test_lookup_unknown_keyframe_raises(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "a", ADAPTIVE_RECORDS, ADAPTIVE_MANIFEST)
    artifact = KeyframeArtifact(artifact_dir)
    with pytest.raises(FrameLookupError, match="not found"):
        artifact.lookup("L25_V085", 999)
    with pytest.raises(FrameLookupError, match="not found"):
        artifact.lookup("L99_V999", 1)


# --- 3. Missing / malformed artifacts ---------------------------------------


def test_missing_artifact_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FrameLookupError, match="does not exist"):
        KeyframeArtifact(tmp_path / "does-not-exist")


def test_artifact_without_frames_ndjson_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FrameLookupError, match=FRAMES_FILENAME):
        KeyframeArtifact(empty_dir)


def test_duplicate_keyframe_record_raises(tmp_path: Path) -> None:
    records = [
        _record_dict("L25_V085", 123, 456),
        _record_dict("L25_V085", 123, 999),
    ]
    artifact_dir = _write_artifact(tmp_path / "dup", records)
    with pytest.raises(FrameLookupError, match="duplicate"):
        KeyframeArtifact(artifact_dir)


def test_invalid_manifest_raises(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "a", ADAPTIVE_RECORDS)
    (artifact_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FrameLookupError, match="manifest"):
        KeyframeArtifact(artifact_dir)


# --- 4. Reference parsing ----------------------------------------------------


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("L25_V085/000123.jpg", ("L25_V085", 123)),
        ("L25_V085/123.jpg", ("L25_V085", 123)),
        ("L25_V085 123", ("L25_V085", 123)),
        ("L25_V085/000123.JPG", ("L25_V085", 123)),
    ],
)
def test_parse_keyframe_reference_accepts_both_formats(reference: str, expected: tuple[str, int]) -> None:
    assert parse_keyframe_reference(reference) == expected


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "   ",
        "L25_V085",
        "L25_V085/000123.jpg/extra",
        "L25_V085 123 456",
        "foo/000123.jpg",
        "L25_V085/abc.jpg",
        "L25_V085 0",
        "L25_V085 -3",
    ],
)
def test_parse_keyframe_reference_rejects_malformed(reference: str) -> None:
    with pytest.raises(FrameLookupError):
        parse_keyframe_reference(reference)


# --- 5. GT frame range derivation -------------------------------------------


def test_compute_gt_range_centers_window_on_keyframe() -> None:
    record = FrameLookupRecord(video_id="L25_V085", keyframe_ordinal=123, frame_index=456, fps=25.0)
    start, end = compute_gt_range(record, 2.0)
    assert start == 431
    assert end == 481
    assert end - start == 50


def test_compute_gt_range_clamps_negative_start_to_zero() -> None:
    record = FrameLookupRecord(video_id="L21_V001", keyframe_ordinal=1, frame_index=2, fps=25.0)
    start, end = compute_gt_range(record, 2.0)
    assert start == 0
    assert end == 27


def test_compute_gt_range_requires_fps() -> None:
    record = FrameLookupRecord(video_id="L25_V085", keyframe_ordinal=123, frame_index=456, fps=None)
    with pytest.raises(FrameLookupError, match="no fps"):
        compute_gt_range(record, 2.0)


def test_compute_gt_range_rejects_non_positive_window() -> None:
    record = FrameLookupRecord(video_id="L25_V085", keyframe_ordinal=123, frame_index=456, fps=25.0)
    with pytest.raises(FrameLookupError, match="window_sec"):
        compute_gt_range(record, 0)
    with pytest.raises(FrameLookupError, match="window_sec"):
        compute_gt_range(record, -1.0)


def test_lookup_with_window_sec_round_trip(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "a", ADAPTIVE_RECORDS, ADAPTIVE_MANIFEST)
    artifact = KeyframeArtifact(artifact_dir)
    record = artifact.lookup("L25_V085", 123)
    start, end = compute_gt_range(record, 2.0)
    assert start <= record.frame_index <= end


# --- 6. Deterministic, stable output ----------------------------------------


def test_format_record_is_deterministic() -> None:
    record = FrameLookupRecord(video_id="L25_V085", keyframe_ordinal=123, frame_index=456, pts_time=4.92, fps=25.0)
    rendered = format_record(record, "images/L25_V085/000123.jpg", (431, 481))
    assert "video_id          : L25_V085" in rendered
    assert "keyframe_ordinal  : 123" in rendered
    assert "frame_index       : 456" in rendered
    assert "pts_time          : 4.92" in rendered
    assert "fps               : 25" in rendered
    assert "image_path        : images/L25_V085/000123.jpg" in rendered
    assert "gt_frame_start    : 431" in rendered
    assert "gt_frame_end      : 481" in rendered
    assert rendered == format_record(record, "images/L25_V085/000123.jpg", (431, 481))


def test_format_record_without_optional_metadata() -> None:
    record = FrameLookupRecord(video_id="L25_V085", keyframe_ordinal=123, frame_index=456)
    rendered = format_record(record, None)
    assert "pts_time          : n/a" in rendered
    assert "fps               : n/a" in rendered
    assert "image_path        : not available" in rendered
    assert "gt_frame_start" not in rendered


def test_reloading_artifact_produces_identical_records(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "a", ADAPTIVE_RECORDS, ADAPTIVE_MANIFEST)
    first = KeyframeArtifact(artifact_dir).lookup("L25_V085", 123)
    second = KeyframeArtifact(artifact_dir).lookup("L25_V085", 123)
    assert first == second


# --- 7. Image path resolution -----------------------------------------------


def test_image_path_returns_path_only_when_image_exists(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(
        tmp_path / "a",
        ADAPTIVE_RECORDS,
        images={("L25_V085", 123): True, ("L25_V085", 124): False},
    )
    artifact = KeyframeArtifact(artifact_dir)
    assert artifact.image_path("L25_V085", 123) == image_relpath("L25_V085", 123)
    assert artifact.image_path("L25_V085", 124) is None


def test_image_relpath_uses_canonical_layout() -> None:
    assert image_relpath("L25_V085", 123) == "images/L25_V085/000123.jpg"


# --- 8. Lenient record parsing ----------------------------------------------


def test_parse_frame_record_accepts_string_video_id() -> None:
    payload = {
        "keyframe": {"video_id": "L25_V085", "keyframe_ordinal": 7},
        "frame_index": 3,
    }
    record = parse_frame_record(payload)
    assert record.video_id == "L25_V085"
    assert record.pts_time is None
    assert record.fps is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"keyframe": {"keyframe_ordinal": 1}, "frame_index": 0},
        {"keyframe": {"video_id": {"value": "L25_V085"}}, "frame_index": 0},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 1}},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 0}, "frame_index": 0},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 1}, "frame_index": -1},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 1}, "frame_index": 0, "fps": 0},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 1}, "frame_index": 0, "pts_time": -0.1},
        {"keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 1}, "frame_index": 0, "fps": "abc"},
    ],
)
def test_parse_frame_record_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(FrameLookupError):
        parse_frame_record(payload)


# --- 9. Config loading ------------------------------------------------------


def test_load_frame_lookup_config_returns_empty_for_none() -> None:
    assert load_frame_lookup_config(None) == {}


def test_load_frame_lookup_config_reads_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("frame_lookup:\n  window_sec: 2.5\n", encoding="utf-8")
    assert load_frame_lookup_config(config_path) == {"window_sec": 2.5}


def test_load_frame_lookup_config_missing_section_is_empty(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("other:\n  foo: bar\n", encoding="utf-8")
    assert load_frame_lookup_config(config_path) == {}


def test_load_frame_lookup_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame_lookup_config(tmp_path / "nope.yaml")


def test_load_frame_lookup_config_rejects_non_mapping_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("frame_lookup: 5\n", encoding="utf-8")
    with pytest.raises(FrameLookupError, match="mapping"):
        load_frame_lookup_config(config_path)