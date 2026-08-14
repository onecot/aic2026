"""Tests for the deduplicated keyframe artifact preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.embedding_artifact import iter_frames
from aic2026.keyframe_dedup import (
    DEDUP_REPORT_SCHEMA_VERSION,
    KEEP_FIRST_STRATEGY,
    DedupError,
    DedupReport,
    deduplicate_keyframes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_records(
    path: Path,
    records: list[tuple[str, int, int, float]],
) -> None:
    """Write canonical frames.ndjson records.

    Each record is ``(video_id, keyframe_ordinal, frame_index, pts_time)``.
    """
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinal, frame_index, pts_time in records:
            record = {
                "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                "pts_time": pts_time,
                "fps": 25.0,
                "frame_index": frame_index,
            }
            handle.write(json.dumps(record) + "\n")


def _read_keys(path: Path) -> list[tuple[str, int]]:
    return [(f.keyframe.video_id.value, f.frame_index) for f in iter_frames(path)]


def _read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Deduplication behavior
# ---------------------------------------------------------------------------


def test_removes_duplicates_and_keeps_first(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V006", 1, 0, 0.0),
            ("L21_V006", 2, 0, 0.4),  # duplicate key (L21_V006, 0), ordinal 2
            ("L21_V006", 3, 2, 0.8),
            ("L21_V007", 1, 0, 0.0),
            ("L21_V007", 2, 0, 0.4),  # duplicate key (L21_V007, 0)
        ],
    )
    out = tmp_path / "out.ndjson"

    report = deduplicate_keyframes(src, out)

    assert report.original_records == 5
    assert report.unique_records == 3
    assert report.removed_duplicates == 2
    assert _read_keys(out) == [("L21_V006", 0), ("L21_V006", 2), ("L21_V007", 0)]

    # keep first occurrence: the kept (L21_V006, 0) record is ordinal 1.
    records = _read_records(out)
    assert records[0]["keyframe"]["keyframe_ordinal"] == 1
    assert records[0]["pts_time"] == 0.0
    # the duplicate ordinal-2 record must not appear.
    assert [r["keyframe"]["keyframe_ordinal"] for r in records] == [1, 3, 1]


def test_preserves_original_ordering(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V001", 1, 0, 0.0),
            ("L21_V002", 1, 0, 0.0),
            ("L21_V001", 2, 1, 0.4),
            ("L21_V001", 3, 0, 0.8),  # duplicate (L21_V001, 0), dropped
            ("L21_V002", 2, 1, 0.4),
        ],
    )
    out = tmp_path / "out.ndjson"
    deduplicate_keyframes(src, out)
    assert _read_keys(out) == [("L21_V001", 0), ("L21_V002", 0), ("L21_V001", 1), ("L21_V002", 1)]


def test_same_frame_index_across_videos_allowed(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [("L21_V001", 1, 0, 0.0), ("L21_V002", 1, 0, 0.0), ("L21_V003", 1, 0, 0.0)],
    )
    out = tmp_path / "out.ndjson"
    report = deduplicate_keyframes(src, out)
    assert report.removed_duplicates == 0
    assert report.unique_records == 3
    assert _read_keys(out) == [("L21_V001", 0), ("L21_V002", 0), ("L21_V003", 0)]


def test_input_never_modified(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V006", 1, 0, 0.0),
            ("L21_V006", 2, 0, 0.4),
            ("L21_V006", 3, 2, 0.8),
        ],
    )
    original = src.read_bytes()
    deduplicate_keyframes(src, tmp_path / "out.ndjson")
    assert src.read_bytes() == original


def test_no_duplicates_is_identity(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [("L21_V001", 1, 0, 0.0), ("L21_V001", 2, 1, 0.4), ("L21_V002", 1, 0, 0.0)],
    )
    out = tmp_path / "out.ndjson"
    report = deduplicate_keyframes(src, out)
    assert report.removed_duplicates == 0
    assert _read_keys(out) == [("L21_V001", 0), ("L21_V001", 1), ("L21_V002", 0)]


def test_empty_input(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    src.write_text("", encoding="utf-8")
    out = tmp_path / "out.ndjson"
    report = deduplicate_keyframes(src, out)
    assert report.original_records == 0
    assert report.unique_records == 0
    assert report.removed_duplicates == 0
    assert _read_records(out) == []
    assert report.duplicate_examples == ()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_correctness(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V006", 1, 0, 0.0),
            ("L21_V006", 2, 0, 0.4),
            ("L21_V007", 1, 0, 0.0),
            ("L21_V007", 2, 0, 0.4),
            ("L21_V008", 1, 0, 0.0),
        ],
    )
    out = tmp_path / "out.ndjson"
    report_path = tmp_path / "dedup_report.json"

    report = deduplicate_keyframes(src, out, report_path)

    assert report.source_path == str(src.resolve())
    assert report.output_path == str(out.resolve())
    assert report.original_records == 5
    assert report.unique_records == 3
    assert report.removed_duplicates == 2
    assert report.strategy == KEEP_FIRST_STRATEGY
    # first duplicate keys in first-seen order.
    assert report.duplicate_examples == (("L21_V006", 0), ("L21_V007", 0))

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DEDUP_REPORT_SCHEMA_VERSION
    assert payload["original_records"] == 5
    assert payload["unique_records"] == 3
    assert payload["removed_duplicates"] == 2
    assert payload["duplicate_examples"] == [["L21_V006", 0], ["L21_V007", 0]]
    assert payload["strategy"] == KEEP_FIRST_STRATEGY

    # round-trip through from_dict.
    assert DedupReport.from_dict(payload) == report


def test_report_default_no_report_path(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0), ("L21_V001", 2, 0, 0.4)])
    report = deduplicate_keyframes(src, tmp_path / "out.ndjson")
    assert report.removed_duplicates == 1
    assert not (tmp_path / "dedup_report.json").exists()


def test_report_max_examples_limits_keys(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V001", 1, 0, 0.0),
            ("L21_V001", 2, 0, 0.4),
            ("L21_V002", 1, 0, 0.0),
            ("L21_V002", 2, 0, 0.4),
            ("L21_V003", 1, 0, 0.0),
            ("L21_V003", 2, 0, 0.4),
        ],
    )
    report = deduplicate_keyframes(src, tmp_path / "out.ndjson", max_examples=2)
    assert report.duplicate_examples == (("L21_V001", 0), ("L21_V002", 0))


def test_report_rejects_unsupported_schema_version() -> None:
    payload = {
        "schema_version": "9.9",
        "source_path": "s",
        "output_path": "o",
        "original_records": 1,
        "unique_records": 1,
        "removed_duplicates": 0,
        "duplicate_examples": [],
        "strategy": KEEP_FIRST_STRATEGY,
    }
    with pytest.raises(ValueError, match="schema version"):
        DedupReport.from_dict(payload)


# ---------------------------------------------------------------------------
# Atomic output
# ---------------------------------------------------------------------------


def test_atomic_output_no_tmp_leftover(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [("L21_V001", 1, 0, 0.0), ("L21_V001", 2, 0, 0.4), ("L21_V002", 1, 0, 0.0)],
    )
    out = tmp_path / "out.ndjson"
    deduplicate_keyframes(src, out)
    assert out.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_output_failure_leaves_no_partial_file(tmp_path: Path) -> None:
    # An invalid line aborts the run; no output or tmp file may be left behind.
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0)])
    with src.open("a", encoding="utf-8") as handle:
        handle.write('{"keyframe": {"video_id": {"value": "L21_V001"}}}\n')  # missing fields
    out = tmp_path / "out.ndjson"
    report = tmp_path / "dedup_report.json"

    with pytest.raises(KeyError):
        deduplicate_keyframes(src, out, report)
    assert not out.exists()
    assert not report.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_output_preserves_existing_output_on_failure(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V001", 1, 0, 0.0),
            ("L21_V001", 2, 0, 0.4),
        ],
    )
    with src.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    out = tmp_path / "out.ndjson"
    out.write_text("previous valid artifact\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        deduplicate_keyframes(src, out)

    # The previous valid artifact must be untouched.
    assert out.read_text(encoding="utf-8") == "previous valid artifact\n"
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Invalid schema handling
# ---------------------------------------------------------------------------


def test_invalid_json_line_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    src.write_text('{"keyframe": {"video_id": {"value": "L21_V001"}, "keyframe_ordinal": 1}, "pts_time": 0.0, "fps": 25.0, "frame_index": 0}\nnot-json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        deduplicate_keyframes(src, tmp_path / "out.ndjson")


def test_missing_video_id_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0)])
    with src.open("a", encoding="utf-8") as handle:
        handle.write('{"keyframe": {"keyframe_ordinal": 1}, "pts_time": 0.0, "fps": 25.0, "frame_index": 0}\n')
    with pytest.raises(KeyError):
        deduplicate_keyframes(src, tmp_path / "out.ndjson")


def test_invalid_video_id_pattern_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0)])
    with src.open("a", encoding="utf-8") as handle:
        handle.write('{"keyframe": {"video_id": {"value": "nope"}, "keyframe_ordinal": 1}, "pts_time": 0.0, "fps": 25.0, "frame_index": 0}\n')
    with pytest.raises(ValueError):
        deduplicate_keyframes(src, tmp_path / "out.ndjson")


def test_negative_frame_index_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0)])
    with src.open("a", encoding="utf-8") as handle:
        handle.write('{"keyframe": {"video_id": {"value": "L21_V001"}, "keyframe_ordinal": 2}, "pts_time": 0.4, "fps": 25.0, "frame_index": -1}\n')
    with pytest.raises(ValueError):
        deduplicate_keyframes(src, tmp_path / "out.ndjson")


def test_missing_input_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        deduplicate_keyframes(tmp_path / "missing.ndjson", tmp_path / "out.ndjson")


def test_invalid_max_examples_raises(tmp_path: Path) -> None:
    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0)])
    with pytest.raises(ValueError, match="max_examples"):
        deduplicate_keyframes(src, tmp_path / "out.ndjson", max_examples=-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_end_to_end(tmp_path: Path) -> None:
    import subprocess
    import sys

    src = tmp_path / "src.ndjson"
    _write_records(
        src,
        [
            ("L21_V006", 1, 0, 0.0),
            ("L21_V006", 2, 0, 0.4),
            ("L21_V007", 1, 0, 0.0),
            ("L21_V007", 2, 0, 0.4),
            ("L21_V008", 1, 0, 0.0),
        ],
    )
    out = tmp_path / "out.ndjson"
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "deduplicate_keyframes.py"),
            "--input",
            str(src),
            "--output",
            str(out),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Original records:   5" in result.stdout
    assert "Unique records:     3" in result.stdout
    assert "Removed duplicates: 2" in result.stdout
    assert "Strategy:           keep_first" in result.stdout
    assert f"Output:             {out.resolve()}" in result.stdout
    assert _read_keys(out) == [("L21_V006", 0), ("L21_V007", 0), ("L21_V008", 0)]
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["removed_duplicates"] == 2


def test_cli_default_report_path(tmp_path: Path) -> None:
    import subprocess
    import sys

    src = tmp_path / "src.ndjson"
    _write_records(src, [("L21_V001", 1, 0, 0.0), ("L21_V001", 2, 0, 0.4)])
    out = tmp_path / "out.ndjson"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "deduplicate_keyframes.py"),
            "--input",
            str(src),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "dedup_report.json").is_file()


def test_cli_missing_input_fails(tmp_path: Path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "deduplicate_keyframes.py"),
            "--input",
            str(tmp_path / "missing.ndjson"),
            "--output",
            str(tmp_path / "out.ndjson"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr
    assert not (tmp_path / "out.ndjson").exists()