"""Tests for deterministic OCR shard merging."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.ocr_generation import (
    MANIFEST_FILENAME,
    OCR_FILENAME,
    OcrLine,
    generate_ocr,
    read_ocr_artifact,
)
from aic2026.ocr_merge import (
    MergeError,
    expected_shard_record_count,
    merge_ocr_shards,
    order_shards,
)

#: (video_id, keyframe_ordinal, frame_index) rows in canonical row order.
LAYOUT_RECORDS = [
    ("L21_V001", 1, 0),
    ("L21_V001", 2, 10),
    ("L21_V001", 3, 20),
    ("L21_V002", 1, 0),
    ("L21_V002", 2, 10),
    ("L22_V001", 1, 0),
    ("L22_V001", 2, 10),
]

NUM_RECORDS = len(LAYOUT_RECORDS)  # 7 rows


def _write_frames(path: Path, records: list[tuple[str, int, int]]) -> None:
    """Write a canonical frames.ndjson for the given ``(video, ordinal, frame_index)`` rows."""
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinal, frame_index in records:
            record = {
                "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                "pts_time": float(frame_index) / 25.0,
                "fps": 25.0,
                "frame_index": frame_index,
            }
            handle.write(json.dumps(record) + "\n")


def _make_keyframes(root: Path) -> Path:
    """Create keyframe images for every record so OCR records are 'ok'."""
    images = root / "images"
    for vid, ordinal, _ in LAYOUT_RECORDS:
        vid_dir = images / vid
        vid_dir.mkdir(parents=True, exist_ok=True)
        (vid_dir / f"{ordinal:06d}.jpg").write_bytes(bytes([1, ordinal, 0]))
    return root


class _StubOcrEngine:
    """Deterministic OCR engine deriving text from image bytes."""

    device = "cpu"
    model_name = "stub:ocr"
    languages = ("vi", "en")

    def detect_text(self, images: list[object]) -> list[list[OcrLine]]:
        out: list[list[OcrLine]] = []
        for image in images:
            data = bytes(image)  # type: ignore[arg-type]
            out.append(
                [
                    OcrLine(
                        text=f"TEXT-v{data[0]}-k{data[1]}",
                        confidence=0.96,
                        bbox=(float(data[0]), float(data[1]), float(data[0]) + 100.0, float(data[1]) + 20.0),
                    )
                ]
            )
        return out


def _image_loader(path: Path) -> bytes:
    return path.read_bytes()


def _make_ocr_shard(
    shard_dir: Path,
    shard_frames: Path,
    keyframes_root: Path,
) -> None:
    """Produce one completed OCR shard from a subset frames file.

    Mirrors the distributed runner: the shard's keyframes view symlinks the
    shard frames and the shared images collection, so the OCR manifest records
    ``source_frames`` as the shard frames file path.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    view = shard_dir.parent / f"view_{shard_dir.name}"
    view.mkdir(parents=True, exist_ok=True)
    frames_link = view / "frames.ndjson"
    images_link = view / "images"
    if frames_link.is_symlink() or frames_link.exists():
        frames_link.unlink()
    if images_link.is_symlink() or images_link.exists():
        images_link.unlink()
    frames_link.symlink_to(shard_frames.resolve())
    images_link.symlink_to((keyframes_root / "images").resolve(), target_is_directory=True)

    generate_ocr(
        keyframes_dir=view,
        output_dir=shard_dir,
        ocr_engine=_StubOcrEngine(),
        image_loader=_image_loader,
    )


@pytest.fixture
def keyframes(tmp_path: Path) -> Path:
    return _make_keyframes(tmp_path / "keyframes")


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    path = tmp_path / "frames.ndjson"
    _write_frames(path, LAYOUT_RECORDS)
    return path


def _split_records(num_shards: int) -> list[list[tuple[str, int, int]]]:
    """Split LAYOUT_RECORDS into contiguous row ranges (balanced, like the runner)."""
    base = (NUM_RECORDS + num_shards - 1) // num_shards
    ranges = [(i * base, min((i + 1) * base, NUM_RECORDS)) for i in range(num_shards)]
    return [LAYOUT_RECORDS[start:end] for start, end in ranges]


def _make_shards(tmp_path: Path, keyframes: Path, num_shards: int = 3) -> list[Path]:
    """Create ``num_shards`` completed OCR shards covering LAYOUT_RECORDS."""
    dirs: list[Path] = []
    for shard_id, records in enumerate(_split_records(num_shards)):
        shard_dir = tmp_path / f"shard_{shard_id:03d}"
        shard_frames = tmp_path / f"frames_{shard_id:03d}.ndjson"
        _write_frames(shard_frames, records)
        _make_ocr_shard(shard_dir, shard_frames, keyframes)
        dirs.append(shard_dir)
    return dirs


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_keys(rows: list[dict]) -> list[tuple[str, int, int]]:
    return [
        (r["video_id"]["value"], int(r["keyframe_ordinal"]), int(r["frame_index"]))
        for r in rows
    ]


def _merge(shards: list[Path], tmp_path: Path, **kwargs: object):
    return merge_ocr_shards(shards, tmp_path / "merged", **kwargs)


# ------------------------------------------------------------------ splitting equivalence


def test_expected_shard_record_count_matches_runner() -> None:
    from modal_ocr_distributed import shard_row_bounds

    for total, num_shards in ((148322, 8), (7, 3), (10, 3), (100, 7), (1, 4), (0, 3)):
        for shard_id in range(num_shards):
            start, end = shard_row_bounds(total, num_shards, shard_id)
            assert expected_shard_record_count(total, num_shards, shard_id) == end - start


def test_expected_shard_record_count_balanced() -> None:
    counts = [expected_shard_record_count(NUM_RECORDS, 3, i) for i in range(3)]
    assert counts == [3, 3, 1]
    assert sum(counts) == NUM_RECORDS


def test_expected_shard_record_count_validates() -> None:
    with pytest.raises(ValueError):
        expected_shard_record_count(7, 0, 0)
    with pytest.raises(ValueError):
        expected_shard_record_count(7, 3, 3)


# ------------------------------------------------------------------ ordering


def test_order_shards_by_index() -> None:
    from aic2026.ocr_merge import _shard_id_from_dirname

    dirs = [Path("shard_002"), Path("shard_000"), Path("shard_001")]
    assert [d.name for d in order_shards(dirs)] == ["shard_000", "shard_001", "shard_002"]
    assert [_shard_id_from_dirname(d) for d in dirs] == [2, 0, 1]
    with pytest.raises(MergeError, match="shard_XXX"):
        order_shards([Path("worker_000")])


def test_order_shards_rejects_duplicate_indices() -> None:
    with pytest.raises(MergeError, match="duplicate shard"):
        order_shards([Path("shard_000"), Path("shard_000")])


def test_order_shards_rejects_empty() -> None:
    with pytest.raises(MergeError, match="at least one"):
        order_shards([])


# ------------------------------------------------------------------ happy path


def test_merge_concatenates_shards_in_order(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    manifest = _merge(shards, tmp_path, source_frames=canonical)

    assert manifest.num_records == NUM_RECORDS
    assert manifest.num_videos == 3
    assert manifest.num_failed == 0
    assert manifest.schema_version == "1.0"
    assert manifest.backend == "paddleocr"
    assert manifest.source_frames == str(canonical.resolve())

    rows = _rows(tmp_path / "merged" / OCR_FILENAME)
    assert len(rows) == NUM_RECORDS
    assert _record_keys(rows) == [(vid, ord_, idx) for vid, ord_, idx in LAYOUT_RECORDS]
    assert (tmp_path / "merged" / MANIFEST_FILENAME).is_file()

    read_manifest, records = read_ocr_artifact(tmp_path / "merged")
    assert read_manifest == manifest
    assert len(records) == NUM_RECORDS


def test_merge_records_preserve_schema(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    _merge(shards, tmp_path, source_frames=canonical)
    rows = _rows(tmp_path / "merged" / OCR_FILENAME)
    for row in rows:
        for key in (
            "video_id", "keyframe_ordinal", "frame_index", "ocr_text",
            "ocr_text_normalized", "ocr_lines", "status",
        ):
            assert key in row
        assert row["status"] == "ok"


def test_merge_deterministic_order(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    merge_ocr_shards(list(reversed(shards)), tmp_path / "merged-a")
    merge_ocr_shards(shards, tmp_path / "merged-b")
    assert _rows(tmp_path / "merged-a" / OCR_FILENAME) == _rows(tmp_path / "merged-b" / OCR_FILENAME)


def test_merge_without_source_frames_still_concatenates(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    manifest = _merge(shards, tmp_path)
    assert manifest.num_records == NUM_RECORDS
    assert len(_rows(tmp_path / "merged" / OCR_FILENAME)) == NUM_RECORDS


# ------------------------------------------------------------------ rejection


def test_merge_rejects_missing_manifest(tmp_path: Path) -> None:
    incomplete = tmp_path / "shard_000"
    incomplete.mkdir()
    with pytest.raises(MergeError, match="not complete"):
        _merge([incomplete], tmp_path)


def test_merge_rejects_inconsistent_artifact(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes, num_shards=1)
    with (shards[0] / OCR_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_rows(shards[0] / OCR_FILENAME)[0]) + "\n")
    with pytest.raises(MergeError, match="inconsistent"):
        _merge(shards, tmp_path / "out")


def test_merge_rejects_mixed_identity(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    payload = json.loads((shards[1] / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["model"] = "paddleocr:PP-OCRv5_server"
    (shards[1] / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MergeError, match="model"):
        _merge(shards, tmp_path / "out")


def test_merge_rejects_duplicate_keyframes(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    # A fourth shard whose rows overlap shard_000's first row.
    dup_frames = tmp_path / "dup_frames.ndjson"
    _write_frames(dup_frames, [LAYOUT_RECORDS[0]])
    overlapping = tmp_path / "shard_003"
    _make_ocr_shard(overlapping, dup_frames, keyframes)
    shards.append(overlapping)
    with pytest.raises(MergeError, match="duplicate keyframe"):
        _merge(shards, tmp_path / "out")


def test_merge_rejects_missing_shard_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _merge([tmp_path / "shard_000"], tmp_path / "out")


def test_merge_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="at least one"):
        _merge([], tmp_path / "out")


# ------------------------------------------------------------------ coverage validation


def test_merge_rejects_total_row_count_mismatch(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    inflated = tmp_path / "frames_inflated.ndjson"
    _write_frames(inflated, LAYOUT_RECORDS + [("L22_V002", 1, 0)])
    with pytest.raises(MergeError, match="contiguous coverage|canonical rows"):
        _merge(shards, tmp_path / "out", source_frames=inflated)


def test_merge_rejects_non_contiguous_shard_indices(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    del shards[1]  # drop shard_001 -> indices [0, 2]
    with pytest.raises(MergeError, match="not contiguous"):
        _merge(shards, tmp_path / "out", source_frames=canonical)


def test_merge_rejects_shard_source_mismatch(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    rows = _rows(shards[0] / OCR_FILENAME)
    tampered = dict(rows[0])
    tampered["frame_index"] = 999
    rows[0] = tampered
    with (shards[0] / OCR_FILENAME).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with pytest.raises(MergeError, match="do not align"):
        _merge(shards, tmp_path / "out", source_frames=canonical)


def test_merge_rejects_missing_source_frames(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    with pytest.raises(FileNotFoundError, match="canonical source frames"):
        _merge(shards, tmp_path / "out", source_frames=tmp_path / "missing.ndjson")


# ------------------------------------------------------------------ output safety


def test_merge_refuses_existing_output(keyframes: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    _merge(shards, tmp_path)
    with pytest.raises(MergeError, match="already exists"):
        merge_ocr_shards(shards, tmp_path / "merged")
    merge_ocr_shards(shards, tmp_path / "merged", overwrite=True)


# ------------------------------------------------------------------ CLI


def _merge_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "merge_ocr_shards.py"


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_merge_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--shards" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--source-frames" in result.stdout


def test_cli_missing_args_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_merge_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--shards" in result.stderr


def test_cli_end_to_end(keyframes: Path, canonical: Path, tmp_path: Path) -> None:
    shards = _make_shards(tmp_path, keyframes)
    merged = tmp_path / "merged-cli"
    result = subprocess.run(
        [
            sys.executable,
            str(_merge_script()),
            "--shards",
            *[str(s) for s in shards],
            "--output-dir",
            str(merged),
            "--source-frames",
            str(canonical),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"Merge complete: {NUM_RECORDS} records from 3 shards" in result.stdout
    assert len(_rows(merged / OCR_FILENAME)) == NUM_RECORDS