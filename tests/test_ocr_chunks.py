"""Tests for deterministic temporal OCR chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.canonical import VideoId
from aic2026.ocr_chunks import (
    MANIFEST_FILENAME,
    build_ocr_chunks,
    read_ocr_chunks,
    validate_ocr_chunks,
    write_ocr_chunks,
)
from aic2026.ocr_generation import (
    MANIFEST_FILENAME as OCR_MANIFEST_FILENAME,
    OCR_FILENAME,
    KeyframeOcr,
    OcrLine,
    OcrManifest,
    normalize_ocr_text,
    write_ocr_records,
)


def _rec(
    video_id: str,
    ordinal: int,
    frame_index: int,
    lines: tuple[str, ...] = (),
    status: str = "ok",
) -> KeyframeOcr:
    """Return one OCR record with the given identity and text lines."""
    return KeyframeOcr(
        video_id=VideoId(value=video_id),
        keyframe_ordinal=ordinal,
        frame_index=frame_index,
        ocr_text=" ".join(lines),
        ocr_text_normalized=" ".join(normalize_ocr_text(line) for line in lines),
        ocr_lines=tuple(
            OcrLine(text=line, confidence=1.0, bbox=(0.0, 0.0, 1.0, 1.0)) for line in lines
        ),
        status=status,
        error="boom" if status == "error" else None,
    )


def _build(records, **kwargs):
    """Build chunks with a fixed timestamp so outputs are directly comparable."""
    kwargs.setdefault("artifact_id", "test")
    kwargs.setdefault("source_ocr_artifact", "source")
    kwargs.setdefault("created_at", "2026-01-01T00:00:00+00:00")
    return build_ocr_chunks(records, **kwargs)


class TestGrouping:
    def test_different_videos_never_share_a_chunk(self) -> None:
        records = [
            _rec("L21_V001", 1, 100, ("A",)),
            _rec("L21_V002", 1, 100, ("B",)),
        ]
        _, chunks = _build(records)
        assert len(chunks) == 2
        assert {chunk.video_id for chunk in chunks} == {"L21_V001", "L21_V002"}
        assert all(chunk.source_count == 1 for chunk in chunks)

    def test_nearby_frames_are_grouped(self) -> None:
        records = [
            _rec("L21_V001", ordinal, ordinal * 60, (f"T{ordinal}",))
            for ordinal in range(1, 6)
        ]
        _, chunks = _build(records)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.frame_start == 60
        assert chunk.frame_end == 300
        assert chunk.source_count == 5
        assert chunk.keyframe_ordinals == (1, 2, 3, 4, 5)

    def test_distant_frames_are_separated(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("early",)),
            _rec("L21_V001", 2, 10_000, ("late",)),
        ]
        _, chunks = _build(records)
        assert [chunk.chunk_id for chunk in chunks] == [
            "L21_V001-ocr-00",
            "L21_V001-ocr-01",
        ]

    def test_window_size_cap_splits_long_runs(self) -> None:
        records = [
            _rec("L21_V001", ordinal, ordinal, ("T",))
            for ordinal in range(1, 26)
        ]
        _, chunks = _build(records, max_records=20)
        assert [chunk.source_count for chunk in chunks] == [20, 5]
        assert chunks[0].frame_indices == tuple(range(1, 21))
        assert chunks[1].frame_indices == tuple(range(21, 26))


class TestTextAggregation:
    def test_duplicate_text_is_collapsed(self) -> None:
        records = [
            _rec("L25_V085", 1, 232, ("THANHNIEN",)),
            _rec("L25_V085", 2, 532, ("THANHNIEN",)),
            _rec("L25_V085", 3, 832, ("THANHNIEN",)),
            _rec("L25_V085", 4, 1132, ("ÔN THI VẬT LÍ NĂM 2024",)),
            _rec("L25_V085", 5, 1432, ("ÔN THI VẬT LÍ NĂM 2024",)),
        ]
        _, chunks = _build(records)
        assert len(chunks) == 1
        assert chunks[0].text == "THANHNIEN ÔN THI VẬT LÍ NĂM 2024"

    def test_case_variants_collapse_to_first_seen_form(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("Thanh Nien",)),
            _rec("L21_V001", 2, 50, ("THANH NIEN",)),
        ]
        _, chunks = _build(records)
        assert chunks[0].text == "Thanh Nien"

    def test_distinct_text_preserved_in_reading_order(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("alpha", "beta")),
            _rec("L21_V001", 2, 40, ("gamma",)),
            _rec("L21_V001", 3, 80, ("beta", "delta")),
        ]
        _, chunks = _build(records)
        assert chunks[0].text == "alpha beta gamma delta"

    def test_error_and_empty_records_produce_no_text(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, status="error"),
            _rec("L21_V001", 2, 40, ()),
            _rec("L21_V001", 3, 80, ("real",)),
        ]
        _, chunks = _build(records)
        assert len(chunks) == 1
        assert chunks[0].text == "real"
        assert chunks[0].source_count == 3


class TestProvenanceAndCoverage:
    def test_representative_frame_is_middle_source_frame(self) -> None:
        records = [
            _rec("L21_V001", ordinal, ordinal * 10, ("T",))
            for ordinal in range(1, 6)
        ]
        _, chunks = _build(records)
        assert chunks[0].representative_frame_index == 30
        assert chunks[0].frame_start <= chunks[0].representative_frame_index <= chunks[0].frame_end

    def test_coverage_accounts_for_every_record(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("a",)),
            _rec("L21_V001", 2, 40, status="error"),
            _rec("L21_V001", 3, 80, ()),
            _rec("L22_V001", 1, 0, status="error"),
            _rec("L22_V001", 2, 40, status="error"),
        ]
        manifest, chunks = _build(records)
        assert len(chunks) == 1
        assert chunks[0].video_id == "L21_V001"
        assert manifest.num_source_records == 5
        assert manifest.num_source_records_chunked == 3
        assert manifest.num_source_records_skipped == 2
        assert (
            manifest.num_source_records_chunked + manifest.num_source_records_skipped
            == manifest.num_source_records
        )

    def test_duplicate_source_records_are_rejected(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("a",)),
            _rec("L21_V001", 1, 0, ("a",)),
        ]
        with pytest.raises(ValueError, match="duplicate source OCR record"):
            _build(records)

    def test_no_duplicate_keys_across_chunks(self) -> None:
        records = [
            _rec("L21_V001", ordinal, ordinal * 10, ("T",))
            for ordinal in range(1, 46)
        ]
        _, chunks = _build(records, max_records=20)
        keys = [
            (chunk.video_id, ordinal, frame_index)
            for chunk in chunks
            for ordinal, frame_index in zip(chunk.keyframe_ordinals, chunk.frame_indices)
        ]
        assert len(keys) == len(set(keys)) == 45


class TestDeterminism:
    def test_output_is_deterministic(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("a",)),
            _rec("L21_V002", 1, 500, ("b",)),
            _rec("L21_V001", 2, 90, ("c",)),
        ]
        manifest_a, chunks_a = _build(records)
        manifest_b, chunks_b = _build(list(reversed(records)))
        assert manifest_a.to_dict() == manifest_b.to_dict()
        assert [chunk.to_dict() for chunk in chunks_a] == [
            chunk.to_dict() for chunk in chunks_b
        ]


class TestArtifactRoundTrip:
    def test_write_read_round_trip(self, tmp_path: Path) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("a",)),
            _rec("L21_V001", 2, 40, ("b",)),
            _rec("L22_V001", 1, 0, ("c",)),
        ]
        manifest, chunks = _build(records)
        output = write_ocr_chunks(tmp_path / "chunks", manifest, chunks)
        read_manifest, read_chunks = read_ocr_chunks(output)
        assert read_manifest == manifest
        assert read_chunks == chunks

    def test_read_rejects_inconsistent_manifest(self, tmp_path: Path) -> None:
        records = [_rec("L21_V001", 1, 0, ("a",))]
        manifest, chunks = _build(records)
        output = write_ocr_chunks(tmp_path / "chunks", manifest, chunks)
        broken = dict(manifest.to_dict(), num_chunks=99)
        (output / MANIFEST_FILENAME).write_text(json.dumps(broken) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="num_chunks"):
            read_ocr_chunks(output)

    def test_validate_reports_non_contiguous_video_blocks(self) -> None:
        records = [
            _rec("L21_V001", 1, 0, ("a",)),
            _rec("L22_V001", 1, 0, ("b",)),
            _rec("L21_V002", 1, 0, ("c",)),
            _rec("L21_V001", 2, 9000, ("d",)),
        ]
        manifest, chunks = _build(records)
        reordered = [chunks[0], chunks[3], chunks[1], chunks[2]]
        problems = validate_ocr_chunks(manifest, reordered)
        assert any("non-contiguous" in problem for problem in problems)


def _write_source_artifact(root: Path, records: list[KeyframeOcr]) -> Path:
    """Write a minimal completed OCR artifact holding ``records``."""
    root.mkdir(parents=True, exist_ok=True)
    write_ocr_records(root / OCR_FILENAME, records)
    manifest = OcrManifest(
        schema_version="1.0",
        artifact_id="src",
        backend="paddleocr",
        model="paddleocr:test",
        languages=("vi",),
        device="cpu",
        batch_size=1,
        source_frames="frames.ndjson",
        num_records=len(records),
        num_videos=len({record.video_id.value for record in records}),
        num_failed=sum(1 for record in records if record.status == "error"),
        canonical_schema_version="1.0",
        created_at="2026-01-01T00:00:00+00:00",
    )
    (root / OCR_MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict()) + "\n", encoding="utf-8"
    )
    return root


class TestCli:
    def test_cli_builds_and_validates(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        records = [
            _rec("L21_V001", 1, 0, ("hello",)),
            _rec("L21_V001", 2, 40, ("world",)),
            _rec("L22_V001", 1, 0, status="error"),
        ]
        source = _write_source_artifact(tmp_path / "ocr", records)
        output = tmp_path / "chunks"
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_ocr_chunks.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--ocr-artifact",
                str(source),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest, chunks = read_ocr_chunks(output)
        assert manifest.num_chunks == 1
        assert chunks[0].text == "hello world"
        assert "Built 1 chunks" in result.stdout

        rerun = subprocess.run(
            [
                sys.executable,
                str(script),
                "--ocr-artifact",
                str(source),
                "--output-dir",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert rerun.returncode == 1
        assert "already exists" in rerun.stderr

        overwrite = subprocess.run(
            [
                sys.executable,
                str(script),
                "--ocr-artifact",
                str(source),
                "--output-dir",
                str(output),
                "--overwrite",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert overwrite.returncode == 0, overwrite.stderr
