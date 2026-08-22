"""Tests for the OCR representation-v2 transform (Phase 5B)."""

from __future__ import annotations

import numpy as np
import pytest

from aic2026.canonical import VideoId
from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_generation import KeyframeOcr, OcrLine, normalize_ocr_text
from aic2026.ocr_representation import (
    RepresentationConfig,
    build_cleaned_representation,
    clean_chunk,
    collect_video_persistent_lines,
    unit_max_scores,
)


def _record(
    video_id: str,
    ordinal: int,
    frame_index: int,
    lines: tuple[str, ...],
) -> KeyframeOcr:
    """Return one ok OCR record with per-line texts."""
    return KeyframeOcr(
        video_id=VideoId(value=video_id),
        keyframe_ordinal=ordinal,
        frame_index=frame_index,
        ocr_text=" ".join(lines),
        ocr_text_normalized=" ".join(normalize_ocr_text(line) for line in lines),
        ocr_lines=tuple(
            OcrLine(text=line, confidence=1.0, bbox=(0.0, 0.0, 1.0, 1.0)) for line in lines
        ),
        status="ok",
    )


def _chunk(chunk_id: str, video_id: str, ordinals: tuple[int, ...]) -> OcrChunk:
    """Return one chunk covering the given ordinals (frames = 10*ordinal)."""
    frames = tuple(10 * ordinal for ordinal in ordinals)
    return OcrChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        frame_start=frames[0],
        frame_end=frames[-1],
        representative_frame_index=frames[len(frames) // 2],
        keyframe_ordinals=ordinals,
        frame_indices=frames,
        text="",
        source_count=len(frames),
    )


class TestPersistentDetection:
    def test_video_local_high_coverage_detected(self) -> None:
        records = [
            _record("L21_V001", ordinal, ordinal * 10, ("WATERMARK", f"content {ordinal}"))
            for ordinal in range(1, 11)
        ]
        persistent = collect_video_persistent_lines(records, min_coverage=0.5, min_records=3)
        assert "watermark" in persistent
        assert "content 5" not in persistent

    def test_same_line_in_other_videos_is_not_removed(self) -> None:
        # WATERMARK is frequent in VA but appears once in VB; VB must not
        # flag it. Detection runs per video.
        records_va = [
            _record("L21_V001", ordinal, ordinal * 10, ("WATERMARK",)) for ordinal in range(1, 11)
        ]
        records_vb = [
            _record("L21_V002", 1, 10, ("WATERMARK", "unique")),
            _record("L21_V002", 2, 20, ("other",)),
            _record("L21_V002", 3, 30, ("more",)),
        ]
        assert collect_video_persistent_lines(records_va, min_coverage=0.5, min_records=3) == {"watermark"}
        assert collect_video_persistent_lines(records_vb, min_coverage=0.5, min_records=3) == set()

    def test_global_frequency_alone_never_removes(self) -> None:
        # The line appears in every record of a SMALL video but below the
        # min_records floor -> kept.
        records = [_record("L21_V001", ordinal, ordinal * 10, ("TITLE",)) for ordinal in range(1, 3)]
        assert collect_video_persistent_lines(records, min_coverage=0.5, min_records=3) == set()

    def test_min_records_floor(self) -> None:
        records = [_record("L21_V001", i, i * 10, ("X", "Y")) for i in range(1, 6)]
        # Both X and Y appear in all 5 records; floor of 6 removes nothing.
        assert collect_video_persistent_lines(records, min_coverage=0.5, min_records=6) == set()


class TestGarbageFilter:
    CONFIG = RepresentationConfig()

    def test_conservative_rules(self) -> None:
        records = [
            _record(
                "L21_V001", 1, 10,
                ("!!!", "8", "2024", "a", "bánh mì", "#", "HTV"),
            )
        ]
        chunk = _chunk("c", "L21_V001", (1,))
        cleaned = clean_chunk(chunk, records, set(), self.CONFIG)
        texts = list(cleaned.units)
        assert "2024" in texts          # digits-as-evidence kept
        assert "a" in texts             # short normal word kept
        assert "bánh mì" in texts       # accented text kept
        assert "!!!" not in texts       # pure symbols dropped
        assert "#" not in texts         # single symbol dropped
        assert "8" in texts or "8" in [p.text for p in cleaned.removed]
        reasons = {(p.text, p.reason) for p in cleaned.removed}
        assert ("!!!", "garbage") in reasons

    def test_single_digit_without_letters_dropped(self) -> None:
        records = [_record("L21_V001", 1, 10, ("8",))]
        cleaned = clean_chunk(_chunk("c", "L21_V001", (1,)), records, set(), self.CONFIG)
        assert cleaned.units == ()
        assert cleaned.removed[0].reason == "garbage"


class TestCleanChunk:
    def test_provenance_and_order(self) -> None:
        records = [
            _record("L21_V001", 1, 10, ("THANHNIEN", "ÔN THI VẬT LÍ")),
            _record("L21_V001", 2, 20, ("THANHNIEN", "đề số 3")),
        ]
        config = RepresentationConfig()
        persistent = collect_video_persistent_lines(records, min_coverage=0.5, min_records=2)
        assert persistent == {"thanhnien"}
        cleaned = clean_chunk(_chunk("c", "L21_V001", (1, 2)), records, persistent, config)
        assert cleaned.units == ("ÔN THI VẬT LÍ", "đề số 3")
        assert cleaned.text == "ÔN THI VẬT LÍ đề số 3"
        removed_persistent = [p for p in cleaned.removed if p.reason == "persistent"]
        assert len(removed_persistent) >= 1 and removed_persistent[0].text == "THANHNIEN"
        # Identity fields preserved from the source chunk.
        assert cleaned.chunk_id == "c" and cleaned.video_id == "L21_V001"
        assert cleaned.frame_start == 10 and cleaned.frame_end == 20

    def test_unit_cap_first_seen_order(self) -> None:
        records = [_record("L21_V001", 1, 10, tuple(f"w{i}" for i in range(20)))]
        config = RepresentationConfig(units_per_chunk=5)
        cleaned = clean_chunk(_chunk("c", "L21_V001", (1,)), records, set(), config)
        assert cleaned.units == tuple(f"w{i}" for i in range(5))
        cap_reasons = [p.reason for p in cleaned.removed if p.reason == "cap"]
        assert len(cap_reasons) == 15

    def test_empty_after_cleaning(self) -> None:
        records = [_record("L21_V001", 1, 10, ("WATERMARK",))]
        cleaned = clean_chunk(
            _chunk("c", "L21_V001", (1,)), records, {"watermark"}, RepresentationConfig()
        )
        assert cleaned.text == "" and cleaned.units == ()

    def test_deterministic(self) -> None:
        records = [
            _record("L21_V001", 1, 10, ("alpha", "beta")),
            _record("L21_V001", 2, 20, ("gamma",)),
        ]
        config = RepresentationConfig()
        a = clean_chunk(_chunk("c", "L21_V001", (1, 2)), records, set(), config)
        b = clean_chunk(_chunk("c", "L21_V001", (1, 2)), records, set(), config)
        assert a == b


class TestBuildRepresentation:
    def test_end_to_end_and_diagnostics(self) -> None:
        chunks = [_chunk("c0", "L21_V001", (1, 2)), _chunk("c1", "L21_V001", (3,))]
        records = [
            _record("L21_V001", 1, 10, ("WATERMARK", "alpha")),
            _record("L21_V001", 2, 20, ("WATERMARK", "beta")),
            _record("L21_V001", 3, 30, ("WATERMARK", "gamma")),
        ]
        cleaned, diagnostics = build_cleaned_representation(
            chunks, records, RepresentationConfig(persistent_min_coverage=0.5, persistent_min_records=2)
        )
        assert [c.chunk_id for c in cleaned] == ["c0", "c1"]
        assert diagnostics["num_chunks"] == 2
        assert diagnostics["removed_reason_counts"].get("persistent", 0) >= 2
        assert cleaned[0].units == ("alpha", "beta")

    def test_duplicate_chunk_ids_rejected(self) -> None:
        chunks = [_chunk("dup", "L21_V001", (1,)), _chunk("dup", "L21_V001", (2,))]
        with pytest.raises(Exception, match="duplicate chunk ids"):
            build_cleaned_representation(chunks, [], RepresentationConfig())


class TestUnitMaxScoring:
    def test_max_over_units(self) -> None:
        unit_matrix = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.71, 0.71]], dtype=np.float32
        )
        norms = np.linalg.norm(unit_matrix, axis=1).astype(np.float32)
        indices = {0: [0, 1], 1: [2]}
        scores = unit_max_scores(np.array([1.0, 0.0]), unit_matrix, norms, indices, 3)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
        assert scores[2] == -np.inf  # chunk without units

    def test_dim_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            unit_max_scores(
                np.array([1.0, 2.0, 3.0]),
                np.zeros((1, 2), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                {},
                1,
            )

    def test_tie_break_deterministic_via_row_order(self) -> None:
        matrix = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
        s1 = unit_max_scores(np.array([0.8, 0.6]), matrix, norms, {0: [0, 1]}, 2)
        s2 = unit_max_scores(np.array([0.8, 0.6]), matrix, norms, {0: [0, 1]}, 2)
        assert float(s1[0]) == float(s2[0])
