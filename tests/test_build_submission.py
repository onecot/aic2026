"""Focused tests for the BTC submission exporter.

Covers filename/task parsing, CSV format, QA 100-char validation,
TRAKE frame-count/order validation, zip structure, and one end-to-end
synthetic submission.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest

from aic2026.btc_queries import (
    QueryFileInfo,
    enumerate_query_files,
    normalize_query_text,
    parse_query_filename,
    read_query_text,
)
from aic2026.submission import (
    KIS_MAX_ROWS,
    QA_ANSWER_MAX_CHARS,
    QA_MAX_ROWS,
    TRAKE_MAX_ROWS,
    QueryMatchError,
    create_submission_zip,
    export_kis_csv,
    export_qa_csv,
    export_trake_csv,
    find_unique_match,
    select_trake_rows,
    _validate_kis_rows,
    _validate_qa_rows,
    _validate_trake_rows,
)
from aic2026.kis_prediction_runner import QueryPredictionResult, RankedKisPrediction
from aic2026.qa.qa_types import QaPrediction, QaQueryResult
from aic2026.retrieval import RetrievalCandidate
from aic2026.trake import TrakeEventResult, TrakeQueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kis_prediction(rank: int, video_id: str, frame_index: int) -> RankedKisPrediction:
    return RankedKisPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=1.0 / rank)


def _kis_result(query_id: str, query_text: str, n: int = 5) -> QueryPredictionResult:
    return QueryPredictionResult(
        query_id=query_id,
        query_text=query_text,
        predictions=tuple(
            _kis_prediction(i + 1, f"L21_V{i:03d}", i * 100)
            for i in range(n)
        ),
    )


def _qa_prediction(rank: int, video_id: str, frame_index: int) -> QaPrediction:
    return QaPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=1.0 / rank)


def _qa_result(
    query_id: str,
    query_text: str,
    answer: str = "red chili peppers",
    has_evidence: bool = True,
) -> QaQueryResult:
    predictions = (_qa_prediction(1, "L26_V134", 2040),)
    evidence = (_qa_prediction(1, "L26_V134", 2040),) if has_evidence else None
    return QaQueryResult(
        query_id=query_id,
        query_text=query_text,
        answer=answer,
        predictions=predictions,
        video_candidates=(),
        window_candidates=(),
        answer_evidence=evidence,
    )


def _trake_event_result(
    event_id: str,
    event_text: str,
    candidates: list[tuple[str, int, float]],
) -> TrakeEventResult:
    """Build a TrakeEventResult from (video_id, frame_index, score) tuples."""
    preds = []
    for rank, (vid, fi, score) in enumerate(candidates, start=1):
        preds.append(
            RetrievalCandidate(
                rank=rank,
                video_id=vid,
                keyframe_ordinal=fi,
                frame_index=fi,
                pts_time=float(fi),
                score=score,
            )
        )
    return TrakeEventResult(
        event_id=event_id,
        event_text=event_text,
        predictions=tuple(preds),
        windows=(),
    )


def _trake_result(query_id: str, query_text: str, events: list[TrakeEventResult]) -> TrakeQueryResult:
    return TrakeQueryResult(
        query_id=query_id,
        query_text=query_text,
        events=tuple(events),
    )


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


# ---------------------------------------------------------------------------
# 1. Filename/task parsing
# ---------------------------------------------------------------------------


class TestParseQueryFilename:
    def test_parses_kis_filename(self) -> None:
        info = parse_query_filename(Path("query-p1-1-kis.txt"))
        assert info.name == "p1-1"
        assert info.task == "kis"

    def test_parses_qa_filename(self) -> None:
        info = parse_query_filename(Path("query-p1-22-qa.txt"))
        assert info.name == "p1-22"
        assert info.task == "qa"

    def test_parses_trake_filename(self) -> None:
        info = parse_query_filename(Path("query-p1-4-trake.txt"))
        assert info.name == "p1-4"
        assert info.task == "trake"

    def test_rejects_invalid_filename(self) -> None:
        with pytest.raises(ValueError, match="does not match pattern"):
            parse_query_filename(Path("invalid-file.txt"))

    def test_rejects_wrong_suffix(self) -> None:
        with pytest.raises(ValueError, match="does not match pattern"):
            parse_query_filename(Path("query-p1-1-txt.txt"))

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="does not match pattern"):
            parse_query_filename(Path("query--kis.txt"))


# ---------------------------------------------------------------------------
# 1b. Query file enumeration
# ---------------------------------------------------------------------------


class TestEnumerateQueryFiles:
    def test_enumerates_matching_files_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "query-p1-1-kis.txt").write_text("a", encoding="utf-8")
        (tmp_path / "query-p1-4-trake.txt").write_text("b", encoding="utf-8")
        (tmp_path / "query-p1-22-qa.txt").write_text("c", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        (tmp_path / "kis.csv").write_text("V,0", encoding="utf-8")

        infos = enumerate_query_files(tmp_path)
        assert [i.name for i in infos] == ["p1-1", "p1-22", "p1-4"]
        assert [i.task for i in infos] == ["kis", "qa", "trake"]

    def test_csv_name_derives_from_filename(self, tmp_path: Path) -> None:
        info = QueryFileInfo(path=tmp_path / "query-p1-1-kis.txt", name="p1-1", task="kis")
        assert info.csv_name == "query-p1-1-kis.csv"

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            enumerate_query_files(tmp_path / "nope")

    def test_raises_when_no_query_files(self, tmp_path: Path) -> None:
        (tmp_path / "irrelevant.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="no 'query-<name>-<task>.txt'"):
            enumerate_query_files(tmp_path)


# ---------------------------------------------------------------------------
# 1c. Strict query matching
# ---------------------------------------------------------------------------


class TestQueryMatching:
    def test_normalize_query_text_collapses_whitespace(self) -> None:
        assert normalize_query_text("  a \n  b\t c \n") == "a b c"

    def test_exact_match(self) -> None:
        results = (
            QueryPredictionResult(query_id="kis_001", query_text="some query", predictions=()),
        )
        matched, method = find_unique_match(
            "some query", results, lambda r: r.query_text
        )
        assert matched.query_id == "kis_001"
        assert method == "exact"

    def test_normalized_match_falls_back(self) -> None:
        artifact_text = "Tìm video một  chiếc\nthuyền"
        results = (
            QueryPredictionResult(
                query_id="kis_001",
                query_text=artifact_text,
                predictions=(),
            ),
        )
        matched, method = find_unique_match(
            "Tìm video một chiếc thuyền", results, lambda r: r.query_text
        )
        assert matched.query_id == "kis_001"
        assert method == "normalized"

    def test_no_match_raises(self) -> None:
        results = (QueryPredictionResult(query_id="kis_001", query_text="a", predictions=()),)
        with pytest.raises(QueryMatchError, match="no exact or whitespace-normalized"):
            find_unique_match("b", results, lambda r: r.query_text)

    def test_ambiguous_match_raises(self) -> None:
        results = (
            QueryPredictionResult(query_id="kis_001", query_text="dup", predictions=()),
            QueryPredictionResult(query_id="kis_002", query_text="dup", predictions=()),
        )
        with pytest.raises(QueryMatchError, match="ambiguous"):
            find_unique_match("dup", results, lambda r: r.query_text)

    def test_duplicate_query_text_resolved_by_query_id(self) -> None:
        # Regression: two BTC queries share identical text; the
        # filename-derived query id must disambiguate them.
        results = (
            QueryPredictionResult(
                query_id="btc-p1-8-kis", query_text="giống hệt nhau", predictions=()
            ),
            QueryPredictionResult(
                query_id="btc-p1-14-kis", query_text="giống hệt nhau", predictions=()
            ),
        )
        matched, method = find_unique_match(
            "giống hệt nhau",
            results,
            lambda r: r.query_text,
            expected_query_id="btc-p1-14-kis",
            get_id=lambda r: r.query_id,
        )
        assert matched.query_id == "btc-p1-14-kis"
        assert method == "query_id"

    def test_whitespace_variant_of_duplicate_text_also_resolved_by_query_id(self) -> None:
        # Even when file text differs only in whitespace from both artifact
        # texts (which would be ambiguous under text matching alone), the
        # query id wins.
        results = (
            QueryPredictionResult(
                query_id="btc-p1-8-kis", query_text="a  b\nc", predictions=()
            ),
            QueryPredictionResult(
                query_id="btc-p1-14-kis", query_text="a b c", predictions=()
            ),
        )
        matched, method = find_unique_match(
            "a\nb\tc",
            results,
            lambda r: r.query_text,
            expected_query_id="btc-p1-8-kis",
            get_id=lambda r: r.query_id,
        )
        assert matched.query_id == "btc-p1-8-kis"
        assert method == "query_id"

    def test_falls_back_to_text_when_expected_query_id_absent(self) -> None:
        results = (
            QueryPredictionResult(query_id="other-id", query_text="unique text", predictions=()),
        )
        matched, method = find_unique_match(
            "unique text",
            results,
            lambda r: r.query_text,
            expected_query_id="btc-p1-99-kis",
            get_id=lambda r: r.query_id,
        )
        assert matched.query_id == "other-id"
        assert method == "exact"

    def test_ambiguous_query_id_raises_instead_of_degrading_to_text(self) -> None:
        results = (
            QueryPredictionResult(query_id="btc-p1-14-kis", query_text="dup", predictions=()),
            QueryPredictionResult(query_id="btc-p1-14-kis", query_text="dup", predictions=()),
        )
        with pytest.raises(QueryMatchError, match="ambiguous query_id match"):
            find_unique_match(
                "dup",
                results,
                lambda r: r.query_text,
                expected_query_id="btc-p1-14-kis",
                get_id=lambda r: r.query_id,
            )

    def test_query_id_with_different_text_fails_loudly(self) -> None:
        # Regression guard: different query batches share the p1-N naming
        # scheme. An id match whose text belongs to another batch must fail
        # instead of exporting unrelated predictions.
        results = (
            QueryPredictionResult(
                query_id="btc-p1-14-kis",
                query_text="Đoạn clip về điêu khắc cát",
                predictions=(),
            ),
        )
        with pytest.raises(QueryMatchError, match="query text differs"):
            find_unique_match(
                "Người đầu bếp đặt nguyên liệu",
                results,
                lambda r: r.query_text,
                expected_query_id="btc-p1-14-kis",
                get_id=lambda r: r.query_id,
            )

    def test_expected_query_id_requires_get_id(self) -> None:
        results = (QueryPredictionResult(query_id="kis_001", query_text="a", predictions=()),)
        with pytest.raises(ValueError, match="get_id is required"):
            find_unique_match("a", results, lambda r: r.query_text, expected_query_id="x")


# ---------------------------------------------------------------------------
# 2. CSV format validation
# ---------------------------------------------------------------------------


class TestKisCsv:
    def test_writes_no_header(self, tmp_path: Path) -> None:
        result = _kis_result("kis_001", "find a boat")
        output = tmp_path / "kis.csv"
        export_kis_csv(result, output)
        rows = _read_csv_rows(output)
        # No header row
        assert rows[0][0] == "L21_V000"

    def test_max_100_rows(self, tmp_path: Path) -> None:
        result = _kis_result("kis_001", "query", n=150)
        output = tmp_path / "kis.csv"
        export_kis_csv(result, output)
        rows = _read_csv_rows(output)
        assert len(rows) == KIS_MAX_ROWS

    def test_utf8_encoding(self, tmp_path: Path) -> None:
        result = _kis_result("kis_001", "query", n=1)
        output = tmp_path / "kis.csv"
        export_kis_csv(result, output)
        content = output.read_bytes()
        assert b"\xef\xbb\xbf" not in content  # No BOM

    def test_comma_delimiter(self, tmp_path: Path) -> None:
        result = _kis_result("kis_001", "query", n=1)
        output = tmp_path / "kis.csv"
        export_kis_csv(result, output)
        content = output.read_text(encoding="utf-8")
        assert content.strip() == "L21_V000,0"

    def test_no_mp4_suffix(self, tmp_path: Path) -> None:
        # Test the validation directly since RankedKisPrediction rejects
        # non-canonical video_id values.  The mp4 check is on the CSV rows.
        from aic2026.submission import _validate_kis_rows
        with pytest.raises(ValueError, match="must not end with .mp4"):
            _validate_kis_rows([["video.mp4", "0"]])

    def test_integer_frame_ids(self, tmp_path: Path) -> None:
        result = _kis_result("kis_001", "query", n=1)
        output = tmp_path / "kis.csv"
        export_kis_csv(result, output)
        rows = _read_csv_rows(output)
        assert rows[0][1] == "0"

    def test_empty_predictions_raises(self, tmp_path: Path) -> None:
        result = QueryPredictionResult(query_id="kis_001", query_text="query", predictions=())
        output = tmp_path / "kis.csv"
        with pytest.raises(ValueError, match="no KIS predictions"):
            export_kis_csv(result, output)


class TestQaCsv:
    def test_writes_three_columns(self, tmp_path: Path) -> None:
        result = _qa_result("qa_001", "question", answer="red chili peppers")
        output = tmp_path / "qa.csv"
        export_qa_csv(result, output)
        rows = _read_csv_rows(output)
        assert len(rows) == 1
        assert len(rows[0]) == 3

    def test_answer_longer_than_100_chars_raises(self, tmp_path: Path) -> None:
        long_answer = "a" * 150
        result = _qa_result("qa_001", "question", answer=long_answer)
        output = tmp_path / "qa.csv"
        with pytest.raises(ValueError, match="exceeds"):
            export_qa_csv(result, output)
        assert not output.exists()

    def test_answer_at_exactly_100_chars_accepted(self, tmp_path: Path) -> None:
        answer = "b" * QA_ANSWER_MAX_CHARS
        result = _qa_result("qa_001", "question", answer=answer)
        output = tmp_path / "qa.csv"
        export_qa_csv(result, output)
        rows = _read_csv_rows(output)
        assert len(rows[0][2]) == QA_ANSWER_MAX_CHARS

    def test_raises_on_empty_answer(self, tmp_path: Path) -> None:
        result = _qa_result("qa_001", "question", answer="", has_evidence=False)
        output = tmp_path / "qa.csv"
        with pytest.raises(ValueError, match="no answer"):
            export_qa_csv(result, output)

    def test_no_header(self, tmp_path: Path) -> None:
        result = _qa_result("qa_001", "question")
        output = tmp_path / "qa.csv"
        export_qa_csv(result, output)
        rows = _read_csv_rows(output)
        assert rows[0][0] != "query_id"

    def test_no_mp4_suffix(self, tmp_path: Path) -> None:
        from aic2026.submission import _validate_qa_rows
        with pytest.raises(ValueError, match="must not end with .mp4"):
            _validate_qa_rows([["L21_V001.mp4", "10", "answer"]])


# ---------------------------------------------------------------------------
# 3. TRAKE validation
# ---------------------------------------------------------------------------


class TestTrakeValidation:
    def test_valid_sequence_accepted(self) -> None:
        events = [
            _trake_event_result("E1", "event1", [("V001", 10, 0.9), ("V001", 50, 0.5)]),
            _trake_event_result("E2", "event2", [("V001", 20, 0.8), ("V001", 60, 0.4)]),
            _trake_event_result("E3", "event3", [("V001", 30, 0.7), ("V001", 70, 0.3)]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b\nE3: c", events)
        rows = select_trake_rows(result)
        assert len(rows) >= 1
        assert rows[0][0] == "V001"
        assert len(rows[0]) == 4  # video_id + 3 frames

    def test_rejects_non_increasing_frames(self) -> None:
        # E1 best frame > E2 best frame for the same video
        events = [
            _trake_event_result("E1", "event1", [("V001", 50, 0.9)]),
            _trake_event_result("E2", "event2", [("V001", 20, 0.8)]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b", events)
        rows = select_trake_rows(result)
        # This video cannot produce a valid sequence
        assert len(rows) == 0

    def test_frame_count_matches_event_count(self) -> None:
        events = [
            _trake_event_result("E1", "event1", [("V001", 10, 0.9)]),
            _trake_event_result("E2", "event2", [("V001", 20, 0.8)]),
            _trake_event_result("E3", "event3", [("V001", 30, 0.7)]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b\nE3: c", events)
        rows = select_trake_rows(result)
        for row in rows:
            assert len(row) == len(events) + 1  # video_id + N frames

    def test_no_header(self, tmp_path: Path) -> None:
        events = [
            _trake_event_result("E1", "event1", [("V001", 10, 0.9)]),
            _trake_event_result("E2", "event2", [("V001", 20, 0.8)]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b", events)
        output = tmp_path / "trake.csv"
        export_trake_csv(result, output)
        rows = _read_csv_rows(output)
        assert rows[0][0] == "V001"

    def test_max_100_rows(self) -> None:
        # Create events with 200 videos that can produce valid sequences
        candidates_per_event = [
            [("V{:03d}".format(i), i * 10 + e, 0.9 - i * 0.001 - e * 0.01)
             for i in range(200)]
            for e in range(2)
        ]
        events = [
            _trake_event_result("E1", "event1", candidates_per_event[0]),
            _trake_event_result("E2", "event2", candidates_per_event[1]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b", events)
        rows = select_trake_rows(result, max_rows=TRAKE_MAX_ROWS)
        assert len(rows) <= TRAKE_MAX_ROWS

    def test_same_video_constraint(self) -> None:
        events = [
            _trake_event_result("E1", "event1", [("V001", 10, 0.9), ("V002", 5, 0.8)]),
            _trake_event_result("E2", "event2", [("V001", 20, 0.85), ("V002", 15, 0.75)]),
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b", events)
        rows = select_trake_rows(result)
        for row in rows:
            assert all(vid == row[0] for vid in [row[0]])

    def test_four_event_sequence_has_five_columns(self) -> None:
        # Mirrors the real BTC TRAKE queries which have 4 events.
        events = [
            _trake_event_result(f"E{i}", f"event{i}", [("V001", i * 10 + 1, 0.9), ("V002", i * 10, 0.1)])
            for i in range(1, 5)
        ]
        result = _trake_result("trake_001", "E1: a\nE2: b\nE3: c\nE4: d", events)
        rows = select_trake_rows(result)
        assert len(rows) >= 1
        assert rows[0][0] == "V001"
        assert len(rows[0]) == 5  # video_id + 4 frames

    def test_validate_trake_rows_rejects_bad_columns(self) -> None:
        with pytest.raises(ValueError, match="expected 5 columns"):
            _validate_trake_rows([["V001", "10", "20"]], num_events=4)

    def test_validate_trake_rows_rejects_decreasing_frames(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            _validate_trake_rows([["V001", "20", "10", "30", "40"]], num_events=4)


# ---------------------------------------------------------------------------
# 4. Zip structure
# ---------------------------------------------------------------------------


class TestZipStructure:
    def test_zip_contains_submission_dir(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        (submission_dir / "query-p1-1-kis.csv").write_text("V001,0\n")
        (submission_dir / "query-p1-2-qa.csv").write_text("V001,0,test\n")

        zip_path = tmp_path / "submission.zip"
        create_submission_zip(submission_dir, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "submission/query-p1-1-kis.csv" in names
            assert "submission/query-p1-2-qa.csv" in names
            assert "submission/" in names  # explicit directory entry
            assert len(names) == 3

    def test_zip_content_matches_files(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir()
        (submission_dir / "query-p1-1-kis.csv").write_text("V001,0\n")

        zip_path = tmp_path / "submission.zip"
        create_submission_zip(submission_dir, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("submission/query-p1-1-kis.csv").decode("utf-8")
            assert content == "V001,0\n"

    def test_zip_creates_parent_dirs(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "output" / "submission"
        submission_dir.mkdir(parents=True)
        (submission_dir / "test.csv").write_text("a,b\n")

        zip_path = tmp_path / "output" / "submission.zip"
        create_submission_zip(submission_dir, zip_path)
        assert zip_path.exists()


# ---------------------------------------------------------------------------
# 5. End-to-end synthetic submission
# ---------------------------------------------------------------------------


class TestEndToEndSubmission:
    def test_full_synthetic_build(self, tmp_path: Path) -> None:
        from aic2026.submission import build_submission

        # Create query files
        query_dir = tmp_path / "queries"
        query_dir.mkdir()

        kis_text = "Tìm video về một chiếc thuyền"
        qa_text = "Trong video có 2 câu thơ"
        trake_text = "E1: Lân quay vòng\nE2: 4 chân chạm đất"

        (query_dir / "query-p1-1-kis.txt").write_text(kis_text, encoding="utf-8")
        (query_dir / "query-p1-22-qa.txt").write_text(qa_text, encoding="utf-8")
        (query_dir / "query-p1-4-trake.txt").write_text(trake_text, encoding="utf-8")

        # Create synthetic KIS artifact
        kis_artifact = tmp_path / "kis_predictions.json"
        import json

        kis_data = {
            "schema_version": "1.0",
            "run_id": "test-run",
            "created_at": "2026-08-21T00:00:00Z",
            "evaluation_set_path": str(tmp_path / "eval.csv"),
            "evaluation_set_query_count": 1,
            "processed_query_count": 1,
            "query_text_field": "query_text_vi",
            "top_k": 5,
            "embedding_space_compatible": True,
            "embedding_space_override_used": False,
            "index_artifact_path": str(tmp_path / "index"),
            "index_manifest": {"artifact_id": "test"},
            "model_embedding_space": {"space_id": "test"},
            "model_config": None,
            "selected_query_ids": None,
            "query_results": [
                {
                    "query_id": "kis_001",
                    "query_text": kis_text,
                    "predictions": [
                        {"rank": i + 1, "video_id": f"L21_V{i:03d}", "frame_index": i * 100, "score": 1.0 / (i + 1)}
                        for i in range(5)
                    ],
                }
            ],
        }
        kis_artifact.write_text(json.dumps(kis_data), encoding="utf-8")

        # Create synthetic QA artifact
        qa_dir = tmp_path / "qa_artifact"
        qa_dir.mkdir()
        (qa_dir / "manifest.json").write_text(json.dumps({
            "schema_version": "1.0",
            "run_id": "qa-test",
            "created_at": "2026-08-21T00:00:00Z",
            "evaluation_set_path": str(tmp_path / "eval.csv"),
            "evaluation_set_query_count": 1,
            "processed_query_count": 1,
            "query_text_field": "query_text_vi",
            "top_k": 10,
            "video_aggregation": None,
            "video_aggregation_top_n": None,
            "window_max_gap": None,
            "embedding_space_compatible": True,
            "embedding_space_override_used": False,
            "index_artifact_path": str(tmp_path / "index"),
            "index_manifest": {"artifact_id": "test"},
            "model_embedding_space": {"space_id": "test"},
            "model_config": None,
            "selected_query_ids": None,
            "answer_generation": None,
        }), encoding="utf-8")
        (qa_dir / "predictions.jsonl").write_text(json.dumps({
            "query_id": "qa_001",
            "query_text": qa_text,
            "answer": "Nguyễn Trung Trực",
            "predictions": [{"rank": 1, "video_id": "L21_V002", "frame_index": 18120, "score": 0.5}],
            "video_candidates": [],
            "window_candidates": [],
            "answer_evidence": [{"rank": 1, "video_id": "L21_V002", "frame_index": 18120, "score": 0.5}],
            "answer_error": None,
        }) + "\n", encoding="utf-8")

        # Create synthetic TRAKE artifact
        trake_dir = tmp_path / "trake_artifact"
        trake_dir.mkdir()
        (trake_dir / "manifest.json").write_text(json.dumps({
            "schema_version": "1.0",
            "run_id": "trake-test",
            "created_at": "2026-08-21T00:00:00Z",
            "evaluation_set_path": str(tmp_path / "eval.csv"),
            "evaluation_set_query_count": 1,
            "processed_query_count": 1,
            "query_text_field": "query_text_vi",
            "top_k": 10,
            "window_max_gap": 60,
            "embedding_space_compatible": True,
            "embedding_space_override_used": False,
            "index_artifact_path": str(tmp_path / "index"),
            "index_manifest": {"artifact_id": "test"},
            "model_embedding_space": {"space_id": "test"},
            "model_config": None,
            "selected_query_ids": None,
        }), encoding="utf-8")
        (trake_dir / "predictions.jsonl").write_text(json.dumps({
            "query_id": "trake_001",
            "query_text": trake_text,
            "events": [
                {
                    "event_id": "E1",
                    "event_text": "Lân quay vòng",
                    "predictions": [
                        {"rank": 1, "video_id": "L24_V044", "keyframe_ordinal": 1, "frame_index": 97, "pts_time": 3.88, "score": 0.4},
                        {"rank": 2, "video_id": "L24_V044", "keyframe_ordinal": 2, "frame_index": 149, "pts_time": 5.96, "score": 0.35},
                    ],
                    "windows": [],
                },
                {
                    "event_id": "E2",
                    "event_text": "4 chân chạm đất",
                    "predictions": [
                        {"rank": 1, "video_id": "L24_V044", "keyframe_ordinal": 3, "frame_index": 200, "pts_time": 8.0, "score": 0.38},
                        {"rank": 2, "video_id": "L24_V044", "keyframe_ordinal": 4, "frame_index": 250, "pts_time": 10.0, "score": 0.33},
                    ],
                    "windows": [],
                },
            ],
        }) + "\n", encoding="utf-8")

        # Build submission
        output_dir = tmp_path / "output"
        result = build_submission(
            query_dir=query_dir,
            output_dir=output_dir,
            kis_artifact=kis_artifact,
            qa_artifact=qa_dir,
            trake_artifact=trake_dir,
        )

        # Verify counts
        assert result["counts"]["kis"] == 1
        assert result["counts"]["qa"] == 1
        assert result["counts"]["trake"] == 1
        assert len(result["generated"]) == 3

        # Verify CSV files exist
        submission_dir = output_dir / "submission"
        assert (submission_dir / "query-p1-1-kis.csv").exists()
        assert (submission_dir / "query-p1-22-qa.csv").exists()
        assert (submission_dir / "query-p1-4-trake.csv").exists()

        # Verify KIS CSV content
        kis_rows = _read_csv_rows(submission_dir / "query-p1-1-kis.csv")
        assert len(kis_rows) == 5
        assert kis_rows[0][0] == "L21_V000"
        assert kis_rows[0][1] == "0"

        # Verify QA CSV content
        qa_rows = _read_csv_rows(submission_dir / "query-p1-22-qa.csv")
        assert len(qa_rows) == 1
        assert qa_rows[0][0] == "L21_V002"
        assert qa_rows[0][1] == "18120"
        assert qa_rows[0][2] == "Nguyễn Trung Trực"

        # Verify TRAKE CSV content
        trake_rows = _read_csv_rows(submission_dir / "query-p1-4-trake.csv")
        assert len(trake_rows) >= 1
        assert trake_rows[0][0] == "L24_V044"
        frame_indices = [int(x) for x in trake_rows[0][1:]]
        assert frame_indices == sorted(frame_indices)
        assert len(frame_indices) == 2  # 2 events

        # Verify zip exists and has correct structure
        zip_path = output_dir / "submission.zip"
        assert zip_path.exists()
        assert result["zip_info"]["valid"] is True
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert len(names) == 4  # submission/ dir + 3 CSVs
            assert all(n.startswith("submission/") for n in names)


# ---------------------------------------------------------------------------
# 6. read_query_text
# ---------------------------------------------------------------------------


class TestReadQueryText:
    def test_reads_text_content(self, tmp_path: Path) -> None:
        qfile = tmp_path / "test.txt"
        qfile.write_text("Hello world\n", encoding="utf-8")
        assert read_query_text(qfile) == "Hello world"

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        qfile = tmp_path / "test.txt"
        qfile.write_text("  spaces  \n", encoding="utf-8")
        assert read_query_text(qfile) == "spaces"

    def test_raises_on_empty_file(self, tmp_path: Path) -> None:
        qfile = tmp_path / "empty.txt"
        qfile.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            read_query_text(qfile)
