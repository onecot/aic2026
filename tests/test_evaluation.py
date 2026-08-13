"""Tests for the CSV-to-canonical evaluation dataset layer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aic2026.canonical import VideoId
from aic2026.evaluation import (
    AnnotationMetadata,
    EvaluationDatasetError,
    EvaluationQuery,
    EvaluationTask,
    EventAnnotation,
    FrameRange,
    load_eval_set,
)

_HEADERS = [
    "query_id",
    "task",
    "query_text_vi",
    "query_text_en",
    "gt_video_id",
    "gt_frame_start",
    "gt_frame_end",
    "gt_answer",
    "gt_events",
    "annotator",
    "note",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "query_id": "kis_001",
        "task": "kis",
        "query_text_vi": "Tìm cảnh một chiếc thuyền.",
        "query_text_en": "Find a boat scene.",
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], headers: list[str] | None = None) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or _HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_kis_rows_normalizes_csv_whitespace_and_preserves_metadata(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [_row(gt_video_id=" L21_V001\r ", note=" useful annotation ")],
    )

    evaluation_set = load_eval_set(path)
    query = evaluation_set.get_query("kis_001")

    assert len(evaluation_set) == 1
    assert query.task is EvaluationTask.KIS
    assert query.query_text_vi == "Tìm cảnh một chiếc thuyền."
    assert query.query_text_en == "Find a boat scene."
    assert query.ground_truth_video_id.value == "L21_V001"
    assert query.ground_truth_frame_range == FrameRange(10, 20)
    assert query.answer is None
    assert query.event is None
    assert query.annotation == AnnotationMetadata(annotator="P1", note="useful annotation")


def test_load_qa_and_trake_task_specific_values(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "eval_set.csv",
        [
            _row(
                query_id="qa_001",
                task="QA",
                gt_answer="Đáp án tiếng Việt",
            ),
            _row(
                query_id="trake_001",
                task="trake",
                gt_events="người bước vào phòng",
            ),
        ],
    )

    evaluation_set = load_eval_set(path)
    qa_query = evaluation_set.get_query("qa_001")
    trake_query = evaluation_set.get_query("trake_001")

    assert qa_query.task is EvaluationTask.QA
    assert qa_query.answer == "Đáp án tiếng Việt"
    assert qa_query.event is None
    assert trake_query.task is EvaluationTask.TRAKE
    assert trake_query.answer is None
    assert trake_query.event == EventAnnotation("người bước vào phòng")
    assert list(evaluation_set.iter_queries("QA")) == [qa_query]
    assert evaluation_set.task_counts() == {EvaluationTask.QA: 1, EvaluationTask.TRAKE: 1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "other"}, "task must be one of"),
        ({"query_id": ""}, "query_id must be a non-empty string"),
        ({"gt_video_id": "bad-id"}, "invalid video id"),
        ({"gt_frame_start": "21", "gt_frame_end": "20"}, "frame range end"),
        ({"gt_frame_start": "not-a-frame"}, "gt_frame_start must be an integer"),
        ({"task": "qa"}, "QA queries require a non-empty answer"),
        ({"task": "trake"}, "TRAKE queries require a non-empty event annotation"),
        ({"task": "kis", "gt_answer": "unexpected"}, "KIS queries must not define"),
    ],
)
def test_invalid_task_specific_or_ground_truth_values_fail(tmp_path: Path, overrides: dict[str, str], message: str) -> None:
    path = _write_csv(tmp_path / "eval_set.csv", [_row(**overrides)])

    with pytest.raises(EvaluationDatasetError, match=message):
        load_eval_set(path)


def test_duplicate_query_ids_and_invalid_headers_fail(tmp_path: Path) -> None:
    duplicate_path = _write_csv(tmp_path / "duplicates.csv", [_row(), _row()])
    with pytest.raises(EvaluationDatasetError, match="query_ids must be unique"):
        load_eval_set(duplicate_path)

    invalid_header_path = _write_csv(
        tmp_path / "bad_header.csv",
        [_row()],
        headers=_HEADERS[:-1],
    )
    with pytest.raises(EvaluationDatasetError, match="columns must be"):
        load_eval_set(invalid_header_path)


def test_query_contract_rejects_invalid_optional_task_fields() -> None:
    with pytest.raises(ValueError, match="TRAKE queries require"):
        EvaluationQuery(
            query_id="trake_001",
            task=EvaluationTask.TRAKE,
            query_text_vi="câu hỏi",
            query_text_en=None,
            ground_truth_video_id=VideoId("L21_V001"),
            ground_truth_frame_range=FrameRange(0, 1),
            answer=None,
            event=None,
            annotation=AnnotationMetadata("P1", None),
        )


def test_real_eval_set_loads_when_available() -> None:
    path = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    if not path.is_file():
        pytest.skip("team evaluation CSV not available")

    evaluation_set = load_eval_set(path)

    assert len(evaluation_set) == 101
    assert evaluation_set.task_counts() == {EvaluationTask.KIS: 101}
    assert all(query.answer is None and query.event is None for query in evaluation_set.queries)
    assert all(
        query.ground_truth_frame_range.start <= query.ground_truth_frame_range.end
        for query in evaluation_set.queries
    )
