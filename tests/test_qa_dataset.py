"""Tests for QA query selection and text resolution (qa_dataset)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aic2026.evaluation import load_eval_set
from aic2026.qa.qa_dataset import iter_qa_queries, resolve_query_text, select_qa_queries

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


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _qa_row(query_id: str, en: str = "answer") -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "qa",
        "query_text_vi": f"cau hoi {query_id}",
        "query_text_en": en,
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "dap an",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }


def _kis_row(query_id: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "task": "kis",
        "query_text_vi": f"truy van {query_id}",
        "query_text_en": "",
        "gt_video_id": "L21_V001",
        "gt_frame_start": "10",
        "gt_frame_end": "20",
        "gt_answer": "",
        "gt_events": "",
        "annotator": "P1",
        "note": "",
    }


def _mixed_set(tmp_path: Path) -> Path:
    return _write_eval_csv(
        tmp_path / "eval_set.csv",
        [_kis_row("kis_001"), _qa_row("qa_001"), _qa_row("qa_002"), _kis_row("kis_002")],
    )


def test_iter_qa_queries_returns_only_qa_in_row_order(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    queries = iter_qa_queries(evaluation_set)
    assert [query.query_id for query in queries] == ["qa_001", "qa_002"]
    assert all(query.task.value == "qa" for query in queries)


def test_iter_qa_queries_empty_when_none_present(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_write_eval_csv(tmp_path / "eval_set.csv", [_kis_row("kis_001")]))
    assert iter_qa_queries(evaluation_set) == ()


def test_select_qa_queries_without_subset_keeps_all(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    assert [query.query_id for query in select_qa_queries(evaluation_set)] == ["qa_001", "qa_002"]


def test_select_qa_queries_subset_preserves_row_order(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    selected = select_qa_queries(evaluation_set, subset_ids=("qa_002", "qa_001"))
    assert [query.query_id for query in selected] == ["qa_001", "qa_002"]


def test_select_qa_queries_unknown_id_raises(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    with pytest.raises(ValueError, match="unknown query ids"):
        select_qa_queries(evaluation_set, subset_ids=("qa_001", "missing"))
    with pytest.raises(ValueError, match="unknown query ids"):
        select_qa_queries(evaluation_set, subset_ids=("kis_001",))


def test_select_qa_queries_subset_of_only_unknown_raises(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    with pytest.raises(ValueError, match="unknown query ids"):
        select_qa_queries(evaluation_set, subset_ids=("kis_001", "kis_002"))


def test_resolve_query_text_vietnamese_by_default(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    query = evaluation_set.get_query("qa_001")
    assert resolve_query_text(query, "query_text_vi") == "cau hoi qa_001"


def test_resolve_query_text_english(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_mixed_set(tmp_path))
    query = evaluation_set.get_query("qa_001")
    assert resolve_query_text(query, "query_text_en") == "answer"


def test_resolve_query_text_missing_english_raises(tmp_path: Path) -> None:
    evaluation_set = load_eval_set(_write_eval_csv(tmp_path / "eval_set.csv", [_qa_row("qa_001", en="")]))
    query = evaluation_set.get_query("qa_001")
    with pytest.raises(ValueError, match="query_text_en was requested but is missing"):
        resolve_query_text(query, "query_text_en")