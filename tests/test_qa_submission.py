"""Tests for the isolated provisional QA submission export."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aic2026.qa.qa_submission import (
    QA_SUBMISSION_CSV_COLUMNS,
    QA_SUBMISSION_FORMAT,
    build_submission_rows,
    write_qa_submission_csv,
)
from aic2026.qa.qa_types import (
    QaPrediction,
    QaPredictionArtifact,
    QaPredictionManifest,
    QaQueryResult,
)
from aic2026.retrieval_candidates import VideoCandidate, WindowCandidate

def _result(query_id: str, answer: str, frame_indices: tuple[int, ...]) -> QaQueryResult:
    return QaQueryResult(
        query_id=query_id,
        query_text=f"text-{query_id}",
        answer=answer,
        predictions=tuple(
            QaPrediction(rank=rank, video_id="L21_V001", frame_index=frame_index, score=1.0 / rank)
            for rank, frame_index in enumerate(frame_indices, start=1)
        ),
        video_candidates=(
            VideoCandidate(video_id="L21_V001", score=1.0, rank=1, frame_count=len(frame_indices), top_frame_index=None),
        ),
        window_candidates=(
            WindowCandidate(video_id="L21_V001", frame_start=10, frame_end=20, score=1.0, rank=1),
        ),
    )


def _manifest(query_count: int) -> QaPredictionManifest:
    return QaPredictionManifest(
        schema_version="1.0",
        run_id="run-1",
        created_at="2026-08-17T00:00:00Z",
        evaluation_set_path="/tmp/eval_set.csv",
        evaluation_set_query_count=query_count,
        processed_query_count=query_count,
        query_text_field="query_text_vi",
        top_k=2,
        video_aggregation=None,
        video_aggregation_top_n=None,
        window_max_gap=None,
        embedding_space_compatible=True,
        embedding_space_override_used=False,
        index_artifact_path="/tmp/index",
        index_manifest={"artifact_id": "test"},
        model_embedding_space={"space_id": "test"},
        model_config=None,
        selected_query_ids=None,
    )


def _artifact(answers: dict[str, str]) -> QaPredictionArtifact:
    results = tuple(
        _result(query_id, answer, (10, 20)) for query_id, answer in answers.items()
    )
    return QaPredictionArtifact(manifest=_manifest(len(results)), query_results=results)


def test_build_submission_rows_one_row_per_prediction() -> None:
    rows = build_submission_rows(_artifact({"qa_001": "answer A", "qa_002": "answer B"}))
    assert len(rows) == 4
    assert rows[0] == {
        "query_id": "qa_001",
        "rank": "1",
        "video_id": "L21_V001",
        "frame_index": "10",
        "answer": "answer A",
    }
    assert {row["query_id"] for row in rows} == {"qa_001", "qa_002"}
    assert all(row["answer"] in ("answer A", "answer B") for row in rows)


def test_write_qa_submission_csv_writes_provisional_header(tmp_path: Path) -> None:
    output = write_qa_submission_csv(_artifact({"qa_001": "answer A"}), tmp_path / "submission.csv")
    assert output == tmp_path / "submission.csv"
    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == QA_SUBMISSION_CSV_COLUMNS
        assert list(reader) == [
            {"query_id": "qa_001", "rank": "1", "video_id": "L21_V001", "frame_index": "10", "answer": "answer A"},
            {"query_id": "qa_001", "rank": "2", "video_id": "L21_V001", "frame_index": "20", "answer": "answer A"},
        ]


def test_empty_answer_raises() -> None:
    artifact = _artifact({"qa_001": ""})
    with pytest.raises(ValueError, match="has no answer"):
        build_submission_rows(artifact)


def test_whitespace_only_answer_raises() -> None:
    artifact = _artifact({"qa_001": "   "})
    with pytest.raises(ValueError, match="has no answer"):
        write_qa_submission_csv(artifact, Path("/tmp/unused.csv"))


def test_query_without_evidence_raises() -> None:
    result = QaQueryResult(
        query_id="qa_001",
        query_text="text",
        answer="answer A",
        predictions=(),
        video_candidates=(),
        window_candidates=(),
    )
    artifact = QaPredictionArtifact(manifest=_manifest(1), query_results=(result,))
    with pytest.raises(ValueError, match="has no evidence"):
        build_submission_rows(artifact)


def test_submission_format_is_provisional() -> None:
    assert QA_SUBMISSION_FORMAT == "provisional-v1"