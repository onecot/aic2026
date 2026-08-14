"""Tests for video-level evaluation of the media-info retrieval branch.

These tests exercise per-query retrieval, aggregate metrics, the CLIP
baseline comparison, validation, report formatting, and the CLI. They use a
synthetic evaluation CSV and a small media-info index; the CLIP baseline is
provided as typed predictions so the scorer/retrieval pipeline is never run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring
from aic2026.kis_scoring import KisPrediction
from aic2026.media_info_evaluation import (
    DEFAULT_TOP_K_VALUES,
    BaselineComparison,
    BaselineQueryRecord,
    VideoRetrievalQueryResult,
    VideoRetrievalSummary,
    category_report_to_dict,
    compare_to_clip_baseline,
    evaluate_media_info_video_retrieval,
    format_category_report,
    format_report,
    retrieve_video_hits,
)
from aic2026.media_info_retrieval import (
    MediaInfoDocument,
    MediaInfoHit,
    MediaInfoIndex,
)

_EVAL_HEADER = (
    "query_id,task,query_text_vi,query_text_en,gt_video_id,"
    "gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note"
)


def _eval_csv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> Path:
    """Write a KIS-only eval CSV with the given (id, vi, en, gt_video, range)."""
    lines = [_EVAL_HEADER]
    for query_id, vi, en, gt_video, frame_range in rows:
        lines.append(
            f"{query_id},kis,{vi},{en},{gt_video},{frame_range},,,P1,"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _index() -> MediaInfoIndex:
    documents = [
        MediaInfoDocument(
            "L21_V001",
            "a boat trip",
            ("travel",),
            "a boat carrying tourists on the river",
        ),
        MediaInfoDocument(
            "L22_V002",
            "car review",
            ("auto",),
            "the new car on the highway",
        ),
        MediaInfoDocument(
            "L23_V003",
            "cooking show",
            ("food",),
            "recipes for every day",
        ),
    ]
    return MediaInfoIndex.build(documents)


def _clip(query_id: str, video_id: str | None, rank: int = 1) -> dict[str, list[KisPrediction]]:
    """Build a one-query CLIP baseline prediction mapping."""
    predictions = [] if video_id is None else [KisPrediction(rank=rank, video_id=video_id, frame_index=0)]
    return {query_id: predictions}


# ---------------------------------------------------------------------------
# retrieve_video_hits
# ---------------------------------------------------------------------------


def test_retrieve_video_hits_returns_ranked_hits() -> None:
    hits = retrieve_video_hits(_index(), "boat", top_k=5)
    assert isinstance(hits, tuple)
    assert [hit.video_id for hit in hits] == ["L21_V001"]
    assert hits[0].rank == 1
    assert hits[0].score > 0.0


def test_retrieve_video_hits_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        retrieve_video_hits(_index(), "boat", top_k=0)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def test_evaluation_returns_metrics_and_summary(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "100,200"),
            ("kis_002", "ô tô", "car", "L22_V002", "50,60"),
            ("kis_003", "nấu ăn", "cooking", "L23_V003", "10,20"),
        ],
    )
    eval_set = load_eval_set(csv_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))

    results, summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)

    assert len(results) == 3
    assert all(result.gt_rank == 1 for result in results)
    assert summary.total == 3
    for cutoff, _fraction in summary.recall_at:
        assert summary.recall_at_cutoff(cutoff) == pytest.approx(1.0)
    assert summary.avg_gt_rank == pytest.approx(1.0)
    assert summary.not_retrieved == ()


def test_evaluation_metrics_reflect_missed_videos(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "100,200"),
            ("kis_002", "ô tô", "car", "L22_V002", "50,60"),
            ("kis_003", "xe máy", "motorcycle", "L23_V003", "10,20"),
        ],
    )
    eval_set = load_eval_set(csv_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))

    results, summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)

    gt_ranks = {result.query_id: result.gt_rank for result in results}
    assert gt_ranks == {"kis_001": 1, "kis_002": 1, "kis_003": None}
    assert summary.avg_gt_rank == pytest.approx(1.0)
    assert summary.not_retrieved == ("kis_003",)
    assert summary.recall_at_cutoff(1) == pytest.approx(2 / 3)
    assert summary.recall_at_cutoff(100) == pytest.approx(2 / 3)


def test_recall_reflects_gt_rank_beyond_cutoff(tmp_path: Path) -> None:
    # GT video appears only at rank 2 for one query.
    documents = [
        MediaInfoDocument("L21_V001", "the quick brown fox", (), "a boat in the river"),
        MediaInfoDocument("L22_V002", "boat", (), "boat boat boat boat"),
    ]
    index = MediaInfoIndex.build(documents)
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L22_V002", "0,10"),
            ("kis_002", "thuyền", "boat", "L21_V001", "0,10"),
        ],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))

    results, summary = evaluate_media_info_video_retrieval(queries, index, top_k=100)

    assert {result.query_id: result.gt_rank for result in results} == {
        "kis_001": 1,
        "kis_002": 2,
    }
    assert summary.recall_at_cutoff(1) == pytest.approx(0.5)
    assert summary.recall_at_cutoff(5) == pytest.approx(1.0)
    assert summary.avg_gt_rank == pytest.approx(1.5)


def test_evaluation_uses_query_text_vi_when_requested(tmp_path: Path) -> None:
    documents = [
        MediaInfoDocument("L21_V001", "thuyền du lịch", (), ""),
        MediaInfoDocument("L22_V002", "món ngon", (), ""),
    ]
    index = MediaInfoIndex.build(documents)
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))

    results, _summary = evaluate_media_info_video_retrieval(
        queries, index, query_text_field="query_text_vi", top_k=100
    )

    assert results[0].gt_rank == 1
    assert results[0].query_text == "thuyền"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_evaluation_rejects_non_kis_queries(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    path.write_text(
        "\n".join(
            [
                _EVAL_HEADER,
                "qa_001,qa,question,question,L21_V001,0,10,answer,,P1,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queries = tuple(load_eval_set(path).iter_queries())
    with pytest.raises(ValueError, match="task='kis'"):
        evaluate_media_info_video_retrieval(queries, _index(), top_k=100)


def test_evaluation_rejects_empty_queries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one query"):
        evaluate_media_info_video_retrieval((), _index(), top_k=100)


def test_evaluation_rejects_missing_query_text_en(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    path.write_text(
        "\n".join(
            [
                _EVAL_HEADER,
                "kis_001,kis,thuyền,,L21_V001,0,10,,,P1,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queries = tuple(load_eval_set(path).iter_queries(EvaluationTask.KIS))
    with pytest.raises(ValueError, match="query_text_en"):
        evaluate_media_info_video_retrieval(queries, _index(), top_k=100)


def test_evaluation_rejects_invalid_query_text_field(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    with pytest.raises(ValueError, match="query_text_field"):
        evaluate_media_info_video_retrieval(
            queries, _index(), query_text_field="nope", top_k=100
        )


def test_evaluation_rejects_invalid_top_k_and_cutoffs(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    with pytest.raises(ValueError, match="top_k"):
        evaluate_media_info_video_retrieval(queries, _index(), top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        evaluate_media_info_video_retrieval(queries, _index(), top_k=5, cutoffs=DEFAULT_TOP_K_VALUES)
    with pytest.raises(ValueError, match="cutoffs"):
        evaluate_media_info_video_retrieval(
            queries, _index(), top_k=100, cutoffs=(5, 1)
        )
    with pytest.raises(ValueError, match="cutoffs"):
        evaluate_media_info_video_retrieval(
            queries, _index(), top_k=100, cutoffs=(0,)
        )


def test_evaluation_is_deterministic(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "ô tô", "car", "L22_V002", "0,10"),
        ],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))

    first = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    second = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)

    assert first[0] == second[0]
    assert first[1] == second[1]


# ---------------------------------------------------------------------------
# Result / summary validation
# ---------------------------------------------------------------------------


def test_video_retrieval_query_result_validation() -> None:
    with pytest.raises(ValueError, match="query_id"):
        VideoRetrievalQueryResult(
            query_id="", query_text="x", gt_video_id="L21_V001", gt_rank=1, hits=()
        )
    with pytest.raises(ValueError, match="rank order"):
        VideoRetrievalQueryResult(
            query_id="kis_001",
            query_text="x",
            gt_video_id="L21_V001",
            gt_rank=None,
            hits=(MediaInfoHit(rank=2, video_id="L21_V001", score=1.0), MediaInfoHit(rank=1, video_id="L22_V002", score=0.5)),
        )
    with pytest.raises(ValueError, match="gt_rank"):
        VideoRetrievalQueryResult(
            query_id="kis_001", query_text="x", gt_video_id="L21_V001", gt_rank=0, hits=()
        )


def test_summary_validation() -> None:
    with pytest.raises(ValueError, match="total"):
        VideoRetrievalSummary(total=0, recall_at=((1, 0.5),), avg_gt_rank=1.0, not_retrieved=())
    with pytest.raises(ValueError, match="cutoffs"):
        VideoRetrievalSummary(total=2, recall_at=((5, 0.5), (1, 1.0)), avg_gt_rank=1.0, not_retrieved=())
    with pytest.raises(ValueError, match="fractions"):
        VideoRetrievalSummary(total=2, recall_at=((1, 1.5),), avg_gt_rank=1.0, not_retrieved=())
    with pytest.raises(ValueError, match="avg_gt_rank"):
        VideoRetrievalSummary(total=2, recall_at=((1, 1.0),), avg_gt_rank=-1.0, not_retrieved=())
    with pytest.raises(ValueError, match="avg_gt_rank"):
        VideoRetrievalSummary(total=2, recall_at=((1, 1.0),), avg_gt_rank=None, not_retrieved=())


def test_summary_to_dict_is_json_safe(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    _results, summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    payload = summary.to_dict()
    assert payload["total"] == 2
    assert payload["recall_at"][0]["k"] == 1
    assert payload["not_retrieved_query_ids"] == ["kis_002"]
    json.dumps(payload)


# ---------------------------------------------------------------------------
# CLIP baseline comparison
# ---------------------------------------------------------------------------


def test_comparison_groups_queries_by_outcome(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "ô tô", "car", "L22_V002", "0,10"),
            ("kis_003", "nấu ăn", "cooking", "L23_V003", "0,10"),
            ("kis_004", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    eval_set = load_eval_set(csv_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    results, _summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    results_by_query = {result.query_id: result for result in results}
    clip_predictions = {
        "kis_001": [KisPrediction(rank=1, video_id="L23_V003", frame_index=0)],  # CLIP fails
        "kis_002": [KisPrediction(rank=1, video_id="L22_V002", frame_index=0)],  # both succeed
        "kis_003": [KisPrediction(rank=1, video_id="L23_V003", frame_index=0)],  # both succeed
        "kis_004": [KisPrediction(rank=1, video_id="L22_V002", frame_index=0)],  # CLIP succeeds, media-info fails
    }

    comparison = compare_to_clip_baseline(queries, results_by_query, clip_predictions, cutoff=100)

    assert [r.query_id for r in comparison.media_info_only] == ["kis_001"]
    assert [r.query_id for r in comparison.both] == ["kis_002", "kis_003"]
    assert [r.query_id for r in comparison.clip_only] == ["kis_004"]
    assert comparison.neither == ()
    assert comparison.cutoff == 100


def test_comparison_respects_cutoff(tmp_path: Path) -> None:
    documents = [
        MediaInfoDocument("L21_V001", "the quick brown fox", (), "a boat in the river"),
        MediaInfoDocument("L22_V002", "boat", (), "boat boat boat boat"),
    ]
    index = MediaInfoIndex.build(documents)
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    results, _summary = evaluate_media_info_video_retrieval(queries, index, top_k=100)
    assert results[0].gt_rank == 2
    results_by_query = {result.query_id: result for result in results}
    clip_predictions = {"kis_001": [KisPrediction(rank=1, video_id="L21_V001", frame_index=0)]}

    strict = compare_to_clip_baseline(queries, results_by_query, clip_predictions, cutoff=1)
    loose = compare_to_clip_baseline(queries, results_by_query, clip_predictions, cutoff=100)

    # GT rank is 2 for media-info: at cutoff 1 media-info fails, at 100 it succeeds.
    assert [r.query_id for r in strict.media_info_only] == []
    assert [r.query_id for r in strict.both] == []
    assert [r.query_id for r in strict.clip_only] == ["kis_001"]
    assert [r.query_id for r in loose.media_info_only] == []
    assert [r.query_id for r in loose.both] == ["kis_001"]
    assert loose.clip_only == ()


def test_comparison_requires_results_and_predictions(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    results, _summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    results_by_query = {result.query_id: result for result in results}

    with pytest.raises(ValueError, match="missing media-info result"):
        compare_to_clip_baseline(queries, {}, _clip("kis_001", "L21_V001"), cutoff=100)
    with pytest.raises(ValueError, match="missing CLIP predictions"):
        compare_to_clip_baseline(queries, results_by_query, {}, cutoff=100)
    with pytest.raises(ValueError, match="cutoff"):
        compare_to_clip_baseline(queries, results_by_query, _clip("kis_001", "L21_V001"), cutoff=0)


def test_comparison_rejects_non_kis_queries(tmp_path: Path) -> None:
    path = tmp_path / "eval.csv"
    path.write_text(
        "\n".join(
            [
                _EVAL_HEADER,
                "qa_001,qa,question,question,L21_V001,0,10,answer,,P1,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    queries = tuple(load_eval_set(path).iter_queries())
    with pytest.raises(ValueError, match="task='kis'"):
        compare_to_clip_baseline(queries, {}, {}, cutoff=100)


def test_comparison_to_dict_is_json_safe(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    results, _summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    results_by_query = {result.query_id: result for result in results}
    comparison = compare_to_clip_baseline(
        queries, results_by_query, _clip("kis_001", "L23_V003"), cutoff=100
    )
    payload = comparison.to_dict()
    assert payload["cutoff"] == 100
    assert [r["query_id"] for r in payload["media_info_only"]] == ["kis_001"]
    json.dumps(payload)


def test_baseline_record_validation() -> None:
    with pytest.raises(ValueError, match="query_id"):
        BaselineQueryRecord(
            query_id="", query_text_vi="v", query_text_en=None, gt_video_id="L21_V001",
            media_info_gt_rank=None, clip_gt_rank=None,
        )
    with pytest.raises(ValueError, match="media_info_gt_rank"):
        BaselineQueryRecord(
            query_id="kis_001", query_text_vi="v", query_text_en=None, gt_video_id="L21_V001",
            media_info_gt_rank=0, clip_gt_rank=None,
        )


def test_comparison_validation() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        BaselineComparison(cutoff=0, media_info_only=(), clip_only=(), both=(), neither=())
    with pytest.raises(ValueError, match="media_info_only"):
        BaselineComparison(
            cutoff=100, media_info_only=(object(),), clip_only=(), both=(), neither=()
        )  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def test_format_report_contains_metrics_and_comparison(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    results, summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    results_by_query = {result.query_id: result for result in results}
    comparison = compare_to_clip_baseline(
        queries,
        results_by_query,
        {
            "kis_001": [KisPrediction(rank=1, video_id="L23_V003", frame_index=0)],
            "kis_002": [KisPrediction(rank=1, video_id="L22_V002", frame_index=0)],
        },
        cutoff=100,
    )

    report = format_report(summary, comparison=comparison)

    assert "Media-info video-level retrieval evaluation" in report
    assert "Queries: 2" in report
    assert "Video Recall@1: 0.5000" in report
    assert "average GT video rank: 1.00" in report
    assert "kis_002" in report
    assert "media-info succeeds, CLIP fails: 1" in report
    assert "CLIP succeeds, media-info fails: 1" in report
    assert "both succeed: 0" in report


def test_format_report_without_comparison(tmp_path: Path) -> None:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    queries = tuple(load_eval_set(csv_path).iter_queries(EvaluationTask.KIS))
    results, summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)

    report = format_report(summary, results=results)

    assert "Comparison vs CLIP baseline" not in report
    assert "Video Recall@1: 1.0000" in report


# ---------------------------------------------------------------------------
# Category report (format + JSON export)
# ---------------------------------------------------------------------------


def _comparison(tmp_path: Path, cutoff: int = 100) -> BaselineComparison:
    csv_path = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "ô tô", "car", "L22_V002", "0,10"),
            ("kis_003", "nấu ăn", "cooking", "L23_V003", "0,10"),
            ("kis_004", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    eval_set = load_eval_set(csv_path)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    results, _summary = evaluate_media_info_video_retrieval(queries, _index(), top_k=100)
    results_by_query = {result.query_id: result for result in results}
    clip_predictions = {
        "kis_001": [KisPrediction(rank=1, video_id="L23_V003", frame_index=0)],  # media-info only
        "kis_002": [KisPrediction(rank=1, video_id="L22_V002", frame_index=0)],  # both
        "kis_003": [KisPrediction(rank=1, video_id="L23_V003", frame_index=0)],  # both
        "kis_004": [KisPrediction(rank=1, video_id="L22_V002", frame_index=0)],  # clip only
    }
    return compare_to_clip_baseline(queries, results_by_query, clip_predictions, cutoff=cutoff)


def test_category_report_to_dict_groups_and_counts(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    payload = category_report_to_dict(comparison)

    assert payload["cutoff"] == 100
    assert payload["total"] == 4
    assert payload["counts"] == {
        "MEDIA_ONLY": 1,
        "CLIP_ONLY": 1,
        "BOTH": 2,
        "NEITHER": 0,
    }
    assert [r["query_id"] for r in payload["categories"]["MEDIA_ONLY"]] == ["kis_001"]
    assert [r["query_id"] for r in payload["categories"]["CLIP_ONLY"]] == ["kis_004"]
    assert [r["query_id"] for r in payload["categories"]["BOTH"]] == ["kis_002", "kis_003"]
    assert payload["categories"]["NEITHER"] == []
    json.dumps(payload)


def test_category_report_to_dict_record_fields(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    payload = category_report_to_dict(comparison)
    record = payload["categories"]["MEDIA_ONLY"][0]

    assert record == {
        "query_id": "kis_001",
        "query_text_vi": "thuyền",
        "query_text_en": "boat",
        "gt_video_id": "L21_V001",
        "media_info_gt_rank": 1,
        "clip_gt_rank": None,
    }


def test_format_category_report_contains_summary_and_records(tmp_path: Path) -> None:
    comparison = _comparison(tmp_path)
    report = format_category_report(comparison)

    assert "Media-info vs CLIP baseline query categorization" in report
    assert "Cutoff: top-100" in report
    assert "MEDIA_ONLY (1): media-info retrieves GT video, CLIP does not" in report
    assert "CLIP_ONLY (1): CLIP retrieves GT video, media-info does not" in report
    assert "BOTH (2): both retrieve the GT video" in report
    assert "NEITHER (0): neither retrieves the GT video" in report
    assert "Summary: MEDIA_ONLY=1, CLIP_ONLY=1, BOTH=2, NEITHER=0 | total=4" in report
    # Full per-query detail: id, texts, GT video, and both ranks.
    assert "kis_001 | EN: boat | VI: thuyền | GT: L21_V001 | media-info rank: 1 | CLIP rank: —" in report
    assert "kis_004 | EN: motorcycle | VI: xe máy | GT: L22_V002 | media-info rank: — | CLIP rank: 1" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_media_info_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "media-info"
    directory.mkdir()
    base = {
        "author": "Test Author",
        "channel_id": "UCHANNEL",
        "channel_url": "https://www.youtube.com/channel/UCHANNEL",
        "description": "test",
        "keywords": ["k1"],
        "length": 42,
        "publish_date": "01/01/2024",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "title": "Test video",
        "watch_url": "https://youtube.com/watch?v=TEST",
    }
    records = {
        "L21_V001": ("a boat trip", ["travel"], "a boat carrying tourists"),
        "L22_V002": ("car review", ["auto"], "the new car on the highway"),
        "L23_V003": ("cooking show", ["food"], "recipes every day"),
    }
    for video_id, (title, keywords, description) in records.items():
        payload = {**base, "title": title, "keywords": keywords, "description": description}
        (directory / f"{video_id}.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_cli_media_info_evaluate_end_to_end(tmp_path: Path) -> None:
    from aic2026.cli import main

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    output_path = tmp_path / "metrics.json"

    exit_code = main(
        [
            "media-info-evaluate",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
            "--query-text-field",
            "query_text_en",
            "--top-k",
            "100",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["total"] == 2
    assert payload["metrics"]["recall_at"][0] == {"k": 1, "recall": 0.5}
    assert payload["comparison"] is None
    assert len(payload["results"]) == 2


def test_cli_media_info_evaluate_with_explicit_clip_baseline(tmp_path: Path) -> None:
    from aic2026.cli import main
    from aic2026.kis_prediction_runner import write_kis_prediction_artifact

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    # Build a minimal prediction artifact that covers the query.
    artifact = {
        "schema_version": "1.0",
        "run_id": "clip-baseline",
        "created_at": "2026-08-14T00:00:00Z",
        "evaluation_set_path": str(eval_csv),
        "evaluation_set_query_count": 1,
        "processed_query_count": 1,
        "query_text_field": "query_text_en",
        "top_k": 100,
        "embedding_space_compatible": True,
        "embedding_space_override_used": False,
        "index_artifact_path": "/tmp/index",
        "index_manifest": {"artifact_id": "dummy"},
        "model_embedding_space": {"backend": "openclip"},
        "model_config": None,
        "selected_query_ids": None,
        "query_results": [
            {
                "query_id": "kis_001",
                "query_text": "boat",
                "predictions": [
                    {"rank": 1, "video_id": "L23_V003", "frame_index": 0, "score": 0.9}
                ],
            }
        ],
    }
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(json.dumps(artifact), encoding="utf-8")
    output_path = tmp_path / "metrics.json"

    exit_code = main(
        [
            "media-info-evaluate",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
            "--clip-predictions",
            str(clip_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"] is not None
    assert [r["query_id"] for r in payload["comparison"]["media_info_only"]] == ["kis_001"]


def test_cli_media_info_evaluate_explicit_clip_mismatch_fails(tmp_path: Path) -> None:
    from aic2026.cli import main

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "clip",
                "created_at": "2026-08-14T00:00:00Z",
                "evaluation_set_path": str(eval_csv),
                "evaluation_set_query_count": 1,
                "processed_query_count": 1,
                "query_text_field": "query_text_en",
                "top_k": 100,
                "embedding_space_compatible": True,
                "embedding_space_override_used": False,
                "index_artifact_path": "/tmp/index",
                "index_manifest": {"artifact_id": "dummy"},
                "model_embedding_space": {"backend": "openclip"},
                "model_config": None,
                "selected_query_ids": None,
                "query_results": [
                    {
                        "query_id": "other",
                        "query_text": "x",
                        "predictions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "media-info-evaluate",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
            "--clip-predictions",
            str(clip_path),
        ]
    )

    assert exit_code == 1


def test_cli_media_info_evaluate_missing_directory(tmp_path: Path) -> None:
    from aic2026.cli import main

    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    exit_code = main(
        [
            "media-info-evaluate",
            "--media-info-dir",
            str(tmp_path / "absent"),
            "--evaluation-set",
            str(eval_csv),
        ]
    )
    assert exit_code == 1


def test_cli_media_info_evaluate_invalid_k1(tmp_path: Path) -> None:
    from aic2026.cli import main

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    exit_code = main(
        [
            "media-info-evaluate",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
            "--k1",
            "-1",
        ]
    )
    assert exit_code == 1


def test_cli_media_info_report_end_to_end(tmp_path: Path) -> None:
    from aic2026.cli import main
    from aic2026.kis_prediction_runner import write_kis_prediction_artifact

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [
            ("kis_001", "thuyền", "boat", "L21_V001", "0,10"),
            ("kis_002", "xe máy", "motorcycle", "L22_V002", "0,10"),
        ],
    )
    clip_path = tmp_path / "clip.json"
    clip_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "clip-baseline",
                "created_at": "2026-08-14T00:00:00Z",
                "evaluation_set_path": str(eval_csv),
                "evaluation_set_query_count": 2,
                "processed_query_count": 2,
                "query_text_field": "query_text_en",
                "top_k": 100,
                "embedding_space_compatible": True,
                "embedding_space_override_used": False,
                "index_artifact_path": "/tmp/index",
                "index_manifest": {"artifact_id": "dummy"},
                "model_embedding_space": {"backend": "openclip"},
                "model_config": None,
                "selected_query_ids": None,
                "query_results": [
                    {
                        "query_id": "kis_001",
                        "query_text": "boat",
                        "predictions": [
                            {"rank": 1, "video_id": "L23_V003", "frame_index": 0, "score": 0.9}
                        ],
                    },
                    {
                        "query_id": "kis_002",
                        "query_text": "motorcycle",
                        "predictions": [
                            {"rank": 1, "video_id": "L22_V002", "frame_index": 0, "score": 0.9}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            "media-info-report",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
            "--clip-predictions",
            str(clip_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["counts"] == {"MEDIA_ONLY": 1, "CLIP_ONLY": 1, "BOTH": 0, "NEITHER": 0}
    assert payload["total"] == 2
    assert [r["query_id"] for r in payload["categories"]["MEDIA_ONLY"]] == ["kis_001"]
    assert [r["query_id"] for r in payload["categories"]["CLIP_ONLY"]] == ["kis_002"]


def test_cli_media_info_report_requires_clip_baseline(tmp_path: Path) -> None:
    from aic2026.cli import main

    media_info_dir = _write_media_info_dir(tmp_path)
    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    exit_code = main(
        [
            "media-info-report",
            "--media-info-dir",
            str(media_info_dir),
            "--evaluation-set",
            str(eval_csv),
        ]
    )
    assert exit_code == 1


def test_cli_media_info_report_missing_directory(tmp_path: Path) -> None:
    from aic2026.cli import main

    eval_csv = _eval_csv(
        tmp_path / "eval.csv",
        [("kis_001", "thuyền", "boat", "L21_V001", "0,10")],
    )
    exit_code = main(
        [
            "media-info-report",
            "--media-info-dir",
            str(tmp_path / "absent"),
            "--evaluation-set",
            str(eval_csv),
        ]
    )
    assert exit_code == 1


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------


def test_real_media_info_evaluation_against_real_baseline() -> None:
    media_info_dir = Path(__file__).resolve().parents[1] / "data/raw/aic2026_batch1/media-info"
    eval_csv = Path(__file__).resolve().parents[1] / "data/eval/eval_set.csv"
    clip_predictions_path = Path(__file__).resolve().parents[1] / "data/interim/predictions/kis-l14-laion2b-full-en.json"
    if not (media_info_dir.is_dir() and eval_csv.is_file() and clip_predictions_path.is_file()):
        pytest.skip("real dataset artifacts not available")
    index = MediaInfoIndex.build_from_dir(media_info_dir)
    eval_set = load_eval_set(eval_csv)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    clip_predictions = load_kis_predictions_for_scoring(str(clip_predictions_path))

    results, summary = evaluate_media_info_video_retrieval(queries, index, top_k=100)
    assert summary.total == 101
    assert 0.0 <= summary.recall_at_cutoff(100) <= 1.0
    assert len(results) == 101

    results_by_query = {result.query_id: result for result in results}
    comparison = compare_to_clip_baseline(queries, results_by_query, clip_predictions, cutoff=100)
    assert comparison.media_info_only
    assert comparison.clip_only
    assert len(comparison.media_info_only) + len(comparison.clip_only) + len(comparison.both) + len(comparison.neither) == 101
