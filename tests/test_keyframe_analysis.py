"""Tests for the keyframe-analysis module and its CLI.

These tests exercise keyframe statistics, GT frame coverage (exact and relaxed
hits), the worst-first per-query error report, the CLIP-baseline comparison,
report serialization, and the CLI. They use a small synthetic
``frames.ndjson`` / evaluation / prediction fixture so no real dataset is
required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.keyframe_analysis import (
    KEYFRAME_COVERED,
    KEYFRAME_NOT_COVERED,
    VIDEO_RETRIEVED_COVERAGE_ISSUE,
    KeyframeAnalysisReport,
    KeyframeMissRecord,
    KeyframeStatistics,
    GtCoverageRecord,
    analyze_clip_comparison,
    analyze_gt_coverage,
    analyze_keyframe_statistics,
    build_error_report,
    format_report,
    load_keyframe_source_stats,
    load_keyframes,
    run_keyframe_analysis,
    write_report,
)
from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    KisPredictionArtifact,
    QueryPredictionResult,
    RankedKisPrediction,
    write_kis_prediction_artifact,
)

_EVAL_HEADER = (
    "query_id,task,query_text_vi,query_text_en,gt_video_id,"
    "gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note"
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_frames(frames_path: Path, videos: dict[str, list[int]]) -> None:
    """Write a frames.ndjson with ``video_id -> [frame_index, ...]`` records."""
    lines = []
    for video_id, frame_indices in videos.items():
        for ordinal, frame_index in enumerate(frame_indices, start=1):
            record = {
                "keyframe": {"video_id": {"value": video_id}, "keyframe_ordinal": ordinal},
                "pts_time": frame_index / 25.0,
                "fps": 25.0,
                "frame_index": frame_index,
            }
            lines.append(json.dumps(record))
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_eval(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write a KIS-only eval CSV with ``(id, gt_video, frame_range)`` rows."""
    lines = [_EVAL_HEADER]
    for query_id, gt_video, frame_range in rows:
        lines.append(
            f"{query_id},kis,vi-{query_id},en-{query_id},{gt_video},{frame_range},,,P1,"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prediction(rank: int, video_id: str, frame_index: int) -> RankedKisPrediction:
    return RankedKisPrediction(rank=rank, video_id=video_id, frame_index=frame_index, score=1.0 / rank)


def _query_result(
    query_id: str, predictions: tuple[RankedKisPrediction, ...]
) -> QueryPredictionResult:
    return QueryPredictionResult(query_id=query_id, query_text="test query", predictions=predictions)


def _artifact(query_results: tuple[QueryPredictionResult, ...]) -> KisPredictionArtifact:
    return KisPredictionArtifact(
        schema_version=KIS_PREDICTION_SCHEMA_VERSION,
        run_id="base-run",
        created_at="2026-08-14T00:00:00Z",
        evaluation_set_path="/tmp/eval_set.csv",
        evaluation_set_query_count=len(query_results),
        processed_query_count=len(query_results),
        query_text_field="query_text_en",
        top_k=100,
        embedding_space_compatible=True,
        embedding_space_override_used=False,
        index_artifact_path="/tmp/index",
        index_manifest={"artifact_id": "dummy-index"},
        model_embedding_space={"backend": "dummy"},
        model_config=None,
        selected_query_ids=None,
        query_results=query_results,
    )


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, list[int]]]:
    """Build synthetic frames.ndjson, eval CSV, and prediction artifact.

    Keyframes:
        - L21_V001: [0, 90, 261]
        - L21_V006: [10, 40]
        - L22_V002: (absent from the keyframe set -> no keyframes)

    Queries:
        - kis_001: L21_V001 [80, 120]   exact hit (frame 90), distance 0
        - kis_002: L21_V001 [200, 210]  miss, nearest frame 261, distance 51
        - kis_003: L21_V006 [200, 300]  miss, nearest frame 40, distance 160
        - kis_004: L22_V002 [5, 10]     miss, no keyframes in video
    """
    frames_path = tmp_path / "frames.ndjson"
    _write_frames(frames_path, {"L21_V001": [0, 90, 261], "L21_V006": [10, 40]})

    eval_path = _write_eval(
        tmp_path / "eval_set.csv",
        [
            ("kis_001", "L21_V001", "80,120"),
            ("kis_002", "L21_V001", "200,210"),
            ("kis_003", "L21_V006", "200,300"),
            ("kis_004", "L22_V002", "5,10"),
        ],
    )

    clip_path = tmp_path / "predictions.json"
    write_kis_prediction_artifact(
        clip_path,
        _artifact(
            (
                _query_result(
                    "kis_001",
                    (
                        _prediction(1, "L22_V009", 5),
                        _prediction(2, "L21_V001", 90),
                        _prediction(3, "L21_V001", 261),
                    ),
                ),
                _query_result(
                    "kis_002",
                    (
                        _prediction(1, "L21_V001", 90),
                        _prediction(2, "L21_V001", 261),
                        _prediction(3, "L22_V009", 0),
                    ),
                ),
                _query_result(
                    "kis_003",
                    (_prediction(1, "L22_V002", 5), _prediction(2, "L22_V009", 0)),
                ),
                _query_result(
                    "kis_004",
                    (_prediction(1, "L22_V002", 5), _prediction(2, "L22_V009", 0)),
                ),
            ),
        ),
    )

    keyframes = {"L21_V001": (0, 90, 261), "L21_V006": (10, 40)}
    return frames_path, eval_path, clip_path, keyframes


def _queries(eval_path: Path) -> tuple:
    from aic2026.evaluation import EvaluationSet

    evaluation_set = load_eval_set(eval_path)
    assert isinstance(evaluation_set, EvaluationSet)
    return tuple(evaluation_set.iter_queries(EvaluationTask.KIS))


# ---------------------------------------------------------------------------
# load_keyframes / statistics
# ---------------------------------------------------------------------------


def test_load_keyframes_groups_sorts_and_deduplicates(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.ndjson"
    _write_frames(frames_path, {"L21_V001": [90, 0, 90, 261]})

    keyframes = load_keyframes(frames_path)

    assert keyframes == {"L21_V001": (0, 90, 261)}


def test_load_keyframes_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_keyframes(tmp_path / "missing.ndjson")


def test_analyze_keyframe_statistics(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.ndjson"
    _write_frames(frames_path, {"L21_V001": [0, 90, 261], "L21_V006": [10, 40], "L22_V001": [5]})

    statistics = analyze_keyframe_statistics(load_keyframes(frames_path))

    assert statistics.total_videos == 3
    assert statistics.total_keyframes == 6
    assert statistics.avg_keyframes_per_video == pytest.approx(2.0)
    assert statistics.min_keyframes == 1
    assert statistics.max_keyframes == 3
    assert statistics.p50 == pytest.approx(2.0)
    assert statistics.p90 == pytest.approx(2.8)
    assert statistics.p95 == pytest.approx(2.9)
    assert statistics.to_dict()["total_videos"] == 3
    # Without explicit record counts the statistics assume unique frames only.
    assert statistics.total_keyframe_records == 6
    assert statistics.duplicate_keyframe_records == 0


def test_analyze_keyframe_statistics_duplicate_records(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.ndjson"
    _write_frames(frames_path, {"L21_V001": [0, 90, 90, 261]})

    keyframes = load_keyframes(frames_path)
    total_records, duplicates = load_keyframe_source_stats(frames_path)
    statistics = analyze_keyframe_statistics(
        keyframes,
        total_keyframe_records=total_records,
        duplicate_keyframe_records=duplicates,
    )

    assert keyframes == {"L21_V001": (0, 90, 261)}
    assert total_records == 4
    assert duplicates == 1
    assert statistics.total_keyframes == 3
    assert statistics.total_keyframe_records == 4
    assert statistics.duplicate_keyframe_records == 1
    assert statistics.avg_keyframes_per_video == pytest.approx(3.0)


def test_analyze_keyframe_statistics_empty() -> None:
    statistics = analyze_keyframe_statistics({})

    assert statistics.total_videos == 0
    assert statistics.total_keyframes == 0
    assert statistics.avg_keyframes_per_video == 0.0
    assert statistics.min_keyframes == 0
    assert statistics.max_keyframes == 0


def test_keyframe_statistics_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        KeyframeStatistics(
            total_videos=-1,
            total_keyframes=3,
            avg_keyframes_per_video=3.0,
            min_keyframes=3,
            max_keyframes=3,
            p50=3.0,
            p90=3.0,
            p95=3.0,
            total_keyframe_records=3,
            duplicate_keyframe_records=0,
        )
    with pytest.raises(ValueError, match="total_keyframe_records cannot be less"):
        KeyframeStatistics(
            total_videos=1,
            total_keyframes=3,
            avg_keyframes_per_video=3.0,
            min_keyframes=3,
            max_keyframes=3,
            p50=3.0,
            p90=3.0,
            p95=3.0,
            total_keyframe_records=2,
            duplicate_keyframe_records=0,
        )
    with pytest.raises(ValueError, match="unique records"):
        KeyframeStatistics(
            total_videos=1,
            total_keyframes=3,
            avg_keyframes_per_video=3.0,
            min_keyframes=3,
            max_keyframes=3,
            p50=3.0,
            p90=3.0,
            p95=3.0,
            total_keyframe_records=4,
            duplicate_keyframe_records=2,
        )


# ---------------------------------------------------------------------------
# GT frame coverage
# ---------------------------------------------------------------------------


def test_analyze_gt_coverage_exact_and_relaxed(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    records, summary = analyze_gt_coverage(_queries(eval_path), keyframes)

    assert [record.query_id for record in records] == ["kis_001", "kis_002", "kis_003", "kis_004"]

    exact = records[0]
    assert exact.exact_hit is True
    assert exact.nearest_keyframe_distance == 0
    assert exact.keyframes_in_range == 1
    assert exact.keyframes_in_video == 3

    near = records[1]
    assert near.exact_hit is False
    assert near.nearest_keyframe_frame_index == 261
    assert near.nearest_keyframe_distance == 51
    assert near.keyframes_in_range == 0

    far = records[2]
    assert far.exact_hit is False
    assert far.nearest_keyframe_frame_index == 40
    assert far.nearest_keyframe_distance == 160

    missing = records[3]
    assert missing.exact_hit is False
    assert missing.nearest_keyframe_frame_index is None
    assert missing.nearest_keyframe_distance is None
    assert missing.keyframes_in_video == 0

    assert summary.total_queries == 4
    assert summary.exact_hit_count == 1
    assert summary.exact_hit_percentage == pytest.approx(25.0)
    assert summary.queries_without_keyframes == 1
    thresholds = {hit.threshold: hit for hit in summary.relaxed_hits}
    assert {hit.threshold for hit in summary.relaxed_hits} == {10, 30, 60, 150, 300}
    assert thresholds[10].count == 1
    assert thresholds[30].count == 1
    assert thresholds[60].count == 2
    assert thresholds[150].count == 2
    assert thresholds[300].count == 3
    assert thresholds[60].percentage == pytest.approx(50.0)
    assert summary.nearest_distance_stats["min"] == pytest.approx(0.0)
    assert summary.nearest_distance_stats["max"] == pytest.approx(160.0)
    assert summary.nearest_distance_stats["p50"] == pytest.approx(51.0)
    assert summary.nearest_distance_stats["mean"] == pytest.approx(70.33333333333333)


def test_analyze_gt_coverage_custom_thresholds(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    _records, summary = analyze_gt_coverage(_queries(eval_path), keyframes, relaxed_thresholds=(60, 300))

    assert [hit.threshold for hit in summary.relaxed_hits] == [60, 300]
    assert summary.relaxed_hits[0].count == 2


def test_analyze_gt_coverage_invalid_thresholds(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    with pytest.raises(ValueError):
        analyze_gt_coverage(_queries(eval_path), keyframes, relaxed_thresholds=(0,))
    with pytest.raises(ValueError):
        analyze_gt_coverage(_queries(eval_path), keyframes, relaxed_thresholds=())


def test_analyze_gt_coverage_rejects_invalid_queries(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    with pytest.raises(ValueError, match="at least one query"):
        analyze_gt_coverage([], keyframes)
    with pytest.raises(ValueError, match="task"):
        analyze_gt_coverage(_non_kis_queries(eval_path), keyframes)


def _non_kis_queries(eval_path: Path) -> list:
    from aic2026.evaluation import EvaluationQuery

    query = _queries(eval_path)[0]
    return [
        EvaluationQuery(
            query_id="qa_001",
            task=EvaluationTask.QA,
            query_text_vi=query.query_text_vi,
            query_text_en=query.query_text_en,
            ground_truth_video_id=query.ground_truth_video_id,
            ground_truth_frame_range=query.ground_truth_frame_range,
            answer="some answer",
            event=None,
            annotation=query.annotation,
        )
    ]


def test_gt_coverage_record_validation() -> None:
    base = dict(
        query_id="kis_001",
        gt_video_id="L21_V001",
        gt_frame_start=80,
        gt_frame_end=120,
        keyframes_in_video=3,
        keyframes_in_range=0,
    )
    with pytest.raises(ValueError, match="exact_hit"):
        GtCoverageRecord(
            exact_hit=True,
            nearest_keyframe_frame_index=None,
            nearest_keyframe_distance=None,
            **base,
        )
    with pytest.raises(ValueError, match="zero nearest_keyframe_distance"):
        GtCoverageRecord(
            exact_hit=False,
            nearest_keyframe_frame_index=90,
            nearest_keyframe_distance=0,
            **base,
        )
    with pytest.raises(ValueError, match="both be None"):
        GtCoverageRecord(
            exact_hit=False,
            nearest_keyframe_frame_index=90,
            nearest_keyframe_distance=None,
            **base,
        )


# ---------------------------------------------------------------------------
# Error report
# ---------------------------------------------------------------------------


def test_build_error_report_worst_first(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    records, _summary = analyze_gt_coverage(_queries(eval_path), keyframes)
    error_report = build_error_report(records)

    assert [record.query_id for record in error_report] == ["kis_004", "kis_003", "kis_002"]
    first = error_report[0]
    assert first.query_id == "kis_004"
    assert first.distance is None
    assert first.closest_keyframe_frame_index is None
    assert first.keyframes_in_video == 0
    last = error_report[-1]
    assert last.query_id == "kis_002"
    assert last.distance == 51
    assert last.closest_keyframe_frame_index == 261
    assert last.to_dict()["gt_video_id"] == "L21_V001"


def test_build_error_report_excludes_exact_hits(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.ndjson"
    _write_frames(frames_path, {"L21_V001": [90]})
    eval_path = _write_eval(tmp_path / "eval.csv", [("kis_001", "L21_V001", "80,120")])

    records, _summary = analyze_gt_coverage(_queries(eval_path), load_keyframes(frames_path))
    assert build_error_report(records) == ()


# ---------------------------------------------------------------------------
# CLIP baseline comparison
# ---------------------------------------------------------------------------


def test_analyze_clip_comparison_categories(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, clip_path, keyframes = fixture
    from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring

    predictions = load_kis_predictions_for_scoring(clip_path)
    records, summary = analyze_clip_comparison(_queries(eval_path), keyframes, predictions)

    by_query = {record.query_id: record for record in records}
    assert by_query["kis_001"].category == KEYFRAME_COVERED
    assert by_query["kis_001"].clip_retrieved_gt_video is True
    assert by_query["kis_001"].clip_retrieved_gt_frame is True
    assert by_query["kis_001"].clip_gt_video_best_rank == 2
    assert by_query["kis_001"].clip_gt_frame_best_rank == 2

    assert by_query["kis_002"].category == VIDEO_RETRIEVED_COVERAGE_ISSUE
    assert by_query["kis_002"].clip_retrieved_gt_video is True
    assert by_query["kis_002"].clip_retrieved_gt_frame is False
    assert by_query["kis_002"].clip_gt_video_best_rank == 1

    assert by_query["kis_003"].category == KEYFRAME_NOT_COVERED
    assert by_query["kis_003"].clip_retrieved_gt_video is False

    assert by_query["kis_004"].category == KEYFRAME_NOT_COVERED
    assert by_query["kis_004"].clip_retrieved_gt_video is True
    assert by_query["kis_004"].clip_retrieved_gt_frame is True

    assert summary.total_queries == 4
    assert summary.clip_retrieved_gt_video_count == 3
    assert summary.clip_retrieved_gt_frame_count == 2
    counts = dict(summary.category_counts)
    assert counts[KEYFRAME_COVERED] == 1
    assert counts[KEYFRAME_NOT_COVERED] == 2
    assert counts[VIDEO_RETRIEVED_COVERAGE_ISSUE] == 1


def test_analyze_clip_comparison_top_k_filtering(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, clip_path, keyframes = fixture
    from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring

    predictions = load_kis_predictions_for_scoring(clip_path)
    records, _summary = analyze_clip_comparison(
        _queries(eval_path), keyframes, predictions, top_k=1
    )

    by_query = {record.query_id: record for record in records}
    assert by_query["kis_001"].clip_retrieved_gt_video is False
    assert by_query["kis_001"].clip_gt_video_best_rank is None


def test_analyze_clip_comparison_missing_predictions(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    with pytest.raises(ValueError, match="missing predictions"):
        analyze_clip_comparison(_queries(eval_path), keyframes, {})


def test_analyze_clip_comparison_invalid_top_k(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, _clip_path, keyframes = fixture

    with pytest.raises(ValueError, match="top_k"):
        analyze_clip_comparison(_queries(eval_path), keyframes, {"kis_001": []}, top_k=0)


def test_clip_comparison_record_validation(fixture: tuple[Path, Path, Path, dict]) -> None:
    from aic2026.keyframe_analysis import ClipComparisonRecord

    frames_path, eval_path, _clip_path, keyframes = fixture
    records, _summary = analyze_gt_coverage(_queries(eval_path), keyframes)

    with pytest.raises(ValueError, match="inconsistent"):
        ClipComparisonRecord(
            query_id="kis_002",
            gt_video_id="L21_V001",
            gt_frame_start=200,
            gt_frame_end=210,
            keyframe_covered=False,
            clip_retrieved_gt_video=True,
            clip_retrieved_gt_frame=False,
            clip_gt_video_best_rank=1,
            clip_gt_frame_best_rank=None,
            category=KEYFRAME_COVERED,
        )


# ---------------------------------------------------------------------------
# Full pipeline / serialization
# ---------------------------------------------------------------------------


def test_run_keyframe_analysis_end_to_end(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, clip_path, _keyframes = fixture

    report = run_keyframe_analysis(
        evaluation_set_path=eval_path,
        frames_path=frames_path,
        clip_predictions_path=clip_path,
    )

    assert isinstance(report, KeyframeAnalysisReport)
    assert report.schema_version == "1.0"
    assert report.keyframe_statistics.total_videos == 2
    assert report.coverage.exact_hit_count == 1
    assert len(report.error_report) == 3
    assert report.clip_comparison.total_queries == 4

    payload = report.to_dict()
    for key in (
        "schema_version",
        "keyframes_source",
        "evaluation_set_path",
        "clip_predictions_path",
        "created_at",
        "keyframe_statistics",
        "gt_coverage",
        "per_query",
        "error_report",
        "clip_comparison",
    ):
        assert key in payload
    assert len(payload["per_query"]) == 4
    assert payload["clip_comparison"]["summary"]["total_queries"] == 4


def test_run_keyframe_analysis_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_keyframe_analysis(
            evaluation_set_path=tmp_path / "missing.csv",
            frames_path=tmp_path / "missing.ndjson",
            clip_predictions_path=tmp_path / "missing.json",
        )


def test_write_report_round_trip(fixture: tuple[Path, Path, Path, dict], tmp_path: Path) -> None:
    frames_path, eval_path, clip_path, _keyframes = fixture
    report = run_keyframe_analysis(
        evaluation_set_path=eval_path,
        frames_path=frames_path,
        clip_predictions_path=clip_path,
    )

    output_path = tmp_path / "nested" / "report.json"
    write_report(output_path, report)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "exact_hit_count" in payload["gt_coverage"]
    assert payload["clip_comparison"]["summary"]["categories"][0]["category"] == KEYFRAME_NOT_COVERED


def test_format_report_contains_sections(fixture: tuple[Path, Path, Path, dict]) -> None:
    frames_path, eval_path, clip_path, _keyframes = fixture
    report = run_keyframe_analysis(
        evaluation_set_path=eval_path,
        frames_path=frames_path,
        clip_predictions_path=clip_path,
    )

    text = format_report(report)

    assert "1. Keyframe statistics" in text
    assert "2. GT frame coverage" in text
    assert "3. Per-query error report" in text
    assert "4. CLIP baseline comparison" in text
    assert "kis_004" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_analyze_keyframes_end_to_end(fixture: tuple[Path, Path, Path, dict], tmp_path: Path) -> None:
    frames_path, eval_path, clip_path, _keyframes = fixture
    output_path = tmp_path / "analysis" / "keyframes" / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframes.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--eval-csv",
            str(eval_path),
            "--frames",
            str(frames_path),
            "--clip-predictions",
            str(clip_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1. Keyframe statistics" in result.stdout
    assert "Keyframe analysis written to:" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"


def test_cli_analyze_keyframes_missing_inputs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframes.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--frames",
            str(tmp_path / "missing.ndjson"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "error:" in result.stderr


def test_cli_analyze_keyframes_invalid_thresholds(
    fixture: tuple[Path, Path, Path, dict],
) -> None:
    frames_path, eval_path, clip_path, _keyframes = fixture
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframes.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--eval-csv",
            str(eval_path),
            "--frames",
            str(frames_path),
            "--clip-predictions",
            str(clip_path),
            "--relaxed-thresholds",
            "0,5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "error:" in result.stderr


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------


def test_real_keyframe_analysis_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    frames_path = root / "data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson"
    eval_path = root / "data/eval/eval_set.csv"
    clip_path = root / "data/interim/predictions/kis-l14-laion2b-full-en-top1000.json"
    if not (frames_path.is_file() and eval_path.is_file() and clip_path.is_file()):
        pytest.skip("real dataset artifacts not available")

    report = run_keyframe_analysis(
        evaluation_set_path=eval_path,
        frames_path=frames_path,
        clip_predictions_path=clip_path,
    )

    assert report.keyframe_statistics.total_keyframes > 0
    assert report.coverage.total_queries == 101
    assert len(report.per_query) == 101
