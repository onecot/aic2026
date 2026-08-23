"""Retriever correctness + candidate foundation primitives.."""

from __future__ import annotations


# ==========================================================================
# Section: test_retrieval
# ==========================================================================

from pathlib import Path

import numpy as np
import pytest

from aic2026.dataset import Dataset
from aic2026.index_builder import build_index
from aic2026.embeddings import Embedding
from aic2026.retrieval import IndexRetriever


def _make_index(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for name in ("clip-features-32", "map-keyframes", "media-info", "keyframes", "objects"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    vids = [("L1_V001", 2), ("L1_V002", 1)]
    dim = 4
    for vid, n in vids:
        with (raw / "map-keyframes" / f"{vid}.csv").open("w", encoding="utf-8") as f:
            f.write("n,pts_time,fps,frame_idx\n")
            for i in range(1, n + 1):
                f.write(f"{i},{(i-1)*0.04},25,{i-1}\n")
        (raw / "media-info" / f"{vid}.json").write_text(
            '{"author":"a","channel_id":"c","channel_url":"u","description":"d","keywords":[],"length":10,'
            '"publish_date":"2020-01-01","thumbnail_url":"t","title":"title","watch_url":"w"}',
            encoding="utf-8",
        )
        if vid == "L1_V001":
            mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float16)
        else:
            mat = np.array([[0, 0, 1, 0]], dtype=np.float16)
        np.save(raw / "clip-features-32" / f"{vid}.npy", mat)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact"
    build_index(ds, artifact, artifact_id="retrieval-test")
    return artifact


def test_exact_cosine_ranking(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    res = retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)
    assert [r.video_id for r in res] == ["L1_V001", "L1_V001", "L1_V002"]
    assert res[0].keyframe_ordinal == 1
    assert pytest.approx(res[0].score, rel=1e-6) == 1.0


def test_top_k(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    res = retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert len(res) == 1
    assert res[0].rank == 1


def test_dimensionality_validation(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    with pytest.raises(ValueError, match="dim"):
        retriever.search(np.array([1, 2], dtype=np.float32), top_k=1)


def test_tie_is_deterministic_by_row_order(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    res = retriever.search(np.array([0, 0, 0, 1], dtype=np.float32), top_k=3)
    assert [r.rank for r in res] == [1, 2, 3]
    assert [(r.video_id, r.keyframe_ordinal) for r in res] == [
        ("L1_V001", 1),
        ("L1_V001", 2),
        ("L1_V002", 1),
    ]


def test_result_field_mapping(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    row = retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)[0]
    assert row.video_id == "L1_V001"
    assert row.keyframe_ordinal == 1
    assert row.frame_index == 0
    assert row.pts_time == 0.0


def test_invalid_inputs(tmp_path: Path):
    retriever = IndexRetriever(_make_index(tmp_path))
    with pytest.raises(ValueError, match="top_k"):
        retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=0)
    with pytest.raises(ValueError, match="1-D"):
        retriever.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=1)
    with pytest.raises(ValueError, match="non-zero"):
        retriever.search(np.array([0, 0, 0, 0], dtype=np.float32), top_k=1)


def test_load_real_index_artifact_format():
    artifact = Path("data/interim/indexes/jina-v2-full")
    if not artifact.is_dir():
        pytest.skip("production index artifact not available")
    retriever = IndexRetriever(artifact)
    space = retriever.manifest.embedding_space
    q = np.zeros(space.embedding_dim, dtype=np.float32)
    q[0] = 1.0
    res = retriever.search(Embedding(kind="text", vector=q, space=space), top_k=1)
    assert len(res) == 1


# ==========================================================================
# Section: test_retrieval_candidates
# ==========================================================================

from pathlib import Path

import numpy as np
import pytest

from aic2026.dataset import Dataset
from aic2026.index_builder import build_index
from aic2026.embeddings import Embedding
from aic2026.retrieval import IndexRetriever, RetrievalCandidate
from aic2026.retrieval_candidates import (
    SUPPORTED_VIDEO_AGGREGATIONS,
    VideoCandidate,
    WindowCandidate,
    aggregate_video_score,
    build_video_candidates,
    group_candidates_by_video,
    group_windows,
    sort_by_score,
)


def _candidate(
    rank: int,
    video_id: str,
    frame_index: int,
    score: float,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        rank=rank,
        video_id=video_id,
        keyframe_ordinal=frame_index + 1,
        frame_index=frame_index,
        pts_time=float(frame_index),
        score=score,
    )


# ------------------------------------------------------------------ grouping


def test_group_candidates_by_video_groups_and_preserves_order() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.5),
        _candidate(2, "L02_V003", 5, 0.9),
        _candidate(3, "L01_V001", 1, 0.7),
    ]

    groups = group_candidates_by_video(candidates)

    assert list(groups) == ["L01_V001", "L02_V003"]
    assert [(c.frame_index, c.score) for c in groups["L01_V001"]] == [(0, 0.5), (1, 0.7)]
    assert [(c.frame_index, c.score) for c in groups["L02_V003"]] == [(5, 0.9)]


def test_group_candidates_by_video_empty() -> None:
    assert group_candidates_by_video([]) == {}


def test_group_candidates_by_video_single_video() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.5),
        _candidate(2, "L01_V001", 1, 0.6),
        _candidate(3, "L01_V001", 2, 0.7),
    ]

    groups = group_candidates_by_video(candidates)

    assert list(groups) == ["L01_V001"]
    assert [c.frame_index for c in groups["L01_V001"]] == [0, 1, 2]


def test_group_candidates_by_video_preserves_duplicate_frames() -> None:
    candidates = [
        _candidate(1, "L01_V001", 10, 0.9),
        _candidate(2, "L01_V001", 10, 0.8),
    ]

    groups = group_candidates_by_video(candidates)

    assert [(c.frame_index, c.score) for c in groups["L01_V001"]] == [(10, 0.9), (10, 0.8)]


# ------------------------------------------------------------------ aggregation


def test_aggregate_video_score_max() -> None:
    assert aggregate_video_score([0.3, 0.9, 0.6], "max") == pytest.approx(0.9)


def test_aggregate_video_score_mean() -> None:
    assert aggregate_video_score([0.9, 0.8, 0.2], "mean") == pytest.approx(0.6333333333333333)


def test_aggregate_video_score_mean_top_n() -> None:
    assert aggregate_video_score([0.9, 0.8, 0.2], "mean_top_n", top_n=2) == pytest.approx(0.85)


def test_aggregate_video_score_mean_top_n_fewer_than_n() -> None:
    assert aggregate_video_score([0.8], "mean_top_n", top_n=5) == pytest.approx(0.8)


def test_aggregate_video_score_defaults_to_max() -> None:
    assert aggregate_video_score([0.2, 0.8]) == pytest.approx(0.8)


def test_aggregate_video_score_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one frame score"):
        aggregate_video_score([])


def test_aggregate_video_score_rejects_unsupported_method() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        aggregate_video_score([0.5], "median")
    with pytest.raises(ValueError, match="unsupported"):
        aggregate_video_score([0.5], "MAX")


def test_aggregate_video_score_rejects_invalid_top_n() -> None:
    for bad in (None, 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            aggregate_video_score([0.5], "mean_top_n", top_n=bad)


def test_supported_aggregations_covered() -> None:
    assert SUPPORTED_VIDEO_AGGREGATIONS == frozenset({"max", "mean", "mean_top_n"})


# ------------------------------------------------------------------ video candidates


def test_build_video_candidates_orders_by_video_score() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.4),
        _candidate(2, "L02_V003", 5, 0.9),
        _candidate(3, "L01_V001", 1, 0.8),
        _candidate(4, "L03_V005", 9, 0.7),
    ]

    videos = build_video_candidates(candidates, method="max")

    assert [(v.video_id, v.score, v.rank) for v in videos] == [
        ("L02_V003", 0.9, 1),
        ("L01_V001", 0.8, 2),
        ("L03_V005", 0.7, 3),
    ]
    assert [v.frame_count for v in videos] == [1, 2, 1]


def test_build_video_candidates_top_frame_index() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.4),
        _candidate(2, "L01_V001", 1, 0.8),
        _candidate(3, "L01_V001", 2, 0.6),
    ]

    videos = build_video_candidates(candidates, method="max")

    assert videos[0].top_frame_index == 1


def test_build_video_candidates_top_frame_index_tie_first_appearance() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.8),
        _candidate(2, "L01_V001", 1, 0.8),
    ]

    videos = build_video_candidates(candidates, method="max")

    assert videos[0].top_frame_index == 0


def test_build_video_candidates_mean_top_n_uses_aggregate() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.9),
        _candidate(2, "L01_V001", 1, 0.8),
        _candidate(3, "L01_V001", 2, 0.2),
    ]

    videos = build_video_candidates(candidates, method="mean_top_n", top_n=2)

    assert videos[0].score == pytest.approx(0.85)


def test_build_video_candidates_empty() -> None:
    assert build_video_candidates([]) == []


def test_build_video_candidates_rank_ordering_is_deterministic_for_ties() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.5),
        _candidate(2, "L02_V003", 5, 0.5),
        _candidate(3, "L03_V005", 9, 0.5),
    ]

    first = build_video_candidates(candidates, method="max")
    second = build_video_candidates(candidates, method="max")

    assert [v.video_id for v in first] == ["L01_V001", "L02_V003", "L03_V005"]
    assert [v.video_id for v in second] == ["L01_V001", "L02_V003", "L03_V005"]


# ------------------------------------------------------------------ ranking


def test_sort_by_score_descending_and_stable() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.5),
        _candidate(2, "L02_V003", 5, 0.9),
        _candidate(3, "L03_V005", 9, 0.5),
        _candidate(4, "L04_V007", 3, 0.7),
    ]

    ordered = sort_by_score(candidates)

    assert [(c.video_id, c.score) for c in ordered] == [
        ("L02_V003", 0.9),
        ("L04_V007", 0.7),
        ("L01_V001", 0.5),
        ("L03_V005", 0.5),
    ]


def test_sort_by_score_ascending() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.8),
        _candidate(2, "L02_V003", 5, 0.2),
    ]

    ordered = sort_by_score(candidates, descending=False)

    assert [c.video_id for c in ordered] == ["L02_V003", "L01_V001"]


def test_sort_by_score_empty() -> None:
    assert sort_by_score([]) == []


def test_sort_by_score_is_deterministic() -> None:
    candidates = [
        _candidate(1, "L01_V001", 0, 0.5),
        _candidate(2, "L02_V003", 5, 0.5),
    ]

    assert [(c.video_id, c.rank) for c in sort_by_score(candidates)] == [
        (c.video_id, c.rank) for c in sort_by_score(candidates)
    ]


# ------------------------------------------------------------------ window grouping


def test_group_windows_contiguous_and_gapped() -> None:
    assert group_windows([0, 1, 2, 5, 6, 9], max_gap=1) == [(0, 2), (5, 6), (9, 9)]


def test_group_windows_single_frame() -> None:
    assert group_windows([4], max_gap=1) == [(4, 4)]


def test_group_windows_empty() -> None:
    assert group_windows([], max_gap=1) == []


def test_group_windows_larger_max_gap_merges() -> None:
    assert group_windows([0, 3, 6], max_gap=3) == [(0, 6)]


def test_group_windows_sorts_unsorted_input() -> None:
    assert group_windows([9, 0, 2, 1], max_gap=1) == [(0, 2), (9, 9)]


def test_group_windows_preserves_duplicate_indices() -> None:
    assert group_windows([0, 0, 1], max_gap=1) == [(0, 1)]


def test_group_windows_rejects_invalid_max_gap() -> None:
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            group_windows([0], max_gap=bad)


def test_group_windows_rejects_negative_indices() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        group_windows([-1, 0], max_gap=1)


# ------------------------------------------------------------------ candidate validation


def test_video_candidate_validation() -> None:
    VideoCandidate(video_id="L01_V001", score=0.9, rank=1, frame_count=2, top_frame_index=3)

    with pytest.raises(ValueError, match="video_id"):
        VideoCandidate(video_id="", score=0.9, rank=1, frame_count=1)
    with pytest.raises(ValueError, match="score"):
        VideoCandidate(video_id="L01_V001", score=True, rank=1, frame_count=1)
    with pytest.raises(ValueError, match="rank"):
        VideoCandidate(video_id="L01_V001", score=0.9, rank=0, frame_count=1)
    with pytest.raises(ValueError, match="frame_count"):
        VideoCandidate(video_id="L01_V001", score=0.9, rank=1, frame_count=0)
    with pytest.raises(ValueError, match="top_frame_index"):
        VideoCandidate(video_id="L01_V001", score=0.9, rank=1, frame_count=1, top_frame_index=-1)


def test_window_candidate_validation() -> None:
    WindowCandidate(video_id="L01_V001", frame_start=10, frame_end=20, score=0.9, rank=1)

    with pytest.raises(ValueError, match="video_id"):
        WindowCandidate(video_id="", frame_start=0, frame_end=1, score=0.9, rank=1)
    with pytest.raises(ValueError, match="frame_start"):
        WindowCandidate(video_id="L01_V001", frame_start=-1, frame_end=1, score=0.9, rank=1)
    with pytest.raises(ValueError, match="frame_end must be"):
        WindowCandidate(video_id="L01_V001", frame_start=5, frame_end=4, score=0.9, rank=1)
    with pytest.raises(ValueError, match="score"):
        WindowCandidate(video_id="L01_V001", frame_start=0, frame_end=1, score="high", rank=1)
    with pytest.raises(ValueError, match="rank"):
        WindowCandidate(video_id="L01_V001", frame_start=0, frame_end=1, score=0.9, rank=0)


# ------------------------------------------------------------------ IndexRetriever compatibility


def _make_index(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for name in ("clip-features-32", "map-keyframes", "media-info", "keyframes", "objects"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    vids = [("L1_V001", 2), ("L1_V002", 1)]
    dim = 4
    for vid, n in vids:
        with (raw / "map-keyframes" / f"{vid}.csv").open("w", encoding="utf-8") as f:
            f.write("n,pts_time,fps,frame_idx\n")
            for i in range(1, n + 1):
                f.write(f"{i},{(i-1)*0.04},25,{i-1}\n")
        (raw / "media-info" / f"{vid}.json").write_text(
            '{"author":"a","channel_id":"c","channel_url":"u","description":"d","keywords":[],"length":10,'
            '"publish_date":"2020-01-01","thumbnail_url":"t","title":"title","watch_url":"w"}',
            encoding="utf-8",
        )
        if vid == "L1_V001":
            mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float16)
        else:
            mat = np.array([[0, 0, 1, 0]], dtype=np.float16)
        np.save(raw / "clip-features-32" / f"{vid}.npy", mat)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact"
    build_index(ds, artifact, artifact_id="retrieval-foundation-test")
    return artifact


def test_grouping_accepts_index_retriever_output(tmp_path: Path) -> None:
    retriever = IndexRetriever(_make_index(tmp_path))
    results = retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)

    groups = group_candidates_by_video(results)

    assert list(groups) == ["L1_V001", "L1_V002"]
    assert len(groups["L1_V001"]) == 2
    assert len(groups["L1_V002"]) == 1
    # per-video order matches the retriever's global rank order
    assert [c.frame_index for c in groups["L1_V001"]] == [0, 1]


def test_build_video_candidates_consumes_index_retriever_output(tmp_path: Path) -> None:
    retriever = IndexRetriever(_make_index(tmp_path))
    results = retriever.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)

    videos = build_video_candidates(results, method="max")

    assert [v.video_id for v in videos] == ["L1_V001", "L1_V002"]
    assert videos[0].score == pytest.approx(1.0, rel=1e-6)
    assert videos[0].top_frame_index == 0
    assert videos[0].frame_count == 2
    assert [v.rank for v in videos] == [1, 2]
    assert videos[1].score == pytest.approx(0.0, abs=1e-6)


def test_sort_by_score_preserves_retriever_tie_order(tmp_path: Path) -> None:
    retriever = IndexRetriever(_make_index(tmp_path))
    results = retriever.search(np.array([0, 0, 0, 1], dtype=np.float32), top_k=3)

    ordered = sort_by_score(results)

    assert [(c.video_id, c.frame_index) for c in ordered] == [
        (c.video_id, c.frame_index) for c in results
    ]
