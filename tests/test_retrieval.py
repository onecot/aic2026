from pathlib import Path

import numpy as np
import pytest

from aic2026.dataset import Dataset
from aic2026.index_builder import build_index
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
    artifact = Path("data/interim/indexes/test-subset")
    if not artifact.is_dir():
        pytest.skip("real subset index artifact not available")
    retriever = IndexRetriever(artifact)
    q = np.zeros(retriever.manifest.embedding_dim, dtype=np.float32)
    q[0] = 1.0
    res = retriever.search(q, top_k=1)
    assert len(res) == 1

