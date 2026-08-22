"""Tests for the OCR-only dense retrieval index and its evaluation helper."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from aic2026.embeddings import Embedding, EmbeddingSpace
from aic2026.evaluation import EvaluationSet, load_eval_set
from aic2026.ocr_chunks import OcrChunk, OcrChunksManifest, write_ocr_chunks
from aic2026.ocr_eval import (
    evaluate_ocr_query,
    evaluate_ocr_retrieval,
    is_ocr_chunk_correct,
)
from aic2026.ocr_index import (
    build_ocr_index,
    read_ocr_index,
    search_ocr_chunks,
    write_ocr_index,
)

DIM = 8


def _space(space_id: str = "fake:enc-1") -> EmbeddingSpace:
    """Return a small fake embedding-space description."""
    return EmbeddingSpace(
        backend="fake",
        model_id="enc-1",
        checkpoint_id="ckpt-1",
        space_id=space_id,
        embedding_dim=DIM,
        output_dtype="float32",
        normalization="l2",
        preprocessing_id="prep-1",
    )


def _vector(text: str) -> np.ndarray:
    """Deterministic bag-of-token vector: same tokens -> similar vectors."""
    vec = np.zeros(DIM, dtype=np.float32)
    for token in text.split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass(frozen=True)
class _FakeEncoder:
    """Minimal text encoder satisfying the encoder contract (no model)."""

    space: EmbeddingSpace

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self.space

    def encode_text(self, text: str) -> Embedding:
        return self.encode_text_batch([text])[0]

    def encode_text_batch(self, texts):
        return [
            Embedding(kind="text", vector=_vector(text), space=self.space) for text in texts
        ]


def _chunk(chunk_id: str, video_id: str, rep: int, start: int, end: int, text: str) -> OcrChunk:
    """Return one OCR chunk with the given identity."""
    ordinals = tuple(range(1, end - start + 2))
    frames = tuple(range(start, end + 1))
    return OcrChunk(
        chunk_id=chunk_id,
        video_id=video_id,
        frame_start=start,
        frame_end=end,
        representative_frame_index=rep,
        keyframe_ordinals=ordinals,
        frame_indices=frames,
        text=text,
        source_count=len(frames),
    )


def _chunks_manifest(chunks: tuple[OcrChunk, ...]) -> OcrChunksManifest:
    """Return a consistent Phase-1 manifest for the given chunks."""
    chunked = sum(chunk.source_count for chunk in chunks)
    return OcrChunksManifest(
        schema_version="1.0",
        artifact_id="test-chunks",
        created_at="2026-01-01T00:00:00+00:00",
        source_ocr_artifact="source",
        max_records=20,
        max_frame_gap=500,
        num_chunks=len(chunks),
        num_videos=len({chunk.video_id for chunk in chunks}),
        num_source_records=chunked,
        num_source_records_chunked=chunked,
        num_source_records_skipped=0,
    )


SAMPLE_CHUNKS = [
    _chunk("L21_V001-ocr-00", "L21_V001", 100, 10, 200, "boat tourists on the water"),
    _chunk("L21_V001-ocr-01", "L21_V001", 400, 210, 500, "school students on stage"),
    _chunk("L22_V003-ocr-00", "L22_V003", 700, 600, 900, "cooking crab with chili peppers"),
]


@pytest.fixture()
def chunks_dir(tmp_path: Path) -> Path:
    """Write a small completed OCR chunk artifact."""
    return write_ocr_chunks(tmp_path / "chunks", _chunks_manifest(SAMPLE_CHUNKS), SAMPLE_CHUNKS)


class TestBuildIndex:
    def test_build_creates_aligned_rows_and_matrix(self, chunks_dir: Path) -> None:
        manifest, embeddings, rows = build_ocr_index(chunks_dir, _FakeEncoder(_space()))
        assert embeddings.shape == (len(SAMPLE_CHUNKS), DIM)
        assert embeddings.dtype == np.float32
        assert [row.chunk_id for row in rows] == [chunk.chunk_id for chunk in SAMPLE_CHUNKS]
        for row, chunk in zip(rows, SAMPLE_CHUNKS):
            assert row.video_id == chunk.video_id
            assert row.representative_frame_index == chunk.representative_frame_index
            assert (row.frame_start, row.frame_end) == (chunk.frame_start, chunk.frame_end)
            assert row.text == chunk.text

    def test_manifest_records_space_and_source(self, chunks_dir: Path) -> None:
        manifest, _, _ = build_ocr_index(
            chunks_dir, _FakeEncoder(_space()), artifact_id="idx", created_at="t0"
        )
        assert manifest.schema_version == "1.0"
        assert manifest.artifact_id == "idx"
        assert manifest.created_at == "t0"
        assert manifest.embedding_dim == DIM
        assert manifest.num_rows == len(SAMPLE_CHUNKS)
        assert manifest.normalization == "l2"
        assert manifest.embedding_space.space_id == "fake:enc-1"
        assert Path(manifest.source_chunks_artifact).name == chunks_dir.name

    def test_save_load_round_trip(self, tmp_path: Path, chunks_dir: Path) -> None:
        manifest, embeddings, rows = build_ocr_index(chunks_dir, _FakeEncoder(_space()))
        output = write_ocr_index(tmp_path / "index", manifest, embeddings, rows)
        handle = read_ocr_index(output)
        assert handle.manifest.to_dict() == manifest.to_dict()
        assert handle.rows == rows
        assert np.array_equal(np.asarray(handle.embeddings), embeddings)


class TestSearch:
    def test_search_ranks_matching_chunk_first(self, chunks_dir: Path) -> None:
        handle = read_handle_for_test(chunks_dir)
        encoder = _FakeEncoder(_space())
        results = search_ocr_chunks(handle, "crab chili cooking", encoder=encoder, top_k=3)
        assert results[0].chunk_id == "L22_V003-ocr-00"
        assert [r.rank for r in results] == [1, 2, 3]
        assert all(-1.0 <= r.score <= 1.0 for r in results)

    def test_search_result_fields(self, chunks_dir: Path) -> None:
        handle = read_handle_for_test(chunks_dir)
        results = handle.search(_FakeEncoder(_space()).encode_text("boat water"), top_k=1)
        top = results[0]
        assert top.chunk_id == "L21_V001-ocr-00"
        assert top.video_id == "L21_V001"
        assert top.representative_frame_index == 100
        assert (top.frame_start, top.frame_end) == (10, 200)
        assert top.text == "boat tourists on the water"

    def test_deterministic_retrieval(self, chunks_dir: Path) -> None:
        handle = read_handle_for_test(chunks_dir)
        encoder = _FakeEncoder(_space())
        a = search_ocr_chunks(handle, "students stage school", encoder=encoder, top_k=3)
        b = search_ocr_chunks(handle, "students stage school", encoder=encoder, top_k=3)
        assert [(r.chunk_id, r.score) for r in a] == [(r.chunk_id, r.score) for r in b]

    def test_top_k_respected(self, chunks_dir: Path) -> None:
        handle = read_handle_for_test(chunks_dir)
        results = handle.search(_FakeEncoder(_space()).encode_text("boat"), top_k=2)
        assert len(results) == 2

    def test_incompatible_space_rejected(self, chunks_dir: Path) -> None:
        handle = read_handle_for_test(chunks_dir)
        other = _FakeEncoder(_space("fake:other"))
        with pytest.raises(ValueError, match="incompatible embedding spaces"):
            search_ocr_chunks(handle, "boat", encoder=other, top_k=3)

    def test_row_metadata_maps_to_source_chunks(self, tmp_path: Path, chunks_dir: Path) -> None:
        manifest, embeddings, rows = build_ocr_index(chunks_dir, _FakeEncoder(_space()))
        output = write_ocr_index(tmp_path / "index", manifest, embeddings, rows)
        stored = [
            json.loads(line)
            for line in (output / "chunks.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [s["chunk_id"] for s in stored] == [c.chunk_id for c in SAMPLE_CHUNKS]
        assert set(stored[0]) == {
            "chunk_id",
            "video_id",
            "representative_frame_index",
            "frame_start",
            "frame_end",
            "text",
        }


def read_handle_for_test(chunks_dir: Path):
    """Build an in-memory index handle from a chunks artifact."""
    manifest, embeddings, rows = build_ocr_index(chunks_dir, _FakeEncoder(_space()))
    from aic2026.ocr_index import OcrIndexHandle

    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
    return OcrIndexHandle(manifest=manifest, embeddings=embeddings, rows=rows, norms=norms)


class TestEvaluationHelper:
    def test_range_mode_hit_and_miss(self) -> None:
        query = _eval_query("kis", "L21_V001", 150, 250)
        miss = _result(1, "L21_V001", rep=100, start=300, end=400)
        wrong_video = _result(2, "L22_V003", rep=180, start=100, end=300)
        hit = _result(3, "L21_V001", rep=100, start=10, end=200)
        assert is_ocr_chunk_correct(query, hit)
        assert not is_ocr_chunk_correct(query, miss)
        assert not is_ocr_chunk_correct(query, wrong_video)
        assert evaluate_ocr_query(query, [miss, wrong_video, hit]).first_correct_rank == 3
        assert evaluate_ocr_query(query, [miss, wrong_video]).first_correct_rank is None

    def test_representative_only_mode(self) -> None:
        query = _eval_query("kis", "L21_V001", 150, 250)
        span_hit = _result(1, "L21_V001", rep=100, start=10, end=200)
        assert is_ocr_chunk_correct(query, span_hit)
        assert not is_ocr_chunk_correct(query, span_hit, representative_only=True)

    def test_evaluate_ocr_retrieval_aggregates(self, chunks_dir: Path) -> None:
        eval_csv = _tmp_csv("kis", "L21_V001", 150, 250)
        eval_set: EvaluationSet = load_eval_set(eval_csv)
        handle = read_handle_for_test(chunks_dir)
        report = evaluate_ocr_retrieval(
            eval_set, handle, _FakeEncoder(_space()), task="kis", top_k=3
        )
        assert report.num_queries == 1
        metrics = dict(report.hit_at_k)
        # The GT range [150, 250] intersects both L21_V001 chunks, so a
        # correct chunk must appear within the retrieved top-3.
        assert metrics[1] in (0.0, 1.0)
        assert metrics[5] == 1.0
        assert metrics[100] == 1.0


def _result(rank: int, video_id: str, rep: int, start: int, end: int):
    from aic2026.ocr_index import OcrSearchResult

    return OcrSearchResult(
        rank=rank,
        chunk_id=f"{video_id}-ocr-00",
        video_id=video_id,
        representative_frame_index=rep,
        frame_start=start,
        frame_end=end,
        score=0.5,
        text="t",
    )


def _eval_query(task: str, video: str, start: int, end: int):
    csv_path = Path(_tmp_csv(task, video, start, end))
    return load_eval_set(csv_path).queries[0]


def _tmp_csv(task: str, video: str, start: int, end: int) -> str:
    import tempfile

    header = (
        "query_id,task,query_text_vi,query_text_en,gt_video_id,"
        "gt_frame_start,gt_frame_end,gt_answer,gt_events,annotator,note\n"
    )
    row = f"{task}_1,{task},một câu truy vấn,one query,{video},{start},{end},,,P1,\n"
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(header + row)
    handle.close()
    return handle.name
