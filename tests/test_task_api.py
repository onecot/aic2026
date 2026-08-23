"""Contract tests for the shared in-memory task API.

Covers the boundaries Phase 3 (API layer) will wrap:

- ``SearchEngine`` construction, search, and embedding-space gating;
- single-query in-memory functions ``search_kis`` / ``search_qa`` /
  ``search_trake``;
- compute/write separation of the prediction artifacts;
- the canonical keyframe image resolver.

The suite runs on the deterministic synthetic index from
:mod:`aic2026.qa.qa_smoke` — no real model, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.kis_prediction_runner import (
    KIS_PREDICTION_SCHEMA_VERSION,
    QueryPredictionResult,
    compute_kis_artifact,
    read_kis_prediction_artifact,
    search_kis,
)
from aic2026.qa.qa_prediction_runner import read_qa_artifact, search_qa
from aic2026.qa.qa_smoke import (
    QaPlaceholderEncoder,
    QaSmokeConfig,
    build_synthetic_index,
    smoke_embedding_space,
)
from aic2026.retrieval_candidates import WindowCandidate
from aic2026.search import SearchEngine
from aic2026.text_query import TextQuery
from aic2026.trake import parse_events, read_trake_artifact, search_trake
from aic2026.frame_images import find_keyframe_image


@pytest.fixture()
def synthetic_index(tmp_path: Path) -> Path:
    """Build the small deterministic index artifact used by these tests."""
    config = QaSmokeConfig(
        evaluation_set_path=tmp_path / "eval.csv",
        work_dir=tmp_path,
        video_ids=("L21_V001", "L21_V002", "L21_V003"),
    )
    return build_synthetic_index(tmp_path / "index", config)


@pytest.fixture()
def engine(synthetic_index: Path) -> SearchEngine:
    """A SearchEngine over the synthetic index with a placeholder encoder."""
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), 8)
    return SearchEngine.from_parts(encoder, synthetic_index, top_k=10)


class TestSearchEngine:
    def test_search_text_returns_ranked_candidates(self, engine: SearchEngine) -> None:
        results = engine.search_text("a red boat on the water")
        assert len(results) == 10
        ranks = [candidate.rank for candidate in results]
        assert ranks == list(range(1, 11))
        scores = [candidate.score for candidate in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_per_call_top_k(self, engine: SearchEngine) -> None:
        results = engine.search_text("anything", top_k=3)
        assert len(results) == 3

    def test_spaces_compatible_without_override(self, engine: SearchEngine) -> None:
        assert engine.spaces_compatible is True
        assert engine.override_used is False

    def test_incompatible_space_rejected_at_construction(
        self, synthetic_index: Path
    ) -> None:
        from aic2026.embeddings import EmbeddingSpace

        other_space = EmbeddingSpace(
            backend="test-backend",
            model_id="different-model",
            checkpoint_id="other-v1",
            space_id="other-space-v1",
            embedding_dim=8,
            output_dtype="float32",
            normalization="l2",
            preprocessing_id="other-preprocess-v1",
        )
        encoder = QaPlaceholderEncoder(other_space, 8)
        with pytest.raises(ValueError, match="incompatible embedding spaces"):
            SearchEngine.from_parts(encoder, synthetic_index)


class TestSearchKis:
    def test_single_query_in_memory(self, engine: SearchEngine) -> None:
        result = search_kis(engine, TextQuery(query_id="q1", query_text="boat on water"))
        assert isinstance(result, QueryPredictionResult)
        assert result.query_id == "q1"
        assert result.query_text == "boat on water"
        assert [p.rank for p in result.predictions] == list(range(1, 11))

    def test_plain_string_query_uses_query_id(self, engine: SearchEngine) -> None:
        result = search_kis(engine, "plain text", query_id="custom-id")
        assert result.query_id == "custom-id"

    def test_compute_artifact_is_not_written_until_told(
        self, engine: SearchEngine, tmp_path: Path
    ) -> None:
        queries = (
            TextQuery(query_id="q1", query_text="one"),
            TextQuery(query_id="q2", query_text="two"),
        )
        artifact = compute_kis_artifact(
            queries,
            engine=engine,
            query_source_path=tmp_path / "queries.csv",
            query_source_count=2,
        )
        output = tmp_path / "nested" / "kis.json"
        assert not output.parent.exists() or not output.exists()

        from aic2026.kis_prediction_runner import write_kis_prediction_artifact

        write_kis_prediction_artifact(output, artifact)
        loaded = read_kis_prediction_artifact(output)
        assert loaded.schema_version == KIS_PREDICTION_SCHEMA_VERSION
        assert loaded.processed_query_count == 2
        assert [r.query_id for r in loaded.query_results] == ["q1", "q2"]


class TestSearchQa:
    def test_single_query_evidence_only(self, engine: SearchEngine) -> None:
        result = search_qa(engine, TextQuery(query_id="qa1", query_text="what color is the boat"))
        assert result.answer == ""
        assert result.answer_error is None
        assert len(result.predictions) > 0
        assert all(v.frame_count >= 1 for v in result.video_candidates)

    def test_windows_generated_when_gap_set(self, engine: SearchEngine) -> None:
        result = search_qa(engine, "question text", window_max_gap=5)
        assert all(isinstance(w, WindowCandidate) for w in result.window_candidates)

    def test_answerer_requires_keyframes_resolver(self, engine: SearchEngine) -> None:
        answerer = object()
        with pytest.raises(ValueError, match="keyframes resolver is required"):
            search_qa(engine, "question", answerer=answerer)


class TestSearchTrake:
    def test_events_parsed_and_retrieved(self, engine: SearchEngine) -> None:
        text = "(1) a man enters the room (2) he turns on the lights"
        events = parse_events(text)
        assert [e.event_id for e in events] == ["E1", "E2"]
        result = search_trake(engine, TextQuery(query_id="t1", query_text=text))
        assert result.query_id == "t1"
        assert [e.event_id for e in result.events] == ["E1", "E2"]
        for event in result.events:
            assert len(event.predictions) == 10
            assert len(event.windows) >= 1

    def test_windows_disabled_with_none_gap(self, engine: SearchEngine) -> None:
        result = search_trake(engine, "(1) first event", window_max_gap=None)
        assert result.events[0].windows == ()

    def test_batch_write_round_trip(self, engine: SearchEngine, tmp_path: Path) -> None:
        from aic2026.trake import run_trake_prediction_for_queries

        queries = (TextQuery(query_id="t1", query_text="(1) one event"),)
        artifact = run_trake_prediction_for_queries(
            queries,
            encoder=engine.encoder,
            index_artifact_dir=engine.retriever.artifact_dir,
            prediction_output_dir=tmp_path / "trake-out",
            top_k=10,
            query_source_path=tmp_path,
            query_source_count=1,
        )
        loaded = read_trake_artifact(tmp_path / "trake-out")
        assert loaded.manifest.run_id == artifact.manifest.run_id
        assert loaded.query_results[0].events[0].event_id == "E1"


class TestFrameResolver:
    def test_materialized_layout_preferred(self, tmp_path: Path) -> None:
        images = tmp_path / "images" / "L21_V001"
        images.mkdir(parents=True)
        (images / "000001.jpg").write_bytes(b"jpeg")
        assert find_keyframe_image(tmp_path, "L21_V001", 1) == images / "000001.jpg"

    def test_legacy_fallback(self, tmp_path: Path) -> None:
        raw = tmp_path / "keyframes" / "L21_V001"
        raw.mkdir(parents=True)
        (raw / "002.png").write_bytes(b"png")
        assert find_keyframe_image(tmp_path, "L21_V001", 2) == raw / "002.png"

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_keyframe_image(tmp_path, "L21_V001", 9)

    def test_invalid_ordinal_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            find_keyframe_image(tmp_path, "L21_V001", 0)
