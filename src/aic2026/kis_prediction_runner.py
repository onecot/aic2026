"""KIS prediction runner and prediction-artifact contract.

This module wires the existing stages:

    EvaluationSet -> text encoder -> IndexRetriever -> KisPrediction

It intentionally does not implement any retrieval-model improvements such as
reranking or query expansion. The output artifact is machine-readable and can
be converted directly to the existing ``evaluate_kis`` input contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from aic2026.embeddings import EmbeddingEncoder, EmbeddingSpace, validate_query_compatibility
from aic2026.evaluation import EvaluationQuery, EvaluationTask, load_eval_set
from aic2026.index_artifact import IndexManifest
from aic2026.kis_scoring import KisPrediction
from aic2026.retrieval import IndexRetriever, RetrievalCandidate
from aic2026.text_query import TextQuery

KIS_PREDICTION_SCHEMA_VERSION = "1.0"
QueryTextField = Literal["query_text_vi", "query_text_en"]


@dataclass(frozen=True)
class KisPredictionRunnerConfig:
    """Configuration for one reproducible KIS prediction run.

    Expected input:
        - evaluation_set_path: CSV path consumed by ``load_eval_set``.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_path: output JSON artifact path.
        - model_config: optional JSON-serializable model configuration.
        - top_k: positive candidate count per query.
        - query_text_field: ``query_text_vi`` (default) or ``query_text_en``.
        - query_subset_ids: optional explicit subset of query ids.
                - allow_embedding_space_mismatch: explicit opt-in override for
                    diagnostic experiments only. Disabled by default.
        - run_id: optional stable run id; auto-generated when omitted.

    Returns:
        - A validated runner configuration.

    Raises:
        - ValueError: if required fields are invalid.
    """

    evaluation_set_path: Path
    index_artifact_dir: Path
    prediction_output_path: Path
    model_config: dict[str, Any] | None = None
    top_k: int = 100
    query_text_field: QueryTextField = "query_text_vi"
    query_subset_ids: tuple[str, ...] | None = None
    allow_embedding_space_mismatch: bool = False
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
        if not isinstance(self.allow_embedding_space_mismatch, bool):
            raise ValueError("allow_embedding_space_mismatch must be a boolean")
        if self.query_subset_ids is not None:
            if len(self.query_subset_ids) == 0:
                raise ValueError("query_subset_ids must be None or a non-empty tuple")
            if any(not isinstance(query_id, str) or query_id.strip() == "" for query_id in self.query_subset_ids):
                raise ValueError("query_subset_ids must contain only non-empty strings")


@dataclass(frozen=True)
class RankedKisPrediction:
    """One ranked prediction with retrieval score for artifact persistence."""

    rank: int
    video_id: str
    frame_index: int
    score: float

    def __post_init__(self) -> None:
        KisPrediction(rank=self.rank, video_id=self.video_id, frame_index=self.frame_index)
        if not isinstance(self.score, (int, float)):
            raise ValueError("prediction score must be numeric")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a JSON-safe dictionary."""
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "score": float(self.score),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RankedKisPrediction:
        """Deserialize one ranked prediction record."""
        return cls(
            rank=int(data["rank"]),
            video_id=str(data["video_id"]),
            frame_index=int(data["frame_index"]),
            score=float(data["score"]),
        )

    def to_kis_prediction(self) -> KisPrediction:
        """Convert to scorer-compatible prediction identity."""
        return KisPrediction(rank=self.rank, video_id=self.video_id, frame_index=self.frame_index)


@dataclass(frozen=True)
class QueryPredictionResult:
    """Ranked predictions for one evaluation query in preserved order."""

    query_id: str
    query_text: str
    predictions: tuple[RankedKisPrediction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.query_text, str) or self.query_text.strip() == "":
            raise ValueError("query_text must be a non-empty string")
        previous_rank = 0
        for prediction in self.predictions:
            if prediction.rank <= previous_rank:
                raise ValueError("prediction ranks must be strictly increasing")
            previous_rank = prediction.rank

    def to_dict(self) -> dict[str, Any]:
        """Serialize this query result to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryPredictionResult:
        """Deserialize one query prediction result."""
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            predictions=tuple(RankedKisPrediction.from_dict(item) for item in data["predictions"]),
        )


@dataclass(frozen=True)
class KisPredictionArtifact:
    """Machine-readable KIS prediction artifact for reproducible evaluation."""

    schema_version: str
    run_id: str
    created_at: str
    evaluation_set_path: str
    evaluation_set_query_count: int
    processed_query_count: int
    query_text_field: QueryTextField
    top_k: int
    embedding_space_compatible: bool
    embedding_space_override_used: bool
    index_artifact_path: str
    index_manifest: dict[str, Any]
    model_embedding_space: dict[str, Any]
    model_config: dict[str, Any] | None
    selected_query_ids: tuple[str, ...] | None
    query_results: tuple[QueryPredictionResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != KIS_PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported KIS prediction schema version {self.schema_version!r}; "
                f"expected {KIS_PREDICTION_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.run_id, str) or self.run_id.strip() == "":
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.created_at, str) or self.created_at.strip() == "":
            raise ValueError("created_at must be a non-empty string")
        if self.evaluation_set_query_count < 1:
            raise ValueError("evaluation_set_query_count must be >= 1")
        if self.processed_query_count < 1:
            raise ValueError("processed_query_count must be >= 1")
        if self.processed_query_count != len(self.query_results):
            raise ValueError("processed_query_count must equal len(query_results)")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(self.embedding_space_compatible, bool):
            raise ValueError("embedding_space_compatible must be a boolean")
        if not isinstance(self.embedding_space_override_used, bool):
            raise ValueError("embedding_space_override_used must be a boolean")
        if self.embedding_space_override_used and self.embedding_space_compatible:
            raise ValueError("embedding_space_override_used cannot be true when spaces are compatible")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full artifact to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "evaluation_set_path": self.evaluation_set_path,
            "evaluation_set_query_count": self.evaluation_set_query_count,
            "processed_query_count": self.processed_query_count,
            "query_text_field": self.query_text_field,
            "top_k": self.top_k,
            "embedding_space_compatible": self.embedding_space_compatible,
            "embedding_space_override_used": self.embedding_space_override_used,
            "index_artifact_path": self.index_artifact_path,
            "index_manifest": self.index_manifest,
            "model_embedding_space": self.model_embedding_space,
            "model_config": self.model_config,
            "selected_query_ids": list(self.selected_query_ids) if self.selected_query_ids is not None else None,
            "query_results": [query_result.to_dict() for query_result in self.query_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KisPredictionArtifact:
        """Deserialize and validate a prediction artifact dictionary."""
        selected = data.get("selected_query_ids")
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            evaluation_set_path=str(data["evaluation_set_path"]),
            evaluation_set_query_count=int(data["evaluation_set_query_count"]),
            processed_query_count=int(data["processed_query_count"]),
            query_text_field=data["query_text_field"],
            top_k=int(data["top_k"]),
            embedding_space_compatible=bool(data["embedding_space_compatible"]),
            embedding_space_override_used=bool(data["embedding_space_override_used"]),
            index_artifact_path=str(data["index_artifact_path"]),
            index_manifest=dict(data["index_manifest"]),
            model_embedding_space=dict(data["model_embedding_space"]),
            model_config=dict(data["model_config"]) if data.get("model_config") is not None else None,
            selected_query_ids=tuple(selected) if selected is not None else None,
            query_results=tuple(QueryPredictionResult.from_dict(item) for item in data["query_results"]),
        )

    def to_kis_predictions_by_query(self) -> dict[str, list[KisPrediction]]:
        """Convert the artifact to scorer-compatible KIS predictions.

        Returns:
            - Mapping from query id to ranked ``KisPrediction`` values.
        """
        return {
            query_result.query_id: [prediction.to_kis_prediction() for prediction in query_result.predictions]
            for query_result in self.query_results
        }


def run_kis_prediction(
    config: KisPredictionRunnerConfig,
    encoder: EmbeddingEncoder,
) -> KisPredictionArtifact:
    """Run KIS prediction generation for all or an explicit subset of queries.

    Expected input:
        - config: validated run configuration.
        - encoder: text encoder implementing the embedding contract.

    Returns:
        - Prediction artifact written to ``config.prediction_output_path`` and
          returned as a typed value.

    Raises:
        - ValueError: if the evaluation data is incompatible with KIS
          prediction, subset ids are invalid, or embedding spaces mismatch.
        - FileNotFoundError: if configured input files are absent.
    """
    evaluation_set = load_eval_set(config.evaluation_set_path)
    all_kis_queries = tuple(evaluation_set.iter_queries(EvaluationTask.KIS))
    if not all_kis_queries:
        raise ValueError("evaluation set contains no KIS queries")

    selected = _select_queries(all_kis_queries, config.query_subset_ids)
    queries = tuple(
        TextQuery(
            query_id=query.query_id,
            query_text=_resolve_query_text(
                query.query_text_vi, query.query_text_en, config.query_text_field
            ),
        )
        for query in selected
    )
    return run_kis_prediction_for_queries(
        queries,
        encoder=encoder,
        index_artifact_dir=config.index_artifact_dir,
        prediction_output_path=config.prediction_output_path,
        top_k=config.top_k,
        query_text_field=config.query_text_field,
        model_config=config.model_config,
        allow_embedding_space_mismatch=config.allow_embedding_space_mismatch,
        run_id=config.run_id,
        query_source_path=Path(config.evaluation_set_path),
        query_source_count=len(all_kis_queries),
        selected_query_ids=config.query_subset_ids,
    )


def run_kis_prediction_for_queries(
    queries: tuple[TextQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_path: Path,
    top_k: int = 100,
    query_text_field: QueryTextField = "query_text_vi",
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
    query_source_path: Path,
    query_source_count: int,
    selected_query_ids: tuple[str, ...] | None = None,
) -> KisPredictionArtifact:
    """Run KIS prediction for source-resolved text queries.

    This is the source-independent KIS prediction core: it consumes queries
    whose text is already resolved (no evaluation CSV is read) and reuses the
    existing retrieval stack unchanged.

    Expected input:
        - queries: at least one resolved text query, in processing order.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_path: output JSON artifact path.
        - top_k: positive candidate count per query.
        - query_text_field: provenance label recorded in the artifact
          manifest (``query_text_vi`` or ``query_text_en``).
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.
        - query_source_path: identity of the query source (eval-set CSV path
          or BTC query directory), recorded as artifact provenance.
        - query_source_count: total queries available at the source for this
          task (>= number processed).
        - selected_query_ids: optional explicit subset recorded in the
          artifact for provenance.

    Returns:
        - Prediction artifact written to ``prediction_output_path`` and
          returned as a typed value.

    Raises:
        - ValueError: if there are no queries, the source count is invalid,
          or embedding spaces mismatch under strict mode.
        - FileNotFoundError: if the index artifact is absent.
    """
    if not queries:
        raise ValueError("queries must contain at least one KIS query")
    if not isinstance(query_source_count, int) or isinstance(query_source_count, bool) or query_source_count < 1:
        raise ValueError("query_source_count must be a positive integer")
    if len(queries) > query_source_count:
        raise ValueError("query_source_count must be >= the number of queries")

    retriever = IndexRetriever(index_artifact_dir)
    spaces_compatible = encoder.embedding_space.is_compatible_with(retriever.manifest.embedding_space)
    override_used = False
    if not spaces_compatible:
        if not allow_embedding_space_mismatch:
            _require_compatible_spaces(encoder.embedding_space, retriever.manifest)
        override_used = True

    results: list[QueryPredictionResult] = []
    for query in queries:
        query_embedding = encoder.encode_text(query.query_text)
        retrieval = _search_candidates(
            retriever=retriever,
            query_embedding=query_embedding,
            top_k=top_k,
            allow_mismatch=allow_embedding_space_mismatch,
        )
        results.append(
            QueryPredictionResult(
                query_id=query.query_id,
                query_text=query.query_text,
                predictions=tuple(_to_ranked_prediction(candidate) for candidate in retrieval),
            )
        )

    artifact = KisPredictionArtifact(
        schema_version=KIS_PREDICTION_SCHEMA_VERSION,
        run_id=run_id or _default_run_id(),
        created_at=_utc_now_iso(),
        evaluation_set_path=str(Path(query_source_path).resolve()),
        evaluation_set_query_count=query_source_count,
        processed_query_count=len(results),
        query_text_field=query_text_field,
        top_k=top_k,
        embedding_space_compatible=spaces_compatible,
        embedding_space_override_used=override_used,
        index_artifact_path=str(Path(index_artifact_dir).resolve()),
        index_manifest=retriever.manifest.to_dict(),
        model_embedding_space=encoder.embedding_space.to_dict(),
        model_config=model_config,
        selected_query_ids=selected_query_ids,
        query_results=tuple(results),
    )
    write_kis_prediction_artifact(prediction_output_path, artifact)
    return artifact


def write_kis_prediction_artifact(path: Path | str, artifact: KisPredictionArtifact) -> None:
    """Write one prediction artifact JSON atomically.

    Expected input:
        - path: target JSON file path.
        - artifact: typed prediction artifact.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2)
        handle.write("\n")
    temp_path.replace(output_path)


def read_kis_prediction_artifact(path: Path | str) -> KisPredictionArtifact:
    """Read and validate one KIS prediction artifact JSON."""
    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return KisPredictionArtifact.from_dict(payload)


def load_kis_predictions_for_scoring(path: Path | str) -> dict[str, list[KisPrediction]]:
    """Load a prediction artifact and return scorer-ready predictions."""
    return read_kis_prediction_artifact(path).to_kis_predictions_by_query()


def _select_queries(
    queries: tuple[EvaluationQuery, ...],
    subset_ids: tuple[str, ...] | None,
) -> tuple[EvaluationQuery, ...]:
    """Return ordered KIS queries, optionally restricted to explicit ids."""
    if subset_ids is None:
        return queries

    expected = {query.query_id for query in queries}
    requested = set(subset_ids)
    unknown = requested - expected
    if unknown:
        raise ValueError(f"query_subset_ids contain unknown query ids: {sorted(unknown)}")

    selected = tuple(query for query in queries if query.query_id in requested)
    if not selected:
        raise ValueError("query_subset_ids selected no queries")
    return selected


def _resolve_query_text(query_text_vi: str, query_text_en: str | None, field: QueryTextField) -> str:
    """Select and validate one query text field from a canonical query."""
    if field == "query_text_vi":
        return query_text_vi
    if query_text_en is None:
        raise ValueError("query_text_en was requested but is missing for at least one query")
    return query_text_en


def _require_compatible_spaces(model_space: EmbeddingSpace, manifest: IndexManifest) -> None:
    """Require model/index embedding spaces to be compatible.

    Uses :meth:`EmbeddingSpace.is_compatible_with`, matching the QA and
    TRAKE runners: build-environment-only ``model_version`` differences are
    ignored, while every comparison-relevant field must match.
    """
    index_space = manifest.embedding_space
    if index_space is None:
        raise ValueError("index artifact is missing embedding-space metadata")
    if not model_space.is_compatible_with(index_space):
        raise ValueError(
            "incompatible embedding spaces between encoder and index: "
            f"model {model_space.space_id!r} ({model_space.backend}/{model_space.model_id}) != "
            f"index {index_space.space_id!r} ({index_space.backend}/{index_space.model_id})"
        )


def _to_ranked_prediction(candidate: RetrievalCandidate) -> RankedKisPrediction:
    """Convert one retrieval candidate to a persisted KIS prediction record."""
    return RankedKisPrediction(
        rank=candidate.rank,
        video_id=candidate.video_id,
        frame_index=candidate.frame_index,
        score=candidate.score,
    )


def _search_candidates(
    retriever: IndexRetriever,
    query_embedding: Any,
    top_k: int,
    allow_mismatch: bool,
) -> list[RetrievalCandidate]:
    """Run retrieval under strict mode or explicit mismatch override.

    Expected input:
        - retriever: loaded index retriever.
        - query_embedding: text embedding generated by the encoder.
        - top_k: retrieval cutoff.
        - allow_mismatch: explicit override gate.

    Returns:
        - Ranked retrieval candidates.

    Raises:
        - ValueError: for incompatible spaces under strict mode, or when
          override is requested for a non-legacy index that cannot accept
          raw vectors by contract.
    """
    if not allow_mismatch:
        validate_query_compatibility(query_embedding, retriever.manifest.embedding_space)
        return retriever.search(query_embedding, top_k=top_k)

    if query_embedding.space.is_compatible_with(retriever.manifest.embedding_space):
        return retriever.search(query_embedding, top_k=top_k)

    index_space = retriever.manifest.embedding_space
    if index_space is None:
        raise ValueError("index artifact is missing embedding-space metadata")
    if index_space.backend != "legacy":
        raise ValueError(
            "embedding-space mismatch override is only supported for legacy indexes; "
            f"index backend is {index_space.backend!r}"
        )
    query_vector = np.asarray(query_embedding.vector)
    return retriever.search(query_vector, top_k=top_k)


def _default_run_id() -> str:
    """Return a deterministic-format run id for artifact provenance."""
    return f"kis-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
