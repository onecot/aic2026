"""Typed QA prediction records and artifact contract.

This module defines the machine-readable QA artifact schema:

- ``manifest.json`` holds run/experiment metadata (one object).
- ``predictions.jsonl`` holds one ``QaQueryResult`` per line, each carrying
  ranked frame-level evidence, optional video-level aggregates, and optional
  temporal window candidates.

The records intentionally reuse the task-independent candidates from
:mod:`aic2026.retrieval_candidates` (``VideoCandidate``,
``WindowCandidate``) so QA never re-implements grouping, aggregation, or
ranking. The artifact is produced by
:func:`aic2026.qa.qa_prediction_runner.run_qa_prediction` and consumed by
the isolated submission formatter (:mod:`aic2026.qa.qa_submission`) and the
future QA scorer (:mod:`aic2026.qa.qa_scoring`).

The textual ``answer`` carried by :class:`QaQueryResult` is a per-query
string that the retrieval foundation leaves empty; a future answer-generation
stage (LLM/VLM) fills it before submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from aic2026.retrieval_candidates import (
    SUPPORTED_VIDEO_AGGREGATIONS,
    VideoCandidate,
    WindowCandidate,
)

QA_PREDICTION_SCHEMA_VERSION = "1.0"
QueryTextField = Literal["query_text_vi", "query_text_en"]


@dataclass(frozen=True)
class QaPrediction:
    """One ranked frame-level QA evidence prediction.

    Expected input:
        - rank: 1-based rank assigned by the candidate source.
        - video_id: canonical video identifier.
        - frame_index: original-video frame index.
        - score: retrieval similarity score (numeric).

    Returns:
        - A validated QA evidence record.

    Raises:
        - ValueError: if any field violates the contract.
    """

    rank: int
    video_id: str
    frame_index: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.video_id, str) or self.video_id == "":
            raise ValueError("video_id must be a non-empty string")
        if not isinstance(self.frame_index, int) or isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise ValueError("score must be numeric")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a JSON-safe dictionary."""
        return {
            "rank": self.rank,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "score": float(self.score),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QaPrediction:
        """Deserialize one QA evidence record."""
        return cls(
            rank=int(data["rank"]),
            video_id=str(data["video_id"]),
            frame_index=int(data["frame_index"]),
            score=float(data["score"]),
        )


@dataclass(frozen=True)
class QaQueryResult:
    """Ranked QA evidence for one evaluation query.

    Expected input:
        - query_id: canonical query identifier.
        - query_text: the resolved query text that was encoded.
        - answer: per-query textual answer; the retrieval foundation leaves
          this empty (a later answer-generation stage fills it).
        - predictions: ranked frame-level evidence, empty when retrieval
          returned no candidates.
        - video_candidates: video-level aggregates, ordered by aggregate
          score.
        - window_candidates: optional temporal windows; empty when no window
          grouping was configured.

    Returns:
        - A validated per-query QA result.

    Raises:
        - ValueError: if any field violates the contract.
    """

    query_id: str
    query_text: str
    answer: str
    predictions: tuple[QaPrediction, ...]
    video_candidates: tuple[VideoCandidate, ...]
    window_candidates: tuple[WindowCandidate, ...]
    answer_evidence: tuple[QaPrediction, ...] | None = None
    answer_error: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("query_id", self.query_id), ("query_text", self.query_text)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.answer, str):
            raise ValueError("answer must be a string (may be empty in the retrieval foundation)")
        if self.answer_evidence is not None and not all(
            isinstance(item, QaPrediction) for item in self.answer_evidence
        ):
            raise ValueError("answer_evidence must contain only typed QaPrediction records")
        if self.answer_error is not None and (
            not isinstance(self.answer_error, str) or self.answer_error == ""
        ):
            raise ValueError("answer_error must be a non-empty string or None")
        for label, candidates in (
            ("predictions", self.predictions),
            ("video_candidates", self.video_candidates),
            ("window_candidates", self.window_candidates),
        ):
            if not all(isinstance(item, (QaPrediction, VideoCandidate, WindowCandidate)) for item in candidates):
                raise ValueError(f"{label} must contain typed candidate records")
            previous = 0
            for item in candidates:
                if item.rank <= previous:
                    raise ValueError(f"{label} ranks must be strictly increasing")
                previous = item.rank

    def to_dict(self) -> dict[str, Any]:
        """Serialize this query result to a JSON-safe dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "answer": self.answer,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "video_candidates": [_video_candidate_to_dict(item) for item in self.video_candidates],
            "window_candidates": [_window_candidate_to_dict(item) for item in self.window_candidates],
            "answer_evidence": (
                [prediction.to_dict() for prediction in self.answer_evidence]
                if self.answer_evidence is not None
                else None
            ),
            "answer_error": self.answer_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QaQueryResult:
        """Deserialize one query result."""
        evidence = data.get("answer_evidence")
        return cls(
            query_id=str(data["query_id"]),
            query_text=str(data["query_text"]),
            answer=str(data["answer"]),
            predictions=tuple(QaPrediction.from_dict(item) for item in data["predictions"]),
            video_candidates=tuple(_video_candidate_from_dict(item) for item in data["video_candidates"]),
            window_candidates=tuple(_window_candidate_from_dict(item) for item in data["window_candidates"]),
            answer_evidence=tuple(QaPrediction.from_dict(item) for item in evidence) if evidence is not None else None,
            answer_error=data.get("answer_error"),
        )


@dataclass(frozen=True)
class QaPredictionManifest:
    """Run-level metadata for one reproducible QA prediction run.

    Expected input:
        - schema_version: must equal ``QA_PREDICTION_SCHEMA_VERSION``.
        - run_id / created_at: provenance strings.
        - evaluation_set_path / evaluation_set_query_count /
          processed_query_count: input identity and processing coverage.
        - query_text_field: ``query_text_vi`` or ``query_text_en``.
        - top_k: frame retrieval cutoff per query.
        - video_aggregation: video-level aggregation strategy used to rank
          candidates (``None`` = frame-level ranking), or one of
          :data:`SUPPORTED_VIDEO_AGGREGATIONS`.
        - video_aggregation_top_n: ``mean_top_n`` parameter (required only
          for that method).
        - window_max_gap: temporal window grouping gap, or ``None``.
        - embedding_space_compatible / embedding_space_override_used:
          embedding-space diagnostics.
        - index_artifact_path / index_manifest / model_embedding_space /
          model_config: artifact provenance.
        - selected_query_ids: explicit subset, or ``None``.
        - answer_generation: optional answer-generation section (model,
          checkpoint, device, top_n, exact prompt, keyframes dir, evidence /
          failed query counts) recorded when answers were generated; ``None``
          for retrieval-only runs.

    Returns:
        - A validated QA prediction manifest.

    Raises:
        - ValueError: if any field violates the contract.
    """

    schema_version: str
    run_id: str
    created_at: str
    evaluation_set_path: str
    evaluation_set_query_count: int
    processed_query_count: int
    query_text_field: QueryTextField
    top_k: int
    video_aggregation: str | None
    video_aggregation_top_n: int | None
    window_max_gap: int | None
    embedding_space_compatible: bool
    embedding_space_override_used: bool
    index_artifact_path: str
    index_manifest: dict[str, Any]
    model_embedding_space: dict[str, Any]
    model_config: dict[str, Any] | None
    selected_query_ids: tuple[str, ...] | None
    answer_generation: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != QA_PREDICTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported QA prediction schema version {self.schema_version!r}; "
                f"expected {QA_PREDICTION_SCHEMA_VERSION!r}"
            )
        for name, value in (("run_id", self.run_id), ("created_at", self.created_at)):
            if not isinstance(value, str) or value.strip() == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.evaluation_set_query_count < 1:
            raise ValueError("evaluation_set_query_count must be >= 1")
        if self.processed_query_count < 1:
            raise ValueError("processed_query_count must be >= 1")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
        if self.video_aggregation is not None and self.video_aggregation not in SUPPORTED_VIDEO_AGGREGATIONS:
            raise ValueError(
                f"unsupported video_aggregation {self.video_aggregation!r}; "
                f"supported: {sorted(SUPPORTED_VIDEO_AGGREGATIONS)}"
            )
        if self.video_aggregation == "mean_top_n":
            if (
                self.video_aggregation_top_n is None
                or not isinstance(self.video_aggregation_top_n, int)
                or isinstance(self.video_aggregation_top_n, bool)
                or self.video_aggregation_top_n < 1
            ):
                raise ValueError("video_aggregation_top_n must be a positive integer for 'mean_top_n'")
        elif self.video_aggregation_top_n is not None:
            raise ValueError("video_aggregation_top_n is only valid with 'mean_top_n'")
        if self.window_max_gap is not None and (
            not isinstance(self.window_max_gap, int)
            or isinstance(self.window_max_gap, bool)
            or self.window_max_gap < 1
        ):
            raise ValueError("window_max_gap must be a positive integer or None")
        for name, value in (
            ("embedding_space_compatible", self.embedding_space_compatible),
            ("embedding_space_override_used", self.embedding_space_override_used),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if self.embedding_space_override_used and self.embedding_space_compatible:
            raise ValueError("embedding_space_override_used cannot be true when spaces are compatible")
        if not isinstance(self.index_manifest, dict) or not isinstance(self.model_embedding_space, dict):
            raise ValueError("index_manifest and model_embedding_space must be dictionaries")
        if self.answer_generation is not None and not isinstance(self.answer_generation, dict):
            raise ValueError("answer_generation must be a dictionary or None")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "evaluation_set_path": self.evaluation_set_path,
            "evaluation_set_query_count": self.evaluation_set_query_count,
            "processed_query_count": self.processed_query_count,
            "query_text_field": self.query_text_field,
            "top_k": self.top_k,
            "video_aggregation": self.video_aggregation,
            "video_aggregation_top_n": self.video_aggregation_top_n,
            "window_max_gap": self.window_max_gap,
            "embedding_space_compatible": self.embedding_space_compatible,
            "embedding_space_override_used": self.embedding_space_override_used,
            "index_artifact_path": self.index_artifact_path,
            "index_manifest": self.index_manifest,
            "model_embedding_space": self.model_embedding_space,
            "model_config": self.model_config,
            "selected_query_ids": list(self.selected_query_ids) if self.selected_query_ids is not None else None,
            "answer_generation": self.answer_generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QaPredictionManifest:
        """Deserialize and validate a manifest dictionary."""
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
            video_aggregation=data.get("video_aggregation"),
            video_aggregation_top_n=data.get("video_aggregation_top_n"),
            window_max_gap=data.get("window_max_gap"),
            embedding_space_compatible=bool(data["embedding_space_compatible"]),
            embedding_space_override_used=bool(data["embedding_space_override_used"]),
            index_artifact_path=str(data["index_artifact_path"]),
            index_manifest=dict(data["index_manifest"]),
            model_embedding_space=dict(data["model_embedding_space"]),
            model_config=dict(data["model_config"]) if data.get("model_config") is not None else None,
            selected_query_ids=tuple(selected) if selected is not None else None,
            answer_generation=dict(data["answer_generation"]) if data.get("answer_generation") is not None else None,
        )


@dataclass(frozen=True)
class QaPredictionArtifact:
    """One complete machine-readable QA prediction artifact.

    Expected input:
        - manifest: run-level metadata; ``processed_query_count`` must equal
          the number of query results.
        - query_results: one result per processed query, in query order.

    Returns:
        - A validated QA prediction artifact.

    Raises:
        - ValueError: if the manifest/result counts disagree.
    """

    manifest: QaPredictionManifest
    query_results: tuple[QaQueryResult, ...]

    def __post_init__(self) -> None:
        if self.manifest.processed_query_count != len(self.query_results):
            raise ValueError("manifest.processed_query_count must equal len(query_results)")


def _video_candidate_to_dict(candidate: VideoCandidate) -> dict[str, Any]:
    """Serialize a task-independent video candidate to JSON-safe fields."""
    return {
        "video_id": candidate.video_id,
        "score": float(candidate.score),
        "rank": candidate.rank,
        "frame_count": candidate.frame_count,
        "top_frame_index": candidate.top_frame_index,
    }


def _video_candidate_from_dict(data: dict[str, Any]) -> VideoCandidate:
    """Deserialize a video candidate from JSON-safe fields."""
    return VideoCandidate(
        video_id=str(data["video_id"]),
        score=float(data["score"]),
        rank=int(data["rank"]),
        frame_count=int(data["frame_count"]),
        top_frame_index=data.get("top_frame_index"),
    )


def _window_candidate_to_dict(candidate: WindowCandidate) -> dict[str, Any]:
    """Serialize a task-independent window candidate to JSON-safe fields."""
    return {
        "video_id": candidate.video_id,
        "frame_start": candidate.frame_start,
        "frame_end": candidate.frame_end,
        "score": float(candidate.score),
        "rank": candidate.rank,
    }


def _window_candidate_from_dict(data: dict[str, Any]) -> WindowCandidate:
    """Deserialize a window candidate from JSON-safe fields."""
    return WindowCandidate(
        video_id=str(data["video_id"]),
        frame_start=int(data["frame_start"]),
        frame_end=int(data["frame_end"]),
        score=float(data["score"]),
        rank=int(data["rank"]),
    )