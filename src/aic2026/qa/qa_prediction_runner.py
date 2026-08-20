"""QA prediction runner and artifact persistence.

This module wires the existing stages for the QA task:

    EvaluationSet (QA rows) -> text encoder -> IndexRetriever
        -> frame / video / window candidates -> QA prediction artifact

It reuses :class:`aic2026.retrieval.IndexRetriever` for frame-level search
and the task-independent foundation in :mod:`aic2026.retrieval_candidates`
(``group_candidates_by_video``, ``build_video_candidates``,
``group_windows``, ``sort_by_score``) for video and window evidence. It does
not implement any retrieval-model improvement, reranking, LLM/VLM answer
generation, or QA scoring; the artifact carries ranked evidence only, with
the per-query textual ``answer`` left empty for a future answer-generation
stage.

Artifacts are written as a directory containing ``manifest.json`` (run
metadata) and ``predictions.jsonl`` (one ``QaQueryResult`` per line), both
written atomically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aic2026.embeddings import EmbeddingEncoder, EmbeddingSpace, validate_query_compatibility
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import IndexManifest
from aic2026.qa.qa_dataset import iter_qa_queries, resolve_query_text, select_qa_queries
from aic2026.qa.qa_types import (
    QaPrediction,
    QaPredictionArtifact,
    QaPredictionManifest,
    QaQueryResult,
    QueryTextField,
)
from aic2026.retrieval import IndexRetriever, RetrievalCandidate
from aic2026.retrieval_candidates import (
    WindowCandidate,
    build_video_candidates,
    group_candidates_by_video,
    group_windows,
    sort_by_score,
)


@dataclass(frozen=True)
class QaPredictionRunnerConfig:
    """Configuration for one reproducible QA prediction run.

    Expected input:
        - evaluation_set_path: CSV path consumed by ``load_eval_set``.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - model_config: optional JSON-serializable model configuration.
        - top_k: frame retrieval cutoff per query.
        - query_text_field: ``query_text_vi`` (default) or ``query_text_en``.
        - video_aggregation: ``None`` (frame-level ranking) or one of the
          supported video aggregation methods (``max``, ``mean``,
          ``mean_top_n``); when set, evidence is ranked at video level.
        - video_aggregation_top_n: ``mean_top_n`` parameter (required only
          for that method).
        - window_max_gap: optional positive gap for temporal window grouping;
          ``None`` disables window candidates.
        - query_subset_ids: optional explicit subset of query ids.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only. Disabled by default.
        - run_id: optional stable run id; auto-generated when omitted.

    Returns:
        - A validated QA runner configuration.

    Raises:
        - ValueError: if required fields are invalid.
    """

    evaluation_set_path: Path
    index_artifact_dir: Path
    prediction_output_dir: Path
    model_config: dict[str, Any] | None = None
    top_k: int = 100
    query_text_field: QueryTextField = "query_text_vi"
    video_aggregation: str | None = None
    video_aggregation_top_n: int | None = None
    window_max_gap: int | None = None
    query_subset_ids: tuple[str, ...] | None = None
    allow_embedding_space_mismatch: bool = False
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.query_text_field not in ("query_text_vi", "query_text_en"):
            raise ValueError("query_text_field must be 'query_text_vi' or 'query_text_en'")
        if self.video_aggregation is not None and self.video_aggregation not in (
            "max",
            "mean",
            "mean_top_n",
        ):
            raise ValueError("video_aggregation must be None or one of: max, mean, mean_top_n")
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
        if not isinstance(self.allow_embedding_space_mismatch, bool):
            raise ValueError("allow_embedding_space_mismatch must be a boolean")
        if self.query_subset_ids is not None:
            if len(self.query_subset_ids) == 0:
                raise ValueError("query_subset_ids must be None or a non-empty tuple")
            if any(not isinstance(query_id, str) or query_id.strip() == "" for query_id in self.query_subset_ids):
                raise ValueError("query_subset_ids must contain only non-empty strings")


def run_qa_prediction(
    config: QaPredictionRunnerConfig,
    encoder: EmbeddingEncoder,
) -> QaPredictionArtifact:
    """Run QA prediction generation for all or an explicit subset of queries.

    Expected input:
        - config: validated run configuration.
        - encoder: text encoder implementing the embedding contract.

    Returns:
        - The QA prediction artifact, written to
          ``config.prediction_output_dir`` (``manifest.json`` +
          ``predictions.jsonl``).

    Raises:
        - ValueError: if the evaluation data has no QA queries, subset ids
          are invalid, or embedding spaces mismatch.
        - FileNotFoundError: if configured input files are absent.
    """
    evaluation_set = load_eval_set(config.evaluation_set_path)
    all_qa_queries = iter_qa_queries(evaluation_set)
    if not all_qa_queries:
        raise ValueError("evaluation set contains no QA queries")

    queries = select_qa_queries(evaluation_set, config.query_subset_ids)
    retriever = IndexRetriever(config.index_artifact_dir)
    spaces_compatible = encoder.embedding_space.is_compatible_with(retriever.manifest.embedding_space)
    override_used = False
    if not spaces_compatible:
        if not config.allow_embedding_space_mismatch:
            _require_compatible_spaces(encoder.embedding_space, retriever.manifest)
        override_used = True

    results: list[QaQueryResult] = []
    for query in queries:
        query_text = resolve_query_text(query, config.query_text_field)
        query_embedding = encoder.encode_text(query_text)
        retrieval = _search_candidates(
            retriever=retriever,
            query_embedding=query_embedding,
            top_k=config.top_k,
            allow_mismatch=config.allow_embedding_space_mismatch,
        )
        predictions, video_candidates, window_candidates = _build_evidence(retrieval, config)
        results.append(
            QaQueryResult(
                query_id=query.query_id,
                query_text=query_text,
                answer="",
                predictions=predictions,
                video_candidates=video_candidates,
                window_candidates=window_candidates,
            )
        )

    artifact = QaPredictionArtifact(
        manifest=QaPredictionManifest(
            schema_version="1.0",
            run_id=config.run_id or _default_run_id(),
            created_at=_utc_now_iso(),
            evaluation_set_path=str(Path(config.evaluation_set_path).resolve()),
            evaluation_set_query_count=len(all_qa_queries),
            processed_query_count=len(results),
            query_text_field=config.query_text_field,
            top_k=config.top_k,
            video_aggregation=config.video_aggregation,
            video_aggregation_top_n=config.video_aggregation_top_n,
            window_max_gap=config.window_max_gap,
            embedding_space_compatible=spaces_compatible,
            embedding_space_override_used=override_used,
            index_artifact_path=str(Path(config.index_artifact_dir).resolve()),
            index_manifest=retriever.manifest.to_dict(),
            model_embedding_space=encoder.embedding_space.to_dict(),
            model_config=config.model_config,
            selected_query_ids=config.query_subset_ids,
        ),
        query_results=tuple(results),
    )
    write_qa_artifact(config.prediction_output_dir, artifact)
    return artifact


def write_qa_artifact(output_dir: Path | str, artifact: QaPredictionArtifact) -> None:
    """Write one QA prediction artifact directory atomically.

    Expected input:
        - output_dir: target directory for ``manifest.json`` and
          ``predictions.jsonl``.
        - artifact: typed QA prediction artifact.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "manifest.json"
    predictions_path = output_path / "predictions.jsonl"
    temp_manifest = manifest_path.with_name("manifest.json.tmp")
    temp_predictions = predictions_path.with_name("predictions.jsonl.tmp")
    with temp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(artifact.manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    with temp_predictions.open("w", encoding="utf-8") as handle:
        for query_result in artifact.query_results:
            handle.write(json.dumps(query_result.to_dict(), ensure_ascii=False) + "\n")
    temp_manifest.replace(manifest_path)
    temp_predictions.replace(predictions_path)


def read_qa_artifact(output_dir: Path | str) -> QaPredictionArtifact:
    """Read and validate one QA prediction artifact directory.

    Expected input:
        - output_dir: path to a directory written by
          :func:`write_qa_artifact`.

    Returns:
        - The validated QA prediction artifact.

    Raises:
        - FileNotFoundError: if the artifact files are absent.
        - ValueError: if the payload violates the artifact contract.
    """
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"
    predictions_path = output_path / "predictions.jsonl"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = QaPredictionManifest.from_dict(json.load(handle))
    results: list[QaQueryResult] = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(QaQueryResult.from_dict(json.loads(line)))
    return QaPredictionArtifact(manifest=manifest, query_results=tuple(results))


def load_qa_predictions_by_query(output_dir: Path | str) -> dict[str, tuple[QaPrediction, ...]]:
    """Load a QA artifact and return per-query frame evidence.

    Expected input:
        - output_dir: path to a QA prediction artifact directory.

    Returns:
        - Mapping from query id to its ranked ``QaPrediction`` evidence.

    Raises:
        - FileNotFoundError / ValueError: see :func:`read_qa_artifact`.
    """
    artifact = read_qa_artifact(output_dir)
    return {
        query_result.query_id: query_result.predictions for query_result in artifact.query_results
    }


def _build_evidence(
    retrieval: list[RetrievalCandidate],
    config: QaPredictionRunnerConfig,
) -> tuple[tuple[QaPrediction, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Compose frame, video, and window evidence from retrieved candidates.

    Expected input:
        - retrieval: ranked frame-level retrieval candidates.
        - config: validated runner configuration.

    Returns:
        - Tuple ``(predictions, video_candidates, window_candidates)`` where
          ``predictions`` are ranked frame evidence, ``video_candidates``
          are video-level aggregates, and ``window_candidates`` are optional
          temporal windows.

    Raises:
        - No exceptions.
    """
    video_candidates = build_video_candidates(
        retrieval,
        method=config.video_aggregation or "max",
        top_n=config.video_aggregation_top_n,
    )
    groups = group_candidates_by_video(retrieval)
    if config.video_aggregation is None:
        predictions = tuple(
            _to_qa_prediction(candidate, candidate.rank) for candidate in retrieval
        )
    else:
        ranked: list[QaPrediction] = []
        for video_candidate in video_candidates:
            ordered = sort_by_score(groups[video_candidate.video_id])
            for candidate in ordered:
                ranked.append(_to_qa_prediction(candidate, len(ranked) + 1))
        predictions = tuple(ranked)
    if config.window_max_gap is None:
        window_candidates = ()
    else:
        window_candidates = _build_window_candidates(groups, config.window_max_gap)
    return predictions, tuple(video_candidates), window_candidates


def _build_window_candidates(
    groups: dict[str, list[RetrievalCandidate]],
    max_gap: int,
) -> tuple[WindowCandidate, ...]:
    """Group each video's frames into ranked temporal window candidates.

    Expected input:
        - groups: frame candidates grouped by video.
        - max_gap: maximum frame-index distance inside one window.

    Returns:
        - Window candidates ranked by score (best window first); empty when
          there are no candidates.

    Raises:
        - No exceptions.
    """
    windows: list[WindowCandidate] = []
    for video_id, frames in groups.items():
        by_index = sorted(frames, key=lambda candidate: candidate.frame_index)
        indices = [candidate.frame_index for candidate in by_index]
        for start, end in group_windows(indices, max_gap):
            in_window = [candidate for candidate in by_index if start <= candidate.frame_index <= end]
            score = max(float(candidate.score) for candidate in in_window)
            windows.append(
                WindowCandidate(
                    video_id=video_id,
                    frame_start=start,
                    frame_end=end,
                    score=score,
                    rank=1,
                )
            )
    ranked = sort_by_score(windows)
    return tuple(replace(window, rank=rank) for rank, window in enumerate(ranked, start=1))


def _to_qa_prediction(candidate: RetrievalCandidate, rank: int) -> QaPrediction:
    """Convert one retrieval candidate to a persisted QA evidence record."""
    return QaPrediction(
        rank=rank,
        video_id=candidate.video_id,
        frame_index=candidate.frame_index,
        score=float(candidate.score),
    )


def _require_compatible_spaces(model_space: EmbeddingSpace, manifest: IndexManifest) -> None:
    """Require model/index embedding spaces to match exactly."""
    index_space = manifest.embedding_space
    if index_space is None:
        raise ValueError("index artifact is missing embedding-space metadata")
    if not model_space.is_compatible_with(index_space):
        raise ValueError(
            "incompatible embedding spaces between encoder and index: "
            f"model {model_space.space_id!r} ({model_space.backend}/{model_space.model_id}) != "
            f"index {index_space.space_id!r} ({index_space.backend}/{index_space.model_id})"
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
        - ValueError: for incompatible spaces under strict mode, or when the
          override is requested for a non-legacy index that cannot accept raw
          vectors by contract.
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
    return f"qa-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with explicit Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")