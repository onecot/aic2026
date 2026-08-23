"""QA prediction runner and artifact persistence.

This module wires the existing stages for the QA task:

    EvaluationSet (QA rows) -> text encoder -> IndexRetriever
        -> frame / video / window candidates -> QA prediction artifact

It reuses :class:`aic2026.retrieval.IndexRetriever` for frame-level search
and the task-independent foundation in :mod:`aic2026.retrieval_candidates`
(``group_candidates_by_video``, ``build_video_candidates``,
``group_windows``, ``sort_by_score``) for video and window evidence. It does
not implement any retrieval-model improvement, reranking, or QA scoring; the
artifact carries ranked evidence only, with the per-query textual ``answer``
left empty unless answer generation is enabled.

When ``generate_answer`` is enabled, the runner fills each query's empty
``answer`` in memory (without recomputing retrieval) using the small QA
answer-generation stage in :mod:`aic2026.qa.answer_generation`: a diverse
evidence subset of the ranked predictions is selected, resolved to materialized
keyframe images, and answered by the Jina VLM. Per-query failures leave the
answer empty and record a clear error; the exact answer prompt and model are
recorded in the manifest's ``answer_generation`` section.

Artifacts are written as a directory containing ``manifest.json`` (run
metadata) and ``predictions.jsonl`` (one ``QaQueryResult`` per line), both
written atomically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic2026.embeddings import EmbeddingEncoder
from aic2026.image_embedder import default_image_loader
from aic2026.evaluation import load_eval_set
from aic2026.qa.answer_generation import (
    ANSWER_DEVICE_CHOICES,
    DEFAULT_ANSWER_TOP_N,
    JinaVlmAnswerGenerator,
    KeyframeEvidence,
    QA_ANSWER_PROMPT,
    answer_query,
)
from aic2026.qa.qa_dataset import iter_qa_queries, resolve_query_text, select_qa_queries
from aic2026.qa.qa_types import (
    QaPrediction,
    QaPredictionArtifact,
    QaPredictionManifest,
    QaQueryResult,
    QueryTextField,
)
from aic2026.retrieval import RetrievalCandidate
from aic2026.retrieval_candidates import (
    build_video_candidates,
    build_window_candidates,
    group_candidates_by_video,
    sort_by_score,
)
from aic2026.search import SearchEngine
from aic2026.task_support import (
    default_run_id,
    portable_source_path,
    utc_now_iso,
    validate_query_text_field,
)
from aic2026.text_query import TextQuery

logger = logging.getLogger(__name__)


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
        - generate_answer: when enabled, the runner fills each query's
          ``answer`` in memory from a small Jina VLM answer-generation stage.
        - answer_top_n: evidence-set size per query (default 5).
        - answer_device: Jina VLM device, ``auto`` or ``mps``.
        - keyframes_dir: keyframe artifact root (``frames.ndjson`` +
          ``images/``) used to resolve evidence frames; required when
          ``generate_answer`` is enabled.

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
    generate_answer: bool = False
    answer_top_n: int = DEFAULT_ANSWER_TOP_N
    answer_device: str = "mps"
    keyframes_dir: Path | None = None

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
        if not isinstance(self.generate_answer, bool):
            raise ValueError("generate_answer must be a boolean")
        if not isinstance(self.answer_top_n, int) or isinstance(self.answer_top_n, bool) or self.answer_top_n < 1:
            raise ValueError("answer_top_n must be a positive integer")
        if self.answer_device not in ANSWER_DEVICE_CHOICES:
            raise ValueError(
                f"answer_device must be one of {sorted(ANSWER_DEVICE_CHOICES)} for the Jina VLM backend"
            )
        if self.generate_answer and self.keyframes_dir is None:
            raise ValueError("keyframes_dir is required when generate_answer is enabled")


def run_qa_prediction(
    config: QaPredictionRunnerConfig,
    encoder: EmbeddingEncoder,
    answer_generator: Any | None = None,
) -> QaPredictionArtifact:
    """Run QA prediction generation for all or an explicit subset of queries.

    Expected input:
        - config: validated run configuration.
        - encoder: text encoder implementing the embedding contract.
        - answer_generator: optional answer generator used instead of the
          production :class:`JinaVlmAnswerGenerator` (test injection; ignored
          when ``config.generate_answer`` is disabled). The injected object
          must expose ``generate_answer(question, images)`` plus the
          ``model_name`` / ``checkpoint`` / ``device`` properties recorded in
          the manifest.

    Returns:
        - The QA prediction artifact, written to
          ``config.prediction_output_dir`` (``manifest.json`` +
          ``predictions.jsonl``).

    Raises:
        - ValueError: if the evaluation data has no QA queries, subset ids
          are invalid, embedding spaces mismatch, or answer-generation
          inputs are missing.
        - FileNotFoundError: if configured input files are absent.
    """
    evaluation_set = load_eval_set(config.evaluation_set_path)
    all_qa_queries = iter_qa_queries(evaluation_set)
    if not all_qa_queries:
        raise ValueError("evaluation set contains no QA queries")

    selected = select_qa_queries(evaluation_set, config.query_subset_ids)
    queries = tuple(
        TextQuery(query_id=query.query_id, query_text=resolve_query_text(query, config.query_text_field))
        for query in selected
    )
    return run_qa_prediction_for_queries(
        queries,
        encoder=encoder,
        index_artifact_dir=config.index_artifact_dir,
        prediction_output_dir=config.prediction_output_dir,
        top_k=config.top_k,
        query_text_field=config.query_text_field,
        video_aggregation=config.video_aggregation,
        video_aggregation_top_n=config.video_aggregation_top_n,
        window_max_gap=config.window_max_gap,
        model_config=config.model_config,
        allow_embedding_space_mismatch=config.allow_embedding_space_mismatch,
        run_id=config.run_id,
        generate_answer=config.generate_answer,
        answer_top_n=config.answer_top_n,
        answer_device=config.answer_device,
        keyframes_dir=config.keyframes_dir,
        answer_generator=answer_generator,
        query_source_path=Path(config.evaluation_set_path),
        query_source_count=len(all_qa_queries),
        selected_query_ids=config.query_subset_ids,
    )


def run_qa_prediction_for_queries(
    queries: tuple[TextQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_dir: Path,
    top_k: int = 100,
    query_text_field: QueryTextField = "query_text_vi",
    video_aggregation: str | None = None,
    video_aggregation_top_n: int | None = None,
    window_max_gap: int | None = None,
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
    generate_answer: bool = False,
    answer_top_n: int = DEFAULT_ANSWER_TOP_N,
    answer_device: str = "mps",
    keyframes_dir: Path | None = None,
    answer_generator: Any | None = None,
    query_source_path: Path,
    query_source_count: int,
    selected_query_ids: tuple[str, ...] | None = None,
) -> QaPredictionArtifact:
    """Run QA prediction for source-resolved text queries.

    This is the source-independent QA prediction core: it consumes queries
    whose text is already resolved (no evaluation CSV is read) and reuses the
    existing retrieval and answer-generation stages unchanged.

    Expected input:
        - queries: at least one resolved text query, in processing order.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - top_k: frame retrieval cutoff per query.
        - query_text_field: provenance label recorded in the manifest.
        - video_aggregation / video_aggregation_top_n / window_max_gap:
          evidence-shaping options (see ``QaPredictionRunnerConfig``).
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.
        - generate_answer: when enabled, fills each query's ``answer`` via
          the answer-generation stage.
        - answer_top_n / answer_device / keyframes_dir: answer-generation
          options; ``keyframes_dir`` required when generating answers.
        - answer_generator: optional injected generator (test seam).
        - query_source_path: identity of the query source, recorded as
          artifact provenance.
        - query_source_count: total queries available at the source for this
          task (>= number processed).
        - selected_query_ids: optional explicit subset recorded in the
          manifest for provenance.

    Returns:
        - The QA prediction artifact written to ``prediction_output_dir``.

    Raises:
        - ValueError: if there are no queries, the source count is invalid,
          embedding spaces mismatch under strict mode, or answer-generation
          inputs are missing.
        - FileNotFoundError: if the index artifact is absent.
    """
    if not queries:
        raise ValueError("queries must contain at least one QA query")
    if not isinstance(query_source_count, int) or isinstance(query_source_count, bool) or query_source_count < 1:
        raise ValueError("query_source_count must be a positive integer")
    if len(queries) > query_source_count:
        raise ValueError("query_source_count must be >= the number of queries")

    engine = SearchEngine.from_parts(
        encoder,
        index_artifact_dir,
        top_k=top_k,
        allow_embedding_space_mismatch=allow_embedding_space_mismatch,
    )
    answerer = None
    keyframes = None
    if generate_answer:
        answerer = answer_generator if answer_generator is not None else JinaVlmAnswerGenerator(
            device=answer_device
        )
        keyframes = KeyframeEvidence(keyframes_dir)

    results: list[QaQueryResult] = []
    evidence_query_count = 0
    failed_query_count = 0
    for query in queries:
        result = search_qa(
            engine,
            query,
            top_k=top_k,
            video_aggregation=video_aggregation,
            video_aggregation_top_n=video_aggregation_top_n,
            window_max_gap=window_max_gap,
            answerer=answerer,
            keyframes=keyframes,
            answer_top_n=answer_top_n,
        )
        if result.answer_evidence:
            evidence_query_count += 1
        if result.answer_error is not None:
            failed_query_count += 1
            logger.warning(
                "QA answer generation failed for query %s: %s",
                result.query_id,
                result.answer_error,
            )
        results.append(result)

    artifact = compute_qa_manifest_and_results(
        tuple(results),
        engine=engine,
        query_text_field=query_text_field,
        top_k=top_k,
        video_aggregation=video_aggregation,
        video_aggregation_top_n=video_aggregation_top_n,
        window_max_gap=window_max_gap,
        model_config=model_config,
        run_id=run_id,
        query_source_path=query_source_path,
        query_source_count=query_source_count,
        selected_query_ids=selected_query_ids,
        generate_answer=generate_answer,
        answerer=answerer,
        answer_top_n=answer_top_n,
        keyframes_dir=keyframes_dir,
        evidence_query_count=evidence_query_count,
        failed_query_count=failed_query_count,
    )
    write_qa_artifact(prediction_output_dir, artifact)
    return artifact


def search_qa(
    engine: SearchEngine,
    query: TextQuery | str,
    *,
    top_k: int | None = None,
    video_aggregation: str | None = None,
    video_aggregation_top_n: int | None = None,
    window_max_gap: int | None = None,
    answerer: Any | None = None,
    keyframes: KeyframeEvidence | None = None,
    answer_top_n: int = DEFAULT_ANSWER_TOP_N,
    query_id: str = "query",
) -> QaQueryResult:
    """Compute one QA query's evidence (and optional answer) entirely in memory.

    This is the single-query entry point for interactive use: no evaluation
    CSV and no artifact writes. Answer generation only runs when an
    ``answerer`` is supplied together with ``keyframes``.

    Expected input:
        - engine: a constructed :class:`aic2026.search.SearchEngine`.
        - query: a :class:`~aic2026.text_query.TextQuery` or plain text.
        - top_k: optional per-call cutoff; defaults to the engine's.
        - video_aggregation / video_aggregation_top_n / window_max_gap:
          evidence-shaping options (see ``QaPredictionRunnerConfig``).
        - answerer: optional generator exposing ``generate_answer(question,
          images) -> str``; requires ``keyframes``.
        - keyframes: keyframe evidence resolver; required with ``answerer``.
        - answer_top_n: evidence-set size for answer generation.
        - query_id: identifier used when ``query`` is plain text.

    Returns:
        - The typed QA result with ranked frame/video/window evidence and,
          when answering, the filled ``answer`` / ``answer_evidence`` /
          ``answer_error`` fields.

    Raises:
        - ValueError: for invalid inputs or incompatible embedding spaces
          under strict mode.
    """
    if isinstance(query, str):
        query = TextQuery(query_id=query_id, query_text=query)
    retrieval = engine.search_text(query.query_text, top_k=top_k)
    predictions, video_candidates, window_candidates = _build_evidence(
        retrieval,
        video_aggregation=video_aggregation,
        video_aggregation_top_n=video_aggregation_top_n,
        window_max_gap=window_max_gap,
    )
    result = QaQueryResult(
        query_id=query.query_id,
        query_text=query.query_text,
        answer="",
        predictions=predictions,
        video_candidates=video_candidates,
        window_candidates=window_candidates,
    )
    if answerer is not None:
        if keyframes is None:
            raise ValueError("keyframes resolver is required when an answerer is provided")
        result = answer_query(result, answerer=answerer, keyframes=keyframes, top_n=answer_top_n)
    return result


def compute_qa_manifest_and_results(
    results: tuple[QaQueryResult, ...],
    *,
    engine: SearchEngine,
    query_text_field: QueryTextField,
    top_k: int,
    video_aggregation: str | None,
    video_aggregation_top_n: int | None,
    window_max_gap: int | None,
    model_config: dict[str, Any] | None,
    run_id: str | None,
    query_source_path: Path | str,
    query_source_count: int,
    selected_query_ids: tuple[str, ...] | None,
    generate_answer: bool,
    answerer: Any | None,
    answer_top_n: int,
    keyframes_dir: Path | None,
    evidence_query_count: int,
    failed_query_count: int,
) -> QaPredictionArtifact:
    """Assemble the typed QA prediction artifact without writing files.

    Expected input:
        - results: computed per-query QA results in processing order.
        - engine: the :class:`aic2026.search.SearchEngine` that produced
          the results (used for manifest provenance).
        - Remaining parameters: run options and answer-generation counters
          recorded in the manifest (see ``run_qa_prediction_for_queries``).

    Returns:
        - The typed QA prediction artifact (not persisted).

    Raises:
        - ValueError: if there are no results or inputs are inconsistent.
    """
    validate_query_text_field(query_text_field)
    if not results:
        raise ValueError("results must contain at least one QA query result")

    answer_generation = None
    if generate_answer:
        if answerer is None:
            raise ValueError("generate_answer requires the answerer used for the run")
        answer_generation = {
            "model": answerer.model_name,
            "checkpoint": answerer.checkpoint,
            "device": answerer.device,
            "top_n": answer_top_n,
            "prompt": QA_ANSWER_PROMPT,
            "max_new_tokens": int(getattr(answerer, "max_new_tokens", 0) or 0),
            "keyframes_dir": portable_source_path(keyframes_dir) if keyframes_dir is not None else "",
            "evidence_query_count": evidence_query_count,
            "failed_query_count": failed_query_count,
        }

    return QaPredictionArtifact(
        manifest=QaPredictionManifest(
            schema_version="1.0",
            run_id=run_id or default_run_id("qa"),
            created_at=utc_now_iso(),
            evaluation_set_path=portable_source_path(query_source_path),
            evaluation_set_query_count=query_source_count,
            processed_query_count=len(results),
            query_text_field=query_text_field,
            top_k=top_k,
            video_aggregation=video_aggregation,
            video_aggregation_top_n=video_aggregation_top_n,
            window_max_gap=window_max_gap,
            embedding_space_compatible=engine.spaces_compatible,
            embedding_space_override_used=engine.override_used,
            index_artifact_path=portable_source_path(engine.retriever.artifact_dir),
            index_manifest=engine.manifest.to_dict(),
            model_embedding_space=engine.embedding_space.to_dict(),
            model_config=model_config,
            selected_query_ids=selected_query_ids,
            answer_generation=answer_generation,
        ),
        query_results=tuple(results),
    )


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
    video_aggregation: str | None,
    video_aggregation_top_n: int | None,
    window_max_gap: int | None,
) -> tuple[tuple[QaPrediction, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Compose frame, video, and window evidence from retrieved candidates.

    Expected input:
        - retrieval: ranked frame-level retrieval candidates.
        - video_aggregation: ``None`` (frame-level ranking) or one of ``max``
          / ``mean`` / ``mean_top_n``.
        - video_aggregation_top_n: ``mean_top_n`` parameter.
        - window_max_gap: temporal window grouping gap; ``None`` disables
          window candidates.

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
        method=video_aggregation or "max",
        top_n=video_aggregation_top_n,
    )
    groups = group_candidates_by_video(retrieval)
    if video_aggregation is None:
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
    if window_max_gap is None:
        window_candidates = ()
    else:
        window_candidates = build_window_candidates(groups, window_max_gap)
    return predictions, tuple(video_candidates), window_candidates


def _to_qa_prediction(candidate: RetrievalCandidate, rank: int) -> QaPrediction:
    """Convert one retrieval candidate to a persisted QA evidence record."""
    return QaPrediction(
        rank=rank,
        video_id=candidate.video_id,
        frame_index=candidate.frame_index,
        score=float(candidate.score),
    )