"""BTC-query inference adapter.

Bridges the BTC experimental query files (``query-<name>-<task>.txt``) to the
existing KIS / QA / TRAKE prediction pipelines without any ground-truth or
evaluation metadata:

    data/sotuyen/thunghiem/*.txt
        -> load_btc_queries (exact file text, no GT)
        -> run_btc_{kis,qa,trake}_prediction (existing prediction cores)
        -> standard KIS / QA / TRAKE prediction artifacts

The artifacts use the unchanged schema contracts of the underlying runners,
so ``scripts/build_submission.py`` can match every generated artifact result
back to its query file by exact query text. Retrieval algorithms, scoring,
and answer generation are reused untouched; this module only adapts the
query source and records provenance:

- ``evaluation_set_path`` holds the resolved BTC query directory (the actual
  input identity for these runs);
- ``evaluation_set_query_count`` is the number of BTC queries of that task;
- run ids default to ``btc-<task>-run-<timestamp>``.

No Modal runs and no GPU runs happen at import time; model execution only
occurs when the runner cores are invoked with a real encoder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.btc_queries import BtcQuery
from aic2026.embeddings import EmbeddingEncoder
from aic2026.kis_prediction_runner import (
    KisPredictionArtifact,
    run_kis_prediction_for_queries,
)
from aic2026.qa.answer_generation import DEFAULT_ANSWER_TOP_N
from aic2026.qa.qa_prediction_runner import QaPredictionArtifact, run_qa_prediction_for_queries
from aic2026.text_query import TextQuery
from aic2026.trake import TrakePredictionArtifact, run_trake_prediction_for_queries


def _select_task_queries(btc_queries: tuple[BtcQuery, ...], task: str) -> tuple[BtcQuery, ...]:
    """Return the BTC queries of one task, failing when there are none."""
    selected = tuple(query for query in btc_queries if query.info.task == task)
    if not selected:
        raise ValueError(f"no BTC {task} queries provided")
    return selected


def _common_query_dir(queries: tuple[BtcQuery, ...]) -> Path:
    """Return the single parent directory shared by all query files."""
    parents = {query.info.path.parent.resolve() for query in queries}
    if len(parents) != 1:
        raise ValueError("all BTC queries must come from one query directory")
    return parents.pop()


def _to_text_queries(queries: tuple[BtcQuery, ...]) -> tuple[TextQuery, ...]:
    """Convert BTC queries to source-resolved text queries."""
    return tuple(
        TextQuery(query_id=query.query_id, query_text=query.query_text)
        for query in queries
    )


def _default_run_id(prefix: str) -> str:
    """Return a timestamped default run id for a BTC task run."""
    return f"{prefix}-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def run_btc_kis_prediction(
    btc_queries: tuple[BtcQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_path: Path,
    top_k: int = 100,
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
) -> KisPredictionArtifact:
    """Run KIS prediction for the KIS queries of a BTC query set.

    Expected input:
        - btc_queries: BTC queries loaded via ``load_btc_queries``; must
          contain at least one KIS query, all from one directory.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_path: output JSON artifact path.
        - top_k: candidate count per query.
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.

    Returns:
        - The KIS prediction artifact written to ``prediction_output_path``.

    Raises:
        - ValueError: if no KIS queries are provided, the queries span
          multiple directories, or embedding spaces mismatch under strict
          mode.
        - FileNotFoundError: if the index artifact is absent.
    """
    queries = _select_task_queries(btc_queries, "kis")
    return run_kis_prediction_for_queries(
        _to_text_queries(queries),
        encoder=encoder,
        index_artifact_dir=index_artifact_dir,
        prediction_output_path=prediction_output_path,
        top_k=top_k,
        query_text_field="query_text_vi",
        model_config=model_config,
        allow_embedding_space_mismatch=allow_embedding_space_mismatch,
        run_id=run_id or _default_run_id("btc-kis"),
        query_source_path=_common_query_dir(queries),
        query_source_count=len(queries),
    )


def run_btc_qa_prediction(
    btc_queries: tuple[BtcQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_dir: Path,
    top_k: int = 100,
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
    generate_answer: bool = False,
    answer_top_n: int = DEFAULT_ANSWER_TOP_N,
    answer_device: str = "mps",
    keyframes_dir: Path | None = None,
    answer_generator: Any | None = None,
) -> QaPredictionArtifact:
    """Run QA prediction for the QA queries of a BTC query set.

    Without ``generate_answer`` the artifact answers stay empty (retrieval
    evidence only); submission export requires generated answers.

    Expected input:
        - btc_queries: BTC queries loaded via ``load_btc_queries``; must
          contain at least one QA query, all from one directory.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - top_k: frame retrieval cutoff per query.
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.
        - generate_answer: enable the Jina VLM answer-generation stage.
        - answer_top_n / answer_device / keyframes_dir: answer-generation
          options; ``keyframes_dir`` required when generating answers.
        - answer_generator: optional injected generator (test seam).

    Returns:
        - The QA prediction artifact written to ``prediction_output_dir``.

    Raises:
        - ValueError: if no QA queries are provided, the queries span
          multiple directories, embedding spaces mismatch under strict mode,
          or answer-generation inputs are missing.
        - FileNotFoundError: if the index artifact is absent.
    """
    queries = _select_task_queries(btc_queries, "qa")
    return run_qa_prediction_for_queries(
        _to_text_queries(queries),
        encoder=encoder,
        index_artifact_dir=index_artifact_dir,
        prediction_output_dir=prediction_output_dir,
        top_k=top_k,
        query_text_field="query_text_vi",
        video_aggregation=None,
        video_aggregation_top_n=None,
        window_max_gap=None,
        model_config=model_config,
        allow_embedding_space_mismatch=allow_embedding_space_mismatch,
        run_id=run_id or _default_run_id("btc-qa"),
        generate_answer=generate_answer,
        answer_top_n=answer_top_n,
        answer_device=answer_device,
        keyframes_dir=keyframes_dir,
        answer_generator=answer_generator,
        query_source_path=_common_query_dir(queries),
        query_source_count=len(queries),
    )


def run_btc_trake_prediction(
    btc_queries: tuple[BtcQuery, ...],
    *,
    encoder: EmbeddingEncoder,
    index_artifact_dir: Path,
    prediction_output_dir: Path,
    top_k: int = 100,
    window_max_gap: int | None = 60,
    model_config: dict[str, Any] | None = None,
    allow_embedding_space_mismatch: bool = False,
    run_id: str | None = None,
) -> TrakePredictionArtifact:
    """Run TRAKE prediction for the TRAKE queries of a BTC query set.

    Events are parsed from each query file's text only; per-event retrieval
    reuses the existing TRAKE pipeline unchanged.

    Expected input:
        - btc_queries: BTC queries loaded via ``load_btc_queries``; must
          contain at least one TRAKE query, all from one directory.
        - encoder: text encoder implementing the embedding contract.
        - index_artifact_dir: path to one index artifact directory.
        - prediction_output_dir: output artifact directory.
        - top_k: frame retrieval cutoff per event.
        - window_max_gap: temporal window grouping gap; ``None`` disables
          window candidates.
        - model_config: optional JSON-serializable model configuration.
        - allow_embedding_space_mismatch: explicit opt-in override for
          diagnostic experiments only.
        - run_id: optional stable run id; auto-generated when omitted.

    Returns:
        - The TRAKE prediction artifact written to
          ``prediction_output_dir``.

    Raises:
        - ValueError: if no TRAKE queries are provided, the queries span
          multiple directories, an event list cannot be parsed, or embedding
          spaces mismatch under strict mode.
        - FileNotFoundError: if the index artifact is absent.
    """
    queries = _select_task_queries(btc_queries, "trake")
    return run_trake_prediction_for_queries(
        _to_text_queries(queries),
        encoder=encoder,
        index_artifact_dir=index_artifact_dir,
        prediction_output_dir=prediction_output_dir,
        top_k=top_k,
        query_text_field="query_text_vi",
        window_max_gap=window_max_gap,
        model_config=model_config,
        allow_embedding_space_mismatch=allow_embedding_space_mismatch,
        run_id=run_id or _default_run_id("btc-trake"),
        query_source_path=_common_query_dir(queries),
        query_source_count=len(queries),
    )
