"""End-to-end QA pipeline smoke test (testing/demo layer).

This module exercises the complete QA pipeline on a small synthetic setup
without any VQA model:

    synthetic QA eval CSV -> load_eval_set -> run_qa_prediction
        -> QA prediction artifact -> placeholder answers -> submission CSV

It intentionally contains no retrieval logic of its own; it composes the
existing shared stages (``load_eval_set``, ``IndexRetriever`` inside
``run_qa_prediction``, the candidate foundation, and ``qa_submission``).
Answers are placeholders only: the smoke layer has no VQA/answer model, so
it fills each query's submission answer with the expected answer from the
synthetic eval set (falling back to a fixed placeholder string).

What this layer provides and validates:

- a deterministic synthetic index builder (:func:`build_synthetic_index`);
- a deterministic placeholder text encoder
  (:class:`QaPlaceholderEncoder`) with an embedding space that matches the
  synthetic index;
- structural artifact validation (:func:`validate_qa_artifact`): every
  query has predictions, ranks are strictly increasing, scores are finite,
  and every ``(video_id, frame_index)`` reference exists in the index;
- a determinism check by running the prediction stage twice;
- a save/load round-trip check via ``read_qa_artifact``;
- submission export through ``qa_submission.write_qa_submission_csv``.

This module is a testing/demo layer only: it does not modify the retrieval
core or KIS behavior, and it never attempts real VQA answering.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId
from aic2026.embeddings import Embedding, EmbeddingEncoder, EmbeddingSpace
from aic2026.evaluation import load_eval_set
from aic2026.index_artifact import (
    EMBEDDINGS_FILENAME,
    FRAMES_FILENAME,
    INDEX_SCHEMA_VERSION,
    NORMS_FILENAME,
    IndexManifest,
    iter_frames,
    write_frames,
    write_manifest,
)
from aic2026.qa.qa_dataset import iter_qa_queries
from aic2026.qa.qa_prediction_runner import (
    QaPredictionRunnerConfig,
    read_qa_artifact,
    run_qa_prediction,
)
from aic2026.qa.qa_submission import write_qa_submission_csv
from aic2026.qa.qa_types import QaPredictionArtifact, QaQueryResult

#: Default synthetic videos used by the smoke index (BTC-style ids).
SMOKE_VIDEO_IDS = ("L21_V001", "L21_V002", "L21_V003", "L21_V004", "L21_V005", "L21_V006")
#: Number of keyframes generated per synthetic video.
SMOKE_FRAMES_PER_VIDEO = 10
#: Spacing between synthetic keyframe frame indices.
SMOKE_FRAME_STRIDE = 10
#: Embedding dimension of the synthetic index and placeholder encoder.
SMOKE_EMBEDDING_DIM = 8
#: Fixed seed making the synthetic index reproducible.
SMOKE_SEED = 42
#: Default placeholder answer used when a query has no expected answer.
SMOKE_PLACEHOLDER_ANSWER = "answer placeholder"


@dataclass(frozen=True)
class QaSmokeConfig:
    """Configuration for one reproducible QA smoke run.

    Expected input:
        - evaluation_set_path: synthetic QA eval CSV (``load_eval_set``
          format).
        - work_dir: directory that receives the regenerated demo artifacts
          (index, prediction artifacts, submission CSV).
        - top_k: frame retrieval cutoff per query.
        - embedding_dim: synthetic embedding dimension.
        - frames_per_video: keyframes per synthetic video.
        - video_ids: synthetic video ids.
        - seed: RNG seed for the synthetic index.
        - run_id: fixed run id so the two determinism runs are comparable.
        - placeholder_answer: fallback answer text for queries without an
          expected answer.

    Returns:
        - A validated smoke configuration.

    Raises:
        - ValueError: if required fields are invalid.
    """

    evaluation_set_path: Path
    work_dir: Path
    top_k: int = 50
    embedding_dim: int = SMOKE_EMBEDDING_DIM
    frames_per_video: int = SMOKE_FRAMES_PER_VIDEO
    video_ids: tuple[str, ...] = SMOKE_VIDEO_IDS
    seed: int = SMOKE_SEED
    run_id: str = "qa-smoke"
    placeholder_answer: str = SMOKE_PLACEHOLDER_ANSWER

    def __post_init__(self) -> None:
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool) or self.embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if (
            not isinstance(self.frames_per_video, int)
            or isinstance(self.frames_per_video, bool)
            or self.frames_per_video < 1
        ):
            raise ValueError("frames_per_video must be a positive integer")
        if not isinstance(self.video_ids, tuple) or not self.video_ids:
            raise ValueError("video_ids must be a non-empty tuple")
        if len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("video_ids must be unique")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.run_id, str) or self.run_id.strip() == "":
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.placeholder_answer, str) or self.placeholder_answer.strip() == "":
            raise ValueError("placeholder_answer must be a non-empty string")


@dataclass(frozen=True)
class QaSmokeResult:
    """Outcome of one QA smoke run.

    Expected input:
        - evaluation_set_path: resolved synthetic eval CSV path.
        - index_dir: generated synthetic index artifact directory.
        - artifact_dir: primary QA prediction artifact directory.
        - determinism_artifact_dir: second prediction artifact directory used
          for the determinism comparison.
        - submission_path: written provisional submission CSV path.
        - query_count: number of processed QA queries.
        - prediction_count: total ranked predictions across all queries.
        - determinism_verified: whether both runs produced identical query
          results.
        - placeholder_answers_filled: number of queries that received a
          non-empty answer for the submission.

    Returns:
        - A validated smoke outcome.

    Raises:
        - No exceptions.
    """

    evaluation_set_path: str
    index_dir: str
    artifact_dir: str
    determinism_artifact_dir: str
    submission_path: str
    query_count: int
    prediction_count: int
    determinism_verified: bool
    placeholder_answers_filled: int


class QaPlaceholderEncoder(EmbeddingEncoder):
    """Deterministic placeholder text encoder for the smoke pipeline.

    Maps text to a fixed random unit vector derived from a stable hash of
    the text, so repeated runs are reproducible. It is not a real
    retrieval model and never produces image embeddings.
    """

    def __init__(self, space: EmbeddingSpace, embedding_dim: int) -> None:
        self._space = space
        self._dim = embedding_dim

    @property
    def embedding_space(self) -> EmbeddingSpace:
        """Return the embedding space of this placeholder encoder."""
        return self._space

    def encode_image(self, image: object) -> Embedding:
        """Unsupported: the placeholder encoder has no image capability."""
        raise NotImplementedError("QaPlaceholderEncoder does not encode images")

    def encode_text(self, text: str) -> Embedding:
        """Encode text into a deterministic placeholder unit vector."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        vector = np.random.default_rng(seed).random(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector = np.ones(self._dim, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
        return Embedding(
            kind="text",
            vector=(vector / norm).astype(np.float32),
            space=self._space,
        )


def smoke_embedding_space() -> EmbeddingSpace:
    """Return the deterministic embedding space of the smoke artifacts."""
    return EmbeddingSpace(
        backend="test-backend",
        model_id="qa-smoke-placeholder",
        checkpoint_id="qa-smoke-v1",
        space_id="qa-smoke-space-v1",
        embedding_dim=SMOKE_EMBEDDING_DIM,
        output_dtype="float32",
        normalization="l2",
        preprocessing_id="qa-smoke-preprocess-v1",
    )


def build_synthetic_index(output_dir: Path | str, config: QaSmokeConfig) -> Path:
    """Build a small deterministic index artifact for the smoke pipeline.

    Expected input:
        - output_dir: target index artifact directory (created as needed).
        - config: smoke configuration; the index uses ``video_ids``,
          ``frames_per_video``, ``embedding_dim``, and ``seed``.

    Returns:
        - The index artifact directory path.

    Raises:
        - OSError: if writing the artifact fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    space = smoke_embedding_space()
    rng = np.random.default_rng(config.seed)
    vectors: list[np.ndarray] = []
    frames: list[KeyframeTiming] = []
    keyframe_ordinal = 0
    for video_id in config.video_ids:
        for local_index in range(config.frames_per_video):
            vector = rng.random(config.embedding_dim).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm).astype(np.float32))
            keyframe_ordinal += 1
            frames.append(
                KeyframeTiming(
                    keyframe=KeyframeKey(
                        video_id=VideoId(video_id),
                        keyframe_ordinal=keyframe_ordinal,
                    ),
                    pts_time=float(local_index),
                    fps=25.0,
                    frame_index=local_index * SMOKE_FRAME_STRIDE,
                )
            )
    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix.astype(np.float32), axis=1).astype(np.float32)
    manifest = IndexManifest(
        schema_version=INDEX_SCHEMA_VERSION,
        artifact_id="qa-smoke-index",
        score_metric="cosine",
        embedding_dim=config.embedding_dim,
        embedding_dtype="float32",
        num_rows=matrix.shape[0],
        num_videos=len(config.video_ids),
        created_at="2026-08-17T00:00:00Z",
        source_dataset_id="synthetic-qa-smoke",
        canonical_schema_version="1.0",
        embedding_space=space,
    )
    write_manifest(output_path / "manifest.json", manifest)
    write_frames(output_path / FRAMES_FILENAME, frames)
    np.save(output_path / EMBEDDINGS_FILENAME, matrix)
    np.save(output_path / NORMS_FILENAME, norms)
    return output_path


def load_index_frame_map(index_dir: Path | str) -> dict[str, set[int]]:
    """Return the valid ``(video_id, frame_index)`` space of an index.

    Expected input:
        - index_dir: an index artifact directory.

    Returns:
        - Mapping from video id to the set of valid original-video frame
          indices present in the artifact.

    Raises:
        - FileNotFoundError: if ``frames.ndjson`` is absent.
    """
    frame_map: dict[str, set[int]] = {}
    for timing in iter_frames(Path(index_dir) / FRAMES_FILENAME):
        video_id = timing.keyframe.video_id.value
        frame_map.setdefault(video_id, set()).add(timing.frame_index)
    return frame_map


def validate_qa_artifact(
    artifact: QaPredictionArtifact,
    frame_map: dict[str, set[int]],
) -> list[str]:
    """Validate the structural integrity of a QA prediction artifact.

    Verifies, for every query result: predictions are present, ranks are
    strictly increasing, scores are finite numbers, and every
    ``(video_id, frame_index)`` reference exists in ``frame_map``.

    Expected input:
        - artifact: a QA prediction artifact.
        - frame_map: output of :func:`load_index_frame_map` for the index
          that produced the artifact.

    Returns:
        - A list of human-readable problems; empty when the artifact is
          structurally valid.

    Raises:
        - No exceptions.
    """
    problems: list[str] = []
    for query_result in artifact.query_results:
        label = f"query {query_result.query_id!r}"
        if not query_result.predictions:
            problems.append(f"{label} produced no predictions")
            continue
        previous_rank = 0
        for prediction in query_result.predictions:
            if prediction.rank <= previous_rank:
                problems.append(f"{label} has non-increasing rank at rank {prediction.rank}")
            previous_rank = prediction.rank
            if not isinstance(prediction.score, (int, float)) or math.isnan(prediction.score) or math.isinf(
                prediction.score
            ):
                problems.append(f"{label} has a non-finite score at rank {prediction.rank}")
            video_frames = frame_map.get(prediction.video_id)
            if video_frames is None:
                problems.append(f"{label} references unknown video {prediction.video_id!r}")
            elif prediction.frame_index not in video_frames:
                problems.append(
                    f"{label} references invalid frame {prediction.frame_index} "
                    f"in video {prediction.video_id!r}"
                )
    return problems


def with_placeholder_answers(
    artifact: QaPredictionArtifact,
    answers_by_query: dict[str, str],
    fallback: str,
) -> QaPredictionArtifact:
    """Return a copy of ``artifact`` with per-query answers filled.

    The QA foundation leaves ``answer`` empty because it has no VQA model.
    This demo helper fills each query result's answer from
    ``answers_by_query`` (typically the expected answers of the synthetic
    eval set), falling back to ``fallback`` when a query is missing.

    Expected input:
        - artifact: source QA prediction artifact.
        - answers_by_query: mapping from query id to answer text.
        - fallback: answer text used for queries without an entry.

    Returns:
        - A new artifact whose query results carry non-empty answers.

    Raises:
        - No exceptions.
    """
    updated = tuple(
        replace(query_result, answer=answers_by_query.get(query_result.query_id, fallback))
        for query_result in artifact.query_results
    )
    return QaPredictionArtifact(manifest=artifact.manifest, query_results=updated)


def run_qa_smoke(config: QaSmokeConfig) -> QaSmokeResult:
    """Run the end-to-end QA smoke pipeline and validate its outputs.

    Expected input:
        - config: validated smoke configuration.

    Returns:
        - The smoke outcome; the generated artifacts are written under
          ``config.work_dir``.

    Raises:
        - ValueError: if the eval set has no QA queries or validation
          fails.
        - FileNotFoundError: if the evaluation CSV is absent.
        - OSError: if writing artifacts fails.
    """
    evaluation_set = load_eval_set(config.evaluation_set_path)
    qa_queries = iter_qa_queries(evaluation_set)
    if not qa_queries:
        raise ValueError("evaluation set contains no QA queries")
    expected_answers = {query.query_id: query.answer or config.placeholder_answer for query in qa_queries}

    work_dir = Path(config.work_dir)
    index_dir = work_dir / "index"
    artifact_dir = work_dir / "qa_predictions"
    determinism_dir = work_dir / "qa_predictions_determinism"
    submission_path = work_dir / "qa_submission.csv"
    for managed in (index_dir, artifact_dir, determinism_dir):
        if managed.exists():
            shutil.rmtree(managed)
    if submission_path.exists():
        submission_path.unlink()

    build_synthetic_index(index_dir, config)
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), config.embedding_dim)
    runner_config = QaPredictionRunnerConfig(
        evaluation_set_path=config.evaluation_set_path,
        index_artifact_dir=index_dir,
        prediction_output_dir=artifact_dir,
        top_k=config.top_k,
        query_text_field="query_text_vi",
        run_id=config.run_id,
    )
    artifact = run_qa_prediction(runner_config, encoder)
    determinism_artifact = run_qa_prediction(
        replace(runner_config, prediction_output_dir=determinism_dir),
        encoder,
    )
    determinism_verified = artifact.query_results == determinism_artifact.query_results

    problems = validate_qa_artifact(artifact, load_index_frame_map(index_dir))
    restored = read_qa_artifact(artifact_dir)
    if restored != artifact:
        problems.append("artifact round-trip mismatch after read_qa_artifact")
    if problems:
        raise ValueError("QA smoke validation failed:\n- " + "\n- ".join(problems))

    submission_artifact = with_placeholder_answers(artifact, expected_answers, config.placeholder_answer)
    write_qa_submission_csv(submission_artifact, submission_path)

    return QaSmokeResult(
        evaluation_set_path=str(Path(config.evaluation_set_path).resolve()),
        index_dir=str(index_dir.resolve()),
        artifact_dir=str(artifact_dir.resolve()),
        determinism_artifact_dir=str(determinism_dir.resolve()),
        submission_path=str(submission_path.resolve()),
        query_count=len(artifact.query_results),
        prediction_count=sum(len(result.predictions) for result in artifact.query_results),
        determinism_verified=determinism_verified,
        placeholder_answers_filled=sum(
            1 for result in submission_artifact.query_results if result.answer.strip()
        ),
    )


def format_smoke_report(result: QaSmokeResult) -> str:
    """Format a smoke outcome as a human-readable report.

    Expected input:
        - result: a smoke outcome.

    Returns:
        - Multi-line report string.

    Raises:
        - No exceptions.
    """
    return "\n".join(
        [
            "QA smoke run complete",
            f"- evaluation set: {result.evaluation_set_path} ({result.query_count} QA queries)",
            f"- index artifact: {result.index_dir}",
            f"- prediction artifact: {result.artifact_dir}",
            f"- determinism artifact: {result.determinism_artifact_dir}",
            f"- predictions: {result.prediction_count} across {result.query_count} queries",
            f"- determinism verified: {result.determinism_verified}",
            f"- placeholder answers filled: {result.placeholder_answers_filled}",
            f"- submission CSV: {result.submission_path}",
        ]
    )


def write_smoke_report_json(result: QaSmokeResult, output_path: Path | str) -> Path:
    """Write a smoke outcome as a JSON report atomically.

    Expected input:
        - result: a smoke outcome.
        - output_path: target JSON file path.

    Returns:
        - The path that was written.

    Raises:
        - OSError: if writing fails.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)
        handle.write("\n")
    temp_path.replace(destination)
    return destination