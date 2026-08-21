"""Minimal QA answer-generation stage: evidence selection + Jina VLM answers.

This module fills the empty ``answer`` of each :class:`QaQueryResult` produced
by the retrieval foundation (:func:`aic2026.qa.qa_prediction_runner.run_qa_prediction`).
It deliberately reuses the existing Jina VLM integration
(:class:`aic2026.caption_generation.JinaVlmMlxCaptioner`, ``jinaai/jina-vlm-mlx``,
the local-first MLX port of ``jinaai/jina-vlm``) instead of introducing a
second VLM backend architecture.

Pipeline (per query):

    ranked frame predictions (existing retrieval output)
        -> select a small, diverse evidence set (default 5 frames)
        -> resolve evidence frames to materialized keyframe images
        -> Jina VLM answer generation
        -> short direct answer recorded on the query result

The Jina VLM MLX interface processes one image per generation call, so a
multi-frame evidence set is answered by (a) one single-image call per frame
and (b) one text-only synthesis call that produces the final concise answer.
A single-frame evidence set uses exactly one image call.

Failure behavior: a failed answer for one query never crashes the run; the
answer stays ``""`` and a human-readable error is recorded in the per-query
``answer_error`` field (and surfaced in the run manifest's
``answer_generation.failed_query_count``).

The exact answer prompt is stored in the run manifest for reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from aic2026.caption_generation import JinaVlmMlxCaptioner
from aic2026.embedding_artifact import FRAMES_FILENAME, iter_frames
from aic2026.embedding_runner import default_image_loader
from aic2026.embedding_shard import keyframe_image_path
from aic2026.qa.qa_types import QaPrediction, QaQueryResult

#: Model checkpoint used for answer generation. Reuses the existing Jina VLM
#: integration (``jinaai/jina-vlm-mlx``, the MLX port of ``jinaai/jina-vlm``)
#: already shipped in :mod:`aic2026.caption_generation`.
DEFAULT_ANSWER_MODEL = "jinaai/jina-vlm-mlx"

#: Default number of evidence frames passed to the Jina VLM per query.
DEFAULT_ANSWER_TOP_N = 5

#: Maximum generated tokens per VLM call (short, direct answers).
DEFAULT_MAX_NEW_TOKENS = 64

#: Minimum distance (original-video frames) between two selected evidence
#: frames of the same video; frames closer than this are treated as
#: near-duplicates of the same time region.
EVIDENCE_MIN_FRAME_GAP = 30

#: Devices accepted by the Jina VLM backend (``auto`` resolves to ``mps``).
ANSWER_DEVICE_CHOICES = ("auto", "mps")

#: Fixed answer prompt. The ``{query}`` placeholder is the resolved query text.
#: Kept explicit so every manifest records the exact instruction used.
QA_ANSWER_PROMPT = """You are answering a video question.

Question:
{query}

Use only the provided video frames as evidence.
Answer the question directly and concisely.
Do not explain your reasoning.
If the answer cannot be determined from the evidence, answer: unknown."""

#: Text-only synthesis prompt combining per-frame observations into one
#: final concise answer (used when the evidence set spans multiple frames,
#: because the Jina VLM MLX interface accepts one image per call).
QA_ANSWER_SYNTHESIS_PROMPT = """You are answering a video question.

Question:
{query}

Below are observations from individual video frames:
{observations}

Use only these observations as evidence.
Answer the question directly and concisely.
Do not explain your reasoning.
If the answer cannot be determined from the evidence, answer: unknown."""


def select_evidence_frames(
    predictions: Sequence[QaPrediction],
    top_n: int,
) -> tuple[QaPrediction, ...]:
    """Select a small, diverse evidence set from ranked frame predictions.

    Frames are taken in retrieval rank order. A frame is skipped when it is a
    near-duplicate of an already-selected frame: same video and within
    ``EVIDENCE_MIN_FRAME_GAP`` original-video frames of a selected frame.
    Ranking order is preserved, so the returned subset is still sorted by
    retrieval rank.

    Expected input:
        - predictions: ranked ``QaPrediction`` evidence records.
        - top_n: maximum number of evidence frames to select.

    Returns:
        - At most ``top_n`` ranked evidence records (fewer when the ranked
          list is exhausted before ``top_n`` diverse frames are found).

    Raises:
        - ValueError: if ``top_n`` is not a positive integer.
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    selected: list[QaPrediction] = []
    selected_frames_by_video: dict[str, list[int]] = {}
    for prediction in predictions:
        if len(selected) >= top_n:
            break
        selected_frames = selected_frames_by_video.get(prediction.video_id, [])
        if not selected_frames:
            selected.append(prediction)
            selected_frames_by_video.setdefault(prediction.video_id, []).append(prediction.frame_index)
            continue
        if min(abs(prediction.frame_index - index) for index in selected_frames) >= EVIDENCE_MIN_FRAME_GAP:
            selected.append(prediction)
            selected_frames_by_video[prediction.video_id].append(prediction.frame_index)
    return tuple(selected)


class KeyframeEvidence:
    """Resolve ``(video_id, frame_index)`` evidence to materialized images.

    Builds a ``(video_id, frame_index) -> keyframe_ordinal`` index from the
    canonical ``frames.ndjson`` of a keyframe artifact and maps each evidence
    reference to the ``images/<video_id>/<ordinal:06d>.jpg`` path.
    """

    def __init__(self, keyframes_root: Path | str) -> None:
        self._root = Path(keyframes_root)
        frames_path = self._root / FRAMES_FILENAME
        if not frames_path.is_file():
            raise ValueError(f"keyframe artifact has no {FRAMES_FILENAME}: {self._root}")
        self._ordinals: dict[tuple[str, int], int] = {}
        for frame in iter_frames(frames_path):
            self._ordinals[
                (frame.keyframe.video_id.value, frame.frame_index)
            ] = frame.keyframe.keyframe_ordinal

    @property
    def keyframes_dir(self) -> Path:
        """The keyframe artifact root directory."""
        return self._root

    def ordinal_for(self, video_id: str, frame_index: int) -> int | None:
        """Return the keyframe ordinal for a video/frame reference, or None."""
        return self._ordinals.get((video_id, frame_index))

    def image_path(self, video_id: str, frame_index: int) -> Path | None:
        """Return the materialized image path for an evidence frame, or None.

        Expected input:
            - video_id: canonical video identifier.
            - frame_index: original-video frame index.

        Returns:
            - The ``images/`` path when the frame exists in the artifact and
              its image file is present, otherwise ``None``.

        Raises:
            - No exceptions.
        """
        ordinal = self.ordinal_for(video_id, frame_index)
        if ordinal is None:
            return None
        candidate = keyframe_image_path(self._root, video_id, ordinal)
        return candidate if candidate.is_file() else None


class JinaVlmAnswerGenerator(JinaVlmMlxCaptioner):
    """Jina VLM answer generator reusing the existing MLX Jina VLM backend.

    Loads the same ``jinaai/jina-vlm-mlx`` checkpoint, device handling, and
    mlx-vlm runtime as :class:`JinaVlmMlxCaptioner`. The Jina VLM MLX
    interface accepts one image per generation call, so multi-frame evidence
    is answered per frame and then synthesized with a text-only call.

    Expected input:
        - model: Jina VLM checkpoint id (default ``jinaai/jina-vlm-mlx``).
        - device: ``auto`` or ``mps`` (MLX runs on Apple Silicon Metal;
          ``cpu``/``cuda`` are rejected by the reused backend).
        - prompt: the answer prompt passed to the model (defaults to
          :data:`QA_ANSWER_PROMPT`).
        - max_new_tokens: maximum generated tokens per VLM call.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANSWER_MODEL,
        *,
        device: str = "mps",
        prompt: str = QA_ANSWER_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__(model=model, device=device, prompt=prompt, max_new_tokens=max_new_tokens)

    @property
    def max_new_tokens(self) -> int:
        """Maximum generated tokens per VLM call."""
        return self._max_new_tokens

    def generate_answer(self, question: str, images: Sequence[object]) -> str:
        """Answer a question from a sequence of evidence images.

        One image is answered with a single VLM call. Multiple images are
        answered with one call per frame followed by a text-only synthesis
        call that returns one concise answer.

        Expected input:
            - question: non-empty question text.
            - images: backend-native images (e.g. RGB PIL images).

        Returns:
            - The stripped answer text (the model answers ``unknown`` when
              the evidence does not support an answer).

        Raises:
            - ValueError: if the question is empty or no image is provided.
            - RuntimeError: if generation fails (the caller records the
              failure per query instead of crashing the run).
        """
        if not isinstance(question, str) or question.strip() == "":
            raise ValueError("question must be a non-empty string")
        if not images:
            raise ValueError("generate_answer requires at least one evidence image")
        prompt_text = self._prompt.format(query=question)
        if len(images) == 1:
            return self._generate(prompt_text, list(images))
        observations = [self._generate(prompt_text, [image]) for image in images]
        return self._synthesize(question, observations)

    def _generate(self, prompt_text: str, images: list[object]) -> str:
        """Run one mlx-vlm generation call and extract the answer text."""
        mlx_vlm = import_module("mlx_vlm")
        formatted = mlx_vlm.prompt_utils.apply_chat_template(
            self._processor,
            self._config,
            prompt_text,
            num_images=len(images),
        )
        if images:
            output = mlx_vlm.generate(
                self._model_obj,
                self._processor,
                formatted,
                list(images),
                max_tokens=self._max_new_tokens,
            )
        else:
            output = mlx_vlm.generate(
                self._model_obj,
                self._processor,
                formatted,
                None,
                max_tokens=self._max_new_tokens,
            )
        return self._coerce_caption(output)

    def _synthesize(self, question: str, observations: Sequence[str]) -> str:
        """Combine per-frame observations into one concise final answer."""
        text = QA_ANSWER_SYNTHESIS_PROMPT.format(
            query=question,
            observations="\n".join(f"- {observation}" for observation in observations),
        )
        return self._generate(text, [])


def answer_query(
    query_result: QaQueryResult,
    answerer: Any,
    keyframes: KeyframeEvidence,
    top_n: int,
    image_loader: Callable[[Path], object] | None = None,
) -> QaQueryResult:
    """Generate the answer for one QA query result from its ranked evidence.

    Expected input:
        - query_result: a QA result whose ``answer`` is the empty string
          (the retrieval foundation output).
        - answerer: an answer generator exposing ``generate_answer(question,
          images) -> str``; ``JinaVlmAnswerGenerator`` is the production
          implementation.
        - keyframes: keyframe evidence resolver exposing
          ``image_path(video_id, frame_index) -> Path | None``.
        - top_n: evidence-set size passed to :func:`select_evidence_frames`.
        - image_loader: callable ``path -> backend-native image``; defaults
          to the shared Pillow-based loader.

    Returns:
        - A copy of the query result with the ``answer`` filled in, the
          ``answer_evidence`` selected frames preserved, and (on failure)
          ``answer_error`` set while ``answer`` stays ``""``.

    Raises:
        - No exceptions: per-query failures are recorded on the result.
    """
    evidence = select_evidence_frames(query_result.predictions, top_n)
    loader = image_loader if image_loader is not None else default_image_loader()
    images: list[object] = []
    load_errors: list[str] = []
    for prediction in evidence:
        try:
            path = keyframes.image_path(prediction.video_id, prediction.frame_index)
            if path is None:
                raise FileNotFoundError(
                    f"no keyframe image for {prediction.video_id} frame {prediction.frame_index}"
                )
            images.append(loader(path))
        except Exception as exc:  # noqa: BLE001 - recorded as a per-query load error
            load_errors.append(f"{prediction.video_id}:{prediction.frame_index}: {exc}")

    answer = ""
    error: str | None = None
    if images:
        try:
            answer = answerer.generate_answer(query_result.query_text, images)
        except Exception as exc:  # noqa: BLE001 - recorded as a per-query error
            error = f"answer generation failed: {exc}"
        if error is None and (answer is None or answer.strip() == ""):
            error = "answer generation returned an empty answer"
            answer = ""
    elif evidence:
        error = "no evidence images could be loaded: " + "; ".join(load_errors[:3])
    return replace(
        query_result,
        answer=answer,
        answer_evidence=evidence if evidence else None,
        answer_error=error,
    )