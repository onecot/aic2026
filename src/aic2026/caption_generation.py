"""Resumable, model-agnostic keyframe caption generation pipeline.

Produces a caption artifact from canonical dataset keyframes using an
injected captioner. The pipeline is model/backend-agnostic: any object
test callers may inject a captioner with the same methods, and batched captioning uses
the captioner's ``generate_captions``. Two backends ship with the pipeline:

- ``qwen2.5-vl`` (:class:`Qwen25VlCaptioner`) via HuggingFace Transformers;
- ``jina-vlm-mlx`` (:class:`JinaVlmMlxCaptioner`) via ``mlx-vlm`` for Apple
  Silicon local inference.

Transformers/Torch (Qwen) and MLX (Jina VLM MLX) are imported only when the
corresponding captioner is constructed, keeping the core package and its unit
tests independent of optional model runtime dependencies.

Public API:
- ``generate_captions(keyframes_dir, output_dir, ...)``
- ``create_captioner(backend, ...)`` / ``CAPTIONER_REGISTRY``
- ``KeyframeCaption``, ``CaptionManifest``, ``CaptionCheckpoint``
- ``read_caption_artifact`` / ``validate_caption_artifact``

Behavior:
- deterministic input order: frames in ``frames.ndjson`` file (row) order;
  row ``i`` of ``captions.ndjson`` always corresponds to line ``i`` of
  ``frames.ndjson``
- resumable via ``checkpoint.json``; already-captioned records are skipped on
  resume and the artifact is considered complete only after ``manifest.json``
  is written atomically (leftover checkpoint/build metadata is deleted)
- graceful degradation: a missing or unreadable keyframe image becomes an
  ``error`` record (empty caption + message) instead of failing the run;
  a caption-generation failure falls back from batch to per-image
- safe writes: the manifest and checkpoint are written atomically; partial
  outputs are never exposed as valid (no ``manifest.json`` until completion)

Caption artifact layout::

    <output-dir>/
        checkpoint.json   resume state (completed count + video ids + identity)
        captions.ndjson   one KeyframeCaption per line (row i <-> frames[i])
        manifest.json     versioned metadata (written on completion)

The keyframe artifact is read-only: this pipeline never modifies
``keyframes_dir`` (frames or images).
"""

from __future__ import annotations

import json
import logging
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from aic2026.canonical import CANONICAL_SCHEMA_VERSION, KeyframeTiming, VideoId
from aic2026.embedding_artifact import FRAMES_FILENAME, iter_frames
from aic2026.embedding_runner import default_image_loader
from aic2026.embedding_shard import keyframe_image_path, select_frames
from aic2026.progress import ProgressReporter, create_progress_reporter

#: Schema version of the caption artifact.
CAPTION_ARTIFACT_SCHEMA_VERSION = "1.0"

#: Relative filenames within a caption artifact directory.
CHECKPOINT_FILENAME = "checkpoint.json"
CAPTIONS_FILENAME = "captions.ndjson"
MANIFEST_FILENAME = "manifest.json"

#: Default Qwen2.5-VL architecture (short) name used by the CLI.
DEFAULT_MODEL = "Qwen2.5-VL-3B-Instruct"

#: Default Jina VLM MLX checkpoint id (Apple Silicon, 4-bit).
DEFAULT_JINA_VLM_MLX_MODEL = "jinaai/jina-vlm-mlx"

#: Supported caption backends selectable via ``--backend`` / the
#: ``backend`` argument of :func:`create_captioner`.
CAPTION_BACKENDS = ("qwen2.5-vl", "jina-vlm-mlx")

#: Backend used when none is explicitly selected (Qwen2.5-VL).
DEFAULT_BACKEND = "qwen2.5-vl"

#: Default HuggingFace organization for the short model name.
DEFAULT_ORG = "Qwen"

#: Minimum Transformers release with native Qwen2.5-VL support.
MIN_TRANSFORMERS_VERSION = "4.49"

#: Runtime packages the Qwen2.5-VL captioner needs when constructed.
REQUIRED_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "PIL",
)

#: Runtime packages the Jina VLM MLX captioner needs when constructed.
REQUIRED_MLX_RUNTIME_PACKAGES = (
    "mlx",
    "mlx_vlm",
    "transformers",
    "PIL",
)

#: Fixed caption prompt. Kept explicit so every artifact records the exact
#: instruction used to produce its captions.
DEFAULT_CAPTION_PROMPT = """Describe this image for a video retrieval system.
Create a short, factual, keyword-rich caption.
Focus on:
specific objects, people, animals, ingredients, tools, machines
actions and interactions
scene context, events, organizations, locations
important visible text from slides, screens, signs, documents, labels, and names
Prefer:specific entity + action + context.
Avoid:
image quality, lighting, composition, camera details
generic descriptions
phrases like "the image shows" or "this is a picture of"
Output only 1-3 concise sentences.
"""

#: Max tokens generated per caption (excluding the prompt).
DEFAULT_MAX_NEW_TOKENS = 256

logger = logging.getLogger(__name__)


class CaptionGenerationError(RuntimeError):
    """Raised when a caption build is interrupted (leaving a resumable state)."""


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a same-directory temp file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read and return a JSON object from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class KeyframeCaption:
    """One caption record for a single keyframe.

    Row ``i`` of ``captions.ndjson`` corresponds to line ``i`` of the source
    ``frames.ndjson``. A record with ``status="error"`` carries an empty
    ``caption`` and a human-readable ``error`` message; row alignment is
    preserved so downstream consumers can rely on stable indexes.

    Fields:
        - video_id: identity of the video.
        - keyframe_ordinal: 1-based keyframe ordinal within the video.
        - frame_index: 0-based keyframe index in the original video.
        - caption: the generated caption text (empty on failure).
        - status: ``ok`` or ``error``.
        - error: optional human-readable failure message.
    """

    video_id: VideoId
    keyframe_ordinal: int
    frame_index: int
    caption: str
    status: str
    error: str | None = None

    def __post_init__(self) -> None:
        _require(
            isinstance(self.keyframe_ordinal, int)
            and not isinstance(self.keyframe_ordinal, bool)
            and self.keyframe_ordinal >= 1,
            f"keyframe_ordinal must be a positive integer, got {self.keyframe_ordinal!r}",
        )
        _require(
            isinstance(self.frame_index, int) and not isinstance(self.frame_index, bool) and self.frame_index >= 0,
            f"frame_index must be a non-negative integer, got {self.frame_index!r}",
        )
        _require(self.status in ("ok", "error"), f"status must be 'ok' or 'error', got {self.status!r}")
        _require(isinstance(self.caption, str), "caption must be a string")
        if self.status == "ok":
            _require(self.error is None, "an 'ok' record must not carry an error message")
        else:
            _require(isinstance(self.error, str) and self.error != "", "an 'error' record requires an error message")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "video_id": self.video_id.to_dict(),
            "keyframe_ordinal": self.keyframe_ordinal,
            "frame_index": self.frame_index,
            "caption": self.caption,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeCaption":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            video_id=VideoId.from_dict(data["video_id"]),
            keyframe_ordinal=int(data["keyframe_ordinal"]),
            frame_index=int(data["frame_index"]),
            caption=str(data["caption"]),
            status=str(data["status"]),
            error=data.get("error"),
        )


def write_captions(path: Path, records: Sequence[KeyframeCaption]) -> int:
    """Append caption records as NDJSON, one per line.

    Expected input:
        - path: a file path (usually ``<artifact_dir>/captions.ndjson``).
        - records: captions in row order (index ``i`` becomes the next line).

    Returns:
        - Number of records written.

    Raises:
        - OSError: if the file cannot be written.
    """
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    return len(records)


def iter_captions(path: Path) -> list[KeyframeCaption]:
    """Return all caption records from an NDJSON file, in file order.

    Expected input:
        - path: a file path to ``captions.ndjson``.

    Returns:
        - List of :class:`KeyframeCaption` in file (row) order.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the caption contract.
    """
    records: list[KeyframeCaption] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            records.append(KeyframeCaption.from_dict(json.loads(line)))
    return records


# ------------------------------------------------------------------ manifest


@dataclass(frozen=True)
class CaptionManifest:
    """Versioned metadata describing one caption artifact.

    Fields:
        - schema_version: caption artifact schema version
          (``CAPTION_ARTIFACT_SCHEMA_VERSION``).
        - artifact_id: unique identifier for this artifact (run identity).
        - model: stable model identity (e.g. ``qwen2.5-vl:Qwen2.5-VL-3B-Instruct``).
        - checkpoint: HuggingFace checkpoint id, e.g. ``Qwen/Qwen2.5-VL-3B-Instruct``.
        - device: concrete runtime device that produced the captions.
        - batch_size: records per captioner invocation.
        - prompt: the exact caption prompt used.
        - source_frames: the input ``frames.ndjson`` path this artifact came from.
        - num_records: number ``N`` of caption records (rows).
        - num_videos: number of distinct videos covered.
        - num_failed: number of records with ``status="error"``.
        - canonical_schema_version: canonical schema version the source
          ``frames.ndjson`` records conform to.
        - created_at: ISO-8601 timestamp of artifact completion.
    """

    schema_version: str
    artifact_id: str
    model: str
    checkpoint: str
    device: str
    batch_size: int
    prompt: str
    source_frames: str
    num_records: int
    num_videos: int
    num_failed: int
    canonical_schema_version: str
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == CAPTION_ARTIFACT_SCHEMA_VERSION,
            f"unsupported caption schema version {self.schema_version!r}",
        )
        _require(isinstance(self.artifact_id, str) and self.artifact_id != "", "artifact_id required")
        _require(isinstance(self.model, str) and self.model != "", "model required")
        _require(isinstance(self.checkpoint, str) and self.checkpoint != "", "checkpoint required")
        _require(isinstance(self.device, str) and self.device != "", "device required")
        _require(
            isinstance(self.batch_size, int) and not isinstance(self.batch_size, bool) and self.batch_size >= 1,
            "batch_size must be >= 1",
        )
        _require(isinstance(self.prompt, str) and self.prompt != "", "prompt required")
        _require(isinstance(self.source_frames, str) and self.source_frames != "", "source_frames required")
        _require(
            isinstance(self.num_records, int) and not isinstance(self.num_records, bool) and self.num_records >= 0,
            "num_records must be >= 0",
        )
        _require(
            isinstance(self.num_videos, int) and not isinstance(self.num_videos, bool) and self.num_videos >= 0,
            "num_videos must be >= 0",
        )
        _require(
            isinstance(self.num_failed, int) and not isinstance(self.num_failed, bool) and self.num_failed >= 0,
            "num_failed must be >= 0",
        )
        _require(
            0 <= self.num_failed <= self.num_records,
            "num_failed must be between 0 and num_records",
        )
        _require(self.canonical_schema_version == CANONICAL_SCHEMA_VERSION, "canonical_schema_version mismatch")
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at required")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "batch_size": self.batch_size,
            "prompt": self.prompt,
            "source_frames": self.source_frames,
            "num_records": self.num_records,
            "num_videos": self.num_videos,
            "num_failed": self.num_failed,
            "canonical_schema_version": self.canonical_schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            model=str(data["model"]),
            checkpoint=str(data["checkpoint"]),
            device=str(data["device"]),
            batch_size=int(data["batch_size"]),
            prompt=str(data["prompt"]),
            source_frames=str(data["source_frames"]),
            num_records=int(data["num_records"]),
            num_videos=int(data["num_videos"]),
            num_failed=int(data["num_failed"]),
            canonical_schema_version=str(data["canonical_schema_version"]),
            created_at=str(data["created_at"]),
        )


@dataclass(frozen=True)
class CaptionCheckpoint:
    """Resume state for a partial caption artifact.

    Fields:
        - completed_record_count: number of records already written.
        - completed_video_ids: sorted video ids fully completed so far.
        - model: captioner model identity (must match on resume).
        - checkpoint: HuggingFace checkpoint id (must match on resume).
        - device: device that produced the written records.
        - batch_size: records per captioner invocation.
        - prompt: the caption prompt used (must match on resume).
        - source_frames: the input frames path (must match on resume).
        - total_records: expected final record count (must match on resume).
        - timestamp: ISO-8601 timestamp of the checkpoint write.
    """

    completed_record_count: int
    completed_video_ids: tuple[str, ...]
    model: str
    checkpoint: str
    device: str
    batch_size: int
    prompt: str
    source_frames: str
    total_records: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "completed_record_count": self.completed_record_count,
            "completed_video_ids": list(self.completed_video_ids),
            "model": self.model,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "batch_size": self.batch_size,
            "prompt": self.prompt,
            "source_frames": self.source_frames,
            "total_records": self.total_records,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionCheckpoint":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            completed_record_count=int(data["completed_record_count"]),
            completed_video_ids=tuple(str(v) for v in data["completed_video_ids"]),
            model=str(data["model"]),
            checkpoint=str(data["checkpoint"]),
            device=str(data["device"]),
            batch_size=int(data["batch_size"]),
            prompt=str(data["prompt"]),
            source_frames=str(data["source_frames"]),
            total_records=int(data["total_records"]),
            timestamp=str(data["timestamp"]),
        )


def resolve_checkpoint(model: str) -> str:
    """Resolve a short model name to a HuggingFace checkpoint id.

    A spec already containing ``/`` (``org/repo``) is returned unchanged;
    otherwise the default organization (``Qwen``) is prepended.

    Expected input:
        - model: a non-empty model name.

    Returns:
        - The resolved checkpoint id.

    Raises:
        - ValueError: if ``model`` is empty.
    """
    if not isinstance(model, str) or model == "":
        raise ValueError("model must be a non-empty string")
    if "/" in model:
        return model
    return f"{DEFAULT_ORG}/{model}"


def _resolve_device(requested: str, torch: Any) -> str:
    """Resolve a portable configured device to an available Torch device.

    Expected input:
        - requested: one of ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - torch: imported Torch module.

    Returns:
        - A concrete device string accepted by Torch/Transformers.

    Raises:
        - ValueError: if the requested device is unsupported or unavailable.
    """
    from aic2026.image_embedder import DEVICE_CHOICES, resolve_device

    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unsupported device {requested!r}; expected auto, cpu, mps, or cuda")
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return resolve_device(
        requested,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


def _parse_version(value: str) -> tuple[int, ...]:
    """Parse ``a.b.c[.suffix]`` into an integer tuple for comparison."""
    parts = []
    for chunk in value.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class QwenCaptionRuntimeError(RuntimeError):
    """Raised when the optional Qwen2.5-VL runtime cannot be loaded or used."""


class Qwen25VlCaptioner:
    """Qwen2.5-VL captioner loaded through HuggingFace Transformers.

    Transformers and Torch are imported only when this class is constructed,
    so importing ``aic2026.caption_generation`` never requires them.

    Expected input:
        - model: short model name (e.g. ``Qwen2.5-VL-3B-Instruct``) or a full
          checkpoint id.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - prompt: the caption prompt passed to the model.
        - max_new_tokens: maximum generated tokens per caption.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        prompt: str = DEFAULT_CAPTION_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model = model
        self._checkpoint = resolve_checkpoint(model)
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
        self._max_new_tokens = max_new_tokens
        self._prompt = prompt

        torch, transformers = self._load_dependencies()
        self._device = _resolve_device(device, torch)
        self._processor = transformers.AutoProcessor.from_pretrained(self._checkpoint)
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model_obj = transformers.AutoModelForVision2Seq.from_pretrained(
            self._checkpoint,
            torch_dtype=dtype,
        )
        self._model_obj.to(self._device)
        self._model_obj.eval()

    @staticmethod
    def _load_dependencies() -> tuple[Any, Any]:
        """Import and pre-flight the optional Qwen2.5-VL runtime.

        Returns:
            - The imported ``torch`` and ``transformers`` modules.

        Raises:
            - QwenCaptionRuntimeError: if a required optional dependency is
              missing or too old.
        """
        missing = []
        for name in REQUIRED_RUNTIME_PACKAGES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)
        if missing:
            raise QwenCaptionRuntimeError(
                "Qwen2.5-VL captioning requires the optional 'caption' dependency "
                f"group; missing packages: {', '.join(missing)}. "
                "Install with pip install -e '.[caption]'"
            )
        transformers = import_module("transformers")
        try:
            installed_version = version("transformers")
        except PackageNotFoundError:
            installed_version = "0"
        if _parse_version(installed_version) < _parse_version(MIN_TRANSFORMERS_VERSION):
            raise QwenCaptionRuntimeError(
                f"Qwen2.5-VL captioning requires transformers >= {MIN_TRANSFORMERS_VERSION}; "
                f"found {installed_version}. Upgrade with pip install -U 'transformers>={MIN_TRANSFORMERS_VERSION}'"
            )
        return import_module("torch"), transformers

    @property
    def device(self) -> str:
        """The concrete runtime device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Stable model identity recorded in manifests."""
        return f"qwen2.5-vl:{self._model}"

    @property
    def checkpoint(self) -> str:
        """The resolved HuggingFace checkpoint id."""
        return self._checkpoint

    def generate_captions(self, images: Sequence[object]) -> list[str]:
        """Return one caption per input image, preserving input order.

        Expected input:
            - images: backend-native images (e.g. RGB PIL images).

        Returns:
            - A caption string per input image.

        Raises:
            - RuntimeError: if generation fails (pipeline falls back to
              per-image captioning and records individual failures).
        """
        if not images:
            return []
        texts = []
        for _ in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<image>"},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ]
            texts.append(
                self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        import torch

        inputs = self._processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            output_ids = self._model_obj.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        captions = self._processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
        )
        return [caption.strip() for caption in captions]


class JinaCaptionRuntimeError(RuntimeError):
    """Raised when the optional Jina VLM MLX runtime cannot be loaded or used."""


class JinaVlmMlxCaptioner:
    """Jina VLM MLX captioner for Apple Silicon via ``mlx-vlm``.

    Native MLX port of Jina-VLM with 4-bit quantization (``~2 GB`` weights),
    optimized for Apple Silicon (Metal) local inference. ``mlx`` and
    ``mlx-vlm`` are imported only when this class is constructed, so importing
    ``aic2026.caption_generation`` never requires them.

    Expected input:
        - model: HuggingFace checkpoint id (default
          ``jinaai/jina-vlm-mlx``).
        - device: ``auto`` or ``mps`` (MLX runs on the Metal device; ``cpu``
          and ``cuda`` are rejected with a clear error).
        - prompt: the caption prompt passed to the model.
        - max_new_tokens: maximum generated tokens per caption.
    """

    def __init__(
        self,
        model: str = DEFAULT_JINA_VLM_MLX_MODEL,
        *,
        device: str = "mps",
        prompt: str = DEFAULT_CAPTION_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model = model
        self._checkpoint = model
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
        self._max_new_tokens = max_new_tokens
        self._prompt = prompt

        if device not in ("auto", "mps"):
            raise ValueError(
                f"unsupported device {device!r} for jina-vlm-mlx; "
                "MLX runs on Apple Silicon Metal and supports only 'auto' or 'mps'"
            )

        mlx_vlm = self._load_dependencies()
        self._device = "mps"
        self._model_obj, self._processor = mlx_vlm.load(self._checkpoint)
        self._config = mlx_vlm.utils.load_config(self._checkpoint)

    @staticmethod
    def _load_dependencies() -> Any:
        """Import and pre-flight the optional Jina VLM MLX runtime.

        Returns:
            - The imported ``mlx_vlm`` module.

        Raises:
            - JinaCaptionRuntimeError: if a required optional dependency is
              missing or the host is not Apple Silicon.
        """
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise JinaCaptionRuntimeError(
                "Jina VLM MLX runs on Apple Silicon (macOS arm64) only; "
                "use the 'qwen2.5-vl' backend on other platforms"
            )
        missing = []
        for name in REQUIRED_MLX_RUNTIME_PACKAGES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)
        if missing:
            raise JinaCaptionRuntimeError(
                "Jina VLM MLX captioning requires the optional 'caption-mlx' dependency "
                f"group; missing packages: {', '.join(missing)}. "
                "Install with pip install -e '.[caption-mlx]'"
            )
        return import_module("mlx_vlm")

    @property
    def device(self) -> str:
        """The concrete runtime device (``mps``)."""
        return self._device

    @property
    def model_name(self) -> str:
        """Stable model identity recorded in manifests."""
        return f"jina-vlm-mlx:{self._model}"

    @property
    def checkpoint(self) -> str:
        """The HuggingFace checkpoint id."""
        return self._checkpoint

    def generate_captions(self, images: Sequence[object]) -> list[str]:
        """Return one caption per input image, preserving input order.

        mlx-vlm's ``generate`` processes a single prompt per call, so each
        image is captioned independently (the pipeline's batch fallback also
        degrades gracefully to per-image calls).

        Expected input:
            - images: backend-native images (e.g. RGB PIL images).

        Returns:
            - A caption string per input image.

        Raises:
            - RuntimeError: if generation fails (pipeline falls back to
              per-image captioning and records individual failures).
        """
        if not images:
            return []
        mlx_vlm = import_module("mlx_vlm")
        captions: list[str] = []
        for image in images:
            formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
                self._processor,
                self._config,
                self._prompt,
                num_images=1,
            )
            output = mlx_vlm.generate(
                self._model_obj,
                self._processor,
                formatted_prompt,
                [image],
                max_tokens=self._max_new_tokens,
            )
            captions.append(self._coerce_caption(output))
        return captions

    @staticmethod
    def _coerce_caption(output: object) -> str:
        """Extract the caption text from an mlx-vlm generation result.

        ``mlx_vlm.generate`` returns an object exposing ``.text`` (or a plain
        string in some versions); both forms are handled defensively.

        Expected input:
            - output: the mlx-vlm generation result.

        Returns:
            - The caption text, stripped.
        """
        text = getattr(output, "text", None)
        if text is None:
            text = str(output)
        return str(text).strip()


def create_captioner(
    backend: str = DEFAULT_BACKEND,
    *,
    model: str | None = None,
    device: str = "auto",
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Any:
    """Create a captioner for a named backend.

    Expected input:
        - backend: one of ``CAPTION_BACKENDS`` (``qwen2.5-vl`` or
          ``jina-vlm-mlx``).
        - model: optional checkpoint id; defaults to the backend's default.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - prompt: the caption prompt passed to the captioner.
        - max_new_tokens: maximum generated tokens per caption.

    Returns:
        - A configured captioner instance for the backend.

    Raises:
        - ValueError: if ``backend`` is not one of ``CAPTION_BACKENDS``.
        - QwenCaptionRuntimeError / JinaCaptionRuntimeError: if the selected
          backend's optional runtime is unavailable.
    """
    if backend == "qwen2.5-vl":
        return Qwen25VlCaptioner(
            model=model or DEFAULT_MODEL,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    if backend == "jina-vlm-mlx":
        return JinaVlmMlxCaptioner(
            model=model or DEFAULT_JINA_VLM_MLX_MODEL,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"unsupported caption backend {backend!r}; expected {CAPTION_BACKENDS}")


# ------------------------------------------------------------------ helpers


def _count_caption_lines(path: Path) -> int:
    """Return the number of complete caption records in an NDJSON file.

    A trailing partial line (crash between line writes) is ignored, so the
    resume point always reflects fully written records.

    Expected input:
        - path: a file path to ``captions.ndjson``.

    Returns:
        - Number of complete records; 0 if the file is missing or empty.
    """
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip() == "":
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                break
            count += 1
    return count


def _trim_captions_overflow(path: Path, keep: int) -> None:
    """Rewrite ``path`` keeping only the first ``keep`` complete records.

    A crash between writing a record and the checkpoint write can leave more
    lines than the checkpoint records; trailing lines (including a partial
    last line) are dropped.

    Expected input:
        - path: a file path to ``captions.ndjson``.
        - keep: number of records to retain.

    Raises:
        - OSError: if the file cannot be rewritten.
    """
    kept: list[str] = []
    overflow = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip() == "":
                continue
            if len(kept) < keep:
                kept.append(line)
            else:
                overflow = True
                break
    if not overflow:
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(record + "\n")


def _last_index_by_video(frames: Sequence[KeyframeTiming]) -> dict[str, int]:
    """Return ``video_id -> last row index`` in the artifact's row order."""
    last: dict[str, int] = {}
    for i, frame in enumerate(frames):
        last[frame.keyframe.video_id.value] = i
    return last


def _completed_video_ids(
    last_index: dict[str, int],
    completed_rows: int,
) -> tuple[str, ...]:
    """Return sorted video ids fully completed by ``completed_rows`` rows."""
    return tuple(sorted(vid for vid, last in last_index.items() if last < completed_rows))


def _caption_batch_safe(
    captioner: Any,
    images: Sequence[object],
) -> list[tuple[str, str | None]]:
    """Caption a batch, falling back to per-image calls on batch failure.

    Expected input:
        - captioner: the captioner to invoke.
        - images: loaded images to caption.

    Returns:
        - One ``(caption, error)`` tuple per image. On success ``error`` is
          ``None``; on failure the caption is ``""`` and the error message is
          populated.
    """
    if not images:
        return []
    try:
        captions = captioner.generate_captions(list(images))
        if len(captions) != len(images):
            raise ValueError(
                f"captioner returned {len(captions)} captions for {len(images)} images"
            )
        return [(caption or "", None) for caption in captions]
    except Exception as batch_exc:  # noqa: BLE001 - recover per image below
        logger.warning("batch captioning failed (%s); retrying per image", batch_exc)
        results: list[tuple[str, str | None]] = []
        for image in images:
            try:
                captions = captioner.generate_captions([image])
                if len(captions) != 1:
                    raise ValueError(f"captioner returned {len(captions)} captions for 1 image")
                results.append((captions[0] or "", None))
            except Exception as exc:  # noqa: BLE001 - recorded as an error record
                logger.warning("per-image captioning failed: %s", exc)
                results.append(("", str(exc)))
        return results


# ------------------------------------------------------------------ pipeline


def generate_captions(
    keyframes_dir: Path | str,
    output_dir: Path | str,
    *,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    batch_size: int = 4,
    device: str = "auto",
    resume: bool = True,
    max_records: int | None = None,
    stop_after_records: int | None = None,
    progress: ProgressReporter | str = "auto",
    captioner: Any | None = None,
    image_loader: Callable[[Path], object] | None = None,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> CaptionManifest:
    """Build a resumable caption artifact from canonical keyframe records.

    Expected input:
        - keyframes_dir: root directory containing the canonical
          ``frames.ndjson`` and the keyframe ``images/`` collection
          (materialized ``adaptive_sbd_v1`` layout).
        - output_dir: target caption artifact directory (created if absent).
        - backend: caption backend name, one of ``CAPTION_BACKENDS``
          (``qwen2.5-vl`` or ``jina-vlm-mlx``); ignored when ``captioner`` is
          given.
        - model: short model name or full checkpoint id used to build the
          backend captioner (ignored when ``captioner`` is given).
        - batch_size: records written (and captioner invocations) per batch.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda`` (ignored when
          ``captioner`` is given).
        - resume: if True, continue a partial artifact when present; an
          already-complete artifact is validated and returned without recompute.
        - max_records: optional row limit selecting the first ``max_records``
          records (for small-scale validation / sharding).
        - stop_after_records: optional testing hook to stop after writing N
          records (no final manifest is written; the artifact stays resumable).
        - progress: progress reporting mode ``auto``/``on``/``off`` or a shared
          :class:`ProgressReporter`.
        - captioner: optional test captioner; production calls create it from
          ``backend``.
        - image_loader: callable ``path -> backend-native image``; defaults to
          a Pillow-based loader.
        - prompt: the caption prompt passed to the captioner and recorded in
          the manifest.

    Returns:
        - Final :class:`CaptionManifest`.

    Raises:
        - ValueError: on invalid inputs or an inconsistent/conflicting partial
          artifact (model, checkpoint, prompt, source, or total_records
          changed). A changed device is allowed (warning only).
        - CaptionGenerationError: when ``stop_after_records`` interrupts the
          build.
        - FileNotFoundError: if the source ``frames.ndjson`` is missing.
    """
    keyframes_dir = Path(keyframes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(prompt, str) or prompt == "":
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a bool")
    if backend not in CAPTION_BACKENDS:
        raise ValueError(f"unsupported caption backend {backend!r}; expected {CAPTION_BACKENDS}")

    if isinstance(progress, ProgressReporter):
        reporter = progress
    elif progress is None:
        reporter = ProgressReporter("off")
    elif isinstance(progress, str):
        reporter = create_progress_reporter(progress)
    else:
        raise ValueError(f"progress must be a ProgressReporter or a mode string, got {progress!r}")
    reporter.begin()
    stages = reporter.stages

    frames_path = keyframes_dir / FRAMES_FILENAME
    if not frames_path.is_file():
        raise FileNotFoundError(f"missing source frames: {frames_path}")
    source = str(frames_path.resolve())

    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    captions_path = output_dir / CAPTIONS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    if captioner is None:
        captioner = create_captioner(
            backend,
            model=model,
            device=device,
            prompt=prompt,
        )
    model_name = captioner.model_name
    checkpoint = captioner.checkpoint
    runtime_device = captioner.device

    with stages.time("load_frames"):
        frames = list(iter_frames(frames_path))
        selected = select_frames(frames, max_records)
        total_records = len(selected)

    # ------------------------------------------------------------ complete fast path
    if resume and manifest_path.is_file():
        existing = CaptionManifest.from_dict(_read_json(manifest_path))
        if existing.model != model_name:
            raise ValueError(
                f"existing caption model {existing.model!r} does not match {model_name!r}"
            )
        if existing.checkpoint != checkpoint:
            raise ValueError(
                f"existing caption checkpoint {existing.checkpoint!r} does not match {checkpoint!r}"
            )
        if existing.prompt != prompt:
            raise ValueError("existing caption prompt does not match the current prompt")
        if existing.source_frames != source:
            raise ValueError(
                f"existing caption source {existing.source_frames!r} != current {source!r}"
            )
        if existing.num_records != total_records:
            raise ValueError(
                f"existing caption num_records {existing.num_records} != selected rows {total_records}"
            )
        return existing

    # ------------------------------------------------------------ decide the resume point
    start_record = 0
    completed_videos: set[str] = set()
    if resume and checkpoint_path.is_file():
        ckpt = CaptionCheckpoint.from_dict(_read_json(checkpoint_path))
        if ckpt.model != model_name:
            raise ValueError(f"checkpoint model {ckpt.model!r} does not match {model_name!r}")
        if ckpt.checkpoint != checkpoint:
            raise ValueError(f"checkpoint checkpoint {ckpt.checkpoint!r} does not match {checkpoint!r}")
        if ckpt.prompt != prompt:
            raise ValueError("checkpoint prompt does not match the current prompt")
        if ckpt.source_frames != source:
            raise ValueError(f"checkpoint source_frames {ckpt.source_frames!r} != current {source!r}")
        if ckpt.total_records != total_records:
            raise ValueError(
                f"checkpoint total_records {ckpt.total_records} != current {total_records}; "
                "source or max_records changed"
            )
        if ckpt.device != runtime_device:
            logger.warning(
                "resuming caption build on device %r (checkpoint was device %r); continuing",
                runtime_device,
                ckpt.device,
            )
        start_record = ckpt.completed_record_count
        completed_videos = set(ckpt.completed_video_ids)

    if start_record > 0:
        if not captions_path.is_file():
            raise ValueError("checkpoint exists but captions.ndjson is missing; cannot resume")
        _trim_captions_overflow(captions_path, start_record)
        if _count_caption_lines(captions_path) != start_record:
            raise ValueError(
                f"captions.ndjson complete records do not match checkpoint count "
                f"({start_record}); cannot resume"
            )

    logger.info(
        "caption build: %d records (model=%s, device=%s, resume_from=%d)",
        total_records,
        model_name,
        runtime_device,
        start_record,
    )
    reporter.set_total(total_records, completed=start_record)

    load_image = image_loader if image_loader is not None else default_image_loader()
    last_index = _last_index_by_video(selected)

    def write_checkpoint(completed: int) -> None:
        _write_json_atomic(
            checkpoint_path,
            CaptionCheckpoint(
                completed_record_count=completed,
                completed_video_ids=_completed_video_ids(last_index, completed),
                model=model_name,
                checkpoint=checkpoint,
                device=runtime_device,
                batch_size=batch_size,
                prompt=prompt,
                source_frames=source,
                total_records=total_records,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict(),
        )

    # ------------------------------------------------------------ stream keyframes through the captioner
    next_record = start_record
    pending: list[tuple[KeyframeTiming, object | None, str | None]] = []

    def write_pending_batch() -> None:
        nonlocal next_record, pending
        if not pending:
            return
        with stages.time("caption"):
            ok_frames = [frame for frame, image, _ in pending if image is not None]
            ok_images = [image for _, image, _ in pending if image is not None]
            results = _caption_batch_safe(captioner, ok_images)
            result_index = 0
            records: list[KeyframeCaption] = []
            for frame, image, load_error in pending:
                if load_error is not None:
                    records.append(
                        KeyframeCaption(
                            video_id=frame.keyframe.video_id,
                            keyframe_ordinal=frame.keyframe.keyframe_ordinal,
                            frame_index=frame.frame_index,
                            caption="",
                            status="error",
                            error=load_error,
                        )
                    )
                    continue
                caption, caption_error = results[result_index]
                result_index += 1
                records.append(
                    KeyframeCaption(
                        video_id=frame.keyframe.video_id,
                        keyframe_ordinal=frame.keyframe.keyframe_ordinal,
                        frame_index=frame.frame_index,
                        caption=caption,
                        status="ok" if caption_error is None else "error",
                        error=caption_error,
                    )
                )
        with stages.time("write_outputs"):
            for idx, record in enumerate(records):
                if stop_after_records is not None and next_record >= stop_after_records:
                    write_checkpoint(next_record)
                    raise CaptionGenerationError(
                        f"interrupted caption build at record {next_record}; "
                        f"checkpoint written to {checkpoint_path}"
                    )
                write_captions(captions_path, [record])
                next_record += 1
            write_checkpoint(next_record)
        pending = []
        reporter.update(next_record, note=f"video={records[0].video_id.value if records else ''}")

    try:
        for frame in selected[start_record:]:
            if stop_after_records is not None and next_record >= stop_after_records:
                write_checkpoint(next_record)
                raise CaptionGenerationError(
                    f"interrupted caption build at record {next_record}; "
                    f"checkpoint written to {checkpoint_path}"
                )
            with stages.time("load_images"):
                video_id = frame.keyframe.video_id.value
                load_error: str | None = None
                image: object | None = None
                try:
                    path = keyframe_image_path(keyframes_dir, video_id, frame.keyframe.keyframe_ordinal)
                    image = load_image(path)
                except Exception as exc:  # noqa: BLE001 - recorded as an error record
                    load_error = f"{type(exc).__name__}: {exc}"
            pending.append((frame, image, load_error))
            if len(pending) >= batch_size:
                write_pending_batch()
        write_pending_batch()
        write_checkpoint(next_record)
    except Exception:
        raise
    finally:
        pass

    if next_record != total_records:
        raise ValueError(f"written records ({next_record}) != expected ({total_records})")

    # ------------------------------------------------------------ validate then write the final manifest
    with stages.time("write_outputs"):
        line_count = _count_caption_lines(captions_path)
        if line_count != total_records:
            raise ValueError(f"captions.ndjson records ({line_count}) != expected ({total_records})")
        records = iter_captions(captions_path)
        num_failed = sum(1 for record in records if record.status == "error")
        manifest = CaptionManifest(
            schema_version=CAPTION_ARTIFACT_SCHEMA_VERSION,
            artifact_id=output_dir.name,
            model=model_name,
            checkpoint=checkpoint,
            device=runtime_device,
            batch_size=batch_size,
            prompt=prompt,
            source_frames=source,
            num_records=total_records,
            num_videos=len({record.video_id.value for record in records}),
            num_failed=num_failed,
            canonical_schema_version=CANONICAL_SCHEMA_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json_atomic(manifest_path, manifest.to_dict())
    for leftover in (checkpoint_path,):
        if leftover.exists():
            leftover.unlink()
    reporter.finish()
    return manifest


# ------------------------------------------------------------------ validation


def validate_caption_artifact(manifest: CaptionManifest, artifact_dir: Path) -> list[str]:
    """Validate that an artifact directory matches its manifest.

    Expected input:
        - manifest: the artifact's manifest.
        - artifact_dir: the directory containing the artifact files.

    Returns:
        - List of human-readable problems; empty when the artifact is
          consistent.

    Raises:
        - No exceptions (problems are accumulated and returned).
    """
    problems: list[str] = []
    captions_path = artifact_dir / CAPTIONS_FILENAME

    if not captions_path.is_file():
        problems.append(f"missing {CAPTIONS_FILENAME}")
        return problems

    try:
        records = iter_captions(captions_path)
    except Exception as exc:  # noqa: BLE001 - surface as a problem
        problems.append(f"{CAPTIONS_FILENAME} unreadable: {exc}")
        return problems

    if len(records) != manifest.num_records:
        problems.append(
            f"{CAPTIONS_FILENAME} records ({len(records)}) != manifest.num_records ({manifest.num_records})"
        )
    num_failed = sum(1 for record in records if record.status == "error")
    if num_failed != manifest.num_failed:
        problems.append(
            f"error records ({num_failed}) != manifest.num_failed ({manifest.num_failed})"
        )
    num_videos = len({record.video_id.value for record in records})
    if num_videos != manifest.num_videos:
        problems.append(
            f"distinct videos ({num_videos}) != manifest.num_videos ({manifest.num_videos})"
        )
    return problems


def read_caption_artifact(artifact_dir: Path | str) -> tuple[CaptionManifest, list[KeyframeCaption]]:
    """Read a completed caption artifact.

    Expected input:
        - artifact_dir: the caption artifact directory.

    Returns:
        - ``(manifest, records)`` with records in row order.

    Raises:
        - FileNotFoundError: if ``manifest.json`` is missing.
        - ValueError: if the manifest or records are invalid.
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing caption manifest: {manifest_path}")
    manifest = CaptionManifest.from_dict(_read_json(manifest_path))
    problems = validate_caption_artifact(manifest, artifact_dir)
    if problems:
        raise ValueError("invalid caption artifact: " + "; ".join(problems))
    return manifest, iter_captions(artifact_dir / CAPTIONS_FILENAME)
