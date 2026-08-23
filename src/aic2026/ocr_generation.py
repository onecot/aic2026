"""Resumable, backend-agnostic keyframe OCR extraction pipeline.

Produces an OCR artifact from canonical dataset keyframes using an injected
OCR engine. The pipeline is backend-agnostic: any object satisfying the
test callers may inject an engine with the same methods. One backend ships
with the pipeline:

- ``paddleocr`` (:class:`PaddleOcrEngine`) via PaddleOCR PP-OCRv5
  multilingual (Vietnamese + English), optimized for subtitles, slides,
  signs, logos, documents, and small text; GPU inference supported.

PaddleOCR/PaddlePaddle are imported only when the engine is constructed,
keeping the core package and its unit tests independent of the optional OCR
runtime dependencies.

Public API:
- ``generate_ocr(keyframes_dir, output_dir, ...)``
- ``create_ocr_engine(backend, ...)``
- ``KeyframeOcr``, ``OcrLine``, ``OcrManifest``, ``OcrCheckpoint``
- ``normalize_ocr_text`` / ``preprocess_image``
- ``read_ocr_artifact`` / ``validate_ocr_artifact``

Behavior:
- deterministic input order: frames in ``frames.ndjson`` file (row) order;
  row ``i`` of ``ocr.ndjson`` always corresponds to line ``i`` of
  ``frames.ndjson``
- resumable via ``checkpoint.json``; already-processed records are skipped on
  resume and the artifact is considered complete only after ``manifest.json``
  is written atomically (leftover checkpoint/build metadata is deleted)
- graceful degradation: a missing or unreadable keyframe image becomes an
  ``error`` record (empty text + message) instead of failing the run;
  an OCR failure falls back from batch to per-image extraction
- safe writes: the manifest and checkpoint are written atomically; partial
  outputs are never exposed as valid (no ``manifest.json`` until completion)

OCR artifact layout::

    <output-dir>/
        checkpoint.json   resume state (completed count + video ids + identity)
        ocr.ndjson        one KeyframeOcr per line (row i <-> frames[i])
        manifest.json     versioned metadata (written on completion)

The keyframe artifact is read-only: this pipeline never modifies
``keyframes_dir`` (frames or images).

OCR is deterministic text extraction: it never translates text. Vietnamese
and English text is kept as-is; ``ocr_text_normalized`` is a lowercase,
whitespace-normalized copy with duplicated symbols removed (accents are
preserved). Retrieval fusion (OCR-text embedding / BM25 fused with CLIP) is
a separate, future stage and is not implemented here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from aic2026.canonical import CANONICAL_SCHEMA_VERSION, KeyframeTiming, VideoId
from aic2026.embedding_artifact import FRAMES_FILENAME, iter_frames
from aic2026.embedding_runner import default_image_loader
from aic2026.frame_images import find_keyframe_image
from aic2026.embedding_shard import select_frames
from aic2026.progress import ProgressReporter, create_progress_reporter

#: Schema version of the OCR artifact.
OCR_ARTIFACT_SCHEMA_VERSION = "1.0"

#: Relative filenames within an OCR artifact directory.
CHECKPOINT_FILENAME = "checkpoint.json"
OCR_FILENAME = "ocr.ndjson"
MANIFEST_FILENAME = "manifest.json"

#: Default PaddleOCR PP-OCRv5 model family (mobile = fast, small).
DEFAULT_OCR_MODEL = "PP-OCRv5_mobile"

#: Supported OCR backends selectable via ``--backend`` / the ``backend``
#: argument of :func:`create_ocr_engine`.
OCR_BACKENDS = ("paddleocr",)

#: Backend used when none is explicitly selected (PaddleOCR).
DEFAULT_BACKEND = "paddleocr"

#: Default OCR languages (Vietnamese + English, both kept as-is).
DEFAULT_LANGUAGES = ("vi", "en")

#: Runtime packages the PaddleOCR engine needs when constructed.
REQUIRED_OCR_RUNTIME_PACKAGES = (
    "paddleocr",
    "paddle",
    "PIL",
)

#: PaddleOCR ``lang`` value used when the languages tuple contains ``vi``.
_PADDLE_LANG_VI = "vi"

#: Symbols collapsed when duplicated in :func:`normalize_ocr_text`.
_SYMBOL_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

logger = logging.getLogger(__name__)


class OcrGenerationError(RuntimeError):
    """Raised when an OCR build is interrupted (leaving a resumable state)."""


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


# ------------------------------------------------------------------ text processing


def normalize_ocr_text(text: str) -> str:
    """Produce a search-friendly copy of OCR text for retrieval.

    Applies, in order: lowercase (Vietnamese accents are preserved by
    Unicode lowercasing), whitespace normalization (runs collapsed to a
    single space), and removal of duplicated symbols (consecutive identical
    punctuation collapsed to one). Text is never translated.

    Expected input:
        - text: raw OCR text.

    Returns:
        - The normalized text.

    Raises:
        - ValueError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {text!r}")
    lowered = text.lower()
    tokens = lowered.split()
    out: list[str] = []
    for token in tokens:
        chars: list[str] = []
        previous: str | None = None
        for char in token:
            if char in _SYMBOL_CHARS and char == previous:
                continue
            chars.append(char)
            previous = char
        out.append("".join(chars))
    return " ".join(out).strip()


def _join_ocr_lines(lines: Sequence[OcrLine]) -> tuple[str, str]:
    """Return ``(raw_text, normalized_text)`` for a list of OCR lines."""
    raw = " ".join(line.text for line in lines if line.text)
    normalized = " ".join(normalize_ocr_text(line.text) for line in lines if line.text)
    return raw, normalized


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class OcrLine:
    """One detected text line within a keyframe.

    Fields:
        - text: the recognized text.
        - confidence: recognition confidence in ``[0.0, 1.0]``.
        - bbox: axis-aligned bounding box ``(x1, y1, x2, y2)`` in image
          pixel coordinates.
    """

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _require(isinstance(self.text, str), "text must be a string")
        _require(
            isinstance(self.confidence, (int, float)) and not isinstance(self.confidence, bool),
            "confidence must be a number",
        )
        _require(0.0 <= self.confidence <= 1.0, f"confidence must be in [0, 1], got {self.confidence!r}")
        _require(
            isinstance(self.bbox, (tuple, list)) and len(self.bbox) == 4,
            "bbox must have 4 elements (x1, y1, x2, y2)",
        )
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        _require(x2 >= x1 and y2 >= y1, f"invalid bbox {self.bbox!r}: x2/y2 must be >= x1/y1")
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrLine":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            text=str(data["text"]),
            confidence=float(data["confidence"]),
            bbox=tuple(float(v) for v in data["bbox"]),
        )


@dataclass(frozen=True)
class KeyframeOcr:
    """One OCR record for a single keyframe.

    Row ``i`` of ``ocr.ndjson`` corresponds to line ``i`` of the source
    ``frames.ndjson``. A record with ``status="error"`` carries an empty
    ``ocr_text`` / ``ocr_lines`` and a human-readable ``error`` message; row
    alignment is preserved so downstream consumers can rely on stable indexes.

    Fields:
        - video_id: identity of the video.
        - keyframe_ordinal: 1-based keyframe ordinal within the video.
        - frame_index: 0-based keyframe index in the original video.
        - ocr_text: raw detected text (lines joined with spaces; empty on
          failure or when no text is found).
        - ocr_text_normalized: normalized copy of ``ocr_text`` (lowercase,
          whitespace-normalized, duplicated symbols removed, accents kept).
        - ocr_lines: detected text lines, in reading order.
        - status: ``ok`` or ``error``.
        - error: optional human-readable failure message.
    """

    video_id: VideoId
    keyframe_ordinal: int
    frame_index: int
    ocr_text: str
    ocr_text_normalized: str
    ocr_lines: tuple[OcrLine, ...]
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
        _require(isinstance(self.ocr_text, str), "ocr_text must be a string")
        _require(isinstance(self.ocr_text_normalized, str), "ocr_text_normalized must be a string")
        _require(isinstance(self.ocr_lines, tuple), "ocr_lines must be a tuple of OcrLine")
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
            "ocr_text": self.ocr_text,
            "ocr_text_normalized": self.ocr_text_normalized,
            "ocr_lines": [line.to_dict() for line in self.ocr_lines],
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyframeOcr":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            video_id=VideoId.from_dict(data["video_id"]),
            keyframe_ordinal=int(data["keyframe_ordinal"]),
            frame_index=int(data["frame_index"]),
            ocr_text=str(data["ocr_text"]),
            ocr_text_normalized=str(data["ocr_text_normalized"]),
            ocr_lines=tuple(OcrLine.from_dict(line) for line in data["ocr_lines"]),
            status=str(data["status"]),
            error=data.get("error"),
        )


def write_ocr_records(path: Path, records: Sequence[KeyframeOcr]) -> int:
    """Append OCR records as NDJSON, one per line.

    Expected input:
        - path: a file path (usually ``<artifact_dir>/ocr.ndjson``).
        - records: OCR records in row order (index ``i`` becomes the next line).

    Returns:
        - Number of records written.

    Raises:
        - OSError: if the file cannot be written.
    """
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")
    return len(records)


def iter_ocr_records(path: Path) -> list[KeyframeOcr]:
    """Return all OCR records from an NDJSON file, in file order.

    Expected input:
        - path: a file path to ``ocr.ndjson``.

    Returns:
        - List of :class:`KeyframeOcr` in file (row) order.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if a line is malformed.
        - ValueError: if a line fails the OCR contract.
    """
    records: list[KeyframeOcr] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            records.append(KeyframeOcr.from_dict(json.loads(line)))
    return records


# ------------------------------------------------------------------ manifest


@dataclass(frozen=True)
class OcrManifest:
    """Versioned metadata describing one OCR artifact.

    Fields:
        - schema_version: OCR artifact schema version
          (``OCR_ARTIFACT_SCHEMA_VERSION``).
        - artifact_id: unique identifier for this artifact (run identity).
        - backend: OCR backend name (e.g. ``paddleocr``).
        - model: stable model identity (e.g. ``paddleocr:PP-OCRv5_mobile``).
        - languages: OCR languages configured (e.g. ``("vi", "en")``).
        - device: concrete runtime device that produced the records.
        - batch_size: records per engine invocation.
        - source_frames: the input ``frames.ndjson`` path this artifact came
          from.
        - num_records: number ``N`` of OCR records (rows).
        - num_videos: number of distinct videos covered.
        - num_failed: number of records with ``status="error"``.
        - canonical_schema_version: canonical schema version the source
          ``frames.ndjson`` records conform to.
        - created_at: ISO-8601 timestamp of artifact completion.
    """

    schema_version: str
    artifact_id: str
    backend: str
    model: str
    languages: tuple[str, ...]
    device: str
    batch_size: int
    source_frames: str
    num_records: int
    num_videos: int
    num_failed: int
    canonical_schema_version: str
    created_at: str

    def __post_init__(self) -> None:
        _require(
            self.schema_version == OCR_ARTIFACT_SCHEMA_VERSION,
            f"unsupported OCR schema version {self.schema_version!r}",
        )
        _require(isinstance(self.artifact_id, str) and self.artifact_id != "", "artifact_id required")
        _require(isinstance(self.backend, str) and self.backend != "", "backend required")
        _require(isinstance(self.model, str) and self.model != "", "model required")
        _require(isinstance(self.languages, tuple) and len(self.languages) >= 1, "languages required")
        _require(all(isinstance(lang, str) and lang != "" for lang in self.languages), "languages must be strings")
        _require(isinstance(self.device, str) and self.device != "", "device required")
        _require(
            isinstance(self.batch_size, int) and not isinstance(self.batch_size, bool) and self.batch_size >= 1,
            "batch_size must be >= 1",
        )
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
            "backend": self.backend,
            "model": self.model,
            "languages": list(self.languages),
            "device": self.device,
            "batch_size": self.batch_size,
            "source_frames": self.source_frames,
            "num_records": self.num_records,
            "num_videos": self.num_videos,
            "num_failed": self.num_failed,
            "canonical_schema_version": self.canonical_schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ValueError: if a value is invalid.
        """
        return cls(
            schema_version=str(data["schema_version"]),
            artifact_id=str(data["artifact_id"]),
            backend=str(data["backend"]),
            model=str(data["model"]),
            languages=tuple(str(lang) for lang in data["languages"]),
            device=str(data["device"]),
            batch_size=int(data["batch_size"]),
            source_frames=str(data["source_frames"]),
            num_records=int(data["num_records"]),
            num_videos=int(data["num_videos"]),
            num_failed=int(data["num_failed"]),
            canonical_schema_version=str(data["canonical_schema_version"]),
            created_at=str(data["created_at"]),
        )


@dataclass(frozen=True)
class OcrCheckpoint:
    """Resume state for a partial OCR artifact.

    Fields:
        - completed_record_count: number of records already written.
        - completed_video_ids: sorted video ids fully completed so far.
        - backend: OCR backend name (must match on resume).
        - model: OCR model identity (must match on resume).
        - languages: OCR languages (must match on resume).
        - device: device that produced the written records.
        - batch_size: records per engine invocation.
        - source_frames: the input frames path (must match on resume).
        - total_records: expected final record count (must match on resume).
        - timestamp: ISO-8601 timestamp of the checkpoint write.
    """

    completed_record_count: int
    completed_video_ids: tuple[str, ...]
    backend: str
    model: str
    languages: tuple[str, ...]
    device: str
    batch_size: int
    source_frames: str
    total_records: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "completed_record_count": self.completed_record_count,
            "completed_video_ids": list(self.completed_video_ids),
            "backend": self.backend,
            "model": self.model,
            "languages": list(self.languages),
            "device": self.device,
            "batch_size": self.batch_size,
            "source_frames": self.source_frames,
            "total_records": self.total_records,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrCheckpoint":
        """Deserialize from a :meth:`to_dict` dict."""
        return cls(
            completed_record_count=int(data["completed_record_count"]),
            completed_video_ids=tuple(str(v) for v in data["completed_video_ids"]),
            backend=str(data["backend"]),
            model=str(data["model"]),
            languages=tuple(str(lang) for lang in data["languages"]),
            device=str(data["device"]),
            batch_size=int(data["batch_size"]),
            source_frames=str(data["source_frames"]),
            total_records=int(data["total_records"]),
            timestamp=str(data["timestamp"]),
        )


def _resolve_paddle_lang(languages: Sequence[str]) -> str:
    """Map the configured languages to a single PaddleOCR ``lang`` value.

    Vietnamese is preferred when present (its recognition model also covers
    English); otherwise the first configured language is used.

    Expected input:
        - languages: non-empty sequence of language codes.

    Returns:
        - A PaddleOCR ``lang`` value.

    Raises:
        - ValueError: if ``languages`` is empty.
    """
    if not languages:
        raise ValueError("languages must contain at least one language")
    if _PADDLE_LANG_VI in languages:
        return _PADDLE_LANG_VI
    return languages[0]


def _resolve_paddle_device(requested: str, paddle: Any) -> str:
    """Resolve a portable device request to a PaddleOCR device string.

    ``auto`` prefers CUDA then CPU. ``mps`` is not supported by PaddlePaddle
    and is mapped to ``cpu`` with a warning.

    Expected input:
        - requested: one of ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - paddle: imported Paddle module.

    Returns:
        - A concrete device string accepted by PaddleOCR (``gpu:0`` or
          ``cpu``).

    Raises:
        - ValueError: if the requested device is unsupported or unavailable.
    """
    from aic2026.image_embedder import DEVICE_CHOICES

    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unsupported device {requested!r}; expected auto, cpu, mps, or cuda")
    cuda_available = bool(
        getattr(paddle.device, "is_compiled_with_cuda", lambda: False)()
        and getattr(paddle.device.cuda, "device_count", lambda: 0)() > 0
    )
    if requested == "cuda":
        if not cuda_available:
            raise ValueError("cuda requested but no CUDA device is available")
        return "gpu:0"
    if requested == "mps":
        logger.warning(
            "PaddlePaddle has no MPS (Metal) backend; using cpu instead for the requested device 'mps'"
        )
        return "cpu"
    if requested == "auto":
        return "gpu:0" if cuda_available else "cpu"
    return "cpu"


class PaddleOcrRuntimeError(RuntimeError):
    """Raised when the optional PaddleOCR runtime cannot be loaded or used."""


class PaddleOcrEngine:
    """PaddleOCR PP-OCRv5 multilingual engine (Vietnamese + English).

    PaddleOCR/PaddlePaddle are imported only when this class is constructed,
    so importing ``aic2026.ocr_generation`` never requires them.

    Expected input:
        - model: PP-OCRv5 model family name (default ``PP-OCRv5_mobile``).
        - device: ``auto``, ``cpu``, ``mps`` (mapped to ``cpu``), or ``cuda``.
        - languages: OCR languages, e.g. ``("vi", "en")`` (Vietnamese +
          English). Text is extracted as-is, never translated.
        - enable_doc_orientation / enable_doc_unwarping / enable_textline_orientation:
          enable the optional PaddleOCR document preprocessing modules. They
          are disabled by default for throughput; typical upright subtitles,
          slides, and signs do not need them.
    """

    def __init__(
        self,
        model: str = DEFAULT_OCR_MODEL,
        *,
        device: str = "auto",
        languages: Sequence[str] = DEFAULT_LANGUAGES,
        enable_doc_orientation: bool = False,
        enable_doc_unwarping: bool = False,
        enable_textline_orientation: bool = False,
    ) -> None:
        self._model = model
        self._languages = tuple(languages)
        if not self._languages:
            raise ValueError("languages must contain at least one language")

        paddleocr = self._load_dependencies()
        paddle = import_module("paddle")
        self._device = _resolve_paddle_device(device, paddle)
        self._engine = paddleocr.PaddleOCR(
            lang=_resolve_paddle_lang(self._languages),
            device=self._device,
            use_doc_orientation_classify=enable_doc_orientation,
            use_doc_unwarping=enable_doc_unwarping,
            use_textline_orientation=enable_textline_orientation,
        )

    @staticmethod
    def _load_dependencies() -> Any:
        """Import and pre-flight the optional PaddleOCR runtime.

        Returns:
            - The imported ``paddleocr`` module.

        Raises:
            - PaddleOcrRuntimeError: if a required optional dependency is
              missing.
        """
        missing = []
        for name in REQUIRED_OCR_RUNTIME_PACKAGES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)
        if missing:
            raise PaddleOcrRuntimeError(
                "PaddleOCR extraction requires the optional 'ocr' dependency "
                f"group; missing packages: {', '.join(missing)}. "
                "Install with pip install -e '.[ocr]'"
            )
        return import_module("paddleocr")

    @property
    def device(self) -> str:
        """The concrete runtime device (``gpu:0`` or ``cpu``)."""
        return self._device

    @property
    def model_name(self) -> str:
        """Stable model identity recorded in manifests."""
        return f"paddleocr:{self._model}"

    @property
    def languages(self) -> tuple[str, ...]:
        """The configured OCR languages."""
        return self._languages

    def detect_text(self, images: Sequence[object]) -> list[list[OcrLine]]:
        """Return one list of OCR lines per input image, preserving order.

        Images are converted to RGB numpy arrays and passed to
        ``PaddleOCR.predict`` in one batch call (PaddleOCR supports batched
        input), so the pipeline can hand over ``batch_size`` frames at once.

        Expected input:
            - images: PIL images (or any object convertible to a numpy array).

        Returns:
            - One list of :class:`OcrLine` per input image.

        Raises:
            - RuntimeError: if extraction fails (pipeline falls back to
              per-image extraction and records individual failures).
        """
        if not images:
            return []
        try:
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover - guarded in deps
            raise PaddleOcrRuntimeError("PaddleOCR extraction requires numpy") from exc
        arrays = [np.asarray(image) for image in images]
        results = list(self._engine.predict(arrays))
        out: list[list[OcrLine]] = []
        for index in range(len(images)):
            result = results[index] if index < len(results) else None
            out.append(self._coerce_result(result))
        return out

    @staticmethod
    def _coerce_result(result: object) -> list[OcrLine]:
        """Extract OCR lines from a PaddleOCR v3 result object/dict.

        PaddleOCR v3 ``predict`` returns per-image result objects that behave
        like dicts exposing ``rec_texts`` (list of str), ``rec_scores``
        (array of floats), ``rec_boxes`` (array of ``[x1, y1, x2, y2]``), and
        ``rec_polys`` (array of 4-point polygons). This coercion reads either
        interface defensively.

        Expected input:
            - result: a PaddleOCR v3 result object (or ``None``).

        Returns:
            - List of :class:`OcrLine` in reading order; empty when the
              result is ``None`` or contains no text.
        """
        if result is None:
            return []
        texts = _result_get(result, "rec_texts")
        if not texts:
            return []
        scores = _result_get(result, "rec_scores")
        boxes = _result_get(result, "rec_boxes")
        polys = _result_get(result, "rec_polys")
        lines: list[OcrLine] = []
        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            confidence = float(scores[index]) if scores is not None and index < len(scores) else 0.0
            lines.append(
                OcrLine(
                    text=text,
                    confidence=min(max(confidence, 0.0), 1.0),
                    bbox=_bbox_for(index, boxes, polys),
                )
            )
        return lines


def _result_get(result: object, key: str) -> object:
    """Read ``result[key]`` (dict) or ``result.key`` (object) defensively."""
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _bbox_for(index: int, boxes: object, polys: object) -> tuple[float, float, float, float]:
    """Compute an axis-aligned ``(x1, y1, x2, y2)`` box for one text line.

    Prefers ``rec_boxes`` (already axis-aligned); falls back to the min/max
    of a ``rec_polys`` polygon.

    Expected input:
        - index: line index.
        - boxes: ``rec_boxes`` (array of ``[x1, y1, x2, y2]``) or ``None``.
        - polys: ``rec_polys`` (array of polygons) or ``None``.

    Returns:
        - ``(x1, y1, x2, y2)`` floats.
    """
    if boxes is not None:
        try:
            row = boxes[index]
            return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        except (IndexError, TypeError, ValueError):
            pass
    if polys is not None:
        try:
            polygon = polys[index]
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            if xs and ys:
                return (min(xs), min(ys), max(xs), max(ys))
        except (IndexError, TypeError, ValueError):
            pass
    return (0.0, 0.0, 0.0, 0.0)


def create_ocr_engine(
    backend: str = DEFAULT_BACKEND,
    *,
    model: str | None = None,
    device: str = "auto",
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    enable_doc_orientation: bool = False,
    enable_doc_unwarping: bool = False,
    enable_textline_orientation: bool = False,
) -> Any:
    """Create an OCR engine for a named backend.

    Expected input:
        - backend: one of ``OCR_BACKENDS`` (``paddleocr``).
        - model: optional model family; defaults to the backend's default.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - languages: OCR languages, e.g. ``("vi", "en")``.
        - enable_doc_orientation / enable_doc_unwarping /
          enable_textline_orientation: optional PaddleOCR preprocessing
          modules (off by default for throughput).

    Returns:
        - A configured OCR engine instance for the backend.

    Raises:
        - ValueError: if ``backend`` is not one of ``OCR_BACKENDS``.
        - PaddleOcrRuntimeError: if the backend's optional runtime is
          unavailable.
    """
    if backend == "paddleocr":
        return PaddleOcrEngine(
            model=model or DEFAULT_OCR_MODEL,
            device=device,
            languages=languages,
            enable_doc_orientation=enable_doc_orientation,
            enable_doc_unwarping=enable_doc_unwarping,
            enable_textline_orientation=enable_textline_orientation,
        )
    raise ValueError(f"unsupported OCR backend {backend!r}; expected {OCR_BACKENDS}")


# ------------------------------------------------------------------ preprocessing


def preprocess_image(
    image: object,
    *,
    upscale_factor: float = 1.0,
    contrast: float = 1.0,
    sharpen: bool = False,
) -> object:
    """Apply optional, configurable preprocessing to a keyframe image.

    RGB conversion is always applied. The remaining transforms are disabled
    by default (do not overprocess): ``upscale_factor`` upscales small text
    (e.g. ``2.0`` to double the resolution), ``contrast`` enhances contrast
    via ``ImageEnhance.Contrast`` (``1.0`` = unchanged, ``> 1`` increases),
    and ``sharpen`` applies a mild sharpening filter.

    Expected input:
        - image: a PIL image (or object with ``convert`` / ``size`` /
          ``resize`` / ``filter``).
        - upscale_factor: positive scale factor (``1.0`` = no upscale).
        - contrast: contrast enhancement factor (``1.0`` = unchanged).
        - sharpen: apply a sharpening filter.

    Returns:
        - The preprocessed image (RGB).

    Raises:
        - ValueError: if ``upscale_factor`` or ``contrast`` are not positive.
    """
    if not isinstance(upscale_factor, (int, float)) or isinstance(upscale_factor, bool) or upscale_factor <= 0:
        raise ValueError(f"upscale_factor must be a positive number, got {upscale_factor!r}")
    if not isinstance(contrast, (int, float)) or isinstance(contrast, bool) or contrast <= 0:
        raise ValueError(f"contrast must be a positive number, got {contrast!r}")
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ModuleNotFoundError as exc:  # pragma: no cover - guarded in deps
        raise ImportError("image preprocessing requires Pillow") from exc

    img = image.convert("RGB") if hasattr(image, "convert") else image
    if upscale_factor != 1.0:
        width, height = img.size
        img = img.resize(
            (max(1, int(width * upscale_factor)), max(1, int(height * upscale_factor))),
            Image.LANCZOS,
        )
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    return img


# ------------------------------------------------------------------ helpers


def _count_ocr_lines(path: Path) -> int:
    """Return the number of complete OCR records in an NDJSON file.

    A trailing partial line (crash between line writes) is ignored, so the
    resume point always reflects fully written records.

    Expected input:
        - path: a file path to ``ocr.ndjson``.

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


def _trim_ocr_overflow(path: Path, keep: int) -> None:
    """Rewrite ``path`` keeping only the first ``keep`` complete records.

    A crash between writing a record and the checkpoint write can leave more
    lines than the checkpoint records; trailing lines (including a partial
    last line) are dropped.

    Expected input:
        - path: a file path to ``ocr.ndjson``.
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


def _ocr_batch_safe(
    engine: Any,
    images: Sequence[object],
) -> list[tuple[list[OcrLine], str | None]]:
    """OCR a batch, falling back to per-image calls on batch failure.

    Expected input:
        - engine: the OCR engine to invoke.
        - images: preprocessed images to extract text from.

    Returns:
        - One ``(lines, error)`` tuple per image. On success ``error`` is
          ``None``; on failure the lines are empty and the error message is
          populated.
    """
    if not images:
        return []
    try:
        results = engine.detect_text(list(images))
        if len(results) != len(images):
            raise ValueError(f"OCR engine returned {len(results)} results for {len(images)} images")
        return [(list(lines), None) for lines in results]
    except Exception as batch_exc:  # noqa: BLE001 - recover per image below
        logger.warning("batch OCR failed (%s); retrying per image", batch_exc)
        results: list[tuple[list[OcrLine], str | None]] = []
        for image in images:
            try:
                lines = engine.detect_text([image])
                if len(lines) != 1:
                    raise ValueError(f"OCR engine returned {len(lines)} results for 1 image")
                results.append((list(lines[0]), None))
            except Exception as exc:  # noqa: BLE001 - recorded as an error record
                logger.warning("per-image OCR failed: %s", exc)
                results.append(([], str(exc)))
        return results


# ------------------------------------------------------------------ pipeline


def generate_ocr(
    keyframes_dir: Path | str,
    output_dir: Path | str,
    *,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    batch_size: int = 16,
    device: str = "auto",
    resume: bool = True,
    max_records: int | None = None,
    stop_after_records: int | None = None,
    progress: ProgressReporter | str = "auto",
    ocr_engine: Any | None = None,
    image_loader: Callable[[Path], object] | None = None,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    upscale_factor: float = 1.0,
    contrast: float = 1.0,
    sharpen: bool = False,
) -> OcrManifest:
    """Build a resumable OCR artifact from canonical keyframe records.

    Expected input:
        - keyframes_dir: root directory containing the canonical
          ``frames.ndjson`` and the keyframe ``images/`` collection
          (materialized ``adaptive_sbd_v1`` layout).
        - output_dir: target OCR artifact directory (created if absent).
        - backend: OCR backend name, one of ``OCR_BACKENDS``
          (``paddleocr``); ignored when ``ocr_engine`` is given.
        - model: model family used to build the backend engine (ignored when
          ``ocr_engine`` is given).
        - batch_size: records written (and engine invocations) per batch.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda`` (ignored when
          ``ocr_engine`` is given).
        - resume: if True, continue a partial artifact when present; an
          already-complete artifact is validated and returned without recompute.
        - max_records: optional row limit selecting the first ``max_records``
          records (for small-scale validation / sharding).
        - stop_after_records: optional testing hook to stop after writing N
          records (no final manifest is written; the artifact stays resumable).
        - progress: progress reporting mode ``auto``/``on``/``off`` or a shared
          :class:`ProgressReporter`.
        - ocr_engine: optional test engine; production calls create the engine
          from ``backend``.
        - image_loader: callable ``path -> backend-native image``; defaults to
          a Pillow-based RGB loader.
        - languages: OCR languages, e.g. ``("vi", "en")``.
        - upscale_factor / contrast / sharpen: preprocessing controls applied
          to every loaded image before extraction.

    Returns:
        - Final :class:`OcrManifest`.

    Raises:
        - ValueError: on invalid inputs or an inconsistent/conflicting partial
          artifact (backend, model, languages, source, or total_records
          changed). A changed device is allowed (warning only).
        - OcrGenerationError: when ``stop_after_records`` interrupts the build.
        - FileNotFoundError: if the source ``frames.ndjson`` is missing.
    """
    keyframes_dir = Path(keyframes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a bool")
    if backend not in OCR_BACKENDS:
        raise ValueError(f"unsupported OCR backend {backend!r}; expected {OCR_BACKENDS}")
    languages = tuple(languages)
    if not languages or any(not isinstance(lang, str) or lang == "" for lang in languages):
        raise ValueError("languages must be a non-empty sequence of non-empty strings")

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
    ocr_path = output_dir / OCR_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    if ocr_engine is None:
        ocr_engine = create_ocr_engine(
            backend,
            model=model,
            device=device,
            languages=languages,
        )
    model_name = ocr_engine.model_name
    runtime_device = ocr_engine.device

    with stages.time("load_frames"):
        frames = list(iter_frames(frames_path))
        selected = select_frames(frames, max_records)
        total_records = len(selected)

    # ------------------------------------------------------------ complete fast path
    if resume and manifest_path.is_file():
        existing = OcrManifest.from_dict(_read_json(manifest_path))
        if existing.backend != backend:
            raise ValueError(f"existing OCR backend {existing.backend!r} does not match {backend!r}")
        if existing.model != model_name:
            raise ValueError(
                f"existing OCR model {existing.model!r} does not match {model_name!r}"
            )
        if existing.languages != languages:
            raise ValueError("existing OCR languages do not match the current languages")
        if existing.source_frames != source:
            raise ValueError(
                f"existing OCR source {existing.source_frames!r} != current {source!r}"
            )
        if existing.num_records != total_records:
            raise ValueError(
                f"existing OCR num_records {existing.num_records} != selected rows {total_records}"
            )
        return existing

    # ------------------------------------------------------------ decide the resume point
    start_record = 0
    completed_videos: set[str] = set()
    if resume and checkpoint_path.is_file():
        ckpt = OcrCheckpoint.from_dict(_read_json(checkpoint_path))
        if ckpt.backend != backend:
            raise ValueError(f"checkpoint backend {ckpt.backend!r} does not match {backend!r}")
        if ckpt.model != model_name:
            raise ValueError(f"checkpoint model {ckpt.model!r} does not match {model_name!r}")
        if ckpt.languages != languages:
            raise ValueError("checkpoint languages do not match the current languages")
        if ckpt.source_frames != source:
            raise ValueError(f"checkpoint source_frames {ckpt.source_frames!r} != current {source!r}")
        if ckpt.total_records != total_records:
            raise ValueError(
                f"checkpoint total_records {ckpt.total_records} != current {total_records}; "
                "source or max_records changed"
            )
        if ckpt.device != runtime_device:
            logger.warning(
                "resuming OCR build on device %r (checkpoint was device %r); continuing",
                runtime_device,
                ckpt.device,
            )
        start_record = ckpt.completed_record_count
        completed_videos = set(ckpt.completed_video_ids)

    if start_record > 0:
        if not ocr_path.is_file():
            raise ValueError("checkpoint exists but ocr.ndjson is missing; cannot resume")
        _trim_ocr_overflow(ocr_path, start_record)
        if _count_ocr_lines(ocr_path) != start_record:
            raise ValueError(
                f"ocr.ndjson complete records do not match checkpoint count "
                f"({start_record}); cannot resume"
            )

    logger.info(
        "OCR build: %d records (backend=%s, model=%s, languages=%s, device=%s, resume_from=%d)",
        total_records,
        backend,
        model_name,
        ",".join(languages),
        runtime_device,
        start_record,
    )
    reporter.set_total(total_records, completed=start_record)

    load_image = image_loader if image_loader is not None else default_image_loader()
    last_index = _last_index_by_video(selected)

    def write_checkpoint(completed: int) -> None:
        _write_json_atomic(
            checkpoint_path,
            OcrCheckpoint(
                completed_record_count=completed,
                completed_video_ids=_completed_video_ids(last_index, completed),
                backend=backend,
                model=model_name,
                languages=languages,
                device=runtime_device,
                batch_size=batch_size,
                source_frames=source,
                total_records=total_records,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).to_dict(),
        )

    # ------------------------------------------------------------ stream keyframes through the engine
    next_record = start_record
    pending: list[tuple[KeyframeTiming, object | None, str | None]] = []

    def write_pending_batch() -> None:
        nonlocal next_record, pending
        if not pending:
            return
        with stages.time("ocr"):
            ok_frames = [frame for frame, image, _ in pending if image is not None]
            ok_images = [
                preprocess_image(
                    image, upscale_factor=upscale_factor, contrast=contrast, sharpen=sharpen
                )
                for _, image, _ in pending
                if image is not None
            ]
            results = _ocr_batch_safe(ocr_engine, ok_images)
            result_index = 0
            records: list[KeyframeOcr] = []
            for frame, image, load_error in pending:
                if load_error is not None:
                    records.append(
                        KeyframeOcr(
                            video_id=frame.keyframe.video_id,
                            keyframe_ordinal=frame.keyframe.keyframe_ordinal,
                            frame_index=frame.frame_index,
                            ocr_text="",
                            ocr_text_normalized="",
                            ocr_lines=(),
                            status="error",
                            error=load_error,
                        )
                    )
                    continue
                lines, ocr_error = results[result_index]
                result_index += 1
                ocr_text, ocr_text_normalized = _join_ocr_lines(lines)
                records.append(
                    KeyframeOcr(
                        video_id=frame.keyframe.video_id,
                        keyframe_ordinal=frame.keyframe.keyframe_ordinal,
                        frame_index=frame.frame_index,
                        ocr_text=ocr_text,
                        ocr_text_normalized=ocr_text_normalized,
                        ocr_lines=tuple(lines),
                        status="ok" if ocr_error is None else "error",
                        error=ocr_error,
                    )
                )
        with stages.time("write_outputs"):
            for idx, record in enumerate(records):
                if stop_after_records is not None and next_record >= stop_after_records:
                    write_checkpoint(next_record)
                    raise OcrGenerationError(
                        f"interrupted OCR build at record {next_record}; "
                        f"checkpoint written to {checkpoint_path}"
                    )
                write_ocr_records(ocr_path, [record])
                next_record += 1
            write_checkpoint(next_record)
        pending = []
        reporter.update(next_record, note=f"video={records[0].video_id.value if records else ''}")

    try:
        for frame in selected[start_record:]:
            if stop_after_records is not None and next_record >= stop_after_records:
                write_checkpoint(next_record)
                raise OcrGenerationError(
                    f"interrupted OCR build at record {next_record}; "
                    f"checkpoint written to {checkpoint_path}"
                )
            with stages.time("load_images"):
                video_id = frame.keyframe.video_id.value
                load_error: str | None = None
                image: object | None = None
                try:
                    path = find_keyframe_image(keyframes_dir, video_id, frame.keyframe.keyframe_ordinal)
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

    if next_record != total_records:
        raise ValueError(f"written records ({next_record}) != expected ({total_records})")

    # ------------------------------------------------------------ validate then write the final manifest
    with stages.time("write_outputs"):
        line_count = _count_ocr_lines(ocr_path)
        if line_count != total_records:
            raise ValueError(f"ocr.ndjson records ({line_count}) != expected ({total_records})")
        records = iter_ocr_records(ocr_path)
        num_failed = sum(1 for record in records if record.status == "error")
        manifest = OcrManifest(
            schema_version=OCR_ARTIFACT_SCHEMA_VERSION,
            artifact_id=output_dir.name,
            backend=backend,
            model=model_name,
            languages=languages,
            device=runtime_device,
            batch_size=batch_size,
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


def validate_ocr_artifact(manifest: OcrManifest, artifact_dir: Path) -> list[str]:
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
    ocr_path = artifact_dir / OCR_FILENAME

    if not ocr_path.is_file():
        problems.append(f"missing {OCR_FILENAME}")
        return problems

    try:
        records = iter_ocr_records(ocr_path)
    except Exception as exc:  # noqa: BLE001 - surface as a problem
        problems.append(f"{OCR_FILENAME} unreadable: {exc}")
        return problems

    if len(records) != manifest.num_records:
        problems.append(
            f"{OCR_FILENAME} records ({len(records)}) != manifest.num_records ({manifest.num_records})"
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


def read_ocr_artifact(artifact_dir: Path | str) -> tuple[OcrManifest, list[KeyframeOcr]]:
    """Read a completed OCR artifact.

    Expected input:
        - artifact_dir: the OCR artifact directory.

    Returns:
        - ``(manifest, records)`` with records in row order.

    Raises:
        - FileNotFoundError: if ``manifest.json`` is missing.
        - ValueError: if the manifest or records are invalid.
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing OCR manifest: {manifest_path}")
    manifest = OcrManifest.from_dict(_read_json(manifest_path))
    problems = validate_ocr_artifact(manifest, artifact_dir)
    if problems:
        raise ValueError("invalid OCR artifact: " + "; ".join(problems))
    return manifest, iter_ocr_records(artifact_dir / OCR_FILENAME)
