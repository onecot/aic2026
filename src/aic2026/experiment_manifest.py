"""Lightweight experiment manifest for tracking embedding/retrieval runs.

This module is metadata/documentation infrastructure only: it records the
identity and provenance of an experiment (model, embedding space, split/merge
inputs, hardware, evaluation artifacts, and free-form parameters) so distinct
model runs (OpenCLIP, Jina, SigLIP, MobileCLIP, ...) cannot be confused.

It does not modify or depend on any retrieval, scorer, evaluation, embedding,
or artifact-schema logic. A manifest is a self-describing JSON document::

    {
      "experiment_id": "jina-v1",
      "experiment_type": "embedding",
      "created_at": "2026-08-14T...",
      "description": "Jina CLIP embedding experiment",
      "model_name": "jina-clip-v2",
      "frames_source": "btc-keyframes-v1",
      "parameters": {}
    }

Serialization is deterministic: keys follow the dataclass field order, nested
dicts (``parameters``, ``metrics``, ``embedding_space``) are written with
sorted keys, and a trailing newline terminates the file. Writes are atomic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.embeddings import EmbeddingSpace

#: Schema version of the experiment manifest.
EXPERIMENT_MANIFEST_SCHEMA_VERSION = "1.0"

#: Known experiment types. Custom types are allowed: validation only requires
#: a non-empty string.
KNOWN_EXPERIMENT_TYPES = ("embedding", "retrieval", "reranking", "fusion")


class ExperimentManifestError(ValueError):
    """Raised when a manifest is invalid or cannot be parsed."""


def _require(condition: bool, message: str) -> None:
    """Raise ExperimentManifestError unless ``condition`` holds."""
    if not condition:
        raise ExperimentManifestError(message)


def _canonicalize(value: Any) -> Any:
    """Recursively sort dict keys so identical content serializes identically."""
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically with deterministic formatting."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def parse_key_value(raw: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI string into a typed ``(key, value)`` pair.

    Values are coerced to bool (``true``/``false``), int, or float when
    possible; otherwise they stay strings.

    Expected input:
        - raw: a ``key=value`` string with a single ``=`` separator.

    Returns:
        - ``(key, value)`` with the parsed value type.

    Raises:
        - ValueError: if the string has no ``=`` or an empty key.
    """
    if "=" not in raw:
        raise ValueError(f"expected key=value, got {raw!r}")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("parameter key must be non-empty")
    value: Any
    lowered = raw_value.strip()
    if lowered in ("true", "false"):
        value = lowered == "true"
    else:
        try:
            value = int(lowered)
        except ValueError:
            try:
                value = float(lowered)
            except ValueError:
                value = lowered
    return key, value


@dataclass(frozen=True)
class ExperimentManifest:
    """Versioned metadata describing one embedding/retrieval experiment.

    Required fields:
        - experiment_id: stable, unique identifier (e.g. ``jina-v1``).
        - experiment_type: ``embedding``, ``retrieval``, ``reranking``,
          ``fusion``, or any non-empty custom type.
        - created_at: ISO-8601 timestamp of manifest creation.
        - description: human-readable experiment description.

    Embedding-related optional fields:
        - model_name: model identity (e.g. ``openclip:ViT-L-14:...`` or
          ``jina-clip-v2``).
        - embedding_space: full :class:`EmbeddingSpace` identity of the
          vectors.
        - embedding_dim: vector length (positive when provided).
        - frames_source: provenance of the keyframe set.
        - split_manifest: path to the split manifest used for a distributed
          run.
        - workers: number of shard workers (positive when provided).
        - hardware: free-form hardware description (e.g. ``mps``, ``cuda``).

    Retrieval-related optional fields:
        - index_path: path to the index artifact evaluated.
        - prediction_artifact: path to the prediction artifact.
        - evaluation_set: path to the evaluation set.
        - metrics: metric name -> value mapping.

    Extra metadata:
        - parameters: free-form experiment parameters.
    """

    experiment_id: str
    experiment_type: str
    created_at: str
    description: str
    model_name: str | None = None
    embedding_space: EmbeddingSpace | None = None
    embedding_dim: int | None = None
    frames_source: str | None = None
    split_manifest: str | None = None
    workers: int | None = None
    hardware: str | None = None
    index_path: str | None = None
    prediction_artifact: str | None = None
    evaluation_set: str | None = None
    metrics: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(isinstance(self.experiment_id, str) and self.experiment_id != "", "experiment_id must be a non-empty string")
        _require(isinstance(self.experiment_type, str) and self.experiment_type != "", "experiment_type must be a non-empty string")
        _require(isinstance(self.created_at, str) and self.created_at != "", "created_at must be a non-empty string")
        _require(isinstance(self.description, str) and self.description != "", "description must be a non-empty string")
        _require(
            self.embedding_dim is None
            or (isinstance(self.embedding_dim, int) and not isinstance(self.embedding_dim, bool) and self.embedding_dim >= 1),
            "embedding_dim must be a positive integer when provided",
        )
        _require(
            self.workers is None
            or (isinstance(self.workers, int) and not isinstance(self.workers, bool) and self.workers >= 1),
            "workers must be a positive integer when provided",
        )
        _require(self.embedding_space is None or isinstance(self.embedding_space, EmbeddingSpace), "embedding_space must be an EmbeddingSpace")
        _require(self.metrics is None or isinstance(self.metrics, dict), "metrics must be a dict when provided")
        _require(isinstance(self.parameters, dict), "parameters must be a dict")
        for name, value in (
            ("model_name", self.model_name),
            ("frames_source", self.frames_source),
            ("split_manifest", self.split_manifest),
            ("hardware", self.hardware),
            ("index_path", self.index_path),
            ("prediction_artifact", self.prediction_artifact),
            ("evaluation_set", self.evaluation_set),
        ):
            _require(value is None or (isinstance(value, str) and value != ""), f"{name} must be a non-empty string when provided")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with deterministic key order."""
        payload: dict[str, Any] = {
            "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "created_at": self.created_at,
            "description": self.description,
            "model_name": self.model_name,
            "embedding_space": self.embedding_space.to_dict() if self.embedding_space else None,
            "embedding_dim": self.embedding_dim,
            "frames_source": self.frames_source,
            "split_manifest": self.split_manifest,
            "workers": self.workers,
            "hardware": self.hardware,
            "index_path": self.index_path,
            "prediction_artifact": self.prediction_artifact,
            "evaluation_set": self.evaluation_set,
            "metrics": self.metrics,
            "parameters": self.parameters,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentManifest":
        """Deserialize from a :meth:`to_dict` dict.

        Raises:
            - KeyError: if a required key is missing.
            - ExperimentManifestError: if a value is invalid.
        """
        embedding_space = data.get("embedding_space")
        return cls(
            experiment_id=data["experiment_id"],
            experiment_type=data["experiment_type"],
            created_at=data["created_at"],
            description=data["description"],
            model_name=data.get("model_name"),
            embedding_space=EmbeddingSpace.from_dict(embedding_space) if embedding_space else None,
            embedding_dim=data.get("embedding_dim"),
            frames_source=data.get("frames_source"),
            split_manifest=data.get("split_manifest"),
            workers=data.get("workers"),
            hardware=data.get("hardware"),
            index_path=data.get("index_path"),
            prediction_artifact=data.get("prediction_artifact"),
            evaluation_set=data.get("evaluation_set"),
            metrics=data.get("metrics"),
            parameters=data.get("parameters", {}),
        )


def _payload(manifest: ExperimentManifest) -> dict[str, Any]:
    """Return a canonicalized, deterministic payload for serialization."""
    payload = manifest.to_dict()
    payload["embedding_space"] = _canonicalize(payload["embedding_space"])
    payload["metrics"] = _canonicalize(payload["metrics"])
    payload["parameters"] = _canonicalize(payload["parameters"])
    return payload


def save_manifest(path: Path | str, manifest: ExperimentManifest) -> None:
    """Persist a manifest to JSON (atomic, deterministic formatting).

    Expected input:
        - path: target JSON file path.
        - manifest: the experiment manifest to write.

    Returns:
        - None.

    Raises:
        - OSError: if the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, _payload(manifest))


def load_manifest(path: Path | str) -> ExperimentManifest:
    """Load and validate a manifest from a JSON file.

    Expected input:
        - path: path to a manifest JSON file written by :func:`save_manifest`.

    Returns:
        - The parsed :class:`ExperimentManifest`.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if the file is not valid JSON.
        - ExperimentManifestError: if the content is invalid.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentManifest.from_dict(json.load(handle))


def new_manifest(
    experiment_id: str,
    experiment_type: str,
    description: str,
    *,
    created_at: str | None = None,
    **kwargs: Any,
) -> ExperimentManifest:
    """Convenience constructor stamping the current UTC timestamp.

    Expected input:
        - experiment_id / experiment_type / description: required fields.
        - created_at: optional explicit ISO-8601 timestamp (defaults to now).
        - kwargs: remaining :class:`ExperimentManifest` fields.

    Returns:
        - A validated :class:`ExperimentManifest`.
    """
    return ExperimentManifest(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        description=description,
        **kwargs,
    )