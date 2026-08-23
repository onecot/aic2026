"""Configuration loading for the AIC 2026 project.

Configuration is file-driven (YAML) with one precedence model:

    configs/default.yaml
            -> explicit YAML override file (--config)

There are no layered experiment configs: an override file replaces the base
wholesale and must carry the same schema. The active model section is a
single ``models`` block; Jina CLIP v2 is the production default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the single active embedding backend.

    Fields:
        - backend: encoder backend identifier (``jina-clip``).
        - model_name: architecture name (e.g. ``jina-clip-v2``).
        - pretrained: HuggingFace repository id or checkpoint path.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - precision: output precision, ``fp32`` or ``fp16``.
        - normalization: image/text vector convention, ``l2`` or ``none``.
        - preprocessing_id: preprocessing pipeline revision identifier.
        - tokenizer_id: tokenizer revision identifier.
    """

    backend: str
    model_name: str
    pretrained: str
    device: str
    precision: str
    normalization: str
    preprocessing_id: str
    tokenizer_id: str

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "model_name",
            "pretrained",
            "preprocessing_id",
            "tokenizer_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be one of: fp32, fp16")
        if self.normalization not in ("l2", "none"):
            raise ValueError("normalization must be one of: l2, none")

    def to_metadata(self) -> dict[str, Any]:
        """Return the section as a JSON-safe provenance dictionary."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "precision": self.precision,
            "normalization": self.normalization,
            "preprocessing_id": self.preprocessing_id,
            "tokenizer_id": self.tokenizer_id,
        }


@dataclass(frozen=True)
class Config:
    """Effective configuration for a run.

    Fields:
        - data_raw_dir: path to the raw dataset root (resolved to an
          absolute path when possible).
        - data_indexes_dir: root directory for derived artifacts such as
          indexes (resolved to an absolute path when possible).
        - data_keyframes_dir: root directory of the materialized keyframe
          artifact (``frames.ndjson`` + ``images/``); authoritative frame
          metadata/image source for the API and QA evidence resolution.
        - model: the single active embedding backend configuration.
    """

    data_raw_dir: Path
    data_indexes_dir: Path
    data_keyframes_dir: Path
    model: ModelConfig

    @property
    def model_config(self) -> dict[str, Any]:
        """JSON-safe provenance dictionary of the active model."""
        return self.model.to_metadata()


def project_root() -> Path:
    """Return the repository root (the parent of ``configs/``).

    Returns:
        - Absolute path to the repository root.

    Raises:
        - No exceptions.
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the path to ``configs/default.yaml``.

    Returns:
        - Absolute path to the default configuration file.

    Raises:
        - No exceptions.
    """
    return project_root() / "configs" / "default.yaml"


_MODEL_FIELDS = (
    "backend",
    "model_name",
    "pretrained",
    "device",
    "precision",
    "normalization",
    "preprocessing_id",
    "tokenizer_id",
)


def load_config(path: Path | str | None = None) -> Config:
    """Load the effective configuration from one YAML file.

    Expected input:
        - path: explicit YAML path overriding ``configs/default.yaml``
          wholesale (the only override mechanism).

    Returns:
        - A :class:`Config` with paths resolved against the repository
          root, independent of where the YAML file lives.

    Raises:
        - FileNotFoundError: if the configuration file does not exist.
        - yaml.YAMLError: if the YAML cannot be parsed.
        - KeyError / ValueError: if required keys are missing or invalid.
    """
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    models_payload = payload["models"]
    return Config(
        data_raw_dir=_resolve(payload["data"]["raw_dir"]),
        data_indexes_dir=_resolve(payload["data"]["indexes_dir"]),
        data_keyframes_dir=_resolve(
            payload["data"].get("keyframes_dir", "data/keyframes/adaptive_sbd_v1")
        ),
        model=ModelConfig(**{field: models_payload[field] for field in _MODEL_FIELDS}),
    )


def _resolve(value: str) -> Path:
    """Resolve a possibly-relative path against the repo root.

    Expected input:
        - value: the raw path value from the YAML.

    Returns:
        - Absolute path if ``value`` is absolute, otherwise resolved
          relative to the repository root.
    """
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    return resolved
