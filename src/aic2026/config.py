"""Configuration loading for the AIC 2026 project.

Configuration is file-driven (YAML preferred) with a precedence model:

    base/default configuration
            -> experiment configuration
            -> explicit CLI overrides

This module loads the base default configuration from
``configs/default.yaml``. CLI overrides are applied by callers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenClipConfig:
    """Configuration for the optional OpenCLIP embedding backend.

    Fields:
        - backend: must be ``"openclip"``.
        - model_name: OpenCLIP architecture name.
        - pretrained: OpenCLIP pretrained/checkpoint identifier.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - precision: output precision, ``fp32`` or ``fp16``.
        - normalization: common image/text vector convention, ``l2`` or
          ``none``.
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
        if self.backend != "openclip":
            raise ValueError(f"openclip backend must be 'openclip', got {self.backend!r}")
        for name, value in (
            ("model_name", self.model_name),
            ("pretrained", self.pretrained),
            ("preprocessing_id", self.preprocessing_id),
            ("tokenizer_id", self.tokenizer_id),
        ):
            if not isinstance(value, str) or value == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be one of: fp32, fp16")
        if self.normalization not in ("l2", "none"):
            raise ValueError("normalization must be one of: l2, none")


@dataclass(frozen=True)
class JinaClipConfig:
    """Configuration for the optional Jina CLIP embedding backend.

    Fields:
        - backend: must be ``"jina-clip"``.
        - model_name: Jina architecture name (e.g. ``jina-clip-v2``).
        - pretrained: HuggingFace repository id or checkpoint path.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.
        - precision: output precision, ``fp32`` or ``fp16``.
        - normalization: common image/text vector convention, ``l2`` or
          ``none``.
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
        if self.backend != "jina-clip":
            raise ValueError(f"jina-clip backend must be 'jina-clip', got {self.backend!r}")
        for name, value in (
            ("model_name", self.model_name),
            ("pretrained", self.pretrained),
            ("preprocessing_id", self.preprocessing_id),
            ("tokenizer_id", self.tokenizer_id),
        ):
            if not isinstance(value, str) or value == "":
                raise ValueError(f"{name} must be a non-empty string")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError("precision must be one of: fp32, fp16")
        if self.normalization not in ("l2", "none"):
            raise ValueError("normalization must be one of: l2, none")


@dataclass(frozen=True)
class Config:
    """Effective configuration for a run.

    Fields:
        - data_raw_dir: path to the raw dataset root (resolved to an
          absolute path when possible).
        - data_indexes_dir: root directory for derived artifacts such as
          indexes (resolved to an absolute path when possible).
        - data_keyframes_dir: root directory of the materialized keyframe
          artifact (``frames.ndjson`` + ``images/``) used by QA answer
          generation to resolve evidence frames (resolved to an absolute
          path when possible).
        - openclip: validated configuration for the OpenCLIP backend.
        - jina: optional validated configuration for the Jina CLIP backend.
          When present, it is the active model section for prediction runs;
          otherwise the OpenCLIP section is used.
    """

    data_raw_dir: Path
    data_indexes_dir: Path
    data_keyframes_dir: Path
    openclip: OpenClipConfig
    jina: JinaClipConfig | None = None

    @property
    def model(self) -> OpenClipConfig | JinaClipConfig:
        """Return the active backend configuration for prediction runs."""
        return self.jina if self.jina is not None else self.openclip

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the active model section as a JSON-safe metadata dictionary."""
        model = self.model
        return {
            "backend": model.backend,
            "model_name": model.model_name,
            "pretrained": model.pretrained,
            "device": model.device,
            "precision": model.precision,
            "normalization": model.normalization,
            "preprocessing_id": model.preprocessing_id,
            "tokenizer_id": model.tokenizer_id,
        }


def project_root() -> Path:
    """Return the repository root (the parent of ``configs/``).

    Expected input:
        - None.

    Returns:
        - Absolute path to the repository root.

    Raises:
        - No exceptions.
    """
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the path to ``configs/default.yaml``.

    Expected input:
        - None.

    Returns:
        - Absolute path to the default configuration file.

    Raises:
        - No exceptions.
    """
    return project_root() / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> Config:
    """Load the default base configuration.

    Expected input:
        - path: optional explicit path to the YAML file; defaults to
          ``configs/default.yaml``.

    Returns:
        - A :class:`Config` with fields resolved from the YAML file.

    Raises:
        - FileNotFoundError: if the configuration file does not exist.
        - yaml.YAMLError: if the YAML cannot be parsed.
        - KeyError: if a required ``data.*`` key is not present.
    """
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    models_payload = payload["models"]
    openclip_payload = models_payload["openclip"]
    jina_payload = models_payload.get("jina")
    jina: JinaClipConfig | None = None
    if jina_payload is not None:
        jina = JinaClipConfig(
            backend=jina_payload["backend"],
            model_name=jina_payload["model_name"],
            pretrained=jina_payload["pretrained"],
            device=jina_payload["device"],
            precision=jina_payload["precision"],
            normalization=jina_payload["normalization"],
            preprocessing_id=jina_payload["preprocessing_id"],
            tokenizer_id=jina_payload["tokenizer_id"],
        )
    return Config(
        data_raw_dir=_resolve(payload["data"]["raw_dir"]),
        data_indexes_dir=_resolve(payload["data"]["indexes_dir"]),
        data_keyframes_dir=_resolve(payload["data"].get("keyframes_dir", "data/keyframes/adaptive_sbd_v1")),
        openclip=OpenClipConfig(
            backend=openclip_payload["backend"],
            model_name=openclip_payload["model_name"],
            pretrained=openclip_payload["pretrained"],
            device=openclip_payload["device"],
            precision=openclip_payload["precision"],
            normalization=openclip_payload["normalization"],
            preprocessing_id=openclip_payload["preprocessing_id"],
            tokenizer_id=openclip_payload["tokenizer_id"],
        ),
        jina=jina,
    )


def _resolve(value: str) -> Path:
    """Resolve a possibly-relative path against the repo root.

    Expected input:
        - value: the raw path value from the YAML.

    Returns:
        - Absolute path if ``value`` is absolute, otherwise resolved
          relative to the repository root (independent of where the config
          file lives, so experiment configs in ``configs/experiments/``
          resolve exactly like the base config in ``configs/``).
    """
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    return resolved