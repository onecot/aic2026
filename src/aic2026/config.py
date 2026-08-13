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
class Config:
    """Effective configuration for a run.

    Fields:
        - data_raw_dir: path to the raw dataset root (resolved to an
          absolute path when possible).
        - data_indexes_dir: root directory for derived artifacts such as
          indexes (resolved to an absolute path when possible).
        - openclip: validated configuration for the OpenCLIP backend.
    """

    data_raw_dir: Path
    data_indexes_dir: Path
    openclip: OpenClipConfig


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


def load_config(path: Path | None = None) -> Config:
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
    config_path = path or default_config_path()
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    openclip_payload = payload["models"]["openclip"]
    return Config(
        data_raw_dir=_resolve(config_path, payload["data"]["raw_dir"]),
        data_indexes_dir=_resolve(config_path, payload["data"]["indexes_dir"]),
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
    )


def _resolve(config_path: Path, value: str) -> Path:
    """Resolve a possibly-relative path against the repo root.

    Expected input:
        - config_path: path to the config file (used to locate the root).
        - value: the raw path value from the YAML.

    Returns:
        - Absolute path if ``value`` is absolute, otherwise resolved
          relative to the repository root.
    """
    resolved = Path(value)
    if not resolved.is_absolute():
        resolved = config_path.parent.parent / resolved
    return resolved