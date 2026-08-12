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
class Config:
    """Effective configuration for a run.

    Fields:
        - data_raw_dir: path to the raw dataset root (resolved to an
          absolute path when possible).
        - data_indexes_dir: root directory for derived artifacts such as
          indexes (resolved to an absolute path when possible).
    """

    data_raw_dir: Path
    data_indexes_dir: Path


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
    return Config(
        data_raw_dir=_resolve(config_path, payload["data"]["raw_dir"]),
        data_indexes_dir=_resolve(config_path, payload["data"]["indexes_dir"]),
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