"""Configuration loading and cross-platform path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_PATH_KEYS = {
    "data_root",
    "features_dir",
    "mapping_dir",
    "keyframes_dir",
    "metadata_dir",
    "objects_dir",
    "path",
    "output_path",
}


def resolve_path(path_value: str, config_path: Path) -> Path:
    """Resolve one configured path relative to its YAML file.

    Expected input: ``path_value`` is a relative or absolute path string and
    ``config_path`` is the path to the YAML configuration file.
    Returns: An absolute, normalized ``Path``.
    Raises: TypeError if ``path_value`` is not a string.
    """
    if not isinstance(path_value, str):
        raise TypeError("Configured path values must be strings.")

    path = Path(path_value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Convert recognized path fields in a configuration mapping to ``Path`` objects.

    Expected input: ``config`` is a nested mapping parsed from YAML and
    ``config_path`` identifies the YAML file that supplied it.
    Returns: A copied configuration mapping with known path fields resolved.
    Raises: TypeError if a configured path field is not a string.
    """
    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_paths(value, config_path)
        elif key in _PATH_KEYS:
            resolved[key] = resolve_path(value, config_path)
        else:
            resolved[key] = value
    return resolved


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file and resolve all known path fields.

    Expected input: ``config_path`` points to a UTF-8 YAML file with a mapping
    at its root.
    Returns: The parsed configuration mapping with configured paths as ``Path`` objects.
    Raises: FileNotFoundError if the config file is missing; ValueError if its root is not a mapping.
    """
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return resolve_config_paths(config, config_path)
