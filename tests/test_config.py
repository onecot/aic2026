"""Configuration contract tests (single active model section, Jina default)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aic2026.config import Config, ModelConfig, default_config_path, load_config


def _write_config(path: Path, payload: dict) -> Path:
    """Write one config payload to YAML and return its path."""
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _payload(**model_overrides: object) -> dict:
    """Return a minimal valid config payload with optional model overrides."""
    model = {
        "backend": "jina-clip",
        "model_name": "jina-clip-v2",
        "pretrained": "jinaai/jina-clip-v2",
        "device": "auto",
        "precision": "fp32",
        "normalization": "l2",
        "preprocessing_id": "prep-v1",
        "tokenizer_id": "tok-v1",
    }
    model.update(model_overrides)
    return {
        "data": {"raw_dir": "data/raw/x", "indexes_dir": "data/interim/indexes"},
        "models": model,
    }


def test_default_config_loads_with_jina_production_default() -> None:
    config = load_config()
    assert isinstance(config, Config)
    assert config.model.backend == "jina-clip"
    assert config.model.model_name == "jina-clip-v2"
    assert config.model.pretrained == "jinaai/jina-clip-v2"
    assert default_config_path().name == "default.yaml"


def test_default_config_paths_resolve_against_repo_root() -> None:
    config = load_config()
    assert config.data_raw_dir.is_absolute()
    assert config.data_indexes_dir.is_absolute()
    assert config.data_keyframes_dir.name == "adaptive_sbd_v1"


def test_model_provenance_metadata_is_json_safe(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "c.yaml", _payload()))
    meta = config.model_config
    assert meta["backend"] == "jina-clip"
    assert set(meta) == {
        "backend", "model_name", "pretrained", "device", "precision",
        "normalization", "preprocessing_id", "tokenizer_id",
    }
    import json

    json.dumps(meta)  # must not raise


def test_explicit_override_file_replaces_base_wholesale(tmp_path: Path) -> None:
    payload = _payload(device="cpu", precision="fp16")
    path = _write_config(tmp_path / "override.yaml", payload)
    config = load_config(path)
    assert config.model.device == "cpu"
    assert config.model.precision == "fp16"


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": ""},
        {"device": "invalid"},
        {"precision": "bf16"},
        {"normalization": "unknown"},
        {"model_name": ""},
        {"pretrained": ""},
    ],
)
def test_invalid_model_values_are_rejected(tmp_path: Path, overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**{**_payload()["models"], **overrides})


def test_missing_required_section_raises(tmp_path: Path) -> None:
    bad = _payload()
    del bad["models"]
    path = _write_config(tmp_path / "bad.yaml", bad)
    with pytest.raises(KeyError):
        load_config(path)
