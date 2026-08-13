"""Tests for the file-driven configuration layer and experiment configs."""

from pathlib import Path

import pytest
import yaml

from aic2026.config import load_config

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"


@pytest.mark.parametrize(
    ("filename", "model_name", "pretrained"),
    [
        ("vit-b-32-openai.yaml", "ViT-B-32", "openai"),
        ("vit-l-14-openai.yaml", "ViT-L-14", "openai"),
        ("vit-l-14-laion2b.yaml", "ViT-L-14", "laion2b_s32b_b82k"),
    ],
)
def test_experiment_configs_load_with_model_and_checkpoint(
    filename: str, model_name: str, pretrained: str
) -> None:
    """The loader accepts each shipped experiment config and its model pair."""
    config = load_config(EXPERIMENTS_DIR / filename)
    assert config.openclip.model_name == model_name
    assert config.openclip.pretrained == pretrained
    assert config.openclip.backend == "openclip"
    # data dirs resolve against the repo root, same as the default config
    assert config.data_raw_dir == Path("data/raw/aic2026_batch1").resolve()
    assert config.data_indexes_dir == Path("data/interim/indexes").resolve()


def test_loader_accepts_changed_model_name_and_pretrained(tmp_path: Path) -> None:
    """A custom config with changed model_name/pretrained flows through load_config."""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": "data/raw/aic2026_batch1",
                    "indexes_dir": "data/interim/indexes",
                },
                "models": {
                    "openclip": {
                        "backend": "openclip",
                        "model_name": "ViT-L-14",
                        "pretrained": "laion2b_s32b_b82k",
                        "device": "cpu",
                        "precision": "fp16",
                        "normalization": "none",
                        "preprocessing_id": "custom-preprocess-v1",
                        "tokenizer_id": "custom-tokenizer-v1",
                    }
                },
            }
        )
    )
    config = load_config(custom)
    assert config.openclip.model_name == "ViT-L-14"
    assert config.openclip.pretrained == "laion2b_s32b_b82k"
    assert config.openclip.device == "cpu"
    assert config.openclip.precision == "fp16"
    assert config.openclip.normalization == "none"


def test_default_config_behavior_unchanged() -> None:
    """The base default config still selects the original model/checkpoint."""
    config = load_config()
    assert config.openclip.backend == "openclip"
    assert config.openclip.model_name == "ViT-B-32"
    assert config.openclip.pretrained == "laion2b_s34b_b79k"
    assert config.openclip.device == "auto"
