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


def test_jina_experiment_config_loads_active_backend() -> None:
    """The Jina CLIP v2 experiment config selects the Jina model section."""
    config = load_config(EXPERIMENTS_DIR / "jina-clip-v2.yaml")

    assert config.jina is not None
    assert config.jina.backend == "jina-clip"
    assert config.jina.model_name == "jina-clip-v2"
    assert config.jina.pretrained == "jinaai/jina-clip-v2"
    assert config.jina.device == "auto"
    assert config.jina.precision == "fp32"
    assert config.jina.normalization == "l2"
    assert config.jina.preprocessing_id == "jina-clip-v2-preprocess-v1"
    assert config.jina.tokenizer_id == "jina-clip-v2-tokenizer-v1"
    # The Jina section is the active model for prediction runs.
    assert config.model is config.jina
    assert config.model_config == {
        "backend": "jina-clip",
        "model_name": "jina-clip-v2",
        "pretrained": "jinaai/jina-clip-v2",
        "device": "auto",
        "precision": "fp32",
        "normalization": "l2",
        "preprocessing_id": "jina-clip-v2-preprocess-v1",
        "tokenizer_id": "jina-clip-v2-tokenizer-v1",
    }


def test_openclip_only_configs_have_no_jina_section() -> None:
    """Configs without a jina section keep OpenCLIP as the active model."""
    for filename in ("vit-b-32-openai.yaml", "vit-l-14-laion2b.yaml"):
        config = load_config(EXPERIMENTS_DIR / filename)
        assert config.jina is None
        assert config.model is config.openclip
        assert config.model_config["backend"] == "openclip"


def test_jina_config_embedding_space_matches_index_manifest() -> None:
    """The Jina config reconstructs the exact space recorded in the index.

    This is the compatibility contract the QA runner enforces before search:
    the query encoder built from the config must report the same space as the
    ``jina-v2-full`` index artifact.
    """
    index_manifest_path = (
        Path(__file__).resolve().parents[1]
        / "data/interim/indexes/jina-v2-full/manifest.json"
    )
    if not index_manifest_path.is_file():
        pytest.skip("jina-v2-full index artifact not available")

    config = load_config(EXPERIMENTS_DIR / "jina-clip-v2.yaml")
    import json

    index_space = json.loads(index_manifest_path.read_text(encoding="utf-8"))["embedding_space"]
    assert index_space["backend"] == "jina-clip"
    assert index_space["model_id"] == "jina-clip-v2"
    assert index_space["checkpoint_id"] == "jinaai/jina-clip-v2"
    assert index_space["embedding_dim"] == 1024
    assert index_space["output_dtype"] == "float32"
    assert index_space["normalization"] == "l2"

    # The space_id construction mirrors model_backends/jina.py and the
    # embedding artifact; the two config fields that build it are the
    # processing/tokenizer revision labels.
    jina = config.jina
    assert jina is not None
    expected_space_id = (
        f"jina-clip:{jina.model_name}:{jina.pretrained}:{jina.precision}:"
        f"{jina.normalization}:{jina.preprocessing_id}:{jina.tokenizer_id}"
    )
    assert expected_space_id == index_space["space_id"]
