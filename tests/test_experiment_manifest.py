"""Tests for the experiment manifest system (metadata infrastructure only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.embeddings import EmbeddingSpace
from aic2026.experiment_manifest import (
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    KNOWN_EXPERIMENT_TYPES,
    ExperimentManifest,
    ExperimentManifestError,
    load_manifest,
    new_manifest,
    parse_key_value,
    save_manifest,
)


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "openclip",
        "model_id": "ViT-L-14",
        "checkpoint_id": "laion2b_s32b_b82k",
        "space_id": "openclip-vit-l-14-v1",
        "embedding_dim": 768,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "clip-vit-l-14",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


def _manifest(**overrides: object) -> ExperimentManifest:
    values: dict[str, object] = {
        "experiment_id": "jina-v1",
        "experiment_type": "embedding",
        "created_at": "2026-08-14T12:00:00+00:00",
        "description": "Jina CLIP embedding experiment",
        "model_name": "jina-clip-v2",
        "embedding_space": _space(),
        "embedding_dim": 768,
        "frames_source": "btc-keyframes-v1",
        "split_manifest": "data/splits/split_manifest.json",
        "workers": 5,
        "hardware": "mps",
        "index_path": None,
        "prediction_artifact": None,
        "evaluation_set": None,
        "metrics": None,
        "parameters": {"batch_size": 32, "seed": 7, "fp16": True},
    }
    values.update(overrides)
    return ExperimentManifest(**values)  # type: ignore[arg-type]


def _script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "create_experiment_manifest.py"


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    save_manifest(path, manifest)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert loaded.embedding_space == _space()
    assert loaded.parameters == {"batch_size": 32, "seed": 7, "fp16": True}


def test_save_load_roundtrip_minimal(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = new_manifest(
        experiment_id="minimal",
        experiment_type="retrieval",
        description="minimal retrieval experiment",
        created_at="2026-08-14T12:00:00+00:00",
    )
    save_manifest(path, manifest)
    assert load_manifest(path) == manifest


def test_save_load_roundtrip_retrieval_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest(
        experiment_type="retrieval",
        model_name=None,
        embedding_space=None,
        embedding_dim=None,
        index_path="data/interim/index.npy",
        prediction_artifact="data/interim/predictions/top1000.json",
        evaluation_set="data/eval/eval_set.csv",
        metrics={"ndcg": 0.42, "hit": 0.9},
    )
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded.metrics == {"ndcg": 0.42, "hit": 0.9}
    assert loaded.index_path == "data/interim/index.npy"


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "manifest.json"
    save_manifest(path, _manifest())
    assert path.is_file()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(path)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_experiment_id_required() -> None:
    with pytest.raises(ExperimentManifestError, match="experiment_id"):
        _manifest(experiment_id="")
    with pytest.raises(ExperimentManifestError, match="experiment_id"):
        _manifest(experiment_id=None)  # type: ignore[arg-type]


def test_experiment_type_required() -> None:
    with pytest.raises(ExperimentManifestError, match="experiment_type"):
        _manifest(experiment_type="")


def test_custom_experiment_type_allowed() -> None:
    manifest = _manifest(experiment_type="custom-pipeline")
    assert manifest.experiment_type == "custom-pipeline"


def test_known_experiment_types_present() -> None:
    assert set(KNOWN_EXPERIMENT_TYPES) >= {"embedding", "retrieval", "reranking", "fusion"}


def test_created_at_and_description_required() -> None:
    with pytest.raises(ExperimentManifestError, match="created_at"):
        _manifest(created_at="")
    with pytest.raises(ExperimentManifestError, match="description"):
        _manifest(description="")


def test_embedding_dim_positive_when_provided() -> None:
    with pytest.raises(ExperimentManifestError, match="embedding_dim"):
        _manifest(embedding_dim=0)
    with pytest.raises(ExperimentManifestError, match="embedding_dim"):
        _manifest(embedding_dim=-4)
    assert _manifest(embedding_dim=512).embedding_dim == 512
    assert _manifest(embedding_dim=None).embedding_dim is None


def test_workers_positive_when_provided() -> None:
    with pytest.raises(ExperimentManifestError, match="workers"):
        _manifest(workers=0)
    with pytest.raises(ExperimentManifestError, match="workers"):
        _manifest(workers=-1)
    assert _manifest(workers=3).workers == 3
    assert _manifest(workers=None).workers is None


def test_optional_strings_non_empty_when_provided() -> None:
    with pytest.raises(ExperimentManifestError, match="model_name"):
        _manifest(model_name="")
    with pytest.raises(ExperimentManifestError, match="frames_source"):
        _manifest(frames_source="")


def test_parameters_must_be_dict() -> None:
    with pytest.raises(ExperimentManifestError, match="parameters"):
        _manifest(parameters="nope")  # type: ignore[arg-type]


def test_metrics_must_be_dict_when_provided() -> None:
    with pytest.raises(ExperimentManifestError, match="metrics"):
        _manifest(metrics=[1, 2])  # type: ignore[arg-type]
    assert _manifest(metrics={"hit": 0.9}).metrics == {"hit": 0.9}


# ---------------------------------------------------------------------------
# deterministic JSON output
# ---------------------------------------------------------------------------


def test_deterministic_json_output(tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_manifest(path_a, _manifest())
    save_manifest(path_b, _manifest())
    assert path_a.read_bytes() == path_b.read_bytes()


def test_deterministic_json_regardless_of_dict_order(tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_manifest(path_a, _manifest(parameters={"b": 1, "a": 2, "c": {"y": 1, "x": 2}}))
    save_manifest(path_b, _manifest(parameters={"c": {"x": 2, "y": 1}, "a": 2, "b": 1}))
    assert path_a.read_bytes() == path_b.read_bytes()


def test_save_load_save_identical(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(path, _manifest())
    first = path.read_bytes()
    loaded = load_manifest(path)
    save_manifest(path, loaded)
    assert path.read_bytes() == first


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(path, _manifest())
    assert not path.with_name(path.name + ".tmp").exists()


def test_json_formatting(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(path, _manifest())
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert payload["schema_version"] == EXPERIMENT_MANIFEST_SCHEMA_VERSION
    assert list(payload)[:4] == ["schema_version", "experiment_id", "experiment_type", "created_at"]


# ---------------------------------------------------------------------------
# parameter parsing
# ---------------------------------------------------------------------------


def test_parse_key_value_types() -> None:
    assert parse_key_value("batch_size=32") == ("batch_size", 32)
    assert parse_key_value("lr=0.001") == ("lr", 0.001)
    assert parse_key_value("fp16=true") == ("fp16", True)
    assert parse_key_value("fp16=false") == ("fp16", False)
    assert parse_key_value("name=jina-clip-v2") == ("name", "jina-clip-v2")
    assert parse_key_value("  key = 7 ") == ("key", 7)


def test_parse_key_value_invalid() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_key_value("noequalsign")
    with pytest.raises(ValueError, match="key"):
        parse_key_value("=5")


# ---------------------------------------------------------------------------
# CLI generation
# ---------------------------------------------------------------------------


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(_script()), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--experiment-id" in result.stdout
    assert "--parameter" in result.stdout


def test_cli_missing_args_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, str(_script())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--experiment-id" in result.stderr


def test_cli_creates_manifest(tmp_path: Path) -> None:
    output = tmp_path / "jina-v1" / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--output",
            str(output),
            "--experiment-id",
            "jina-v1",
            "--type",
            "embedding",
            "--model",
            "jina-clip-v2",
            "--frames-source",
            "btc-keyframes-v1",
            "--description",
            "Jina CLIP embedding experiment",
            "--parameter",
            "batch_size=32",
            "--parameter",
            "fp16=true",
            "--parameter",
            "lr=0.001",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Experiment manifest written" in result.stdout
    assert output.is_file()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "jina-v1"
    assert payload["experiment_type"] == "embedding"
    assert payload["model_name"] == "jina-clip-v2"
    assert payload["frames_source"] == "btc-keyframes-v1"
    assert payload["description"] == "Jina CLIP embedding experiment"
    assert payload["parameters"] == {"batch_size": 32, "fp16": True, "lr": 0.001}

    loaded = load_manifest(output)
    assert loaded.experiment_id == "jina-v1"


def test_cli_duplicate_parameter_rejected(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--output",
            str(tmp_path / "manifest.json"),
            "--experiment-id",
            "x",
            "--type",
            "embedding",
            "--description",
            "d",
            "--parameter",
            "a=1",
            "--parameter",
            "a=2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "duplicate parameter key" in result.stderr


def test_cli_invalid_parameter_exits_1(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--output",
            str(tmp_path / "manifest.json"),
            "--experiment-id",
            "x",
            "--type",
            "embedding",
            "--description",
            "d",
            "--parameter",
            "noequals",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_cli_validation_error_exits_1(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--output",
            str(tmp_path / "manifest.json"),
            "--experiment-id",
            "",
            "--type",
            "embedding",
            "--description",
            "d",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "experiment_id" in result.stderr