"""Tests for the model-independent embedding boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.config import Config, ModelConfig
from aic2026.embeddings import (
    Embedding,
    EmbeddingSpace,
    validate_query_compatibility,
)
from aic2026.index_artifact import IndexManifest, write_manifest, write_frames
from aic2026.retrieval import IndexRetriever
from aic2026.canonical import KeyframeKey, KeyframeTiming, VideoId


def _space(**overrides: object) -> EmbeddingSpace:
    values: dict[str, object] = {
        "backend": "test-backend",
        "model_id": "test-model",
        "checkpoint_id": "test-checkpoint",
        "space_id": "test-model-space-v1",
        "embedding_dim": 3,
        "output_dtype": "float32",
        "normalization": "l2",
        "preprocessing_id": "preprocess-v1",
        "model_version": "1.0",
    }
    values.update(overrides)
    return EmbeddingSpace(**values)  # type: ignore[arg-type]


def _embedding(kind: str, space: EmbeddingSpace | None = None) -> Embedding:
    return Embedding(
        kind=kind,  # type: ignore[arg-type]
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        space=space or _space(),
    )


def test_valid_embedding_space_metadata_round_trip() -> None:
    """Model identity metadata is valid and JSON serializable."""
    space = _space()
    restored = EmbeddingSpace.from_dict(json.loads(json.dumps(space.to_dict())))
    assert restored == space


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": ""},
        {"embedding_dim": 0},
        {"output_dtype": "int32"},
        {"normalization": "unit"},
        {"preprocessing_id": ""},
    ],
)
def test_embedding_space_rejects_invalid_metadata(overrides: dict[str, object]) -> None:
    """Incomplete or invalid model metadata fails at the boundary."""
    with pytest.raises(ValueError):
        _space(**overrides)


def test_image_and_text_embeddings_preserve_vector_contract() -> None:
    """Both supported embedding kinds retain dimension, dtype, and shape."""
    image = _embedding("image")
    text = _embedding("text")
    assert image.kind == "image"
    assert text.kind == "text"
    assert image.embedding_dim == 3
    assert text.dtype == np.dtype("float32")
    assert image.shape == (3,)


def test_embedding_serialization_round_trip() -> None:
    """Embedding serialization preserves vector bytes and model metadata."""
    embedding = _embedding("text")
    restored = Embedding.from_dict(json.loads(json.dumps(embedding.to_dict())))
    assert restored.kind == "text"
    assert restored.space == embedding.space
    np.testing.assert_array_equal(restored.vector, embedding.vector)


def test_embedding_rejects_vector_that_disagrees_with_space() -> None:
    """Vector dimensionality and dtype cannot disagree with model metadata."""
    with pytest.raises(ValueError, match="dim"):
        Embedding("text", np.zeros(2, dtype=np.float32), _space())
    with pytest.raises(ValueError, match="dtype"):
        Embedding("text", np.zeros(3, dtype=np.float16), _space())


def test_compatible_query_space_is_accepted() -> None:
    """Text queries are accepted when their full space identity matches."""
    query = _embedding("text")
    validate_query_compatibility(query, _space())


def test_incompatible_query_space_is_rejected() -> None:
    """Matching dimensions do not permit cross-model comparison."""
    query = _embedding("text")
    other = _space(model_id="other-model", space_id="other-space")
    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        validate_query_compatibility(query, other)


def test_model_version_is_not_comparison_relevant() -> None:
    """A differing build-env model runtime version does not split a space."""
    query = _embedding("text")
    index = _space(model_version="4.46.3")
    assert query.space.is_compatible_with(index)
    validate_query_compatibility(query, index)


def test_is_compatible_with_still_requires_identity_match() -> None:
    """Compatibility excludes only model_version, not the space identity."""
    base = _space()
    assert base.is_compatible_with(_space(model_version="9.9.9"))
    assert not base.is_compatible_with(_space(space_id="other-space"))
    assert not base.is_compatible_with(_space(embedding_dim=5))
    assert not base.is_compatible_with(_space(normalization="none"))


def test_retrieval_requires_compatible_typed_query_for_identified_index(tmp_path: Path) -> None:
    """Retrieval validates space metadata before running cosine similarity."""
    space = _space()
    artifact = tmp_path / "identified-index"
    artifact.mkdir()
    manifest = IndexManifest(
        schema_version="1.1",
        artifact_id="identified-index",
        score_metric="cosine",
        embedding_dim=3,
        embedding_dtype="float32",
        num_rows=1,
        num_videos=1,
        created_at="2026-08-13T00:00:00Z",
        source_dataset_id="test",
        canonical_schema_version="1.0",
        embedding_space=space,
    )
    write_manifest(artifact / "manifest.json", manifest)
    write_frames(
        artifact / "frames.ndjson",
        [
            KeyframeTiming(
                keyframe=KeyframeKey(VideoId("L1_V001"), 1),
                pts_time=0.0,
                fps=25.0,
                frame_index=0,
            )
        ],
    )
    np.save(artifact / "embeddings.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    np.save(artifact / "norms.npy", np.array([1.0], dtype=np.float32))

    retriever = IndexRetriever(artifact)
    assert retriever.search(_embedding("text"), top_k=1)[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="incompatible embedding spaces"):
        retriever.search(_embedding("text", _space(space_id="other-space")), top_k=1)
    with pytest.raises(ValueError, match="require an Embedding query"):
        retriever.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)


# ============================================================================
# Section: image-embedder registry + image loader seam
# ============================================================================


def test_default_backend_is_jina_clip() -> None:
    from aic2026.image_embedder import DEFAULT_BACKEND, EMBEDDER_FACTORIES

    assert DEFAULT_BACKEND == "jina-clip"
    assert "jina-clip" in EMBEDDER_FACTORIES


def test_unknown_model_spec_backend_rejected() -> None:
    import pytest as _pytest

    from aic2026.image_embedder import create_image_embedder

    with _pytest.raises(ValueError, match="unknown model backend"):
        create_image_embedder("siglip:some-model")


def test_default_image_loader_returns_rgb_image(tmp_path: Path) -> None:
    from aic2026.image_embedder import default_image_loader
    from PIL import Image

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    loaded = default_image_loader()(image_path)
    assert loaded.size == (4, 4)
    assert loaded.mode == "RGB"


def test_create_text_encoder_rejects_unknown_backend() -> None:
    import pytest as _pytest

    from aic2026.config import load_config
    from aic2026.image_embedder import create_text_encoder

    real = load_config()
    broken = Config(
        data_raw_dir=real.data_raw_dir,
        data_indexes_dir=real.data_indexes_dir,
        data_keyframes_dir=real.data_keyframes_dir,
        model=ModelConfig(
            backend="siglip",
            model_name="x",
            pretrained="y",
            device="cpu",
            precision="fp32",
            normalization="l2",
            preprocessing_id="p",
            tokenizer_id="t",
        ),
    )
    with _pytest.raises(ValueError, match="unknown embedding backend"):
        create_text_encoder(broken)
