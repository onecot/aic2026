import json
from pathlib import Path
import numpy as np
import os

from aic2026.dataset import Dataset
from aic2026.index_builder import build_index
from aic2026.embeddings import EmbeddingSpace
from aic2026.index_artifact import read_manifest, iter_frames


def _write_map_csv(path: Path, n_rows: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("n,pts_time,fps,frame_idx\n")
        for i in range(1, n_rows + 1):
            f.write(f"{i},{(i-1)*0.04},25,{(i-1)*2}\n")


def _write_media_info(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "author": "a",
        "channel_id": "c",
        "channel_url": "u",
        "description": "d",
        "keywords": [],
        "length": 10,
        "publish_date": "2020-01-01",
        "thumbnail_url": "t",
        "title": "title",
        "watch_url": "w",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_clip_npy(path: Path, mat: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mat)


def make_raw_dataset(tmp_path: Path):
    raw = tmp_path / "raw"
    # create collection dirs
    for name in ("clip-features-32", "map-keyframes", "media-info", "keyframes", "objects"):
        (raw / name).mkdir(parents=True, exist_ok=True)
    # two videos
    vids = [("L1_V001", 2), ("L1_V002", 1)]
    dim = 8
    for vid, n in vids:
        _write_map_csv(raw / "map-keyframes" / f"{vid}.csv", n)
        _write_media_info(raw / "media-info" / f"{vid}.json")
        mat = (np.arange(n * dim).reshape(n, dim).astype(np.float16) + 1.0)
        _write_clip_npy(raw / "clip-features-32" / f"{vid}.npy", mat)
    return raw


def test_build_index_basic(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact"
    manifest = build_index(ds, artifact, artifact_id="test-1")
    # check files
    assert (artifact / "frames.ndjson").exists()
    assert (artifact / "embeddings.npy").exists()
    assert (artifact / "norms.npy").exists()
    assert (artifact / "manifest.json").exists()
    # validate manifest
    m = read_manifest(artifact / "manifest.json")
    assert m.num_rows == 3
    assert m.num_videos == 2
    emb = np.load(artifact / "embeddings.npy")
    norms = np.load(artifact / "norms.npy")
    assert emb.shape[0] == 3
    assert norms.shape[0] == 3
    # check alignment
    frames = list(iter_frames(artifact / "frames.ndjson"))
    assert len(frames) == 3
    # check that manifest embedding dim matches file
    assert emb.shape[1] == m.embedding_dim


def test_new_build_accepts_explicit_embedding_space(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    artifact = tmp_path / "model_identified_artifact"
    embedding_space = EmbeddingSpace(
        backend="test-backend",
        model_id="test-model",
        checkpoint_id="test-checkpoint",
        space_id="test-space-v1",
        embedding_dim=8,
        output_dtype="float16",
        normalization="l2",
        preprocessing_id="test-preprocess-v1",
    )

    manifest = build_index(
        Dataset(raw),
        artifact,
        artifact_id="model-identified-test",
        embedding_space=embedding_space,
    )

    assert manifest.embedding_space == embedding_space
    assert read_manifest(artifact / "manifest.json").embedding_space == embedding_space


def test_resume_behavior(tmp_path: Path):
    raw = make_raw_dataset(tmp_path)
    ds = Dataset(raw)
    artifact = tmp_path / "artifact_resume"
    # first run: stop after 1 row
    try:
        build_index(ds, artifact, artifact_id="test-resume", stop_after_rows=1)
    except RuntimeError:
        pass
    # manifest should not exist
    assert not (artifact / "manifest.json").exists()
    # checkpoint should exist
    assert (artifact / ".checkpoint.json").exists()
    # now resume full build
    manifest = build_index(ds, artifact, artifact_id="test-resume")
    assert (artifact / "manifest.json").exists()
    m = read_manifest(artifact / "manifest.json")
    assert m.num_rows == 3
    emb = np.load(artifact / "embeddings.npy")
    norms = np.load(artifact / "norms.npy")
    assert emb.shape[0] == 3
    assert norms.shape[0] == 3
