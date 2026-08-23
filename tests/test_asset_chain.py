"""End-to-end offline asset-chain smoke test.

Covers the rebuild path  raw videos -> keyframes -> materialized images ->
embeddings -> index  on tiny synthetic fixtures:

- keyframe extraction uses the synthetic frame reader (no ffmpeg) and the
  deterministic synthetic shot detector;
- materialization runs only when ffmpeg is available (skipped otherwise);
- embedding shards use a stub embedder, so no model is loaded;
- the final index is retrieved through :class:`IndexRetriever`.

This module replaces the former twelve per-stage asset test files; it
protects the public chain contract, not internal bookkeeping.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from aic2026.assets.embedding_merge import merge_shards as merge_embedding_shards
from aic2026.assets.embedding_shard import embed_shard
from aic2026.assets.keyframe_extraction import (
    ExtractionConfig,
    VideoEntry,
    VideoManifest,
    extract_keyframes,
    save_video_manifest,
)
from aic2026.assets.keyframe_extraction_merge import merge_shards as merge_keyframe_shards
from aic2026.assets.keyframe_extraction_split import split_video_manifest
from aic2026.assets.keyframe_extraction_verify import verify_keyframe_extraction
from aic2026.assets.keyframe_extractor import write_synthetic_video
from aic2026.embeddings import EmbeddingSpace
from aic2026.index_builder import build_index
from aic2026.retrieval import IndexRetriever

ffmpeg_available = shutil.which("ffmpeg") is not None


# --------------------------------------------------------------------- fakes


def _scene_values(num_frames: int, scene_length: int = 10) -> list[float]:
    return [float((i // scene_length) * 100.0) for i in range(num_frames)]


def _make_videos(tmp_path: Path, count: int = 3, frames: int = 120) -> list[VideoEntry]:
    entries: list[VideoEntry] = []
    for i in range(count):
        vid = f"L21_V{i + 1:03d}"
        path = write_synthetic_video(tmp_path / f"{vid}.json", _scene_values(frames))
        entries.append(VideoEntry(video_id=vid, path=str(path), num_frames=frames))
    return entries


def _extraction_config(frame_reader: str = "synthetic") -> ExtractionConfig:
    return ExtractionConfig(
        frame_reader=frame_reader,
        target_keyframes_per_video=5,
        min_temporal_distance=4,
    )


class _StubImageEmbedder:
    """Deterministic embedder deriving vectors from keyframe image bytes."""

    def __init__(self, space: EmbeddingSpace) -> None:
        self._space = space

    @property
    def model_name(self) -> str:
        return "stub:model"

    @property
    def embedding_dim(self) -> int:
        return self._space.embedding_dim

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def embedding_space(self) -> EmbeddingSpace:
        return self._space

    def encode_images(self, batch_images: list[object]) -> np.ndarray:
        out = np.zeros((len(batch_images), self._space.embedding_dim), dtype=np.float32)
        for i, image in enumerate(batch_images):
            data = bytes(image)  # type: ignore[arg-type]
            out[i, 0] = float(data[0]) if data else 0.0
            out[i, 1 % self._space.embedding_dim] = float(len(data)) % 7.0
        return out


# ------------------------------------------------------------- keyframe chain


def test_keyframe_chain_split_extract_merge_verify(tmp_path: Path) -> None:
    from aic2026.assets.keyframe_extraction import save_video_manifest, load_video_manifest

    videos = _make_videos(tmp_path)
    manifest = VideoManifest(schema_version="1.0", videos=tuple(videos))
    manifest_path = tmp_path / "videos.json"
    save_video_manifest(manifest_path, manifest)

    split_root = tmp_path / "splits"
    split_manifest = split_video_manifest(manifest_path, num_workers=2, output_dir=split_root)
    assert split_manifest.num_workers == 2

    shard_dirs = []
    for worker_id in range(2):
        worker_json = split_root / f"worker_{worker_id:03d}" / "videos.json"
        out_dir = tmp_path / f"worker_{worker_id:03d}"
        result = extract_keyframes(worker_json, out_dir, _extraction_config())
        shard_dirs.append(out_dir)
        assert result.num_videos > 0
        assert result.num_keyframes > 0

    merged_dir = tmp_path / "keyframes_full"
    merged = merge_keyframe_shards(shard_dirs, merged_dir)
    assert merged.num_videos == len(videos)

    report = verify_keyframe_extraction(merged_dir)
    assert report.valid


@pytest.mark.skipif(not ffmpeg_available, reason="materialization requires ffmpeg on PATH")
def test_materialization_produces_image_artifact(tmp_path: Path) -> None:
    from aic2026.assets.keyframe_materialize import materialize_keyframes

    video_path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=3",
            "-pix_fmt", "yuv420p", "-y", str(video_path),
        ],
        check=True, capture_output=True,
    )
    videos = tmp_path / "videos.json"
    manifest = VideoManifest(
        schema_version="1.0",
        videos=(VideoEntry(video_id="L21_V001", path=str(video_path), num_frames=75),),
    )
    save_video_manifest(videos, manifest)

    extracted = extract_keyframes(videos, tmp_path / "extracted", _extraction_config(frame_reader="ffmpeg"))
    assert extracted.num_keyframes >= 1

    out = tmp_path / "adaptive_sbd_v1"
    materialize_keyframes(
        videos,
        tmp_path / "extracted" / "frames.ndjson",
        out,
    )
    assert (out / "images" / "L21_V001").is_dir()
    images = list((out / "images" / "L21_V001").glob("*.jpg"))
    assert images, "materialized images missing"


# ------------------------------------------------------------ embedding chain


def _write_frames(path: Path, rows: list[tuple[str, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for video_id, ordinal, frame_index in rows:
            handle.write(
                json.dumps(
                    {
                        "keyframe": {
                            "video_id": {"value": video_id},
                            "keyframe_ordinal": ordinal,
                        },
                        "pts_time": frame_index / 25.0,
                        "fps": 25.0,
                        "frame_index": frame_index,
                    }
                )
                + "\n"
            )


def _space() -> EmbeddingSpace:
    from aic2026.embeddings import legacy_embedding_space

    return legacy_embedding_space(embedding_dim=4, output_dtype="float32")


def test_embedding_chain_shard_merge_index_retrieve(tmp_path: Path) -> None:
    frames_file = tmp_path / "frames.ndjson"
    rows = [("L21_V001", o, o * 10) for o in range(1, 4)]
    rows += [("L21_V002", o, o * 10) for o in range(1, 4)]
    _write_frames(frames_file, rows)

    # raw-layout images so the stub loader can read per-keyframe files
    raw = tmp_path / "raw"
    for video_id, ordinal, _ in rows:
        image_dir = raw / "keyframes" / video_id
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{ordinal:03d}.jpg").write_bytes(bytes([ordinal % 255, 7, 9]))

    space = _space()
    shard_dirs = []
    for shard_index, chunk in enumerate((rows[:3], rows[3:])):
        shard_frames = tmp_path / f"frames_{shard_index}.ndjson"
        _write_frames(shard_frames, chunk)
        embedder = _StubImageEmbedder(space)
        manifest = embed_shard(
            shard_frames,
            tmp_path / f"shard_{shard_index}",
            embedder,
            keyframes_root=raw,
            image_loader=lambda p: Path(p).read_bytes(),
        )
        assert manifest.num_embeddings == 3
        shard_dirs.append(tmp_path / f"shard_{shard_index}")

    merged_dir = tmp_path / "embeddings_all"
    merged = merge_embedding_shards(shard_dirs, merged_dir)
    assert merged.total_embeddings == 6

    # embeddings -> index -> retrieve through the public runtime boundary
    from aic2026.embedding_dataset import EmbeddingArtifactDataset

    dataset = EmbeddingArtifactDataset(merged_dir)
    index_dir = tmp_path / "index"
    index_manifest = build_index(dataset, index_dir, artifact_id="chain-test")
    assert index_manifest.artifact_id == "chain-test"

    retriever = IndexRetriever(index_dir)
    rng = np.random.default_rng(7)
    query_vector = rng.standard_normal(space.embedding_dim).astype(np.float32)
    hits = retriever.search(query_vector, top_k=3)
    assert len(hits) == 3
    assert [hit.rank for hit in hits] == [1, 2, 3]
