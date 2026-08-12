"""Fixture helpers that build a small, synthetic mirror of the raw dataset.

The fixture mirrors the real BTC layout on a tiny scale so inspection
logic can be tested without the full dataset.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aic2026.contracts import COLLECTION_NAMES

#: Layout of the synthetic dataset: video id -> (num_keyframes, embedding_dim)
VIDEO_LAYOUT = {
    "L21_V001": (3, 512),
    "L21_V002": (2, 512),
    "L22_V001": (1, 512),
}


def _write_keyframes(keyframes_dir: Path, n: int) -> None:
    """Write n placeholder keyframe images ``001.jpg ... NNN.jpg``."""
    keyframes_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        (keyframes_dir / f"{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")

    # macOS artifacts are intentionally included so inspection ignores them.
    (keyframes_dir / ".DS_Store").write_bytes(b"")


def _write_objects(objects_dir: Path, n: int) -> None:
    """Write n detection JSONs ``001.json ... NNN.json``."""
    objects_dir.mkdir(parents=True)
    for i in range(1, n + 1):
        payload = {
            "detection_scores": ["0.99"],
            "detection_class_names": ["/m/01jfsr"],
            "detection_class_entities": ["Lantern"],
            "detection_class_labels": ["84"],
            "detection_boxes": [["0.0", "0.0", "1.0", "1.0"]],
        }
        (objects_dir / f"{i:03d}.json").write_text(json.dumps(payload))
    (objects_dir / ".DS_Store").write_bytes(b"")


def _write_map(csv_path: Path, n: int) -> None:
    """Write a map-keyframes CSV with n rows @ fps 25."""
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        for i in range(1, n + 1):
            writer.writerow([i, (i - 1) * 0.4, 25.0, (i - 1) * 10])


def _write_media(media_path: Path) -> None:
    """Write a minimal media-info JSON."""
    payload = {
        "author": "Test Author",
        "channel_id": "UCHANNEL",
        "channel_url": "https://www.youtube.com/channel/UCHANNEL",
        "description": "test",
        "keywords": ["k1"],
        "length": 42,
        "publish_date": "01/01/2024",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "title": "Test video",
        "watch_url": "https://youtube.com/watch?v=TEST",
    }
    media_path.write_text(json.dumps(payload))


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Build a synthetic raw dataset mirror and return its root.

    Returns:
        - Path to the synthetic dataset root.
    """
    root = tmp_path / "raw"
    for collection in COLLECTION_NAMES:
        (root / collection).mkdir(parents=True, exist_ok=True)
    for video_id, (n_keyframes, embed_dim) in VIDEO_LAYOUT.items():
        _write_keyframes(root / "keyframes" / video_id, n_keyframes)
        _write_objects(root / "objects" / video_id, n_keyframes)
        _write_map(root / "map-keyframes" / f"{video_id}.csv", n_keyframes)
        _write_media(root / "media-info" / f"{video_id}.json")
        matrix = np.zeros((n_keyframes, embed_dim), dtype=np.float16)
        np.save(root / "clip-features-32" / f"{video_id}.npy", matrix)
    return root