"""Canonical keyframe image resolution.

One resolver for the whole codebase: given a keyframe artifact root and a
``(video_id, keyframe_ordinal)`` pair, return the image file path. It
prefers the materialized ``images/`` collection of the current keyframe
artifact (``adaptive_sbd_v1``: ``images/<video_id>/<ordinal:06d>.jpg``) and
falls back to the legacy BTC-style ``keyframes/<video_id>/<ordinal:03d>.jpg``
layout when no materialized collection exists.

Consumers (embedding shards, QA evidence resolution, OCR/caption generation)
must use this function instead of re-implementing lookup conventions.
"""

from __future__ import annotations

from pathlib import Path

#: Name of the materialized per-video image collection inside a keyframe
#: artifact (``adaptive_sbd_v1`` layout).
IMAGES_DIRNAME = "images"


def image_filename(keyframe_ordinal: int) -> str:
    """Return the canonical materialized image filename for one ordinal.

    Expected input:
        - keyframe_ordinal: 1-based keyframe ordinal.

    Returns:
        - Filename of the form ``<ordinal:06d>.jpg``.
    """
    return f"{keyframe_ordinal:06d}.jpg"


def find_keyframe_image(
    keyframes_root: Path | str,
    video_id: str,
    keyframe_ordinal: int,
) -> Path:
    """Resolve the keyframe image path for one record.

    Expected input:
        - keyframes_root: root directory containing either an ``images/``
          collection or a legacy ``keyframes/`` collection.
        - video_id: the ``L*_V*`` id.
        - keyframe_ordinal: the 1-based keyframe ordinal.

    Returns:
        - The image file path for that keyframe.

    Raises:
        - ValueError: if ``keyframe_ordinal`` is not a positive integer.
        - FileNotFoundError: if no keyframe image exists for the ordinal.
    """
    if not isinstance(keyframe_ordinal, int) or isinstance(keyframe_ordinal, bool) or keyframe_ordinal < 1:
        raise ValueError(f"keyframe_ordinal must be a positive integer, got {keyframe_ordinal!r}")
    root = Path(keyframes_root)
    materialized = root / IMAGES_DIRNAME / video_id / image_filename(keyframe_ordinal)
    if materialized.is_file():
        return materialized
    if (root / IMAGES_DIRNAME).is_dir():
        raise FileNotFoundError(
            f"keyframe image missing for {video_id} ordinal {keyframe_ordinal}: "
            f"{materialized}"
        )
    directory = root / "keyframes" / video_id
    for extension in ("jpg", "jpeg", "png"):
        candidate = directory / f"{keyframe_ordinal:03d}.{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"keyframe image missing for {video_id} ordinal {keyframe_ordinal}: {directory}"
    )
