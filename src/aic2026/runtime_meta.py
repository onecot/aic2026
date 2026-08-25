"""Runtime metadata + per-query telemetry assembly for the HTTP API.

The loaded runtime is the single source of truth: values are derived from
the live :class:`aic2026.search.SearchEngine` (embedding space + index
manifest), the :class:`aic2026.frame_catalog.FrameCatalog`, and the
keyframe artifact's identity manifest. Nothing here reads embeddings or
exposes filesystem paths — path-like provenance fields are deliberately
dropped in favor of logical identifiers only.

Framework-free: plain functions returning JSON-safe dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_keyframes_identity(keyframes_root: Path | str) -> dict[str, Any]:
    """Read the keyframe artifact's identity manifest, ignoring paths.

    Returns:
        - ``{"artifact_id", "num_videos", "num_keyframes"}`` when available;
          absent keys are omitted. Missing/corrupt manifests yield ``{}``.

    Raises:
        - No exceptions.
    """
    try:
        with (Path(keyframes_root) / "manifest.json").open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    identity: dict[str, Any] = {}
    for key in ("artifact_id", "num_videos", "num_keyframes"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            identity[key] = value
    return identity


def _count_catalog(catalog: Any) -> tuple[int, int]:
    """Return ``(total_keyframes, total_videos)`` known to the catalog."""
    total_frames = 0
    video_ids = catalog.video_ids()
    for video_id in video_ids:
        total_frames += len(catalog.list_video_frames(video_id, offset=0, limit=1_000_000_000))
    return total_frames, len(video_ids)


#: Active runtime capabilities. These describe what the running backend can
#: actually serve today — not optional dependencies or deleted experiments.
ACTIVE_FEATURES = {
    "visual_embedding": True,  # SearchEngine over the Jina CLIP v2 index
    "ocr": False,  # OCR retrieval was removed from production (Phase 1)
    "asr": False,  # never implemented
    "metadata_search": False,  # lexical media-info branch removed (Phase 1)
    "temporal_navigation": True,  # FrameCatalog browsing endpoints active
}

#: Per-search signals mirror the active features plus pipeline stages that
#: do not run today (no reranking stage exists in the serving path).
SEARCH_SIGNALS = {
    "visual_embedding": True,
    "ocr": False,
    "asr": False,
    "metadata": False,
    "temporal_reranking": False,
}


def build_system_metadata(
    *,
    engine: Any,
    catalog: Any,
    keyframes_root: Path | str,
    vlm_enabled: bool = False,
) -> dict[str, Any]:
    """Assemble the ``GET /api/system`` payload from loaded runtime objects.

    Expected input:
        - engine: loaded :class:`aic2026.search.SearchEngine`.
        - catalog: loaded :class:`aic2026.frame_catalog.FrameCatalog`.
        - keyframes_root: keyframe artifact root (identity manifest only;
          the path itself is never exposed).
        - vlm_enabled: whether VLM answer generation is callable through
          this API today (it is lazy server-side and not exposed yet).

    Returns:
        - JSON-safe system metadata dict.

    Raises:
        - No exceptions beyond malformed runtime objects (which indicate a
          startup bug, surfaced by the readiness guard before this runs).
    """
    space = engine.embedding_space
    index_manifest = engine.manifest
    identity = _load_keyframes_identity(keyframes_root)
    catalog_frames, catalog_videos = _count_catalog(catalog)

    return {
        "status": "ready",
        "retrieval": {
            "model": {
                "name": space.model_id,
                "backend": space.backend,
            },
            "index": {
                "id": index_manifest.artifact_id,
                "total_keyframes": int(index_manifest.num_rows),
                "total_videos": int(index_manifest.num_videos),
                "embedding_dimension": int(index_manifest.embedding_dim),
                "similarity": index_manifest.score_metric,
            },
            "keyframes": {
                # Logical identity only; provenance paths are dropped.
                "version": identity.get("artifact_id"),
                "total_keyframes": catalog_frames,
                "total_videos": catalog_videos,
            },
        },
        "features": dict(ACTIVE_FEATURES),
        "generation": {
            "vlm": {
                "enabled": bool(vlm_enabled),
                "model": _default_vlm_model(),
                "load_policy": "lazy",
            }
        },
    }


def build_search_meta(
    *,
    task: str,
    returned: int,
    total_ms: float,
    encode_ms: float | None = None,
    retrieve_ms: float | None = None,
) -> dict[str, Any]:
    """Assemble the additive ``meta`` object of a search response.

    Timings are milliseconds, rounded to one decimal; unsupported splits
    stay ``None`` instead of being faked.

    Expected input:
        - task: served task identifier (``kis`` / ``trake``).
        - returned: number of top-level items returned (hits for KIS,
          events for TRAKE).
        - total_ms: full handling time measured by the route.
        - encode_ms / retrieve_ms: split timings when cleanly measurable.

    Returns:
        - JSON-safe telemetry dict.
    """
    def _ms(value: float | None) -> float | None:
        return None if value is None else round(max(0.0, float(value)), 1)

    return {
        "task": task,
        "returned": int(returned),
        "timing": {
            "total_ms": _ms(total_ms),
            "encode_ms": _ms(encode_ms),
            "retrieve_ms": _ms(retrieve_ms),
        },
        "signals": dict(SEARCH_SIGNALS),
    }


def _default_vlm_model() -> str | None:
    """The configured default QA VLM checkpoint id (no model loading)."""
    try:
        from aic2026.qa.answer_generation import DEFAULT_ANSWER_MODEL

        return DEFAULT_ANSWER_MODEL
    except Exception:  # noqa: BLE001 - optional dependency metadata only
        return None
