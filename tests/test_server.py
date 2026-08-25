"""Minimal contract test for the conventional server entrypoint.

Importing :mod:`aic2026.server` must build the FastAPI application without
loading heavy resources (the Jina encoder, index, and FrameCatalog load
during the startup lifespan, not at import time).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_server_exposes_fastapi_app_without_heavy_load() -> None:
    before = set(sys.modules)

    from fastapi import FastAPI

    from aic2026 import server

    # Only modules *newly* imported by building the app matter: earlier
    # tests in the same process may legitimately have loaded model code.
    newly = set(sys.modules) - before

    assert isinstance(server.app, FastAPI)
    # Managed mode: not ready before the lifespan runs, and no heavy model
    # runtime was imported as a side effect of building the app.
    assert getattr(server.app.state, "ready", False) is False
    assert not any(m.startswith("aic2026.model_backends") for m in newly)
    assert "torch" not in newly


def test_server_routes_match_api_surface() -> None:
    from aic2026 import server

    paths = {route.path for route in server.app.routes}
    assert {
        "/api/health",
        "/api/search",
        "/api/videos/{video_id}",
        "/api/videos/{video_id}/frames",
        "/api/frames/{video_id}/{frame_index}",
    } <= paths
