"""Runtime server entrypoint for the AIC2026 retrieval API.

Exposes the conventional ASGI target:

    uvicorn aic2026.server:app --host 127.0.0.1 --port 8000

This module is intentionally a one-line composition over
:func:`aic2026.api.create_default_app`; all routes, lifespan behavior,
startup loading, readiness state, and dependency injection live in
:mod:`aic2026.api`. Importing this module builds the application object but
does **not** load heavy resources — the Jina encoder, index, and
FrameCatalog load during the startup lifespan (see ``aic2026.api``).
"""

from __future__ import annotations

from aic2026.api import create_default_app

app = create_default_app()
