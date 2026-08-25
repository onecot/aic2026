"""Minimal HTTP API over the Phase-2 retrieval core.

The HTTP layer is thin: request validation, service call, response
serialization, HTTP error mapping. All capability lives behind two plain
Python services that are fully usable without FastAPI:

    RetrievalService.search(query, limit)   -> ranked hits
    FrameCatalog.get_video/list/frames_around

Startup lifecycle
-----------------

Two construction modes:

- **Injected services** (:func:`create_app` with ``engine``/``catalog``):
  used by tests; the runtime is ready immediately.
- **Managed mode** (:func:`create_app` without services, e.g.
  :func:`create_default_app`): configuration, the Jina encoder + index, and
  the FrameCatalog are loaded exactly once during application startup
  (FastAPI lifespan), *before* the server reports READY. A startup failure
  is logged concisely and aborts startup instead of serving a half-ready
  backend. While initialization is in flight, ``GET /api/health`` reports
  ``status="starting"`` (HTTP 503) and retrieval endpoints return 503;
  requests never trigger lazy resource construction.

No prediction artifacts are written as a side effect of any request; no
filesystem paths, embeddings, or internal artifact details are exposed.

Run locally::

    uvicorn aic2026.server:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aic2026.config import Config, load_config
from aic2026.frame_catalog import FrameCatalog
from aic2026.frame_images import find_keyframe_image
from aic2026.retrieval import RetrievalCandidate
from aic2026.search import SearchEngine

logger = logging.getLogger(__name__)

#: Default production index used when no explicit index artifact is given.
DEFAULT_SEARCH_INDEX = Path("data/interim/indexes/jina-v2-full")

_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


# ---------------------------------------------------------------------------
# Service layer (framework-free)
# ---------------------------------------------------------------------------


class RetrievalService:
    """Text-to-candidates search service over one :class:`SearchEngine`.

    Expected input:
        - engine: a constructed :class:`aic2026.search.SearchEngine`.

    Raises:
        - No exceptions at construction.
    """

    def __init__(self, engine: SearchEngine) -> None:
        self._engine = engine

    @property
    def engine(self) -> SearchEngine:
        """The bound search engine."""
        return self._engine

    def search(self, query: str, limit: int = 100) -> list[RetrievalCandidate]:
        """Rank indexed keyframes for one free-text query.

        Expected input:
            - query: non-empty query text.
            - limit: positive number of candidates to return.

        Returns:
            - Ranked candidates (deterministic for identical inputs).

        Raises:
            - ValueError: for invalid inputs or embedding-space mismatch.
        """
        return list(self._engine.search_text(query, top_k=limit))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for ``POST /api/search``."""

    query: str = Field(min_length=1)
    task: Literal["kis", "trake"] = "kis"
    limit: int = Field(default=100, ge=1, le=1000)


class SearchHit(BaseModel):
    """One ranked candidate exposed by the search endpoint."""

    rank: int
    video_id: str
    frame_index: int
    keyframe_ordinal: int
    pts_time: float
    score: float


class WindowPayload(BaseModel):
    """One temporal window candidate of a TRAKE event."""

    video_id: str
    frame_start: int
    frame_end: int
    score: float
    rank: int


class EventPayload(BaseModel):
    """One TRAKE event with its ranked candidates and windows.

    Mirrors the TRAKE prediction-artifact line schema so UI/API consumers
    work against one stable shape.
    """

    event_id: str
    event_text: str
    predictions: list[SearchHit]
    windows: list[WindowPayload]


class SearchResponse(BaseModel):
    """Response body for ``POST /api/search``."""

    query: str
    results: list[SearchHit]
    events: list[EventPayload] | None = None


class VideoMetadataResponse(BaseModel):
    """Response body for ``GET /api/videos/{video_id}``."""

    video_id: str
    frame_count: int
    first_frame_index: int
    last_frame_index: int


class FrameRecord(BaseModel):
    """One indexed keyframe exposed while browsing a video."""

    keyframe_ordinal: int
    frame_index: int
    pts_time: float
    fps: float


class VideoFramesResponse(BaseModel):
    """Response body for ``GET /api/videos/{video_id}/frames``."""

    video_id: str
    frames: list[FrameRecord]
    total: int


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    status: Literal["ready", "starting"]
    model: bool
    index: bool
    frame_catalog: bool


def _hit(candidate: RetrievalCandidate) -> SearchHit:
    """Serialize one retrieval candidate for the search response."""
    return SearchHit(
        rank=candidate.rank,
        video_id=candidate.video_id,
        frame_index=candidate.frame_index,
        keyframe_ordinal=candidate.keyframe_ordinal,
        pts_time=float(candidate.pts_time),
        score=float(candidate.score),
    )


def _frame(timing: Any) -> FrameRecord:
    """Serialize one canonical timing record for frame browsing."""
    return FrameRecord(
        keyframe_ordinal=timing.keyframe.keyframe_ordinal,
        frame_index=timing.frame_index,
        pts_time=float(timing.pts_time),
        fps=float(timing.fps),
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def _load_managed_runtime(config: Config | None, index_artifact: Path | str | None,
                          keyframes_root: Path | str | None) -> tuple[Any, Any, Any]:
    """Construct (engine, catalog, retrieval service); logs once per process.

    Raises:
        - Exception (propagated): any failure during model/index/catalog
          loading aborts startup so the server never reports READY.
    """
    effective_config = config if config is not None else load_config()
    resolved_keyframes = (
        Path(keyframes_root)
        if keyframes_root is not None
        else effective_config.data_keyframes_dir
    )
    if index_artifact is None:
        index_artifact = DEFAULT_SEARCH_INDEX
    logger.info("API startup: loading search engine (model+index)...")
    engine = SearchEngine.from_config(effective_config, Path(index_artifact))
    logger.info("API startup: loading FrameCatalog...")
    catalog = FrameCatalog.from_keyframes_root(resolved_keyframes)
    logger.info("API startup: runtime ready")
    return engine, catalog, RetrievalService(engine)


def create_app(
    *,
    engine: SearchEngine | None = None,
    catalog: FrameCatalog | None = None,
    config: Config | None = None,
    index_artifact: Path | str | None = None,
    keyframes_root: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI application around injected or default services.

    Two modes:

    - **Injected** (tests): both ``engine`` and ``catalog`` provided; the
      runtime is ready immediately and no lifespan loading occurs.
    - **Managed**: at least one service omitted; heavy resources are loaded
      exactly once during startup (see module docstring). Requests never
      construct resources lazily.

    Expected input:
        - engine / catalog: pre-built services (injected mode).
        - config: effective configuration; defaults to
          :func:`aic2026.config.load_config`.
        - index_artifact: index artifact directory (managed mode);
          defaults to the production ``jina-v2-full`` index.
        - keyframes_root: keyframe artifact root (managed mode).

    Returns:
        - The configured FastAPI application.

    Raises:
        - No exceptions at factory time in managed mode; startup failures
          surface during the lifespan with a concise error log.
    """
    managed = engine is None or catalog is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if managed:
            try:
                eng, cat, svc = _load_managed_runtime(config, index_artifact, keyframes_root)
                app.state.engine, app.state.catalog, app.state.retrieval = eng, cat, svc
                app.state.keyframes_root = (
                    Path(keyframes_root)
                    if keyframes_root is not None
                    else (config if config is not None else load_config()).data_keyframes_dir
                )
                app.state.ready = True
            except Exception as exc:  # noqa: BLE001 - clear startup failure
                logger.error("Backend startup failed: %s: %s", type(exc).__name__, exc)
                app.state.ready = False
                raise
        else:
            app.state.ready = True
        yield

    app = FastAPI(title="AIC2026 Retrieval API", version="1.1", lifespan=lifespan)

    if not managed:
        # Injected mode: services exist before serving.
        app.state.engine = engine
        app.state.catalog = catalog
        app.state.retrieval = RetrievalService(engine)
        resolved_keyframes = (
            Path(keyframes_root)
            if keyframes_root is not None
            else (config if config is not None else load_config()).data_keyframes_dir
        )
        app.state.keyframes_root = resolved_keyframes
        app.state.ready = True
    else:
        app.state.ready = False

    def _require_ready() -> None:
        """Guard: retrieval endpoints refuse to serve before initialization."""
        if not getattr(app.state, "ready", False):
            raise HTTPException(status_code=503, detail="Backend is not ready")

    # -------------------------------------------------------------- errors
    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> Any:
        raise HTTPException(status_code=400, detail=str(exc))

    @app.exception_handler(KeyError)
    async def _key_error_handler(request: Request, exc: KeyError) -> Any:
        # Unknown video/frame identifiers map to 404 without echoing inputs
        # beyond the plain identifier message.
        message = str(exc.args[0]) if exc.args else "not found"
        raise HTTPException(status_code=404, detail=message)

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(request: Request, exc: RequestValidationError) -> Any:
        # Invalid request *bodies* keep FastAPI's standard 422; invalid
        # query parameters (pagination/radius) map to the stable 400.
        if any(err.get("loc", ("?",))[0] == "query" for err in exc.errors()):
            raise HTTPException(status_code=400, detail="invalid query parameters")
        raise HTTPException(status_code=422, detail="invalid request body")

    # ------------------------------------------------------------ endpoints
    @app.get("/api/health", response_model=HealthResponse)
    def health() -> Any:
        ready: bool = bool(getattr(app.state, "ready", False))
        if ready:
            return HealthResponse(status="ready", model=True, index=True, frame_catalog=True)
        # Not ready: still describe component state, but signal that
        # retrieval cannot be served yet (flat, stable body — not the
        # usual {"detail": ...} error envelope).
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "starting", "model": False, "index": False, "frame_catalog": False},
        )

    @app.post("/api/search", response_model=SearchResponse)
    def search(payload: SearchRequest) -> SearchResponse:
        _require_ready()
        retrieval: RetrievalService = app.state.retrieval
        if payload.task == "trake":
            from aic2026.text_query import TextQuery
            from aic2026.trake import search_trake

            result = search_trake(
                retrieval.engine, TextQuery(query_id="api-query", query_text=payload.query),
                top_k=payload.limit,
            )
            events = [
                EventPayload(
                    event_id=event.event_id,
                    event_text=event.event_text,
                    predictions=[_hit(candidate) for candidate in event.predictions],
                    windows=[
                        WindowPayload(
                            video_id=w.video_id,
                            frame_start=w.frame_start,
                            frame_end=w.frame_end,
                            score=float(w.score),
                            rank=w.rank,
                        )
                        for w in event.windows
                    ],
                )
                for event in result.events
            ]
            return SearchResponse(query=payload.query, results=[], events=events)
        hits = [_hit(candidate) for candidate in retrieval.search(payload.query, payload.limit)]
        return SearchResponse(query=payload.query, results=hits)

    @app.get("/api/videos/{video_id}", response_model=VideoMetadataResponse)
    def video_metadata(video_id: str) -> VideoMetadataResponse:
        _require_ready()
        summary: FrameVideoSummary = app.state.catalog.get_video(video_id)
        return VideoMetadataResponse(
            video_id=summary.video_id,
            frame_count=summary.frame_count,
            first_frame_index=summary.first_frame_index,
            last_frame_index=summary.last_frame_index,
        )

    @app.get("/api/videos/{video_id}/frames", response_model=VideoFramesResponse)
    def video_frames(
        video_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        center: Annotated[int | None, Query(ge=0)] = None,
        radius: Annotated[int | None, Query(ge=0)] = None,
    ) -> VideoFramesResponse:
        _require_ready()
        cat: FrameCatalog = app.state.catalog
        total = cat.get_video(video_id).frame_count
        if center is not None:
            timings = cat.frames_around(video_id, center, radius=radius, limit=limit)
        else:
            timings = cat.list_video_frames(video_id, offset=offset, limit=limit)
        return VideoFramesResponse(
            video_id=video_id,
            frames=[_frame(timing) for timing in timings],
            total=total,
        )

    @app.get("/api/frames/{video_id}/{frame_index}")
    def frame_image(video_id: str, frame_index: int) -> FileResponse:
        _require_ready()
        if frame_index < 0:
            raise HTTPException(status_code=400, detail="frame_index must be non-negative")
        cat: FrameCatalog = app.state.catalog
        ordinal = cat.ordinal_for(video_id, frame_index)
        if ordinal is None:
            raise HTTPException(status_code=404, detail="frame not found")
        try:
            path = find_keyframe_image(app.state.keyframes_root, video_id, ordinal)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="frame image not available") from None
        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media_type)

    return app


def create_default_app() -> FastAPI:
    """Build the production-default application (managed lifecycle).

    Heavy resources (Jina encoder + index + FrameCatalog) load exactly once
    during startup; the server does not report READY until they finish, and
    a startup failure aborts serving with a concise error log.

    Environment overrides (optional):
        - ``AIC_INDEX_DIR``: search index artifact directory; defaults to
          ``data/interim/indexes/jina-v2-full``.
        - ``AIC_KEYFRAMES_DIR``: keyframe artifact root; defaults to the
          configured ``data.keyframes_dir``.

    Returns:
        - The configured FastAPI application (managed mode).

    Raises:
        - No exceptions at factory time.
    """
    env_index = os.environ.get("AIC_INDEX_DIR")
    env_keyframes = os.environ.get("AIC_KEYFRAMES_DIR")
    return create_app(
        index_artifact=Path(env_index) if env_index else DEFAULT_SEARCH_INDEX,
        keyframes_root=Path(env_keyframes) if env_keyframes else None,
    )
