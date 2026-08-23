"""Minimal HTTP API over the Phase-2 retrieval core.

The HTTP layer is thin: request validation, service call, response
serialization, HTTP error mapping. All capability lives behind two plain
Python services that are fully usable without FastAPI:

    RetrievalService.search(query, limit)   -> ranked hits
    FrameCatalog.get_video/list/frames_around

No prediction artifacts are written as a side effect of any request; no
filesystem paths, embeddings, or internal artifact details are exposed.

Run locally::

    uvicorn aic2026.api:create_default_app --factory --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aic2026.config import Config, load_config
from aic2026.frame_catalog import FrameCatalog, FrameVideoSummary
from aic2026.frame_images import find_keyframe_image
from aic2026.retrieval import RetrievalCandidate
from aic2026.search import SearchEngine

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
    task: Literal["kis"] = "kis"
    limit: int = Field(default=100, ge=1, le=1000)


class SearchHit(BaseModel):
    """One ranked candidate exposed by the search endpoint."""

    rank: int
    video_id: str
    frame_index: int
    keyframe_ordinal: int
    pts_time: float
    score: float


class SearchResponse(BaseModel):
    """Response body for ``POST /api/search``."""

    query: str
    results: list[SearchHit]


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


def create_app(
    *,
    engine: SearchEngine | None = None,
    catalog: FrameCatalog | None = None,
    config: Config | None = None,
    index_artifact: Path | str | None = None,
    keyframes_root: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI application around injected or default services.

    Services are constructed exactly once here; requests never reload the
    model, the index, or ``frames.ndjson``.

    Expected input:
        - engine: pre-built :class:`aic2026.search.SearchEngine` (tests);
          when omitted, one is built from ``config`` + ``index_artifact``.
        - catalog: pre-built :class:`aic2026.frame_catalog.FrameCatalog`
          (tests); when omitted, built from ``keyframes_root`` or the
          configured ``data.keyframes_dir``.
        - config: effective configuration; defaults to
          :func:`aic2026.config.load_config`.
        - index_artifact: index artifact directory for the default engine;
          required when ``engine`` is omitted.
        - keyframes_root: keyframe artifact root for the default catalog.

    Returns:
        - The configured FastAPI application.

    Raises:
        - ValueError: if no engine was injected and no index artifact was
          provided.
    """
    effective_config = config if config is not None else load_config()
    resolved_keyframes = (
        Path(keyframes_root)
        if keyframes_root is not None
        else effective_config.data_keyframes_dir
    )
    if engine is None:
        if index_artifact is None:
            raise ValueError(
                "create_app requires either an injected engine or an "
                "index_artifact directory"
            )
        engine = SearchEngine.from_config(effective_config, Path(index_artifact))
    if catalog is None:
        catalog = FrameCatalog.from_keyframes_root(resolved_keyframes)

    app = FastAPI(title="AIC2026 Retrieval API", version="1.0")
    retrieval = RetrievalService(engine)

    app.state.engine = engine
    app.state.catalog = catalog
    app.state.retrieval = retrieval
    app.state.keyframes_root = resolved_keyframes

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
    @app.post("/api/search", response_model=SearchResponse)
    def search(payload: SearchRequest) -> SearchResponse:
        hits = [ _hit(candidate) for candidate in retrieval.search(payload.query, payload.limit) ]
        return SearchResponse(query=payload.query, results=hits)

    @app.get("/api/videos/{video_id}", response_model=VideoMetadataResponse)
    def video_metadata(video_id: str) -> VideoMetadataResponse:
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
    """Build the production-default application from environment/config.

    Environment overrides (optional):
        - ``AIC_INDEX_DIR``: search index artifact directory; defaults to
          ``data/interim/indexes/jina-v2-full``.
        - ``AIC_KEYFRAMES_DIR``: keyframe artifact root; defaults to the
          configured ``data.keyframes_dir``.

    Returns:
        - The configured FastAPI application.

    Raises:
        - ValueError / FileNotFoundError: see :func:`create_app`.
    """
    config = load_config()
    env_index = os.environ.get("AIC_INDEX_DIR")
    env_keyframes = os.environ.get("AIC_KEYFRAMES_DIR")
    return create_app(
        config=config,
        index_artifact=Path(env_index) if env_index else DEFAULT_SEARCH_INDEX,
        keyframes_root=Path(env_keyframes) if env_keyframes else None,
    )
