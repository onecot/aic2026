"""Tiny HTTP client binding the inspection UI to the FastAPI backend.

The Streamlit process must never instantiate the retrieval runtime. This
module is its only coupling point: plain ``urllib`` calls to the running
FastAPI server (``AIC_API_URL``, default ``http://127.0.0.1:8000``).

Functions return parsed JSON payloads (or raw bytes for images) and raise
:class:`ApiError` with concise, user-safe messages — never stack traces or
internal paths.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 30.0


class ApiError(RuntimeError):
    """Raised when the backend cannot fulfill a request.

    Attributes:
        - status: HTTP status code, or None for connection-level failures.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def base_url() -> str:
    """Return the configured backend URL (``AIC_API_URL`` or localhost)."""
    return os.environ.get("AIC_API_URL", DEFAULT_API_URL).rstrip("/")


def _request(path: str, *, payload: dict[str, Any] | None = None,
             timeout: float = DEFAULT_TIMEOUT) -> Any:
    """Perform one JSON request; raise :class:`ApiError` on failure."""
    url = f"{base_url()}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        raise ApiError(_error_detail(exc), exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(f"cannot reach backend at {base_url()}") from None


# ---------------------------------------------------------------- operations


def health(timeout: float = 5.0) -> dict[str, Any]:
    """Return the health payload.

    Returns:
        - ``{"status": "ready"|"starting", "model": bool, "index": bool,
          "frame_catalog": bool}``.

    Raises:
        - ApiError: backend unreachable or returned a non-health error.
    """
    result: dict[str, Any] = _request("/api/health", timeout=timeout)
    return result


def search(query: str, *, task: str = "kis", limit: int = 100) -> dict[str, Any]:
    """Run one retrieval search.

    Returns:
        - KIS/general: ``{"query", "results": [hit...]}``.
        - TRAKE: additionally ``"events": [{event_id, event_text,
          predictions, windows}]``.

    Raises:
        - ApiError: non-2xx (503 not-ready, 400 invalid input, ...).
    """
    return _request("/api/search", payload={"query": query, "task": task, "limit": limit})


def get_video(video_id: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch video metadata (frame_count / first / last frame index)."""
    return _request(f"/api/videos/{video_id}", timeout=timeout)


def get_frames(
    video_id: str,
    *,
    offset: int | None = None,
    limit: int = 100,
    center: int | None = None,
    radius: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Browse indexed frames (paging mode or around-center mode)."""
    params = f"limit={int(limit)}"
    if center is not None:
        params += f"&center={int(center)}"
        if radius is not None:
            params += f"&radius={int(radius)}"
    elif offset is not None:
        params += f"&offset={int(offset)}"
    return _request(f"/api/videos/{video_id}/frames?{params}", timeout=timeout)


def get_frame_image(video_id: str, frame_index: int, timeout: float = DEFAULT_TIMEOUT) -> bytes | None:
    """Fetch one keyframe image through the backend.

    Returns:
        - Image bytes, or None when the backend reports 404 (unknown
          frame/image).

    Raises:
        - ApiError: for other failures (503 not ready, connection errors).
    """
    url = f"{base_url()}/api/frames/{video_id}/{int(frame_index)}"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ApiError(_short_reason(exc), exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(f"cannot reach backend at {base_url()}") from None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Extract a concise user-safe message from an HTTP error response.

    Handles both ``{"detail": ...}`` envelopes and flat structured bodies
    (e.g. the health endpoint's ``{"status": "starting", ...}`` on 503).
    """
    try:
        parsed = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - fallback to the reason phrase
        return str(exc.reason)
    if isinstance(parsed, dict):
        if "detail" in parsed:
            detail = parsed["detail"]
            if isinstance(detail, dict):
                return str(detail.get("status", exc.reason))
            return str(detail)
        if "status" in parsed:
            return str(parsed["status"])
    return str(parsed)


def _short_reason(exc: urllib.error.HTTPError) -> str:
    """Alias of :func:`_error_detail` kept for readability at call sites."""
    return _error_detail(exc)
