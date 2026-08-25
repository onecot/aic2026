"""Rendering helpers for the inspection workspace (HTTP-backed).

All data arrives from the FastAPI backend through :mod:`app.api_client`;
images are fetched as bytes via ``GET /api/frames/{video}/{frame_index}``
— the UI never touches the filesystem or any retrieval runtime.

Streamlit APIs follow docs.streamlit.io v1.61: ``st.image`` with the
documented ``width`` parameter, buttons in ``st.columns``, and a fragment
for temporal paging (``st.rerun(scope="fragment")`` stays local).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from app import api_client

THUMBNAIL_WIDTH = 168
STRIP_WIDTH = 140
INSPECTOR_WIDTH = 520


def _looks_like_image(data: bytes) -> bool:
    """Cheap signature check guarding against truncated/corrupt payloads."""
    if len(data) < 32:
        return False
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"


def image_bytes_or_none(video_id: str, frame_index: int) -> bytes | None:
    """Fetch one keyframe image through the backend; None on any failure.

    Expected input:
        - video_id / frame_index: logical frame identity.

    Returns:
        - Verified image bytes, or None when unavailable/unreadable.
          Failures degrade gracefully to placeholders instead of crashing
          the app (per the MVP image-handling contract).
    """
    try:
        data = api_client.get_frame_image(video_id, frame_index)
    except api_client.ApiError:
        return None
    if data is None or not _looks_like_image(data):
        return None
    return data


def hit_image(record: dict[str, Any]) -> bytes | None:
    """Image bytes for one search-hit / evidence record."""
    ordinal = record.get("keyframe_ordinal")
    if ordinal is None:
        return None
    return image_bytes_or_none(record["video_id"], int(record["frame_index"]))


def render_hit_grid(
    ss: MutableMapping[str, Any],
    records: list[dict[str, Any]],
    *,
    columns: int = 4,
) -> dict[str, Any] | None:
    """Render the rank-preserving candidate grid; return a clicked record.

    Rank order is preserved exactly; rank stays visible, score is secondary.
    Pagination ("Show more") is grid-local.
    """
    from app.state import bump_result_limit, select_hit

    if not records:
        st.info("Run a search to see ranked candidates.")
        return None

    limit = int(ss.get("result_limit", 24))
    visible = records[:limit]

    clicked: dict[str, Any] | None = None
    for row_start in range(0, len(visible), columns):
        row = visible[row_start : row_start + columns]
        cols = st.columns(columns)
        for col, record in zip(cols, row):
            with col:
                image = hit_image(record)
                if image is not None:
                    st.image(image, width=THUMBNAIL_WIDTH)
                else:
                    st.caption("image unavailable")
                pts_time = record.get("pts_time")
                time_text = f" @ {pts_time:.1f}s" if isinstance(pts_time, (int, float)) else ""
                st.markdown(
                    f"**#{record['rank']}** · {record['video_id']} · f{record['frame_index']}"
                    f"{time_text} · {record['score']:.3f}"
                )
                if st.button("Inspect", key=f"hit_{record['video_id']}_{record['frame_index']}"):
                    clicked = record

    remaining = len(records) - limit
    if remaining > 0:
        st.button(f"Show more ({remaining} hidden)", on_click=bump_result_limit, args=(ss,))
    if clicked is not None:
        select_hit(
            ss,
            video_id=clicked["video_id"],
            frame_index=clicked["frame_index"],
            rank=clicked.get("rank"),
            score=clicked.get("score"),
            keyframe_ordinal=clicked.get("keyframe_ordinal"),
            pts_time=clicked.get("pts_time"),
        )
    return clicked


def render_inspector(record: dict[str, Any] | None) -> None:
    """Render the shared Frame Inspector for the current selection."""
    st.subheader("Inspector")
    if record is None:
        st.info("Select a candidate, window frame, or temporal neighbor.")
        return
    image = image_bytes_or_none(record["video_id"], int(record["frame_index"]))
    if image is None:
        st.warning("Image unavailable for this frame.")
    else:
        st.image(image, width=INSPECTOR_WIDTH)
    rank = record.get("rank")
    lines = [
        f"**video**: {record['video_id']}",
        f"**frame_index**: {record['frame_index']}",
        f"**keyframe_ordinal**: {record.get('keyframe_ordinal', '—')}",
        f"**pts_time**: {record.get('pts_time', '—')}",
    ]
    if rank is not None:
        score = record.get("score")
        lines.append(
            f"**rank**: {rank}" + (f"  |  **score**: {score:.4f}" if isinstance(score, (int, float)) else "")
        )
    st.markdown("\n\n".join(lines))


@st.fragment
def render_temporal_strip(
    ss: MutableMapping[str, Any],
    *,
    radius: int = 400,
    page_size: int = 5,
) -> None:
    """Temporal neighbors of the selected frame via ``GET .../frames``.

    The neighbor window is fetched once per selection; ◀▶ paging reruns the
    fragment locally. Picking a neighbor updates the shared selection and
    reruns the app so the Inspector follows.
    """
    from app.state import select_hit

    video_id = ss.get("selected_video_id")
    center = ss.get("selected_frame_index")
    if not video_id or center is None:
        st.caption("Select a candidate to browse its temporal neighborhood.")
        return

    try:
        payload = api_client.get_frames(
            video_id, center=int(center), radius=radius, limit=40
        )
    except api_client.ApiError as exc:
        st.caption(f"temporal neighbors unavailable ({exc})")
        return
    frames = payload.get("frames", [])
    if not frames:
        st.caption("No indexed keyframes within range.")
        return

    positions = [i for i, fr in enumerate(frames) if fr["frame_index"] == int(center)]
    pos = positions[0] if positions else 0
    offset = int(ss.get("temporal_offset", 0))
    start = max(0, min(pos - offset, max(0, len(frames) - page_size)))
    window = frames[start : start + page_size]

    total = payload.get("total", len(frames))
    st.caption(f"Indexed neighbors of frame {center} (±{radius}); video has {total} keyframes")
    cols = st.columns(len(window))
    for col, frame in zip(cols, window):
        with col:
            image = image_bytes_or_none(video_id, int(frame["frame_index"]))
            if image is not None:
                st.image(image, width=STRIP_WIDTH)
            else:
                st.caption("n/a")
            is_selected = frame["frame_index"] == int(center)
            st.markdown("**here**" if is_selected else str(frame["frame_index"]))
            if not is_selected and st.button(
                "View", key=f"temporal_{video_id}_{frame['frame_index']}"
            ):
                # A temporal neighbor is no longer the ranked hit.
                select_hit(ss, video_id=video_id, frame_index=int(frame["frame_index"]))
                st.rerun()

    nav_left, nav_right = st.columns(2)
    with nav_left:
        if start > 0 and st.button("◀ earlier", key="temporal_prev"):
            ss["temporal_offset"] = offset + page_size
            st.rerun(scope="fragment")
    with nav_right:
        if start + page_size < len(frames) and st.button("later ▶", key="temporal_next"):
            ss["temporal_offset"] = max(0, offset - page_size)
            st.rerun(scope="fragment")


def render_evidence_thumbnail(item: dict[str, Any]) -> None:
    """One QA evidence thumbnail + caption."""
    image = hit_image(item)
    if image is not None:
        st.image(image, width=THUMBNAIL_WIDTH)
    else:
        st.caption("image unavailable")
    st.caption(f"{item['video_id']} · f{item['frame_index']}")
