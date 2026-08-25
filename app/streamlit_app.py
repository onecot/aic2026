"""AIC2026 interactive human-inspection workspace (Streamlit MVP, HTTP client).

The UI is a **pure API client**: it never loads the Jina model, the index,
a FrameCatalog, or any VLM. All capability comes from the FastAPI backend
(``uvicorn aic2026.server:app``) through :mod:`app.api_client`.

Backend URL: ``AIC_API_URL`` (default ``http://127.0.0.1:8000``).

Startup gate:

- READY       -> full search workspace.
- STARTING    -> "Retrieval backend is starting..." + Retry.
- UNAVAILABLE -> "Retrieval backend is unavailable." + start hint + Retry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app import api_client, components
from app.state import (
    TASKS,
    init_state,
    qa_add_evidence,
    qa_remove_evidence,
    reset_for_new_search,
    selected_record,
    select_hit,
    set_kis_selection,
    store_search_result,
    trake_set_window_selection,
)

st.set_page_config(page_title="AIC2026 Inspection", page_icon="🔎", layout="wide")

init_state(st.session_state)


# ------------------------------------------------------------------ actions


def _on_search() -> None:
    """Callback: run the active task's search through the backend.

    Callbacks execute before the script reruns; the result swap is atomic
    (reset + store). Connection failures land in ``search_error`` and keep
    prior results intact.
    """
    task = st.session_state["task"]
    text = (st.session_state.get("query_input") or "").strip()
    if not text:
        return
    api_task = "trake" if task == "TRAKE" else "kis"  # QA evidence uses general search
    try:
        payload = api_client.search(text, task=api_task)
    except api_client.ApiError as exc:
        st.session_state["search_error"] = str(exc)
        return
    st.session_state["search_error"] = None
    store_search_result(st.session_state, task, payload)


def _on_select_kis_frame() -> None:
    """KIS action: mark the currently inspected frame as the final pick."""
    record = selected_record(st.session_state)
    if record is not None:
        set_kis_selection(st.session_state, record)


# ------------------------------------------------------------- health gate


def _backend_gate() -> bool:
    """Render backend status; return True when the workspace may proceed.

    Shows one of READY / STARTING / UNAVAILABLE per spec; never exposes a
    traceback. Includes an explicit Retry action (no auto-spawn).
    """
    try:
        status = api_client.health().get("status")
    except api_client.ApiError:
        status = None
    if status == "ready":
        return True
    st.title("AIC2026 — Human Inspection Workspace")
    if status == "starting":
        st.warning("Retrieval backend is starting...")
    else:
        st.error("Retrieval backend is unavailable.")
        st.caption(f"Start the FastAPI server and retry (expected at {api_client.base_url()}).")
    st.button("Retry", key="retry_backend", on_click=lambda: None, type="primary")
    st.stop()


if not _backend_gate():
    st.stop()

# ------------------------------------------------------------------ helpers


def _render_trake_results() -> None:
    """Event-centric rendering of the TRAKE HTTP payload (never flattened)."""
    payload = st.session_state["trake_results"]
    for position, event in enumerate(payload.get("events", [])):
        st.markdown(f"### Event {position + 1} — {event['event_text']}")
        windows = sorted(event.get("windows", []), key=lambda w: w["rank"])[:5]
        if not windows:
            st.caption("No candidate windows.")
            continue
        chosen = st.session_state["trake_selections"].get(position)
        for window in windows:
            window_key = (
                f"trake_{position}_{window['video_id']}_"
                f"{window['frame_start']}_{window['frame_end']}"
            )
            reps = _window_representatives(window, limit=4)
            head, strip, action = st.columns([3, 5, 2])
            with head:
                st.markdown(
                    f"`{window['video_id']}` frames "
                    f"**{window['frame_start']}–{window['frame_end']}**"
                )
                st.caption(f"rank {window['rank']} · score {window['score']:.3f}")
            with strip:
                rep_cols = st.columns(len(reps) or 1)
                for col, frame in zip(rep_cols, reps):
                    with col:
                        image = components.image_bytes_or_none(
                            frame["video_id"], int(frame["frame_index"])
                        )
                        if image is not None:
                            st.image(image, width=110)
                        else:
                            st.caption("n/a")
                        if st.button(
                            "inspect",
                            key=f"{window_key}_rep_{frame['frame_index']}",
                        ):
                            select_hit(
                                st.session_state,
                                video_id=frame["video_id"],
                                frame_index=int(frame["frame_index"]),
                                rank=window["rank"],
                                score=window["score"],
                                keyframe_ordinal=(
                                    int(frame["keyframe_ordinal"])
                                    if frame.get("keyframe_ordinal") is not None else None
                                ),
                                pts_time=(
                                    float(frame["pts_time"])
                                    if frame.get("pts_time") is not None else None
                                ),
                            )
                            st.rerun()
            with action:
                already_chosen = bool(
                    chosen
                    and chosen["video_id"] == window["video_id"]
                    and chosen["start"] == window["frame_start"]
                )
                label = "Selected ✓" if already_chosen else "Select window"
                if st.button(label, key=f"{window_key}_select", disabled=already_chosen):
                    anchor = reps[0]["frame_index"] if reps else int(window["frame_start"])
                    trake_set_window_selection(
                        st.session_state,
                        position,
                        video_id=window["video_id"],
                        start=int(window["frame_start"]),
                        end=int(window["frame_end"]),
                        frame_index=int(anchor),
                    )
                    st.rerun()
        st.divider()


def _window_representatives(window: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    """Representative indexed frames inside one TRAKE event window.

    Uses the around-center browsing mode of the existing frames endpoint;
    no temporal logic lives client-side. Each returned record carries its
    window's ``video_id`` (frame rows are video-scoped on the wire).
    """
    start, end = int(window["frame_start"]), int(window["frame_end"])
    center = (start + end) // 2
    radius = max(1, (end - start) // 2)
    try:
        payload = api_client.get_frames(
            window["video_id"], center=center, radius=radius, limit=limit
        )
    except api_client.ApiError:
        return []
    frames = []
    for frame in payload.get("frames", []):
        enriched = {**frame, "video_id": window["video_id"]}
        frames.append(enriched)
    return frames


def _render_trake_actions() -> None:
    """Summary of per-event human selections."""
    selections = st.session_state["trake_selections"]
    if not selections:
        st.caption("Select one candidate window per event above.")
        return
    for position in sorted(selections):
        picked = selections[position]
        st.success(
            f"Event {position + 1}: {picked['video_id']} "
            f"frames {picked['start']}–{picked['end']} "
            f"(anchor f{picked['frame_index']})"
        )


# -------------------------------------------------------------------- chrome

st.title("AIC2026 — Human Inspection Workspace")

with st.sidebar:
    st.header("Task")
    st.radio("Active task", list(TASKS), key="task")
    st.divider()
    st.success("Backend: READY", icon="✅")
    st.caption("Score = ranking score, not calibrated probability.")
    st.caption(
        "QA note: answer generation is not available through the current "
        "HTTP API; enter answers manually."
    )
    if st.button("Reset workspace"):
        reset_for_new_search(st.session_state)
        st.rerun()

# --------------------------------------------------------------- query bar

query_cols = st.columns([4, 1])
query_cols[0].text_input(
    "Query",
    key="query_input",
    placeholder="e.g. một người đang nấu ăn trong bếp",
)
query_cols[1].markdown("&nbsp;")
query_cols[1].button("Search", key="search_button", on_click=_on_search)

if error := st.session_state.get("search_error"):
    st.error(f"Search failed: {error}")

task = st.session_state["task"]
results_slot = st.session_state[f"{task.lower()}_results"]

# ------------------------------------------------------------- main split

left, right = st.columns([3, 2])

with right:
    current = selected_record(st.session_state)
    components.render_inspector(current)

with left:
    st.subheader(f"Results · {task}")
    if results_slot is None:
        st.info("Enter a query and press Search.")
    elif task == "TRAKE":
        _render_trake_results()
    else:
        records = results_slot.get("results", [])
        clicked = components.render_hit_grid(st.session_state, records)
        if clicked is not None:
            st.rerun()  # propagate selection to the Inspector zone

# ------------------------------------------------------ temporal navigation

st.divider()
components.render_temporal_strip(st.session_state)

# ------------------------------------------------------- task action panel

st.divider()
st.subheader(f"Action · {task}")

if task == "KIS":
    if current is not None:
        st.button(
            "Select this frame",
            key="kis_select_button",
            on_click=_on_select_kis_frame,
            type="primary",
        )
    picked = st.session_state["kis_selected"]
    if picked:
        st.success(
            f"Selected frame: {picked['video_id']} @ f{picked['frame_index']} "
            f"(ordinal {picked.get('keyframe_ordinal', '—')})"
        )
    else:
        st.caption("No final frame selected yet.")

elif task == "QA":
    evidence = st.session_state["qa_evidence"]
    evidence_col, answer_col = st.columns([2, 2])
    with evidence_col:
        st.markdown("**Selected evidence**")
        if not evidence:
            st.caption("Inspect a frame, then add it as evidence.")
        for index, item in enumerate(evidence):
            item_cols = st.columns([3, 1])
            with item_cols[0]:
                components.render_evidence_thumbnail(item)
            with item_cols[1]:
                if st.button("✕", key=f"qa_ev_rm_{index}"):
                    qa_remove_evidence(st.session_state, index)
                    st.rerun()
    with answer_col:
        if current is not None:
            if st.button("Add inspected frame as evidence", key="qa_ev_add"):
                if qa_add_evidence(st.session_state, current):
                    st.rerun()
        st.text_area("Answer", key="qa_answer", height=96)
        st.button("Generate answer (unavailable via HTTP API)", key="qa_gen_button", disabled=True)
        st.caption("Manual entry above; generation returns with a QA API endpoint.")

elif task == "TRAKE":
    _render_trake_actions()

st.caption("MVP: selections live in session state only; nothing is persisted.")