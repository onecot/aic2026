"""Session-state schema and pure interaction logic for the inspection UI.

This module deliberately imports **no Streamlit and no retrieval runtime**:
every function takes the session mapping (``st.session_state`` in the app,
a plain dict in tests). Result slots hold **HTTP payloads** exactly as
returned by ``app.api_client`` (dicts), never runtime objects.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

TASKS = ("KIS", "QA", "TRAKE")

#: Session-state schema. Every key is initialized explicitly so reruns never
#: see missing keys; values are small JSON-ish structures, never bulk data.
STATE_DEFAULTS: dict[str, Any] = {
    "task": "KIS",
    "query_text": "",
    # One HTTP-result slot per task; a new search replaces all three
    # atomically (old results belong to the old query).
    "kis_results": None,  # {"query", "results": [hit...]}
    "qa_results": None,  # same shape as kis_results (general frame search)
    "trake_results": None,  # {"query", "results": [], "events": [...]}
    # Shared Frame Inspector selection.
    "selected_video_id": None,
    "selected_frame_index": None,
    "selected_rank": None,
    "selected_score": None,
    "selected_keyframe_ordinal": None,
    "selected_pts_time": None,
    # Per-task human actions.
    "kis_selected": None,  # final KIS frame record (dict)
    "qa_evidence": [],  # list[dict] of evidence records (deduplicated)
    "qa_answer": "",  # manual answer text (editable)
    "trake_selections": {},  # event position -> selection dict
    # Grid-local pagination and temporal-strip offset.
    "result_limit": 24,
    "temporal_offset": 0,
}


def init_state(ss: MutableMapping[str, Any]) -> None:
    """Initialize every session-state key explicitly (idempotent).

    Expected input:
        - ss: ``st.session_state`` or any mutable mapping.

    Returns:
        - None.

    Raises:
        - No exceptions.
    """
    for key, value in STATE_DEFAULTS.items():
        if key not in ss:
            ss[key] = value


def reset_for_new_search(ss: MutableMapping[str, Any]) -> None:
    """Reset query-scoped state atomically before storing new results.

    Keeps the active task and typed query text; clears all result slots and
    every human action belonging to the previous query. The backend process
    (and its loaded model/index/catalog) is untouched.
    """
    ss["kis_results"] = None
    ss["qa_results"] = None
    ss["trake_results"] = None
    ss["selected_video_id"] = None
    ss["selected_frame_index"] = None
    ss["selected_rank"] = None
    ss["selected_score"] = None
    ss["selected_keyframe_ordinal"] = None
    ss["selected_pts_time"] = None
    ss["kis_selected"] = None
    ss["qa_evidence"] = []
    ss["qa_answer"] = ""
    ss["trake_selections"] = {}
    ss["result_limit"] = STATE_DEFAULTS["result_limit"]
    ss["temporal_offset"] = 0


def store_search_result(
    ss: MutableMapping[str, Any],
    task: str,
    payload: dict[str, Any],
) -> None:
    """Store one task's HTTP search payload after an atomic reset.

    Raises:
        - ValueError: for an unknown task.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    reset_for_new_search(ss)
    ss[f"{task.lower()}_results"] = payload


def select_hit(
    ss: MutableMapping[str, Any],
    *,
    video_id: str,
    frame_index: int,
    rank: int | None = None,
    score: float | None = None,
    keyframe_ordinal: int | None = None,
    pts_time: float | None = None,
) -> None:
    """Point the shared Frame Inspector at one candidate."""
    ss["selected_video_id"] = video_id
    ss["selected_frame_index"] = int(frame_index)
    ss["selected_rank"] = rank
    ss["selected_score"] = score
    ss["selected_keyframe_ordinal"] = keyframe_ordinal
    ss["selected_pts_time"] = pts_time
    ss["temporal_offset"] = 0  # re-center temporal browsing


def selected_record(ss: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Build full inspector metadata for the current selection.

    Hits from ``POST /api/search`` already carry ``keyframe_ordinal`` /
    ``pts_time``, so this is pure state assembly; returns ``None`` when
    nothing is selected.
    """
    video_id = ss.get("selected_video_id")
    frame_index = ss.get("selected_frame_index")
    if not video_id or frame_index is None:
        return None
    return {
        "video_id": video_id,
        "frame_index": int(frame_index),
        "keyframe_ordinal": ss.get("selected_keyframe_ordinal"),
        "pts_time": ss.get("selected_pts_time"),
        "rank": ss.get("selected_rank"),
        "score": ss.get("selected_score"),
    }


# --------------------------------------------------------------------- KIS


def set_kis_selection(ss: MutableMapping[str, Any], record: dict[str, Any]) -> None:
    """Store the final human-selected KIS frame."""
    ss["kis_selected"] = dict(record)


# ---------------------------------------------------------------------- QA


def qa_add_evidence(ss: MutableMapping[str, Any], record: dict[str, Any]) -> bool:
    """Add the inspected frame as QA evidence (idempotent per frame).

    Returns:
        - True when the evidence list changed.
    """
    key = (record["video_id"], int(record["frame_index"]))
    existing = {(e["video_id"], int(e["frame_index"])) for e in ss["qa_evidence"]}
    if key in existing:
        return False
    evidence = {
        k: record[k]
        for k in ("video_id", "frame_index", "keyframe_ordinal")
        if record.get(k) is not None
    }
    if record.get("pts_time") is not None:
        evidence["pts_time"] = record["pts_time"]
    ss["qa_evidence"] = [*ss["qa_evidence"], evidence]
    return True


def qa_remove_evidence(ss: MutableMapping[str, Any], index: int) -> bool:
    """Remove one evidence entry by position.

    Returns:
        - True when the evidence list changed.
    """
    evidence = list(ss["qa_evidence"])
    if 0 <= index < len(evidence):
        del evidence[index]
        ss["qa_evidence"] = evidence
        return True
    return False


# ------------------------------------------------------------------- TRAKE


def trake_set_window_selection(
    ss: MutableMapping[str, Any],
    event_position: int,
    *,
    video_id: str,
    start: int,
    end: int,
    frame_index: int,
) -> None:
    """Record the human-selected window/frame for one event."""
    selections = dict(ss["trake_selections"])
    selections[int(event_position)] = {
        "video_id": video_id,
        "start": int(start),
        "end": int(end),
        "frame_index": int(frame_index),
    }
    ss["trake_selections"] = selections


def bump_result_limit(ss: MutableMapping[str, Any], step: int = 24, cap: int = 200) -> None:
    """Grow the visible-results window (grid-local pagination)."""
    ss["result_limit"] = min(int(ss.get("result_limit", 24)) + step, cap)
