"""Focused tests for the inspection UI and its HTTP client.

Layers tested:

1. ``app.state`` pure interaction logic (Streamlit-free, always runs).
2. ``app.api_client`` against a real FastAPI backend running on a local
   ephemeral port with injected stub services (no model loads).
3. Full Streamlit flows through ``streamlit.testing.v1.AppTest`` pointed at
   that same backend (``AIC_API_URL``).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import pytest

from app.state import (
    STATE_DEFAULTS,
    bump_result_limit,
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

CATALOG_VIDEO = "L21_V001"


# ------------------------------------------------------------- state logic


def _fresh() -> OrderedDict:
    ss: OrderedDict = OrderedDict()
    init_state(ss)
    return ss


def test_init_state_fills_every_default_exactly_once() -> None:
    ss = _fresh()
    assert set(ss) == set(STATE_DEFAULTS)
    ss["query_text"] = "changed"
    init_state(ss)
    assert ss["query_text"] == "changed"  # idempotent, never overwrites


def test_store_search_resets_query_scoped_state_atomically() -> None:
    ss = _fresh()
    select_hit(ss, video_id=CATALOG_VIDEO, frame_index=100, rank=1, score=0.5)
    set_kis_selection(ss, {"video_id": CATALOG_VIDEO, "frame_index": 100})
    qa_add_evidence(ss, {"video_id": CATALOG_VIDEO, "frame_index": 100, "keyframe_ordinal": 1})
    ss["qa_answer"] = "something"
    trake_set_window_selection(ss, 0, video_id=CATALOG_VIDEO, start=1, end=9, frame_index=1)
    ss["result_limit"] = 100

    store_search_result(ss, "QA", {"query": "x", "results": []})

    assert ss["kis_results"] is None and ss["trake_results"] is None
    assert ss["qa_results"] is not None
    assert ss["selected_video_id"] is None and ss["selected_frame_index"] is None
    assert ss["kis_selected"] is None
    assert ss["qa_evidence"] == [] and ss["qa_answer"] == ""
    assert ss["trake_selections"] == {}
    assert ss["result_limit"] == STATE_DEFAULTS["result_limit"]


def test_store_search_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        store_search_result(_fresh(), "NOPE", {})


def test_select_hit_recenters_temporal_offset() -> None:
    ss = _fresh()
    ss["temporal_offset"] = 10
    select_hit(ss, video_id=CATALOG_VIDEO, frame_index=40)
    assert ss["selected_video_id"] == CATALOG_VIDEO
    assert ss["temporal_offset"] == 0


def test_selected_record_assembles_from_state() -> None:
    ss = _fresh()
    assert selected_record(ss) is None
    select_hit(ss, video_id=CATALOG_VIDEO, frame_index=40, rank=2, score=0.25)
    record = selected_record(ss)
    assert record is not None
    assert record["frame_index"] == 40 and record["rank"] == 2


def test_kis_selection_round_trip_is_isolated() -> None:
    ss = _fresh()
    record = {"video_id": CATALOG_VIDEO, "frame_index": 100, "keyframe_ordinal": 7}
    set_kis_selection(ss, record)
    record["frame_index"] = 999
    assert ss["kis_selected"]["frame_index"] == 100


def test_qa_evidence_add_is_deduplicated_and_remove_works() -> None:
    ss = _fresh()
    record = {"video_id": CATALOG_VIDEO, "frame_index": 100, "keyframe_ordinal": 7}
    assert qa_add_evidence(ss, record) is True
    assert qa_add_evidence(ss, dict(record)) is False
    assert len(ss["qa_evidence"]) == 1
    assert qa_remove_evidence(ss, 0) is True
    assert ss["qa_evidence"] == []
    assert qa_remove_evidence(ss, 5) is False


def test_trake_selection_per_event_with_overwrite() -> None:
    ss = _fresh()
    trake_set_window_selection(ss, 0, video_id="A", start=10, end=50, frame_index=20)
    trake_set_window_selection(ss, 1, video_id="B", start=5, end=9, frame_index=6)
    assert ss["trake_selections"][1]["video_id"] == "B"
    trake_set_window_selection(ss, 0, video_id="C", start=1, end=2, frame_index=1)
    assert len(ss["trake_selections"]) == 2
    assert ss["trake_selections"][0]["video_id"] == "C"


def test_bump_result_limit_caps() -> None:
    ss = _fresh()
    bump_result_limit(ss)
    assert ss["result_limit"] == 48
    ss["result_limit"] = 199
    bump_result_limit(ss)
    assert ss["result_limit"] == 200


# ------------------------------------------------- live backend fixture


class _StubEngine:
    """Deterministic placeholder engine over the synthetic smoke index."""

    def __init__(self, index_dir: Path) -> None:
        from aic2026.qa.qa_smoke import QaPlaceholderEncoder, smoke_embedding_space
        from aic2026.search import SearchEngine

        self._inner = SearchEngine.from_parts(
            QaPlaceholderEncoder(smoke_embedding_space(), 8), index_dir, top_k=10
        )

    @property
    def embedding_space(self):
        return self._inner.embedding_space

    @property
    def retriever(self):
        return self._inner.retriever

    @property
    def manifest(self):
        return self._inner.manifest

    def encode_text(self, text: str):
        return self._inner.encode_text(text)

    def encode_query(self, text: str):
        return self._inner.encode_query(text)

    def search_text(self, text: str, *, top_k=None):
        return self._inner.search_text(text, top_k=top_k)

    def search_embedding(self, embedding, *, top_k=None):
        return self._inner.search_embedding(embedding, top_k=top_k)


@pytest.fixture(scope="module")
def backend() -> Iterator[str]:
    """Run one FastAPI backend (stub services) on an ephemeral local port."""
    import uvicorn
    from fastapi.testclient import TestClient

    from aic2026.api import create_app
    from aic2026.frame_catalog import FrameCatalog
    from aic2026.qa.qa_smoke import QaSmokeConfig, build_synthetic_index
    from aic2026.artifact_io import write_ndjson
    from aic2026.canonical import KeyframeKey, KeyframeTiming

    tmp = Path(__file__).parent / "_ui_backend_fixtures"
    tmp.mkdir(exist_ok=True)
    index = build_synthetic_index(
        tmp / "index",
        QaSmokeConfig(evaluation_set_path=tmp / "eval.csv", work_dir=tmp,
                      video_ids=("L21_V001", "L21_V002", "L21_V003")),
    )
    frames_path = tmp / "keyframes" / "frames.ndjson"
    if not frames_path.is_file():
        records = []
        images = tmp / "keyframes" / "images" / CATALOG_VIDEO
        images.mkdir(parents=True, exist_ok=True)
        for ordinal in range(1, 11):
            frame_index = (ordinal - 1) * 10
            records.append({
                "keyframe": {"video_id": {"value": CATALOG_VIDEO}, "keyframe_ordinal": ordinal},
                "pts_time": frame_index / 30.0,
                "fps": 30.0,
                "frame_index": frame_index,
            })
            (images / f"{ordinal:06d}.jpg").write_bytes(b"\xff\xd8\xff\xe0stub")
        write_ndjson(frames_path, records)
    catalog = FrameCatalog.from_keyframes_root(tmp / "keyframes")
    app = create_app(engine=_StubEngine(index), catalog=catalog,
                     keyframes_root=tmp / "keyframes")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time as _time

    deadline = _time.time() + 10
    last_error: Exception | None = None
    while _time.time() < deadline:
        try:
            if _probe(port):
                break
        except Exception as exc:  # noqa: BLE001 - retry until ready/deadline
            last_error = exc
        _time.sleep(0.05)
    else:
        raise RuntimeError(f"backend did not become ready: {last_error}")

    patcher = pytest.MonkeyPatch()
    patcher.setenv("AIC_API_URL", f"http://127.0.0.1:{port}")
    yield f"http://127.0.0.1:{port}"
    patcher.undo()
    server.should_exit = True
    thread.join(timeout=5)


def _probe(port: int) -> bool:
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
        return r.status == 200


streamlit = pytest.importorskip("streamlit")


# --------------------------------------------------------- api_client tests


def test_client_health_ready(backend: str) -> None:
    from app import api_client

    payload = api_client.health()
    assert payload == {"status": "ready", "model": True, "index": True, "frame_catalog": True}


def test_client_search_returns_hits_with_full_identity(backend: str) -> None:
    from app import api_client

    body = api_client.search("a red boat on the water", limit=5)
    assert len(body["results"]) == 5
    # additive telemetry tolerated/preserved by the client layer
    assert body["meta"]["task"] == "kis"
    assert isinstance(body["meta"]["timing"]["total_ms"], float)
    hit = body["results"][0]
    assert set(hit) >= {"rank", "video_id", "frame_index", "keyframe_ordinal", "pts_time", "score"}
    ranks = [h["rank"] for h in body["results"]]
    assert ranks == [1, 2, 3, 4, 5]


def test_client_trake_search_returns_events(backend: str) -> None:
    from app import api_client

    body = api_client.search("(1) a man enters the room (2) lights on", task="trake", limit=4)
    assert [e["event_id"] for e in body["events"]] == ["E1", "E2"]
    for event in body["events"]:
        assert event["predictions"]


def test_client_frame_image_bytes_and_404(backend: str) -> None:
    from app import api_client

    meta = api_client.get_video(CATALOG_VIDEO)
    first = api_client.get_frames(CATALOG_VIDEO, offset=0, limit=1)["frames"][0]
    image = api_client.get_frame_image(CATALOG_VIDEO, int(first["frame_index"]))
    assert isinstance(image, bytes) and image[:2] == b"\xff\xd8"
    assert api_client.get_frame_image(CATALOG_VIDEO, 999999) is None


def test_client_unreachable_backend_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import api_client

    monkeypatch.setenv("AIC_API_URL", "http://127.0.0.1:59999")  # closed port
    with pytest.raises(api_client.ApiError, match="cannot reach backend"):
        api_client.health()


def test_unavailable_backend_shows_retry_state(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pointing the UI at a dead port renders the unavailable gate + Retry."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("AIC_API_URL", "http://127.0.0.1:59999")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("unavailable" in e.value for e in at.error)
    assert any(b.key == "retry_backend" for b in at.button)
    # full workspace is hidden behind the gate
    assert not any(b.key == "search_button" for b in at.button)


def test_client_maps_structured_503_to_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend 503 carrying the structured health payload maps to 'starting'."""
    from app import api_client

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url, 503, "Service Unavailable", hdrs={}, fp=json_file
        )

    import io

    json_file = io.BytesIO(json.dumps({"status": "starting"}).encode())
    monkeypatch.setattr(api_client.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(api_client.ApiError) as excinfo:
        api_client.health()
    assert excinfo.value.status == 503
    assert str(excinfo.value) == "starting"


# ------------------------------------------------------ AppTest UI wiring


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _boot(backend: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def test_app_shows_ready_and_boots_without_exception(backend: str) -> None:
    at = _boot(backend)
    assert not at.exception
    assert at.radio(key="task").value == "KIS"
    assert any("READY" in c.value for c in at.success)


def test_kis_flow_search_select_and_pick(backend: str) -> None:
    at = _boot(backend)
    at.text_input(key="query_input").set_value("a red boat on the water").run()
    at.button(key="search_button").click().run()
    assert not at.exception
    results = at.session_state["kis_results"]
    assert results is not None and len(results["results"]) > 0

    hit_key = next(b.key for b in at.button if b.key and b.key.startswith("hit_"))
    at.button(key=hit_key).click().run()
    assert at.session_state["selected_video_id"] is not None

    at.button(key="kis_select_button").click().run()
    picked = at.session_state["kis_selected"]
    assert picked is not None
    assert picked["video_id"] == at.session_state["selected_video_id"]
    assert "Selected frame" in at.success[0].value


def test_qa_flow_evidence_and_manual_answer(backend: str) -> None:
    at = _boot(backend)
    at.radio(key="task").set_value("QA").run()
    at.text_input(key="query_input").set_value("what ingredient is red?").run()
    at.button(key="search_button").click().run()
    assert not at.exception
    assert at.session_state["qa_results"] is not None

    hit_key = next(b.key for b in at.button if b.key and b.key.startswith("hit_"))
    at.button(key=hit_key).click().run()
    before = len(at.session_state["qa_evidence"])
    at.button(key="qa_ev_add").click().run()
    assert len(at.session_state["qa_evidence"]) == before + 1

    generate_button = next(b for b in at.button if b.key == "qa_gen_button")
    assert generate_button.disabled  # generation disabled without a QA endpoint

    at.text_area(key="qa_answer").set_value("red chili peppers").run()
    assert at.session_state["qa_answer"] == "red chili peppers"


def test_trake_flow_two_events_selectable(backend: str) -> None:
    at = _boot(backend)
    at.radio(key="task").set_value("TRAKE").run()
    at.text_input(key="query_input").set_value(
        "(1) a man enters the room (2) he turns on the lights"
    ).run()
    at.button(key="search_button").click().run()
    assert not at.exception
    events = at.session_state["trake_results"]["events"]
    assert [e["event_id"] for e in events] == ["E1", "E2"]

    first_per_event: dict[int, str] = {}
    for b in at.button:
        if b.key and b.key.endswith("_select"):
            position = int(b.key.split("_")[1])
            first_per_event.setdefault(position, b.key)
    assert set(first_per_event) == {0, 1}
    at.button(key=first_per_event[0]).click().run()
    assert 0 in at.session_state["trake_selections"]
    at.button(key=first_per_event[1]).click().run()
    assert {0, 1} <= set(at.session_state["trake_selections"])
