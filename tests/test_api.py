"""Integration tests for the minimal HTTP API (synthetic fixtures only).

The suite builds a tiny placeholder search engine (deterministic hash
encoder over a synthetic index) and a tiny frames.ndjson + images fixture,
so no real model or large runtime asset is required. One optional smoke
test exercises the production assets when they are present locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="httpx is required for the test client")

from fastapi.testclient import TestClient

from aic2026.api import create_app
from aic2026.canonical import KeyframeKey, KeyframeTiming
from aic2026.artifact_io import write_ndjson
from aic2026.frame_catalog import FrameCatalog
from aic2026.qa.qa_smoke import (
    QaPlaceholderEncoder,
    QaSmokeConfig,
    build_synthetic_index,
    smoke_embedding_space,
)
from aic2026.search import SearchEngine

CATALOG_VIDEO = "L21_V001"


def _timing(ordinal: int, frame_index: int) -> KeyframeTiming:
    """Build one canonical timing record for the catalog fixture video."""
    return KeyframeTiming(
        keyframe=KeyframeKey.from_dict(
            {"video_id": {"value": CATALOG_VIDEO}, "keyframe_ordinal": ordinal}
        ),
        pts_time=frame_index / 30.0,
        fps=30.0,
        frame_index=frame_index,
    )


@pytest.fixture()
def keyframes_root(tmp_path: Path) -> Path:
    """Tiny materialized keyframes artifact: frames.ndjson + images/."""
    records = [_timing(i, i * 100).to_dict() for i in range(1, 11)]
    write_ndjson(tmp_path / "frames.ndjson", records)
    images = tmp_path / "images" / CATALOG_VIDEO
    images.mkdir(parents=True)
    for i in range(1, 11):
        (images / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return tmp_path


@pytest.fixture()
def engine(tmp_path: Path) -> SearchEngine:
    """Deterministic placeholder engine over a synthetic index."""
    config = QaSmokeConfig(
        evaluation_set_path=tmp_path / "eval.csv",
        work_dir=tmp_path,
        video_ids=("L21_V001", "L21_V002", "L21_V003"),
    )
    index = build_synthetic_index(tmp_path / "index", config)
    encoder = QaPlaceholderEncoder(smoke_embedding_space(), 8)
    return SearchEngine.from_parts(encoder, index, top_k=10)


@pytest.fixture()
def client(engine: SearchEngine, keyframes_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient whose working directory is an empty scratch dir."""
    app = create_app(engine=engine, catalog=FrameCatalog.from_keyframes_root(keyframes_root), keyframes_root=keyframes_root)
    monkeypatch.chdir(tmp_path / "scratch" if False else tmp_path)
    return TestClient(app, raise_server_exceptions=True)


class TestSearchEndpoint:
    def test_search_returns_deterministic_ranked_candidates(self, client: TestClient) -> None:
        payload = {"query": "a red boat on the water", "limit": 5}
        first = client.post("/api/search", json=payload)
        second = client.post("/api/search", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        body = first.json()
        assert body["query"] == payload["query"]
        hits = body["results"]
        assert len(hits) == 5
        assert [h["rank"] for h in hits] == [1, 2, 3, 4, 5]
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)
        # every hit carries full frame identity for the UI
        for hit in hits:
            assert set(hit) == {"rank", "video_id", "frame_index", "keyframe_ordinal", "pts_time", "score"}
        assert hits == second.json()["results"]

    def test_search_rejects_invalid_body(self, client: TestClient) -> None:
        assert client.post("/api/search", json={"query": "", "limit": 5}).status_code == 422
        assert client.post("/api/search", json={"query": "x", "limit": 0}).status_code == 422

    def test_search_does_not_write_prediction_artifacts(self, client: TestClient, tmp_path: Path) -> None:
        before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
        response = client.post("/api/search", json={"query": "kitchen cooking", "limit": 3})
        assert response.status_code == 200
        after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
        assert before == after


class TestVideoEndpoints:
    def test_video_metadata(self, client: TestClient) -> None:
        response = client.get(f"/api/videos/{CATALOG_VIDEO}")
        assert response.status_code == 200
        assert response.json() == {
            "video_id": CATALOG_VIDEO,
            "frame_count": 10,
            "first_frame_index": 100,
            "last_frame_index": 1000,
        }

    def test_unknown_video_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/videos/L99_V999")
        assert response.status_code == 404

    def test_offset_limit_browsing(self, client: TestClient) -> None:
        response = client.get(f"/api/videos/{CATALOG_VIDEO}/frames?offset=2&limit=3")
        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == CATALOG_VIDEO
        assert body["total"] == 10
        assert [f["frame_index"] for f in body["frames"]] == [300, 400, 500]

    def test_center_radius_browsing_temporal_order(self, client: TestClient) -> None:
        response = client.get(f"/api/videos/{CATALOG_VIDEO}/frames?center=450&radius=150&limit=10")
        assert response.status_code == 200
        frames = response.json()["frames"]
        indices = [f["frame_index"] for f in frames]
        assert indices == sorted(indices)
        assert set(indices) <= {300, 400, 500, 600}
        assert all("image_path" not in f for f in frames)

    def test_frame_browsing_validates_pagination(self, client: TestClient) -> None:
        assert client.get(f"/api/videos/{CATALOG_VIDEO}/frames?limit=0").status_code == 400
        assert client.get(f"/api/videos/{CATALOG_VIDEO}/frames?offset=-1").status_code == 400


class TestFrameImageEndpoint:
    def test_image_returned_as_binary(self, client: TestClient) -> None:
        response = client.get(f"/api/frames/{CATALOG_VIDEO}/400")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")
        assert response.content.startswith(b"\xff\xd8\xff")

    def test_unknown_frame_returns_404(self, client: TestClient) -> None:
        # valid video but a frame index that is not an indexed keyframe
        assert client.get(f"/api/frames/{CATALOG_VIDEO}/401").status_code == 404
        assert client.get("/api/frames/L99_V999/100").status_code == 404

    def test_negative_frame_index_returns_400(self, client: TestClient) -> None:
        assert client.get(f"/api/frames/{CATALOG_VIDEO}/-5").status_code == 400


class TestInitialization:
    def test_catalog_loaded_once_and_shared(self, engine: SearchEngine, keyframes_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        built = {"count": 0}
        original_init = FrameCatalog.__init__

        def counting_init(self: FrameCatalog, path: object) -> None:
            built["count"] += 1
            original_init(self, path)

        monkeypatch.setattr(FrameCatalog, "__init__", counting_init)
        app = create_app(engine=engine, keyframes_root=keyframes_root)
        client = TestClient(app)
        assert client.get(f"/api/videos/{CATALOG_VIDEO}").status_code == 200
        assert client.get(f"/api/videos/{CATALOG_VIDEO}/frames?limit=1").status_code == 200
        assert built["count"] == 1  # construction only; requests reuse the instance

    def test_create_app_requires_engine_or_index(self, keyframes_root: Path) -> None:
        import pytest

        with pytest.raises(ValueError, match="index_artifact"):
            create_app(keyframes_root=keyframes_root)

    def test_services_usable_without_http(self, engine: SearchEngine, keyframes_root: Path) -> None:
        from aic2026.api import RetrievalService

        service = RetrievalService(engine)
        catalog = FrameCatalog.from_keyframes_root(keyframes_root)
        assert len(service.search("boat", limit=3)) == 3
        assert catalog.get_video(CATALOG_VIDEO).frame_count == 10


class TestRealDataSmoke:
    def test_real_assets_end_to_end(self) -> None:
        """Full read-only pass over the production assets (skipped when absent)."""
        index_dir = Path("data/interim/indexes/jina-v2-full")
        keyframes_dir = Path("data/keyframes/adaptive_sbd_v1")
        if not (index_dir / "manifest.json").is_file() or not (keyframes_dir / "frames.ndjson").is_file():
            pytest.skip("production assets not available on this machine")
        from fastapi.testclient import TestClient as _TC

        app = create_app(index_artifact=index_dir, keyframes_root=keyframes_dir)
        client = _TC(app)
        search = client.post("/api/search", json={"query": "một người đang nấu ăn trong bếp", "limit": 3})
        assert search.status_code == 200
        top = search.json()["results"][0]
        video_id = top["video_id"]
        meta = client.get(f"/api/videos/{video_id}")
        assert meta.status_code == 200
        center = top["frame_index"]
        browse = client.get(f"/api/videos/{video_id}/frames?center={center}&radius=2000&limit=5")
        assert browse.status_code == 200
        image = client.get(f"/api/frames/{video_id}/{center}")
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/")
