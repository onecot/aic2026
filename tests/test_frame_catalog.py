"""Contract tests for the FrameCatalog navigation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from aic2026.canonical import KeyframeKey, KeyframeTiming
from aic2026.frame_catalog import FrameCatalog

VIDEO_ID = "L21_V001"
VIDEO_ID_2 = "L22_V001"


def _timing(ordinal: int, frame_index: int, pts_time: float) -> KeyframeTiming:
    """Build one canonical timing record for the synthetic video."""
    return KeyframeTiming(
        keyframe=KeyframeKey.from_dict(
            {"video_id": {"value": VIDEO_ID}, "keyframe_ordinal": ordinal}
        ),
        pts_time=pts_time,
        fps=30.0,
        frame_index=frame_index,
    )


@pytest.fixture()
def frames_file(tmp_path: Path) -> Path:
    """Write a tiny canonical frames.ndjson: two videos, 6 + 2 records.

    L21_V001: ordinals 1..6 at frame_index 10,20,...,60 (evenly spaced).
    """
    from aic2026.artifact_io import write_ndjson

    records = [_timing(i, i * 10, i * 10 / 30.0).to_dict() for i in range(1, 7)]
    records.append(
        KeyframeTiming(
            keyframe=KeyframeKey.from_dict(
                {"video_id": {"value": VIDEO_ID_2}, "keyframe_ordinal": 1}
            ),
            pts_time=0.5,
            fps=25.0,
            frame_index=5,
        ).to_dict()
    )
    path = tmp_path / "frames.ndjson"
    write_ndjson(path, records)
    return path


@pytest.fixture()
def catalog(frames_file: Path) -> FrameCatalog:
    return FrameCatalog(frames_file)


class TestGetVideo:
    def test_summary_metadata(self, catalog: FrameCatalog) -> None:
        summary = catalog.get_video(VIDEO_ID)
        assert summary.video_id == VIDEO_ID
        assert summary.frame_count == 6
        assert summary.first_frame_index == 10
        assert summary.last_frame_index == 60
        assert summary.first_pts_time == pytest.approx(10 / 30.0)
        assert summary.last_pts_time == pytest.approx(60 / 30.0)

    def test_unknown_video_raises_key_error(self, catalog: FrameCatalog) -> None:
        with pytest.raises(KeyError, match="unknown video"):
            catalog.get_video("L99_V999")

    def test_video_ids_and_has_video(self, catalog: FrameCatalog) -> None:
        assert set(catalog.video_ids()) == {VIDEO_ID, VIDEO_ID_2}
        assert catalog.has_video(VIDEO_ID_2)
        assert not catalog.has_video("L99_V999")


class TestListVideoFrames:
    def test_full_page_is_temporally_ordered(self, catalog: FrameCatalog) -> None:
        page = catalog.list_video_frames(VIDEO_ID)
        indices = [t.frame_index for t in page]
        assert indices == [10, 20, 30, 40, 50, 60]

    def test_offset_and_limit(self, catalog: FrameCatalog) -> None:
        page = catalog.list_video_frames(VIDEO_ID, offset=2, limit=2)
        assert [t.keyframe.keyframe_ordinal for t in page] == [3, 4]

    def test_offset_beyond_end_returns_empty(self, catalog: FrameCatalog) -> None:
        assert catalog.list_video_frames(VIDEO_ID, offset=99) == ()

    def test_invalid_pagination_rejected(self, catalog: FrameCatalog) -> None:
        with pytest.raises(ValueError, match="offset"):
            catalog.list_video_frames(VIDEO_ID, offset=-1)
        with pytest.raises(ValueError, match="limit"):
            catalog.list_video_frames(VIDEO_ID, limit=0)


class TestFramesAround:
    def test_center_between_records_returns_nearest_in_temporal_order(
        self, catalog: FrameCatalog
    ) -> None:
        # Nearest-first selection with limit=3 picks 30 (d=4), 40 (d=6), then
        # the earlier of the tied pair (20 beats 50 at d=14); the returned
        # sequence itself is temporal order.
        rows = catalog.frames_around(VIDEO_ID, center_frame_index=34, limit=3)
        assert [t.frame_index for t in rows] == [20, 30, 40]
        assert list(rows) == sorted(rows, key=lambda t: t.frame_index)

    def test_exact_center_hit_first(self, catalog: FrameCatalog) -> None:
        rows = catalog.frames_around(VIDEO_ID, center_frame_index=50, limit=1)
        assert rows[0].frame_index == 50

    def test_radius_excludes_far_records(self, catalog: FrameCatalog) -> None:
        rows = catalog.frames_around(VIDEO_ID, center_frame_index=34, radius=5, limit=10)
        assert [t.frame_index for t in rows] == [30]

    def test_limit_caps_result_size(self, catalog: FrameCatalog) -> None:
        assert len(catalog.frames_around(VIDEO_ID, center_frame_index=35, limit=3)) == 3

    def test_unknown_video_raises_key_error(self, catalog: FrameCatalog) -> None:
        with pytest.raises(KeyError):
            catalog.frames_around("L99_V999", center_frame_index=10)

    def test_invalid_radius_rejected(self, catalog: FrameCatalog) -> None:
        with pytest.raises(ValueError, match="radius"):
            catalog.frames_around(VIDEO_ID, center_frame_index=10, radius=-5)


class TestResolve:
    def test_exact_hit(self, catalog: FrameCatalog) -> None:
        timing = catalog.resolve(VIDEO_ID, 40)
        assert timing is not None
        assert timing.keyframe.keyframe_ordinal == 4
        assert catalog.ordinal_for(VIDEO_ID, 40) == 4

    def test_nonindexed_frame_misses(self, catalog: FrameCatalog) -> None:
        assert catalog.resolve(VIDEO_ID, 41) is None
        assert catalog.ordinal_for(VIDEO_ID, 41) is None


class TestLoadOnce:
    def test_root_factory_reads_single_file_once(
        self, frames_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"count": 0}
        real_open = Path.open

        def counting_open(self: Path, *args: object, **kwargs: object):
            if self.name == "frames.ndjson":
                calls["count"] += 1
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", counting_open)
        catalog = FrameCatalog.from_keyframes_root(frames_file.parent)
        catalog.get_video(VIDEO_ID)
        catalog.list_video_frames(VIDEO_ID, offset=1, limit=2)
        catalog.frames_around(VIDEO_ID, center_frame_index=30, limit=2)
        # construction reads once; queries never touch the filesystem again
        assert calls["count"] == 1
