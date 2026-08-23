"""In-video frame navigation over one canonical ``frames.ndjson`` artifact.

``FrameCatalog`` loads a keyframe artifact's frame metadata **once** and
serves read-only navigation queries:

    get_video(video_id)                        -> summary metadata
    list_video_frames(video_id, offset, limit) -> ordinal page
    frames_around(video_id, center, radius)    -> nearest indexed keyframes
    resolve / ordinal_for                      -> frame_index -> record

It reuses the canonical :class:`aic2026.canonical.KeyframeTiming` model (no
duplicate type), never synthesizes frame indices that are not in the
artifact, and performs no I/O after construction. The UI/API layer wraps
this class directly; it is usable without FastAPI.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from aic2026.canonical import KeyframeTiming
from aic2026.embedding_artifact import FRAMES_FILENAME, iter_frames


@dataclass(frozen=True)
class FrameVideoSummary:
    """Per-video frame metadata derived from the catalog.

    Fields:
        - video_id: canonical video identifier.
        - frame_count: number of indexed keyframes for the video.
        - first_frame_index / last_frame_index: original-video frame index
          range covered by indexed keyframes (temporal order).
        - first_pts_time / last_pts_time: presentation timestamps of the
          first and last indexed keyframes in seconds.

    Duration is intentionally not reported: it cannot be derived exactly
    from sparse keyframe timestamps without inventing metadata.
    """

    video_id: str
    frame_count: int
    first_frame_index: int
    last_frame_index: int
    first_pts_time: float
    last_pts_time: float


def _require_limit(name: str, value: int) -> None:
    """Validate one positive-integer query parameter."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class FrameCatalog:
    """Read-only, load-once index over one artifact's ``frames.ndjson``.

    Expected input:
        - frames_path: path to a canonical ``frames.ndjson`` file (row order
          follows the artifact contract: grouped by video, ascending
          keyframe ordinal).

    Raises:
        - FileNotFoundError: if the file does not exist.
        - ValueError: if any row violates the canonical contract.
    """

    def __init__(self, frames_path: Path | str) -> None:
        self._frames_path = Path(frames_path)
        by_video: dict[str, list[KeyframeTiming]] = {}
        for timing in iter_frames(self._frames_path):
            by_video.setdefault(timing.keyframe.video_id.value, []).append(timing)
        for records in by_video.values():
            records.sort(key=lambda timing: timing.frame_index)
        self._by_video: dict[str, list[KeyframeTiming]] = by_video

    @classmethod
    def from_keyframes_root(cls, keyframes_root: Path | str) -> "FrameCatalog":
        """Build a catalog from a keyframe artifact root directory.

        Expected input:
            - keyframes_root: directory containing ``frames.ndjson`` (the
              materialized artifact root, e.g. ``adaptive_sbd_v1``).

        Returns:
            - A loaded :class:`FrameCatalog`.

        Raises:
            - FileNotFoundError / ValueError: see :meth:`__init__`.
        """
        return cls(Path(keyframes_root) / FRAMES_FILENAME)

    @property
    def frames_path(self) -> Path:
        """The backing ``frames.ndjson`` path."""
        return self._frames_path

    def video_ids(self) -> tuple[str, ...]:
        """Return all known video ids in first-seen artifact order."""
        return tuple(self._by_video)

    def has_video(self, video_id: str) -> bool:
        """Return whether the catalog contains this video."""
        return video_id in self._by_video

    def _records(self, video_id: str) -> list[KeyframeTiming]:
        """Return the sorted record list for one video (validated)."""
        try:
            return self._by_video[video_id]
        except KeyError:
            raise KeyError(f"unknown video: {video_id}") from None

    def get_video(self, video_id: str) -> FrameVideoSummary:
        """Return summary metadata for one video.

        Expected input:
            - video_id: canonical video identifier.

        Returns:
            - The :class:`FrameVideoSummary` derived from indexed keyframes.

        Raises:
            - KeyError: if the video is unknown.
        """
        records = self._records(video_id)
        return FrameVideoSummary(
            video_id=video_id,
            frame_count=len(records),
            first_frame_index=records[0].frame_index,
            last_frame_index=records[-1].frame_index,
            first_pts_time=float(records[0].pts_time),
            last_pts_time=float(records[-1].pts_time),
        )

    def list_video_frames(
        self,
        video_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[KeyframeTiming, ...]:
        """Return one ordinal page of a video's indexed keyframes.

        Only actual indexed keyframes are returned; missing frame indices
        are never synthesized.

        Expected input:
            - video_id: canonical video identifier.
            - offset: non-negative number of records to skip.
            - limit: positive maximum number of records to return.

        Returns:
            - At most ``limit`` records in temporal (frame_index) order.

        Raises:
            - KeyError: if the video is unknown.
            - ValueError: for invalid pagination parameters.
        """
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        _require_limit("limit", limit)
        records = self._records(video_id)
        return tuple(records[offset : offset + limit])

    def frames_around(
        self,
        video_id: str,
        center_frame_index: int,
        radius: int | None = None,
        limit: int = 50,
    ) -> tuple[KeyframeTiming, ...]:
        """Return indexed keyframes nearest to a requested center frame.

        Nearest records are selected with binary search plus outward
        expansion (ties prefer the earlier frame), then returned in temporal
        (frame_index) order. When ``radius`` is given, only records within
        ``|frame_index - center| <= radius`` are eligible.

        Expected input:
            - video_id: canonical video identifier.
            - center_frame_index: non-negative original-video frame index;
              it does not need to be an indexed keyframe itself.
            - radius: optional non-negative eligibility window in frames.
            - limit: positive maximum number of records to return.

        Returns:
            - At most ``limit`` records in temporal order (possibly empty
              when ``radius`` excludes every indexed keyframe).

        Raises:
            - KeyError: if the video is unknown.
            - ValueError: for invalid radius/limit/center parameters.
        """
        if not isinstance(center_frame_index, int) or isinstance(center_frame_index, bool) or center_frame_index < 0:
            raise ValueError("center_frame_index must be a non-negative integer")
        if radius is not None and (
            not isinstance(radius, int) or isinstance(radius, bool) or radius < 0
        ):
            raise ValueError("radius must be a non-negative integer or None")
        _require_limit("limit", limit)
        records = self._records(video_id)

        indices = [timing.frame_index for timing in records]
        right = bisect_left(indices, center_frame_index)
        left = right - 1

        def distance(index: int) -> int:
            return abs(indices[index] - center_frame_index)

        def eligible(index: int) -> bool:
            return radius is None or distance(index) <= radius

        selected: list[int] = []
        while len(selected) < limit and (left >= 0 or right < len(records)):
            if left >= 0 and right < len(records):
                take_left = distance(left) <= distance(right)  # tie prefers earlier frame
            elif left >= 0:
                take_left = True
            else:
                take_left = False
            index = left if take_left else right
            if not eligible(index):
                break  # nearest-first order: every remaining record is farther
            selected.append(index)
            if take_left:
                left -= 1
            else:
                right += 1

        selected.sort()
        return tuple(records[index] for index in selected)

    def resolve(
        self,
        video_id: str,
        frame_index: int,
    ) -> KeyframeTiming | None:
        """Return the exact indexed record for a frame index, or None.

        Expected input:
            - video_id: canonical video identifier.
            - frame_index: original-video frame index.

        Returns:
            - The matching :class:`KeyframeTiming`, or ``None`` when that
              exact frame index is not an indexed keyframe.

        Raises:
            - KeyError: if the video is unknown.
        """
        records = self._records(video_id)
        indices = [timing.frame_index for timing in records]
        position = bisect_left(indices, frame_index)
        if position < len(records) and indices[position] == frame_index:
            return records[position]
        return None

    def ordinal_for(self, video_id: str, frame_index: int) -> int | None:
        """Return the keyframe ordinal for an exact frame index, or None."""
        timing = self.resolve(video_id, frame_index)
        return None if timing is None else timing.keyframe.keyframe_ordinal
