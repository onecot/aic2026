"""OmniShotCut-based adaptive keyframe extractor.

Extracts keyframes from a video in two stages:

1. shot detection (via a :class:`~aic2026.assets.shot_detection.ShotDetector`,
   defaulting to the OmniShotCut model);
2. per-shot adaptive sampling (via :mod:`aic2026.assets.keyframe_sampling`).

The extractor produces the same canonical :class:`Keyframe` records and the
same ``frames.ndjson``/``map-keyframes`` artifact contract as the DAKE
extractor, so downstream embedding, retrieval, indexing, and evaluation code
consumes it unchanged.

Sampling policy (``adaptive_v1``):

- shots shorter than ``short_shot_seconds``: the middle frame only;
- shots up to ``long_shot_seconds``: the middle frame plus a uniform sample
  every ``uniform_interval_seconds``;
- longer shots: a uniform sample plus high-change frames (when per-frame
  change scores are available).

Change scores are computed only when at least one shot is "long", reusing the
same frame reader / change scorer as the DAKE pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aic2026.assets.keyframe_extractor import (
    ChangeScorer,
    FrameDifferenceScorer,
    FrameReader,
    Keyframe,
    KeyframeExtractionError,
)
from aic2026.assets.keyframe_sampling import create_sampler
from aic2026.assets.shot_detection import ShotDetectionError, ShotDetector, normalize_shots

logger = logging.getLogger(__name__)


class OmniShotCutAdaptiveExtractor:
    """Extract keyframes via OmniShotCut shot detection + adaptive sampling.

    Expected input:
        - video_id: the ``L*_V*`` video id stamped on every :class:`Keyframe`.
        - shot_detector: a shared :class:`ShotDetector` (the OmniShotCut model
          is loaded once per run and reused across videos).
        - frame_reader: a :class:`FrameReader` for change-score computation on
          long shots (must decode the full video; the DAKE readers qualify).
        - scorer: optional :class:`ChangeScorer` (default frame-difference).
        - short_shot_seconds: shots shorter than this get only their middle
          frame.
        - long_shot_seconds: shots at least this long use the long-shot policy
          (uniform + high-change).
        - uniform_interval_seconds: uniform sample spacing for non-short
          shots.
        - min_temporal_distance / peak_threshold: high-change selection
          parameters for long shots.
        - fps: optional override of the video frame rate (otherwise taken from
          the shot detector).

    Returns:
        - An extractor instance.

    Raises:
        - ValueError: on invalid parameters.
    """

    extractor_name: str = "omnishotcut_adaptive"
    extractor_version: str = "1.0"

    def __init__(
        self,
        video_id: str,
        shot_detector: ShotDetector,
        frame_reader: FrameReader,
        *,
        scorer: ChangeScorer | None = None,
        short_shot_seconds: float = 5.0,
        long_shot_seconds: float = 60.0,
        uniform_interval_seconds: float = 10.0,
        min_temporal_distance: int = 30,
        peak_threshold: float = 8.0,
        fps: float | None = None,
    ) -> None:
        if float(short_shot_seconds) <= 0.0 or float(long_shot_seconds) <= float(short_shot_seconds):
            raise ValueError(
                f"need 0 < short_shot_seconds < long_shot_seconds, got short={short_shot_seconds!r}, long={long_shot_seconds!r}"
            )
        if float(uniform_interval_seconds) <= 0.0:
            raise ValueError(f"uniform_interval_seconds must be > 0, got {uniform_interval_seconds!r}")
        if not isinstance(min_temporal_distance, int) or isinstance(min_temporal_distance, bool) or min_temporal_distance < 1:
            raise ValueError(f"min_temporal_distance must be >= 1, got {min_temporal_distance!r}")
        if float(peak_threshold) < 0.0:
            raise ValueError(f"peak_threshold must be >= 0, got {peak_threshold!r}")
        if fps is not None and float(fps) <= 0.0:
            raise ValueError("fps must be > 0 when provided")

        self._video_id = video_id
        self._shot_detector = shot_detector
        self._frame_reader = frame_reader
        self._scorer = scorer if scorer is not None else FrameDifferenceScorer()
        self._short_shot_seconds = float(short_shot_seconds)
        self._long_shot_seconds = float(long_shot_seconds)
        self._uniform_interval_seconds = float(uniform_interval_seconds)
        self._min_temporal_distance = min_temporal_distance
        self._peak_threshold = float(peak_threshold)
        self._fps = float(fps) if fps is not None else None
        #: Number of source frames scanned by the most recent :meth:`extract`.
        self._last_frames_scanned = 0

    @property
    def video_id(self) -> str:
        """Return the video id this extractor stamps keyframes with."""
        return self._video_id

    @property
    def last_frames_scanned(self) -> int:
        """Return the number of source frames scanned by the last extract() call.

        Returns:
            - Total frames decoded for change scores (>= 0).
        """
        return self._last_frames_scanned

    def extract(self, video_path: Path | str) -> list[Keyframe]:
        """Extract keyframes from a video.

        Expected input:
            - video_path: path to the video to process.

        Returns:
            - Selected :class:`Keyframe` records ordered by frame index.

        Raises:
            - FileNotFoundError: if the video is absent.
            - KeyframeExtractionError: if detection, decoding, or scoring fails.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")

        try:
            shots = self._shot_detector.detect(path)
        except ShotDetectionError as exc:
            raise KeyframeExtractionError(f"shot detection failed for {path}: {exc}") from exc
        except KeyframeExtractionError:
            raise
        fps = self._fps
        if fps is None and self._shot_detector.last_fps is not None:
            fps = self._shot_detector.last_fps
        num_frames = self._shot_detector.last_num_frames
        if fps is None or num_frames is None:
            probed_fps, _, probed_frames = self._probe(path)
            fps = fps if fps is not None else probed_fps
            num_frames = num_frames if num_frames is not None else probed_frames
        if fps is None or fps <= 0.0 or num_frames is None or num_frames < 0:
            raise KeyframeExtractionError(
                f"cannot determine fps/num_frames for {path} (fps={fps!r}, num_frames={num_frames!r})"
            )

        shots = normalize_shots(shots, num_frames)
        sampler = create_sampler(
            "adaptive_v1",
            fps=fps,
            short_shot_seconds=self._short_shot_seconds,
            long_shot_seconds=self._long_shot_seconds,
            uniform_interval_seconds=self._uniform_interval_seconds,
            min_temporal_distance=self._min_temporal_distance,
            peak_threshold=self._peak_threshold,
        )

        # Change scores are only needed when at least one shot is "long".
        has_long = any(
            (shot.end_frame - shot.start_frame + 1) / fps > self._long_shot_seconds
            for shot in shots
        )
        scores: np.ndarray | None = None
        if has_long:
            scores = self._compute_scores(path)
            self._last_frames_scanned = int(len(scores))
        else:
            self._last_frames_scanned = num_frames

        selected: set[int] = set()
        for shot in shots:
            frame_indices = list(range(shot.start_frame, shot.end_frame + 1))
            chosen = sampler.sample(shot, frame_indices, scores=scores)
            selected.update(chosen)

        keyframes: list[Keyframe] = []
        for ordinal, frame_index in enumerate(sorted(selected), start=1):
            keyframes.append(
                Keyframe(
                    video_id=self._video_id,
                    keyframe_ordinal=ordinal,
                    pts_time=frame_index / fps,
                    fps=fps,
                    frame_index=frame_index,
                )
            )
        return keyframes

    def _compute_scores(self, video_path: Path) -> np.ndarray:
        """Score all frames of the video, returning one score per frame.

        Returns:
            - ``scores[i]`` for the ``i``-th decoded frame.

        Raises:
            - KeyframeExtractionError: if decoding or scoring fails.
        """
        try:
            return np.asarray(self._scorer.compute(self._frame_reader.read(video_path)), dtype=np.float64)
        except KeyframeExtractionError:
            raise
        except Exception as exc:
            raise KeyframeExtractionError(f"failed to score video {video_path}: {exc}") from exc

    @staticmethod
    def _probe(path: Path) -> tuple[float, float, int]:
        """Probe ``(fps, duration_seconds, num_frames)`` via ffprobe.

        Returns:
            - The probed video metadata.

        Raises:
            - KeyframeExtractionError: if probing fails.
        """
        from aic2026.assets.video_manifest import VideoManifestError, _probe_video

        try:
            return _probe_video(path)
        except VideoManifestError as exc:
            raise KeyframeExtractionError(str(exc)) from exc


def extract_video_keyframes_adaptive(
    video_id: str,
    video_path: Path | str,
    *,
    shot_detector: ShotDetector,
    frame_reader: FrameReader,
    scorer: ChangeScorer | None = None,
    short_shot_seconds: float = 5.0,
    long_shot_seconds: float = 60.0,
    uniform_interval_seconds: float = 10.0,
    min_temporal_distance: int = 30,
    peak_threshold: float = 8.0,
    fps: float | None = None,
) -> list[Keyframe]:
    """Convenience wrapper: adaptively extract keyframes from one video.

    Expected input:
        - video_id: the ``L*_V*`` video id.
        - video_path: path to the video.
        - shot_detector / frame_reader / scorer / sampling parameters: see
          :class:`OmniShotCutAdaptiveExtractor`.

    Returns:
        - Selected :class:`Keyframe` records ordered by frame index.

    Raises:
        - FileNotFoundError: if the video is absent.
        - KeyframeExtractionError: if detection, decoding, or scoring fails.
    """
    extractor = OmniShotCutAdaptiveExtractor(
        video_id,
        shot_detector,
        frame_reader,
        scorer=scorer,
        short_shot_seconds=short_shot_seconds,
        long_shot_seconds=long_shot_seconds,
        uniform_interval_seconds=uniform_interval_seconds,
        min_temporal_distance=min_temporal_distance,
        peak_threshold=peak_threshold,
        fps=fps,
    )
    return extractor.extract(video_path)
