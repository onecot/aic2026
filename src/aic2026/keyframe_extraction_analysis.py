"""Dataset-level analysis of the OmniShotCut adaptive keyframe extractor.

This module profiles how the adaptive pipeline (shot detection + per-shot
sampling) would behave over a video manifest *before* running the full
extraction. It is analysis-only infrastructure: it never writes a canonical
artifact and shares no code with embedding/retrieval/evaluation stages.

For every video in the manifest it:

- runs the (shared) shot detector and, exactly like the extractor, normalizes
  the segmentation (gaps filled, overlaps merged) so the union covers the
  whole video; a video that yields no detected shots at all is rejected;
- validates the normalized segmentation:
  - every video yields at least one shot;
  - every shot is a valid inclusive range (``start_frame <= end_frame``);
  - shots never overlap;
  - the shots cover the video timeline (``[0, num_frames - 1]``);
  - the estimated keyframe count is at least 1;
- estimates the number of keyframes the adaptive sampler would select per
  video (using :func:`~aic2026.keyframe_sampling.create_sampler` with no
  change scores, so long-shot high-change frames are not counted; the estimate
  is therefore a conservative lower bound);
- aggregates shot counts, shot durations, and estimated keyframes across the
  manifest into distribution statistics;
- flags outlier videos (unusually high shot counts or estimated keyframes).

The returned report is a deterministic, JSON-serializable structure whose
required keys match the spec:

.. code-block:: python

    {
        "schema_version": "1.0",
        "meta": {...},
        "total_videos": int,
        "shots": {...},
        "shot_duration_seconds": {...},
        "estimated_keyframes": {...},
        "outliers": [...]
    }

Validation failures raise :class:`ShotValidationError` naming the offending
video, so a broken manifest fails fast instead of producing a misleading
profile.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aic2026.keyframe_extraction import VideoManifest, load_video_manifest
from aic2026.keyframe_sampling import create_sampler
from aic2026.progress import ProgressReporter
from aic2026.shot_detection import Shot, ShotDetector, normalize_shots

#: Schema version of the shot-distribution report.
REPORT_SCHEMA_VERSION = "1.0"

#: Default sampling parameters used when ``sampler_config`` omits a key.
DEFAULT_SAMPLER_CONFIG: dict[str, Any] = {
    "sampler": "adaptive_v1",
    "short_shot_seconds": 5.0,
    "long_shot_seconds": 60.0,
    "uniform_interval_seconds": 10.0,
    "min_temporal_distance": 30,
    "peak_threshold": 8.0,
}


class ShotValidationError(ValueError):
    """Raised when a video's shot segmentation fails a verification check."""


@dataclass(frozen=True)
class CountStats:
    """Distribution statistics over a per-video integer quantity.

    Fields:
        - total: sum across videos.
        - avg: mean across videos.
        - min / median / p90 / max: distribution percentiles.
    """

    total: int
    avg: float
    min: int
    median: float
    p90: float
    max: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "total": self.total,
            "avg": self.avg,
            "min": self.min,
            "median": self.median,
            "p90": self.p90,
            "max": self.max,
        }


@dataclass(frozen=True)
class ShotDurationStats:
    """Distribution statistics over detected shot durations (seconds).

    Fields:
        - mean / median / p90 / max: duration percentiles over all shots.
    """

    mean: float
    median: float
    p90: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"mean": self.mean, "median": self.median, "p90": self.p90, "max": self.max}


@dataclass(frozen=True)
class ShotDistributionReport:
    """Structured shot-distribution and keyframe-estimation report.

    Fields:
        - schema_version: report schema version.
        - meta: self-describing run metadata (detector, sampler config).
        - total_videos: number of videos analyzed.
        - shots: per-video shot-count statistics.
        - shot_duration_seconds: shot-duration statistics over all shots.
        - estimated_keyframes: per-video estimated-keyframe statistics.
        - outliers: videos with unusually high shot counts or keyframe counts.
    """

    schema_version: str
    meta: dict[str, Any]
    total_videos: int
    shots: CountStats
    shot_duration_seconds: ShotDurationStats
    estimated_keyframes: CountStats
    outliers: list[dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "schema_version": self.schema_version,
            "meta": self.meta,
            "total_videos": self.total_videos,
            "shots": self.shots.to_dict(),
            "shot_duration_seconds": self.shot_duration_seconds.to_dict(),
            "estimated_keyframes": self.estimated_keyframes.to_dict(),
            "outliers": self.outliers,
        }


# ---------------------------------------------------------------------- math


def _null_timer() -> Any:
    """Return a no-op context manager (used when no progress reporter exists)."""
    from contextlib import nullcontext

    return nullcontext()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return the ``percentile``-th percentile of a sorted list (0..100).

    Uses linear interpolation between the nearest ranks (matching NumPy's
    default). An empty list yields ``0.0``.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac


def _median(sorted_values: list[float]) -> float:
    """Return the median of a sorted list (``0.0`` when empty)."""
    if not sorted_values:
        return 0.0
    return float(statistics.median(sorted_values))


def _p90(sorted_values: list[float]) -> float:
    """Return the 90th percentile of a sorted list (``0.0`` when empty)."""
    return _percentile(sorted_values, 90.0)


# ------------------------------------------------------------------ validation


def validate_shots(
    shots: list[Shot],
    num_frames: int,
    *,
    video_id: str,
) -> None:
    """Verify a detected shot segmentation against the video timeline.

    Checks, in order: at least one shot exists; every shot is a valid
    inclusive range; shots never overlap; and the shots jointly cover the
    full video timeline ``[0, num_frames - 1]``.

    Expected input:
        - shots: detected shot segments (any order).
        - num_frames: total source frames of the video.
        - video_id: the video being validated (used in error messages).

    Returns:
        - None.

    Raises:
        - ShotValidationError: if any check fails.
    """
    if not shots:
        raise ShotValidationError(f"{video_id}: detected 0 shots")
    if num_frames < 1:
        raise ShotValidationError(f"{video_id}: invalid frame count {num_frames}")
    ordered = sorted(shots, key=lambda shot: (shot.start_frame, shot.end_frame))
    previous_end = -1
    for shot in ordered:
        if shot.start_frame < 0 or shot.start_frame > shot.end_frame or shot.end_frame >= num_frames:
            raise ShotValidationError(
                f"{video_id}: invalid shot range {shot.to_dict()} for {num_frames} frames"
            )
        if shot.start_frame <= previous_end:
            raise ShotValidationError(f"{video_id}: overlapping shots in {shot.to_dict()}")
        if previous_end >= 0 and shot.start_frame != previous_end + 1:
            raise ShotValidationError(
                f"{video_id}: shots do not cover the video timeline "
                f"(gap between frame {previous_end} and frame {shot.start_frame})"
            )
        previous_end = shot.end_frame
    # Coverage: the union must be exactly [0, num_frames - 1].
    if ordered[-1].end_frame != num_frames - 1 or ordered[0].start_frame != 0:
        raise ShotValidationError(
            f"{video_id}: shots do not cover the video timeline "
            f"(first={ordered[0].start_frame}, last={ordered[-1].end_frame}, frames={num_frames})"
        )


# ------------------------------------------------------------------ estimation


def estimate_keyframes(
    shots: list[Shot],
    fps: float,
    sampler_config: Mapping[str, Any],
) -> int:
    """Estimate the number of keyframes the adaptive sampler would select.

    Uses :func:`~aic2026.keyframe_sampling.create_sampler` with the supplied
    configuration and no change scores. Because change scores are unavailable
    during profiling, long-shot high-change frames are not counted, so the
    result is a conservative lower bound.

    Expected input:
        - shots: validated, timeline-covering shot segments.
        - fps: frame rate of the video (positive).
        - sampler_config: sampler name plus adaptive parameters.

    Returns:
        - The estimated total keyframe count across ``shots`` (>= 1).

    Raises:
        - ValueError: if the sampler configuration is invalid.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    config = {**DEFAULT_SAMPLER_CONFIG, **dict(sampler_config)}
    sampler = create_sampler(
        config["sampler"],
        fps=fps,
        short_shot_seconds=config["short_shot_seconds"],
        long_shot_seconds=config["long_shot_seconds"],
        uniform_interval_seconds=config["uniform_interval_seconds"],
        min_temporal_distance=config["min_temporal_distance"],
        peak_threshold=config["peak_threshold"],
    )
    total = 0
    for shot in shots:
        frames = list(range(shot.start_frame, shot.end_frame + 1))
        total += len(sampler.sample(shot, frames, scores=None))
    return total


# ------------------------------------------------------------------ report


def _count_stats(values: list[int]) -> CountStats:
    """Build :class:`CountStats` from a list of per-video integer values."""
    if not values:
        return CountStats(total=0, avg=0.0, min=0, median=0.0, p90=0.0, max=0)
    ordered = sorted(float(v) for v in values)
    return CountStats(
        total=int(sum(values)),
        avg=float(statistics.mean(values)),
        min=int(min(values)),
        median=_median(ordered),
        p90=_p90(ordered),
        max=int(max(values)),
    )


def _outliers(
    num_shots_by_video: list[tuple[str, int]],
    estimated_by_video: list[tuple[str, int]],
    *,
    std_threshold: float,
) -> list[dict[str, int]]:
    """Flag videos whose shot count or keyframe estimate exceeds the mean by
    ``std_threshold`` standard deviations (when the distribution has spread).

    Returns:
        - Sorted (by estimated keyframes, descending) outlier records.
    """
    if len(num_shots_by_video) < 2:
        return []
    shot_values = [int(n) for _, n in num_shots_by_video]
    kf_values = [int(k) for _, k in estimated_by_video]
    shot_mean = float(statistics.mean(shot_values))
    shot_std = float(statistics.stdev(shot_values)) if len(shot_values) > 1 else 0.0
    kf_mean = float(statistics.mean(kf_values))
    kf_std = float(statistics.stdev(kf_values)) if len(kf_values) > 1 else 0.0

    outliers: list[dict[str, int]] = []
    for (video_id, num_shots), (_, estimated) in zip(
        num_shots_by_video, estimated_by_video
    ):
        is_outlier = False
        if shot_std > 0 and num_shots > shot_mean + std_threshold * shot_std:
            is_outlier = True
        if kf_std > 0 and estimated > kf_mean + std_threshold * kf_std:
            is_outlier = True
        if is_outlier:
            outliers.append({"video_id": video_id, "num_shots": num_shots, "estimated_keyframes": estimated})
    outliers.sort(key=lambda record: (record["estimated_keyframes"], record["video_id"]), reverse=True)
    return outliers


def progress_unit_for_manifest(manifest: VideoManifest) -> str:
    """Return the progress unit a reporter should use for a manifest.

    Detection throughput is meaningful in frames per second; the frame-based
    unit is used when every video carries a known ``num_frames``, otherwise
    progress falls back to per-video counting.

    Expected input:
        - manifest: a loaded video manifest.

    Returns:
        - ``"frames"`` when every entry has ``num_frames``, else ``"videos"``.
    """
    if manifest.videos and all(entry.num_frames is not None for entry in manifest.videos):
        return "frames"
    return "videos"


def analyze_shot_distribution(
    videos_manifest: Path | str | VideoManifest,
    detector: ShotDetector,
    sampler_config: Mapping[str, Any] | None = None,
    *,
    progress: ProgressReporter | None = None,
    outlier_std_threshold: float = 2.0,
) -> ShotDistributionReport:
    """Profile shot distribution and adaptive sampling output over a manifest.

    Expected input:
        - videos_manifest: a ``videos.json`` path or a loaded
          :class:`~aic2026.keyframe_extraction.VideoManifest`.
        - detector: a shared :class:`ShotDetector` (loaded once, reused across
          videos). Must expose ``detect``, ``last_fps``, and
          ``last_num_frames`` (the OmniShotCut and synthetic detectors do).
        - sampler_config: optional sampler parameters overriding
          :data:`DEFAULT_SAMPLER_CONFIG`.
        - progress: optional reporter; ``set_total`` is called with the total
          frame count (when every entry has ``num_frames``) or the video
          count, and one ``update`` is emitted per video.
        - outlier_std_threshold: videos whose shot count or keyframe estimate
          exceed the mean by this many standard deviations are flagged as
          outliers (default 2.0).

    Returns:
        - A :class:`ShotDistributionReport` describing the manifest.

    Raises:
        - FileNotFoundError: if the manifest path does not exist.
        - ValueError: if the manifest is empty or the detector's metadata is
          missing.
        - ShotValidationError: if a video's shot segmentation is invalid.
    """
    if isinstance(videos_manifest, VideoManifest):
        manifest = videos_manifest
    else:
        manifest = load_video_manifest(videos_manifest)
    if not manifest.videos:
        raise ValueError("video manifest is empty; nothing to analyze")

    config = {**DEFAULT_SAMPLER_CONFIG, **(dict(sampler_config) if sampler_config else {})}
    frame_based_progress = progress_unit_for_manifest(manifest) == "frames"
    if progress is not None:
        progress.begin()
        if frame_based_progress:
            total_frames = sum(
                entry.num_frames for entry in manifest.videos if entry.num_frames is not None
            )
            progress.set_total(total_frames)
        else:
            progress.set_total(len(manifest.videos))

    num_shots_by_video: list[tuple[str, int]] = []
    estimated_by_video: list[tuple[str, int]] = []
    all_shot_durations: list[float] = []

    processed_frames = 0
    for completed, entry in enumerate(manifest.videos, start=1):
        with progress.stages.time("detect_shots") if progress is not None else _null_timer():
            shots = detector.detect(entry.path)
        fps = detector.last_fps
        num_frames = detector.last_num_frames
        if fps is None or fps <= 0.0:
            raise ValueError(
                f"{entry.video_id}: detector did not report a valid fps ({fps!r})"
            )
        if num_frames is None or num_frames < 1:
            raise ValueError(
                f"{entry.video_id}: detector did not report a valid frame count ({num_frames!r})"
            )
        if not shots:
            raise ShotValidationError(f"{entry.video_id}: detected 0 shots")
        shots = normalize_shots(shots, num_frames)
        validate_shots(shots, num_frames, video_id=entry.video_id)

        with progress.stages.time("estimate_sampling") if progress is not None else _null_timer():
            estimated = estimate_keyframes(shots, fps, config)
        if estimated < 1:
            raise ShotValidationError(
                f"{entry.video_id}: estimated 0 keyframes for {len(shots)} shots"
            )

        num_shots_by_video.append((entry.video_id, len(shots)))
        estimated_by_video.append((entry.video_id, estimated))
        all_shot_durations.extend((shot.end_frame - shot.start_frame + 1) / fps for shot in shots)

        if progress is not None:
            processed_frames += num_frames
            progress.update(
                processed_frames if frame_based_progress else completed,
                note=f"video={entry.video_id} shots={len(shots)} estimated={estimated}",
            )

    duration_sorted = sorted(all_shot_durations)
    report = ShotDistributionReport(
        schema_version=REPORT_SCHEMA_VERSION,
        meta={
            "detector": detector.name,
            "sampler_config": config,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        },
        total_videos=len(manifest.videos),
        shots=_count_stats([n for _, n in num_shots_by_video]),
        shot_duration_seconds=ShotDurationStats(
            mean=float(statistics.mean(duration_sorted)) if duration_sorted else 0.0,
            median=_median(duration_sorted),
            p90=_p90(duration_sorted),
            max=duration_sorted[-1] if duration_sorted else 0.0,
        ),
        estimated_keyframes=_count_stats([k for _, k in estimated_by_video]),
        outliers=_outliers(
            num_shots_by_video,
            estimated_by_video,
            std_threshold=outlier_std_threshold,
        ),
    )
    if progress is not None:
        progress.finish(note=f"analyzed {len(manifest.videos)} videos")
    return report


def write_report(path: Path | str, report: ShotDistributionReport) -> Path:
    """Write a shot-distribution report as JSON (atomic).

    Expected input:
        - path: target JSON file path.
        - report: the report to persist.

    Returns:
        - The written path.

    Raises:
        - OSError: if writing fails.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
        handle.write("\n")
    temp_path.replace(output_path)
    return output_path


def format_report(report: ShotDistributionReport) -> str:
    """Render a human-readable summary of a shot-distribution report.

    Expected input:
        - report: the report to format.

    Returns:
        - Multi-line human-readable summary (trailing newline).
    """
    lines = [
        f"Shot distribution (detector={report.meta.get('detector', '?')})",
        f"  videos: {report.total_videos}",
        f"  total shots: {report.shots.total}",
        f"  shots/video: avg={report.shots.avg:.1f} min={report.shots.min} "
        f"median={report.shots.median:.1f} p90={report.shots.p90:.1f} max={report.shots.max}",
        f"  shot duration (s): mean={report.shot_duration_seconds.mean:.1f} "
        f"median={report.shot_duration_seconds.median:.1f} "
        f"p90={report.shot_duration_seconds.p90:.1f} max={report.shot_duration_seconds.max:.1f}",
        f"  estimated keyframes/video: total={report.estimated_keyframes.total} "
        f"avg={report.estimated_keyframes.avg:.1f} min={report.estimated_keyframes.min} "
        f"median={report.estimated_keyframes.median:.1f} "
        f"p90={report.estimated_keyframes.p90:.1f} max={report.estimated_keyframes.max}",
    ]
    if report.outliers:
        lines.append("  outliers:")
        for record in report.outliers:
            lines.append(
                f"    {record['video_id']}: shots={record['num_shots']} "
                f"estimated_keyframes={record['estimated_keyframes']}"
            )
    else:
        lines.append("  outliers: none")
    return "\n".join(lines) + "\n"
