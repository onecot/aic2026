"""Runtime progress reporting and stage timing for long-running pipelines.

This module provides pure, reusable helpers (duration/ETA/throughput formatting)
plus a lightweight :class:`ProgressReporter` and :class:`StageTimer` used by the
embedding shard pipeline (:mod:`aic2026.embedding_shard`) and its CLI to report
progress without hurting throughput.

Design constraints:

- reporting is throttled: at most one update per ``interval_seconds`` (or one
  per batch), so it never prints per-frame;
- two render modes exist:
    - ``interactive``: single-line carriage-return updates to a stream;
    - ``periodic``: throttled ``logging`` lines for non-interactive/log-file
      runs;
    - ``off``: no-op, zero overhead.
- ``auto`` picks ``interactive`` when attached to a terminal and ``periodic``
  otherwise.

Progress reporting is metadata/logging only: it never affects embedding
outputs or checkpoint/artifact schemas.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

#: Valid ``--progress`` CLI values.
PROGRESS_MODES = ("auto", "on", "off")

#: Width of the interactive progress bar.
_BAR_WIDTH = 20


def format_duration(seconds: float) -> str:
    """Format seconds as ``X.Ys`` or ``HH:MM:SS``.

    Expected input:
        - seconds: a non-negative duration.

    Returns:
        - ``0.0s`` .. ``59.9s`` as ``X.Ys``; otherwise ``HH:MM:SS``.
    """
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def compute_speed(completed: int, elapsed: float) -> float:
    """Compute the average throughput in frames/second.

    Expected input:
        - completed: number of processed units.
        - elapsed: wall-clock seconds elapsed (>= 0).

    Returns:
        - ``completed / elapsed``, or ``0.0`` when ``elapsed`` is not positive.
    """
    if elapsed <= 0 or completed <= 0:
        return 0.0
    return float(completed) / float(elapsed)


def estimate_eta(completed: int, total: int, elapsed: float) -> float:
    """Estimate the remaining time (seconds) given current progress.

    Uses the average throughput so far.

    Expected input:
        - completed: number of processed units.
        - total: total number of units (>= completed).
        - elapsed: wall-clock seconds elapsed (>= 0).

    Returns:
        - Estimated remaining seconds (>= 0), or ``0.0`` when progress or
          elapsed is not positive.
    """
    if completed <= 0 or elapsed <= 0:
        return 0.0
    remaining = float(total - completed) * float(elapsed) / float(completed)
    return max(0.0, remaining)


def format_progress_line(
    completed: int,
    total: int,
    elapsed: float,
    *,
    bar: bool = True,
    note: str | None = None,
) -> str:
    """Format a single-line progress report.

    Expected input:
        - completed: number of processed frames.
        - total: total number of frames.
        - elapsed: wall-clock seconds elapsed.
        - bar: include an ASCII progress bar (interactive mode).
        - note: optional context suffix (e.g. current video).

    Returns:
        - A single line such as::

            [=======>........] 42.5% | 75200/177321 frames  \
elapsed=01:35:20 speed=13.1 frames/s ETA=02:12:40

    """
    percent = 100.0 * completed / total if total > 0 else 100.0
    prefix = ""
    if bar:
        filled = int(round(_BAR_WIDTH * percent / 100.0)) if total > 0 else _BAR_WIDTH
        prefix = "[" + "=" * filled + "." * (_BAR_WIDTH - filled) + "] "
    speed = compute_speed(completed, elapsed)
    eta = estimate_eta(completed, total, elapsed)
    line = (
        f"{prefix}{percent:5.1f}% | {completed}/{total} frames  "
        f"elapsed={format_duration(elapsed)} "
        f"speed={speed:.1f} frames/s "
        f"ETA={format_duration(eta)}"
    )
    if note:
        line += f"  {note}"
    return line


class StageTimer:
    """Accumulate wall-clock time per named stage.

    Stages may be started/stopped repeatedly; elapsed time accumulates across
    calls. Safe for nested or overlapping use via :meth:`time`.
    """

    def __init__(self) -> None:
        self._started: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing ``name`` (idempotent while already started)."""
        if name not in self._started:
            self._started[name] = time.monotonic()

    def stop(self, name: str) -> None:
        """Stop timing ``name`` and accumulate its elapsed seconds."""
        started = self._started.pop(name, None)
        if started is not None:
            self._elapsed[name] = self._elapsed.get(name, 0.0) + time.monotonic() - started

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """Time the enclosed block under ``name`` (accumulating)."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def elapsed(self) -> dict[str, float]:
        """Return ``stage name -> accumulated seconds`` (copy)."""
        return dict(self._elapsed)

    def format(self) -> str:
        """Render the stage timing report (or ``""`` when empty)."""
        elapsed = self.elapsed()
        if not elapsed:
            return ""
        lines = ["Stage timing:"]
        for name in sorted(elapsed):
            lines.append(f"{name}: {format_duration(elapsed[name])}")
        return "\n".join(lines)


class ProgressReporter:
    """Throttled runtime progress reporter for embedding-style loops.

    Expected input:
        - mode: ``interactive`` (single-line \r updates), ``periodic``
          (throttled logging), or ``off`` (no-op).
        - interval_seconds: minimum seconds between reports (0 disables
          throttling); irrelevant when ``mode`` is ``off``.
        - stream: destination for interactive updates (default stdout).
        - logger_name: logger used for ``periodic`` mode.

    The reporter measures wall-clock time from :meth:`begin` and reports at
    most once per ``interval_seconds`` (plus a final report at :meth:`finish`).
    :meth:`begin` is idempotent, so a caller and the pipeline can both call it.
    """

    def __init__(
        self,
        mode: str = "off",
        *,
        interval_seconds: float = 2.0,
        stream: TextIO | None = None,
        logger_name: str = "aic2026.progress",
    ) -> None:
        if mode not in ("interactive", "periodic", "off"):
            raise ValueError(
                f"unsupported reporter mode {mode!r}; "
                "expected interactive, periodic, or off"
            )
        self._mode = mode
        self._interval = max(0.0, float(interval_seconds))
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._logger = logging.getLogger(logger_name)
        self._started = False
        self._begin_time = 0.0
        self._last_report: float | None = None
        self._total = 0
        self._completed = 0
        #: Stage accumulator; callers may record stage timings into it.
        self.stages = StageTimer()

    @property
    def mode(self) -> str:
        """Return the concrete render mode."""
        return self._mode

    @property
    def enabled(self) -> bool:
        """True when reporting is active (not ``off``)."""
        return self._mode != "off"

    def begin(self) -> None:
        """Start the wall clock (idempotent)."""
        if self._started:
            return
        self._started = True
        self._begin_time = time.monotonic()
        self._last_report = self._begin_time

    def set_total(self, total: int, completed: int = 0) -> None:
        """Set the expected total and the already-completed count."""
        self._total = max(0, int(total))
        self._completed = max(0, int(completed))

    def update(self, completed: int, *, note: str | None = None) -> None:
        """Report progress for ``completed`` units (throttled)."""
        if not self._started or self._mode == "off":
            self._completed = max(0, int(completed))
            return
        self._completed = max(0, int(completed))
        # The final 100% line is always rendered by finish().
        if self._total <= 0 or self._completed >= self._total:
            return
        now = time.monotonic()
        if self._last_report is not None and now - self._last_report < self._interval:
            return
        self._last_report = now
        self._render(note=note)

    def finish(self, *, note: str | None = None) -> None:
        """Emit the final report and the stage-timing summary."""
        if not self._started or self._mode == "off":
            return
        if self._total > 0:
            self._render(note=note, final=True)
        self._render_stages()

    def _elapsed(self) -> float:
        return time.monotonic() - self._begin_time

    def _render(self, *, note: str | None = None, final: bool = False) -> None:
        line = format_progress_line(
            self._completed,
            self._total,
            self._elapsed(),
            bar=self._mode == "interactive",
            note=note,
        )
        if self._mode == "interactive":
            print("\r" + line, file=self._stream, end="", flush=True)
            if final:
                print(file=self._stream)
        else:
            self._logger.info("embedding progress: %s", line)

    def _render_stages(self) -> None:
        report = self.stages.format()
        if not report:
            return
        if self._mode == "interactive":
            print(report, file=self._stream)
        else:
            self._logger.info(report.replace("\n", "; "))


def create_progress_reporter(mode: str = "auto", **kwargs: Any) -> ProgressReporter:
    """Create a :class:`ProgressReporter` from a CLI-style mode.

    Expected input:
        - mode: ``auto``, ``on``, or ``off``.
        - kwargs: forwarded to :class:`ProgressReporter`.

    Returns:
        - ``auto``: interactive when attached to a terminal, else periodic.
        - ``on``: interactive.
        - ``off``: disabled.

    Raises:
        - ValueError: if ``mode`` is not one of the valid values.
    """
    if mode not in PROGRESS_MODES:
        raise ValueError(f"unsupported progress mode {mode!r}; expected auto, on, or off")
    if mode == "off":
        return ProgressReporter("off", **kwargs)
    if mode == "on":
        return ProgressReporter("interactive", **kwargs)
    interactive = bool(sys.stdout.isatty())
    return ProgressReporter("interactive" if interactive else "periodic", **kwargs)