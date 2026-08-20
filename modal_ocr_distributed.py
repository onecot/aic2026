#!/usr/bin/env python3
"""Distributed Modal OCR runner: sharded, GPU-parallel, resumable.

Splits the canonical keyframe collection (``frames.ndjson`` + ``images/``)
into ``num_shards`` deterministic, disjoint row ranges and runs one
independent, resumable OCR artifact per shard on a Modal GPU worker.

Use ``--num-workers 1`` for a single-GPU run. This module:

- shards by contiguous rows of the canonical ``frames.ndjson`` so each shard
  is deterministic given the same source and shard count;
- writes each shard's filtered ``frames.ndjson`` to the Volume at
  ``/data/interim/ocr/keyframes/shard_XXX/frames.ndjson`` (stable, durable
  source path recorded in the manifest);
- runs ``generate_ocr`` unchanged per shard, writing the OCR artifact to
  ``/data/interim/ocr/shards/shard_XXX/``;
- prints a flush=True heartbeat line on every batch so the worker's
  progress (processed/current/total, frames_per_min, elapsed, ETA, GPU info)
  is observable from the Modal logs.

The canonical keyframe artifact and the ``generate_ocr`` contract are never
modified. Each shard is fully resumable: an interrupted shard restarts from
its own checkpoint on the Volume.

Usage:
    modal run modal_ocr_distributed.py                       # 8 shards, L4
    modal run modal_ocr_distributed.py --num-workers 16
    modal run modal_ocr_distributed.py --shard-size 20000
    modal run modal_ocr_distributed.py --gpu A10G --num-workers 4
    modal run modal_ocr_distributed.py --validate           # small smoke run
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import modal

from aic2026.embedding_artifact import FRAMES_FILENAME
from aic2026.keyframe_materialize import IMAGES_DIRNAME
from aic2026.ocr_generation import generate_ocr
from aic2026.progress import (
    ProgressReporter,
    compute_speed,
    estimate_eta,
    format_duration,
    format_eta,
)

REPO_ROOT = Path(__file__).resolve().parent

VOLUME_NAME = "aic2026-data"
VOLUME_MOUNT_PATH = "/data"
KEYFRAMES_ZIP = f"{VOLUME_MOUNT_PATH}/keyframes/adaptive_sbd_v1.zip"
KEYFRAMES_DIR = f"{VOLUME_MOUNT_PATH}/keyframes/adaptive_sbd_v1"
OCR_TIMEOUT = 12 * 60 * 60

volume = modal.Volume.from_name(VOLUME_NAME).with_mount_options(sub_path="data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "paddlepaddle-gpu==3.2.0",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install("paddleocr>=3.0", "numpy>=1.24", "Pillow>=9.0")
    .env({"PYTHONPATH": "/app/src"})
    .add_local_dir(REPO_ROOT / "src", remote_path="/app/src")
)


def ensure_keyframes_extracted() -> None:
    """Materialize the canonical keyframe artifact in the mounted Volume."""
    target = Path(KEYFRAMES_DIR)
    if (target / FRAMES_FILENAME).is_file():
        return
    zip_path = Path(KEYFRAMES_ZIP)
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing keyframe zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target.parent)
    if not (target / FRAMES_FILENAME).is_file():
        raise RuntimeError(f"extraction produced no {target / FRAMES_FILENAME}")
    volume.commit()

#: Volume paths for sharded artifacts (mounted at /data via ``volume``).
SHARDS_ROOT = f"{VOLUME_MOUNT_PATH}/interim/ocr/shards"
KEYFRAMES_SHARDS_ROOT = f"{VOLUME_MOUNT_PATH}/interim/ocr/keyframes"
SHARDS_VALIDATE_ROOT = f"{VOLUME_MOUNT_PATH}/interim/ocr/shards-validate"

#: Container-local root used to build each shard's keyframes view. Symlinks
#: on the local container disk always work, so images resolve from the shared
#: Volume path without relying on Volume-internal symlink support.
LOCAL_KEYFRAMES_ROOT = Path("/work")

#: Default GPU and worker count when the CLI omits them.
DEFAULT_GPU = "L4"
DEFAULT_NUM_WORKERS = 8

#: Records processed by ``--validate`` (small-scale validation before a full run).
VALIDATE_MAX_RECORDS = 4

#: Records handed to ``generate_ocr`` per batch (matches the single-GPU CLI).
BATCH_SIZE = 64

app = modal.App("aic2026-ocr-distributed")


# ---------------------------------------------------------------- pure helpers


def shard_row_bounds(total_records: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Return the deterministic ``[start, end)`` row range for one shard.

    Expected input:
        - total_records: total number of rows in the canonical frames file.
        - num_shards: number of shards (>= 1).
        - shard_id: shard index in ``[0, num_shards)``.

    Returns:
        - ``(start, end)`` inclusive-exclusive row bounds. Bounds are
          balanced (each shard differs by at most one row) and cover the
          whole range exactly once.

    Raises:
        - ValueError: on invalid ``total_records``, ``num_shards``, or
          ``shard_id``.
    """
    if not isinstance(total_records, int) or isinstance(total_records, bool) or total_records < 0:
        raise ValueError(f"total_records must be a non-negative integer, got {total_records!r}")
    if not isinstance(num_shards, int) or isinstance(num_shards, bool) or num_shards < 1:
        raise ValueError(f"num_shards must be a positive integer, got {num_shards!r}")
    if not isinstance(shard_id, int) or isinstance(shard_id, bool):
        raise ValueError(f"shard_id must be an integer, got {shard_id!r}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id!r}")
    if total_records == 0:
        return (0, 0)
    base = math.ceil(total_records / num_shards)
    start = shard_id * base
    end = min(start + base, total_records)
    return (start, end)


def write_shard_frames(
    canonical_path: Path | str,
    target_path: Path | str,
    start: int,
    end: int,
    *,
    limit: int | None = None,
) -> int:
    """Write rows ``[start, end)`` of a canonical frames file to ``target_path``.

    Rows are copied verbatim (each line is a full canonical
    ``KeyframeTiming`` record) so the shard's ``frames.ndjson`` is byte-faithful
    to the source. The write is atomic (temp file + replace).

    Expected input:
        - canonical_path: source ``frames.ndjson`` (NDJSON, one record/line).
        - target_path: destination path (parent must exist).
        - start: first row index (inclusive).
        - end: last row index (exclusive); clamped to the file's length.
        - limit: optional cap on the number of rows written (validated against
          ``end - start``); used by ``--validate``.

    Returns:
        - Number of rows written.

    Raises:
        - FileNotFoundError: if ``canonical_path`` is missing.
        - ValueError: on invalid bounds or ``limit``.
    """
    canonical = Path(canonical_path)
    if not canonical.is_file():
        raise FileNotFoundError(f"missing canonical frames: {canonical}")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError(f"start must be a non-negative integer, got {start!r}")
    if not isinstance(end, int) or isinstance(end, bool) or end < start:
        raise ValueError(f"end must be >= start, got start={start!r} end={end!r}")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError(f"limit must be a non-negative integer or None, got {limit!r}")

    with canonical.open(encoding="utf-8") as handle:
        rows = handle.readlines()
    total = len(rows)
    end = min(end, total)
    if limit is not None:
        end = min(end, start + limit)
    selected = rows[start:end]

    target = Path(target_path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(selected)
    os.replace(tmp, target)
    return len(selected)


def build_shard_keyframes_dir(
    keyframes_dir: Path | str,
    shard_frames_path: Path | str,
    shared_images_dir: Path | str,
) -> None:
    """Build a container-local keyframes view for one shard.

    Creates ``keyframes_dir`` containing:
    - ``frames.ndjson`` -> symlink to the shard's Volume frames file;
    - ``images`` -> symlink to the shared canonical images collection.

    Both symlinks live on the local container disk, so ``generate_ocr``'s
    ``keyframe_image_path`` resolution (``<root>/images/<video>/<ordinal>.jpg``)
    works through them without relying on Volume-internal symlink support.
    ``Path.resolve()`` then records the durable Volume paths in the manifest.

    Expected input:
        - keyframes_dir: directory to create (parent must exist).
        - shard_frames_path: absolute target for the ``frames.ndjson`` symlink.
        - shared_images_dir: absolute target for the ``images`` symlink.

    Returns:
        - None.

    Raises:
        - OSError: if the directory or symlinks cannot be created.
    """
    root = Path(keyframes_dir)
    root.mkdir(parents=True, exist_ok=True)

    frames_link = root / FRAMES_FILENAME
    images_link = root / IMAGES_DIRNAME
    for link in (frames_link, images_link):
        if link.is_symlink() or link.exists():
            link.unlink()

    frames_link.symlink_to(Path(shard_frames_path))
    images_link.symlink_to(Path(shared_images_dir), target_is_directory=True)


def _gpu_info() -> str:
    """Return a short GPU description (name and VRAM) for heartbeats.

    Queries ``nvidia-smi`` when available; falls back to ``"unknown"``. This is
    best-effort heartbeat metadata, not pipeline state.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, mem_mb = result.stdout.strip().splitlines()[0].split(",")
            return f"{name.strip()}-{int(float(mem_mb)) // 1024}GB"
    except (OSError, ValueError):
        pass
    return "unknown"


def format_heartbeat_line(
    worker_id: str,
    processed: int,
    current: int,
    total: int,
    elapsed: float,
    gpu_info: str,
) -> str:
    """Format one heartbeat line for a shard worker.

    Expected input:
        - worker_id: shard worker label, e.g. ``shard_000``.
        - processed: rows completed in this session (excludes a resume offset).
        - current: total rows completed including a resume offset.
        - total: total rows in the shard.
        - elapsed: wall-clock seconds since the run began.
        - gpu_info: short GPU description.

    Returns:
        - A single line::

            [shard_000] processed=0/current=123/total=18540 | 45.2 frames/min | elapsed=00:01:23 | ETA 2h41m | gpu=L4-15GB

    Raises:
        - ValueError: on invalid numeric inputs or an empty ``worker_id``.
    """
    if not isinstance(worker_id, str) or worker_id == "":
        raise ValueError("worker_id must be a non-empty string")
    for name, value in (("processed", processed), ("current", current), ("total", total)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if current > total:
        raise ValueError(f"current ({current}) exceeds total ({total})")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
        raise ValueError(f"elapsed must be a non-negative number, got {elapsed!r}")

    frames_per_min = compute_speed(processed, elapsed) * 60.0
    eta = estimate_eta(current, total, elapsed, initial_completed=current - processed)
    return (
        f"[{worker_id}] processed={processed}/current={current}/total={total} "
        f"| {frames_per_min:.1f} frames/min | elapsed={format_duration(elapsed)} "
        f"| ETA {format_eta(eta)} | gpu={gpu_info}"
    )


class ShardHeartbeatReporter(ProgressReporter):
    """Progress reporter that prints a flush=True heartbeat on every batch.

    Subclasses :class:`ProgressReporter` so it can be passed directly to
    ``generate_ocr``. ``interval_seconds=0`` disables throttling, so every
    batch update renders one heartbeat line to ``stream`` with ``flush=True``.
    """

    def __init__(
        self,
        worker_id: str,
        gpu_info: str,
        *,
        stream=None,
    ) -> None:
        super().__init__("interactive", interval_seconds=0, stream=stream, unit="frames")
        self._worker_id = worker_id
        self._gpu_info = gpu_info

    def _render(self, *, note: str | None = None, final: bool = False) -> None:
        processed = max(0, self._completed - self._initial_completed)
        line = format_heartbeat_line(
            self._worker_id,
            processed,
            self._completed,
            self._total,
            self._elapsed(),
            self._gpu_info,
        )
        print(line, file=self._stream, flush=True)
        if note:
            print(f"[{self._worker_id}] note: {note}", file=self._stream, flush=True)

    def _render_stages(self) -> None:
        report = self.stages.format()
        if report:
            joined = report.replace(chr(10), "; ")
            print(f"[{self._worker_id}] {joined}", file=self._stream, flush=True)


# ------------------------------------------------------------- Modal function


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=OCR_TIMEOUT,
    volumes={VOLUME_MOUNT_PATH: volume},
)
def run_ocr_shard(
    shard_id: int,
    num_shards: int,
    max_records: int | None = None,
    shards_root: str = SHARDS_ROOT,
) -> None:
    """Run one deterministic OCR shard on a Modal GPU worker.

    Expected input:
        - shard_id: shard index in ``[0, num_shards)``.
        - num_shards: total number of shards (same for every worker).
        - max_records: optional cap on rows processed by this shard (used by
          ``--validate``).
        - shards_root: Volume root for shard artifacts (overridden for
          ``--validate`` so real shards are untouched).

    Returns:
        - None. The shard's OCR artifact (``checkpoint.json``,
          ``ocr.ndjson``, ``manifest.json``) is written to
          ``<shards_root>/shard_XXX`` and its filtered source frames to
          ``/data/interim/ocr/keyframes/shard_XXX/frames.ndjson``.

    Raises:
        - RuntimeError: if the shard's OCR build fails.
    """
    ensure_keyframes_extracted()

    worker_id = f"shard_{shard_id:03d}"
    print(f"[{worker_id}] starting shard {shard_id}/{num_shards - 1}", flush=True)

    canonical = Path(KEYFRAMES_DIR) / FRAMES_FILENAME
    with canonical.open(encoding="utf-8") as handle:
        total_records = sum(1 for _ in handle)
    start, end = shard_row_bounds(total_records, num_shards, shard_id)

    shard_frames_path = Path(KEYFRAMES_SHARDS_ROOT) / worker_id / FRAMES_FILENAME
    shard_frames_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = write_shard_frames(
        canonical, shard_frames_path, start, end, limit=max_records
    )
    print(
        f"[{worker_id}] wrote {rows_written} rows [{start}, {end}) "
        f"-> {shard_frames_path}",
        flush=True,
    )
    volume.commit()

    local_keyframes = LOCAL_KEYFRAMES_ROOT / f"keyframes_{worker_id}"
    shared_images = Path(KEYFRAMES_DIR) / IMAGES_DIRNAME
    build_shard_keyframes_dir(local_keyframes, shard_frames_path, shared_images)

    output_dir = Path(shards_root) / worker_id
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = ShardHeartbeatReporter(worker_id, _gpu_info())
    print(
        f"[{worker_id}] running OCR -> {output_dir} (gpu={_gpu_info()})",
        flush=True,
    )
    try:
        generate_ocr(
            keyframes_dir=local_keyframes,
            output_dir=output_dir,
            backend="paddleocr",
            device="cuda",
            batch_size=BATCH_SIZE,
            resume=True,
            max_records=max_records,
            progress=reporter,
            languages=("vi", "en"),
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure with context
        raise RuntimeError(f"shard {worker_id} OCR failed: {exc}") from exc
    finally:
        volume.commit()

    print(f"[{worker_id}] finished shard {worker_id} at {output_dir}", flush=True)


# ----------------------------------------------------------------- launcher


def _local_canonical_frames() -> Path:
    """Return the repo-local canonical frames file (used to compute shard count)."""
    return REPO_ROOT / "data" / "keyframes" / "adaptive_sbd_v1" / FRAMES_FILENAME


def _count_local_frames(path: Path) -> int:
    """Count non-empty lines in a local frames file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"cannot compute --shard-size without {path}; "
            "use --num-workers instead or provide the data directory"
        )
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() != "")


@app.local_entrypoint()
def main(
    num_workers: int = DEFAULT_NUM_WORKERS,
    shard_size: int | None = None,
    gpu: str = DEFAULT_GPU,
    validate: bool = False,
) -> None:
    """Launch the distributed OCR job.

    ``--num-workers N`` splits the collection into N shards. ``--shard-size S``
    is an alternative that splits into ``ceil(total / S)`` shards (requires the
    canonical frames file locally to count rows). ``--gpu`` selects the worker
    GPU (default ``L4``). ``--validate`` runs a tiny shard into
    ``shards-validate`` so the real shard artifacts are never polluted.
    """
    if validate:
        run_ocr_shard.remote(
            shard_id=0,
            num_shards=1,
            max_records=VALIDATE_MAX_RECORDS,
            shards_root=SHARDS_VALIDATE_ROOT,
        )
        return

    if not isinstance(gpu, str) or gpu == "":
        raise ValueError(f"gpu must be a non-empty string, got {gpu!r}")

    if shard_size is not None:
        if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
            raise ValueError(f"shard_size must be a positive integer, got {shard_size!r}")
        total = _count_local_frames(_local_canonical_frames())
        num_shards = math.ceil(total / shard_size)
        if num_shards < 1:
            raise ValueError(f"shard_size {shard_size} exceeds the {total} available rows")
        print(
            f"shard_size={shard_size} rows -> {num_shards} shards over {total} rows",
            flush=True,
        )
    else:
        if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 1:
            raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")
        num_shards = num_workers

    fn = run_ocr_shard.with_options(gpu=gpu)
    print(f"dispatching {num_shards} shards on gpu={gpu}", flush=True)
    list(fn.map(range(num_shards), kwargs={"num_shards": num_shards}))
    print(f"all {num_shards} shards completed", flush=True)
