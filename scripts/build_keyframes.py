"""Offline asset builder: raw videos -> canonical keyframe artifact.

Consolidates the former per-stage scripts into one entrypoint with stable
subcommands; each subcommand wraps the same assets-module implementation and
flag surface as the previous standalone script:

    manifest            discover .mp4 inputs -> videos.json
    split               partition videos.json into worker manifests
    extract             resumable keyframe extraction (omnishotcut_adaptive)
    merge               merge completed shards -> frames.ndjson artifact
    verify              validate an extracted artifact
    materialize         decode selected frames -> adaptive_sbd_v1 images/
    verify-materialize  validate a materialized artifact
    verify-split        validate worker split manifests

Typical rebuild flow:

    python scripts/build_keyframes.py manifest --video-dir data/raw/aic2026_batch1 --output data/interim/videos/videos.json
    python scripts/build_keyframes.py split --videos data/interim/videos/videos.json --output-dir data/keyframe-splits --workers 5
    python scripts/build_keyframes.py extract --videos <worker videos.json> --output-dir outputs/keyframes_worker_000
    python scripts/build_keyframes.py merge --shard-dir outputs/keyframes_worker_000 ... --output-dir outputs/keyframes_full
    python scripts/build_keyframes.py materialize --artifact-dir outputs/keyframes_full --output-dir data/keyframes/adaptive_sbd_v1
"""

from __future__ import annotations

import sys



def _cmd_manifest(argv: list[str] | None = None) -> int:
    """discover .mp4 videos and write the canonical videos.json manifest (formerly scripts/build_video_manifest)."""
    #!/usr/bin/env python3
    """CLI: build a canonical video inventory manifest (``videos.json``).

    Discovers ``*.mp4`` files under ``--video-dir``, derives each ``video_id``
    from the filename stem, probes per-video metadata with ``ffprobe`` (unless
    ``--no-metadata``), and writes a deterministic, atomic ``videos.json``
    manifest consumed by the keyframe extraction split/execution workflow.

    This is a thin CLI; the generation logic lives in
    ``aic2026.assets.video_manifest``.
    """


    import argparse
    import sys

    from aic2026.assets.video_manifest import (
        VideoManifestError,
        build_video_manifest,
        save_video_manifest,
    )


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="build_video_manifest",
            description=(
                "Discover .mp4 videos and write a canonical videos.json inventory "
                "for keyframe extraction."
            ),
        )
        parser.add_argument(
            "--video-dir",
            required=True,
            help="directory containing the .mp4 video files",
        )
        parser.add_argument(
            "--output",
            required=True,
            help="target videos.json manifest path",
        )
        parser.add_argument(
            "--no-metadata",
            action="store_true",
            help="skip ffprobe metadata probing (fps/duration/frames left unknown)",
        )
        return parser


    def _format_bytes(size: int) -> str:
        """Return a compact human-readable byte size."""
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                if unit == "B":
                    return f"{int(value)} B"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{size} B"


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            manifest = build_video_manifest(args.video_dir, include_metadata=not args.no_metadata)
            save_video_manifest(args.output, manifest)
        except (FileNotFoundError, ValueError, VideoManifestError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        total_size = sum(entry.file_size_bytes for entry in manifest.videos)
        fps_available = sum(1 for entry in manifest.videos if entry.fps is not None)
        frames_available = sum(1 for entry in manifest.videos if entry.num_frames is not None)

        print("Video manifest created")
        print()
        print("Videos:")
        print(len(manifest.videos))
        print()
        print("Total size:")
        print(_format_bytes(total_size))
        print()
        print("Metadata:")
        print("fps available:")
        print(fps_available)
        print("frames available:")
        print(frames_available)
        return 0
    return main(argv)


def _cmd_split(argv: list[str] | None = None) -> int:
    """partition videos.json into deterministic worker manifests (formerly scripts/split_keyframe_extraction_shards)."""
    #!/usr/bin/env python3
    """CLI: deterministically split a video manifest into worker inputs.

    Reads a ``videos.json`` manifest and partitions its videos into ``N`` worker
    manifests (``worker_000/videos.json``, ...), keeping each video intact and
    balancing estimated frame counts per worker (using the optional ``num_frames``
    weight). The split is deterministic: the same input and ``--num-workers``
    always produce identical worker files. Writes a ``split_manifest.json``
    describing the split.

    This is a thin CLI; the splitting logic lives in
    ``aic2026.assets.keyframe_extraction_split``.
    """


    import argparse
    import sys

    from aic2026.assets.keyframe_extraction_split import split_video_manifest


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="split_keyframe_extraction_shards",
            description=(
                "Partition a videos.json manifest into balanced, deterministic "
                "worker manifests, keeping videos intact."
            ),
        )
        parser.add_argument(
            "--videos",
            required=True,
            help="path to a videos.json manifest to split",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            required=True,
            help="number of worker splits to produce (>= 1)",
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory to write worker_*/videos.json and split_manifest.json",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing split instead of failing when present",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            manifest = split_video_manifest(
                args.videos,
                args.num_workers,
                args.output_dir,
                overwrite=args.overwrite,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Split complete: {sum(w.num_videos for w in manifest.workers)} videos "
            f"across {manifest.num_workers} workers"
        )
        for worker in manifest.workers:
            print(
                f"  {worker.worker_id:03d}: {worker.num_videos} videos "
                f"({worker.estimated_frames} estimated frames)"
            )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_extract(argv: list[str] | None = None) -> int:
    """resumable keyframe extraction over a videos.json manifest (formerly scripts/extract_keyframes)."""
    #!/usr/bin/env python3
    """CLI: run resumable keyframe extraction over a video manifest.

    Reads a ``videos.json`` manifest (``video_id`` + ``path`` entries), runs the
    DAKE-inspired extractor over every video, and writes a canonical keyframe
    artifact: ``frames.ndjson``, ``map-keyframes/<VIDEO_ID>.csv``,
    ``manifest.json``, and (during processing) ``checkpoint.json``. Re-running the
    same command after an interruption resumes from the checkpoint; a changed
    configuration on resume is rejected.

    This is a thin CLI; the pipeline lives in ``aic2026.assets.keyframe_extraction``.
    """


    import argparse
    import logging
    import sys

    from aic2026.assets.keyframe_extraction import (
        FRAMES_FILENAME,
        ExtractionConfig,
        extract_keyframes,
    )
    from aic2026.progress import PROGRESS_MODES, create_progress_reporter


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="extract_keyframes",
            description=(
                "Extract keyframes from a video manifest into a canonical "
                "frames.ndjson + map-keyframes artifact (resumable)."
            ),
        )
        parser.add_argument(
            "--videos",
            type=str,
            required=True,
            help="path to a videos.json manifest (video_id + path entries)",
        )
        parser.add_argument(
            "--video-root",
            type=str,
            default=None,
            help=(
                "root directory that relative video paths in the manifest are "
                "resolved against (overrides the manifest's source_dir); required "
                "when the manifest is relocatable and videos live outside "
                "source_dir"
            ),
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=True,
            help=(
                f"directory to write {FRAMES_FILENAME}, map-keyframes/, "
                "manifest.json, and checkpoint.json"
            ),
        )
        target_group = parser.add_mutually_exclusive_group()
        target_group.add_argument(
            "--target-keyframes-per-video",
            type=int,
            default=None,
            help="hard cap on total keyframes per video, including anchors (mutually exclusive with --sampling-ratio; dake extractor only)",
        )
        target_group.add_argument(
            "--sampling-ratio",
            type=float,
            default=None,
            help="fraction of frames selected as keyframes, e.g. 0.1 (mutually exclusive with --target-keyframes-per-video; dake extractor only)",
        )
        parser.add_argument(
            "--extractor",
            choices=("dake", "omnishotcut_adaptive"),
            default="dake",
            help="keyframe extractor: dake (fixed-budget, DAKE-inspired) or omnishotcut_adaptive (OmniShotCut shots + adaptive sampling) (default: dake)",
        )
        parser.add_argument(
            "--detector",
            choices=("omnishotcut", "synthetic"),
            default="omnishotcut",
            help="shot detector used by the omnishotcut_adaptive extractor (default: omnishotcut)",
        )
        parser.add_argument(
            "--sampler",
            choices=("adaptive_v1",),
            default="adaptive_v1",
            help="per-shot sampler used by the omnishotcut_adaptive extractor (default: adaptive_v1)",
        )
        parser.add_argument(
            "--short-shot-seconds",
            type=float,
            default=5.0,
            help="shots shorter than this (seconds) get only their middle frame (default: 5.0)",
        )
        parser.add_argument(
            "--long-shot-seconds",
            type=float,
            default=60.0,
            help="shots at least this long (seconds) also get high-change frames (default: 60.0)",
        )
        parser.add_argument(
            "--uniform-interval-seconds",
            type=float,
            default=10.0,
            help="uniform keyframe spacing (seconds) for non-short shots (default: 10.0)",
        )
        parser.add_argument(
            "--omnishotcut-mode",
            choices=("clean_shot", "default"),
            default="clean_shot",
            help="OmniShotCut prediction mode (default: clean_shot)",
        )
        parser.add_argument(
            "--omnishotcut-overlap",
            type=int,
            default=20,
            help="OmniShotCut overlap frames between inference windows (default: 20)",
        )
        parser.add_argument(
            "--device",
            choices=("auto", "cpu", "mps", "cuda"),
            default="auto",
            help="device for the OmniShotCut model (default: auto)",
        )
        parser.add_argument(
            "--omnishotcut-checkpoint",
            type=str,
            default=None,
            help="OmniShotCut checkpoint: local path or HuggingFace repo id (default: uva-cv-lab/OmniShotCut)",
        )
        parser.add_argument(
            "--omnishotcut-filename",
            type=str,
            default="OmniShotCut_ckpt.pth",
            help="OmniShotCut checkpoint filename inside a HuggingFace repo id (default: OmniShotCut_ckpt.pth)",
        )
        parser.add_argument(
            "--frame-reader",
            choices=("ffmpeg", "synthetic"),
            default="ffmpeg",
            help="frame source: ffmpeg (real videos) or synthetic (test descriptors) (default: ffmpeg)",
        )
        parser.add_argument(
            "--scorer",
            choices=("frame_difference", "jpeg_size"),
            default="frame_difference",
            help="change-scoring method: frame_difference (mean abs pixel change) or jpeg_size (DAKE-style JPEG-compressed-size change) (default: frame_difference)",
        )
        parser.add_argument(
            "--jpeg-quality",
            type=int,
            default=70,
            help="JPEG quality in [1, 100] used by the jpeg_size scorer (default: 70)",
        )
        parser.add_argument(
            "--anchor-interval-seconds",
            type=float,
            default=None,
            help="add a uniform keyframe anchor every N seconds between the always-kept first and last frames (default: none)",
        )
        parser.add_argument(
            "--min-temporal-distance",
            type=int,
            default=30,
            help="minimum frame gap between selected keyframes (default: 30)",
        )
        parser.add_argument(
            "--peak-threshold",
            type=float,
            default=8.0,
            help="minimum change score for a candidate peak (default: 8.0)",
        )
        parser.add_argument(
            "--max-side",
            type=int,
            default=64,
            help="longest side (pixels) of downscaled grayscale frames (default: 64)",
        )
        parser.add_argument(
            "--fps",
            type=float,
            default=None,
            help="override the video frame rate (default: probe from the video)",
        )
        parser.add_argument(
            "--progress",
            choices=PROGRESS_MODES,
            default="auto",
            help=(
                "progress reporting: auto (interactive on a terminal, periodic "
                "logs otherwise), on (force interactive), or off (default: auto)"
            ),
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            default=True,
            help="resume from an existing matching checkpoint (default: enabled)",
        )
        parser.add_argument(
            "--no-resume",
            action="store_false",
            dest="resume",
            help="ignore any existing checkpoint and start fresh (refused if the output is non-empty without --overwrite)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing artifact instead of failing when present",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="enable verbose logging.",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

        config = ExtractionConfig(
            extractor=args.extractor,
            scorer=args.scorer,
            frame_reader=args.frame_reader,
            detector=args.detector,
            sampler=args.sampler,
            short_shot_seconds=args.short_shot_seconds,
            long_shot_seconds=args.long_shot_seconds,
            uniform_interval_seconds=args.uniform_interval_seconds,
            omnishotcut_mode=args.omnishotcut_mode,
            omnishotcut_overlap=args.omnishotcut_overlap,
            device=args.device,
            checkpoint=args.omnishotcut_checkpoint,
            omnishotcut_filename=args.omnishotcut_filename,
            min_temporal_distance=args.min_temporal_distance,
            peak_threshold=args.peak_threshold,
            target_keyframes_per_video=args.target_keyframes_per_video,
            sampling_ratio=args.sampling_ratio,
            max_side=args.max_side,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            anchor_interval_seconds=args.anchor_interval_seconds,
        )
        reporter = create_progress_reporter(args.progress, unit="videos")
        reporter.begin()
        try:
            manifest = extract_keyframes(
                args.videos,
                args.output_dir,
                config,
                progress=reporter,
                resume=args.resume,
                overwrite=args.overwrite,
                video_root=args.video_root,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Extraction complete: {manifest.num_keyframes} keyframes across "
            f"{manifest.num_videos} videos (config_hash={manifest.config_hash[:8]})"
        )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_merge(argv: list[str] | None = None) -> int:
    """merge completed extraction shards into one canonical artifact (formerly scripts/merge_keyframe_shards)."""
    #!/usr/bin/env python3
    """CLI: merge completed keyframe-extraction shards into one artifact.

    Reads completed extraction directories (each with ``manifest.json`` and
    ``frames.ndjson``), validates that they share the same extractor, scorer,
    frame reader, and extraction configuration, rejects any duplicate keyframes or
    videos spanning multiple shards, and concatenates them in deterministic order
    into a merged ``frames.ndjson``, ``map-keyframes/<VIDEO_ID>.csv``, and
    ``manifest.json``.

    This is a thin CLI; the merge logic lives in
    ``aic2026.assets.keyframe_extraction_merge``.
    """


    import argparse
    import sys
    from pathlib import Path

    from aic2026.assets.keyframe_extraction_merge import merge_shards


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="merge_keyframe_shards",
            description=(
                "Merge completed keyframe-extraction shards into one canonical "
                "artifact, validating config consistency and rejecting duplicates."
            ),
        )
        parser.add_argument(
            "--shards",
            nargs="+",
            type=Path,
            required=True,
            help="one or more completed extraction directories to merge",
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory to write merged frames.ndjson, map-keyframes/, manifest.json",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing merged artifact instead of failing when present",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            manifest = merge_shards(
                args.shards,
                args.output_dir,
                overwrite=args.overwrite,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Merge complete: {manifest.num_keyframes} keyframes for "
            f"{manifest.num_videos} videos from {len(manifest.source_shards)} shards"
        )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_verify(argv: list[str] | None = None) -> int:
    """verify an extracted worker/merged keyframe artifact (formerly scripts/verify_keyframe_extraction)."""
    #!/usr/bin/env python3
    """CLI: verify a completed keyframe extraction artifact before upload.

    Reads an extraction output (``outputs/keyframes_worker_XXX/`` or a merged
    ``outputs/keyframes_full/`` with ``frames.ndjson`` + ``map-keyframes/`` +
    ``manifest.json``) and verifies it is internally consistent. Worker artifacts
    (``source_videos``) and merged artifacts (``source_shards``) are both
    supported. When ``--videos`` (a worker's assigned ``videos.json``) is
    provided, coverage is checked against it for worker artifacts.

    This is a thin CLI; the verification logic lives in
    ``aic2026.assets.keyframe_extraction_verify.verify_keyframe_extraction``.
    """


    import argparse
    import sys

    from aic2026.assets.keyframe_extraction_verify import verify_keyframe_extraction


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="verify_keyframe_extraction",
            description=(
                "Validate a completed keyframe extraction artifact (worker or "
                "merged): manifest, frame records, deterministic ordering, video "
                "coverage, and per-video map-keyframes files."
            ),
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory containing frames.ndjson, map-keyframes/, and manifest.json",
        )
        parser.add_argument(
            "--videos",
            default=None,
            help="optional videos.json assigned to this worker (checks coverage)",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code (0 valid, 1 invalid)."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            report = verify_keyframe_extraction(args.output_dir, videos_path=args.videos)
        except (FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        status = "PASS" if report.valid else "FAIL"
        print(f"Keyframe extraction verification: {status}")
        print()
        print(f"Artifact type: {report.artifact_type}")
        print()
        print("Videos:")
        print(f"  expected: {report.expected_videos}")
        print(f"  found: {report.found_videos}")
        print()
        print("Keyframes:")
        print(f"  total: {report.total_keyframes}")
        print(f"  duplicates: {report.duplicates}")
        print()
        print("Manifest:")
        print(f"  extractor: {report.extractor}")
        print(f"  detector: {report.detector}")
        print(f"  sampler: {report.sampler}")

        if not report.valid:
            print()
            print("Validation failed:")
            for issue in report.issues:
                print(f"  [{issue.section}] {issue.message}")
            return 1

        print()
        print("OK: extraction artifact is valid")
        return 0
    return main(argv)


def _cmd_materialize(argv: list[str] | None = None) -> int:
    """decode selected frames into the adaptive_sbd_v1 images/ artifact (formerly scripts/materialize_keyframes)."""
    #!/usr/bin/env python3
    """CLI: materialize adaptive keyframe images from the original videos.

    Reads a ``videos.json`` manifest and a canonical artifact ``frames.ndjson``
    (produced by extraction or merge), decodes the selected frames from each
    original mp4, and writes ``images/<video_id>/<NNN>.jpg`` (the ``adaptive_sbd_v1``
    artifact) plus a validated copy of ``frames.ndjson`` and ``manifest.json``.
    The output replaces the BTC-provided keyframe image collection as the input
    for the embedding step.

    This is a thin CLI; the pipeline lives in ``aic2026.assets.keyframe_materialize``.
    """


    import argparse
    import logging
    import sys

    from aic2026.assets.keyframe_extraction import FRAMES_FILENAME, MANIFEST_FILENAME
    from aic2026.assets.keyframe_materialize import (
        IMAGE_EXTENSION,
        IMAGES_DIRNAME,
        materialize_keyframes,
        verify_materialization,
    )
    from aic2026.progress import PROGRESS_MODES


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="materialize_keyframes",
            description=(
                "Decode the selected keyframes from the original videos into "
                f"{IMAGES_DIRNAME}/<video_id>/<ordinal>.{IMAGE_EXTENSION} images "
                "for the embedding step."
            ),
        )
        parser.add_argument(
            "--videos",
            type=str,
            required=True,
            help="path to a videos.json manifest (video_id + path entries)",
        )
        parser.add_argument(
            "--frames",
            type=str,
            required=True,
            help=f"path to a canonical artifact {FRAMES_FILENAME} (extraction or merge output)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=True,
            help=(
                f"directory to write {IMAGES_DIRNAME}/, {FRAMES_FILENAME}, and "
                f"{MANIFEST_FILENAME} (the adaptive_sbd_v1 image artifact)"
            ),
        )
        parser.add_argument(
            "--video-root",
            type=str,
            default=None,
            help=(
                "root directory that relative video paths in the manifest are "
                "resolved against (overrides the manifest's source_dir); required "
                "when the manifest is relocatable and videos live outside "
                "source_dir"
            ),
        )
        parser.add_argument(
            "--jpeg-quality",
            type=int,
            default=2,
            help="ffmpeg mjpeg quality -q:v in [1, 31]; lower is better (default: 2)",
        )
        parser.add_argument(
            "--progress",
            choices=PROGRESS_MODES,
            default="auto",
            help=(
                "progress reporting: auto (interactive on a terminal, periodic "
                "logs otherwise), on (force interactive), or off (default: auto)"
            ),
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing artifact instead of failing when present",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="enable verbose logging.",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.WARNING,
            format="%(levelname)s %(name)s: %(message)s",
        )

        try:
            manifest = materialize_keyframes(
                args.videos,
                args.frames,
                args.output_dir,
                video_root=args.video_root,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
                progress=args.progress,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Materialization complete: {manifest.num_keyframes} keyframe images "
            f"across {manifest.num_videos} videos"
        )
        print(f"Output: {args.output_dir}")
        report = verify_materialization(args.output_dir)
        if not report.valid:
            for issue in report.issues:
                print(f"verification issue [{issue.section}]: {issue.message}", file=sys.stderr)
            return 1
        print(
            f"Verification: valid ({report.num_keyframes} images, "
            f"{report.missing_images} missing)"
        )
        return 0
    return main(argv)


def _cmd_verify_materialize(argv: list[str] | None = None) -> int:
    """verify a materialized images/ artifact (formerly scripts/verify_keyframe_materialization)."""
    #!/usr/bin/env python3
    """CLI: verify a completed keyframe-image materialization artifact.

    Reads a materialized ``adaptive_sbd_v1`` artifact (``images/`` +
    ``frames.ndjson`` + ``manifest.json``) and verifies it is internally
    consistent: artifact existence, manifest counts, and image coverage (every
    record has its ``images/<video_id>/<ordinal:06d>.jpg`` and the image count
    matches the canonical frame records).

    This is a thin CLI; the verification logic lives in
    ``aic2026.assets.keyframe_materialize.verify_materialization``.
    """


    import argparse
    import sys

    from aic2026.assets.keyframe_materialize import verify_materialization


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="verify_keyframe_materialization",
            description=(
                "Validate a materialized adaptive_sbd_v1 keyframe image artifact: "
                "manifest, frame records, and image coverage."
            ),
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory containing images/, frames.ndjson, and manifest.json",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code (0 valid, 1 invalid)."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            report = verify_materialization(args.output_dir)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        status = "PASS" if report.valid else "FAIL"
        print(f"Keyframe materialization verification: {status}")
        print()
        print("Artifact:")
        print("  type: adaptive_sbd_v1")
        print()
        print("Keyframes:")
        print(f"  records: {report.num_keyframes}")
        print(f"  images: {report.images_present}")
        print(f"  missing: {report.missing_images}")

        if not report.valid:
            print()
            print("Validation failed:")
            for issue in report.issues:
                print(f"  [{issue.section}] {issue.message}")
            return 1

        print()
        print("OK: materialized keyframe artifact is valid")
        return 0
    return main(argv)


def _cmd_verify_split(argv: list[str] | None = None) -> int:
    """verify worker split manifests against the source inventory (formerly scripts/verify_video_split)."""
    #!/usr/bin/env python3
    """CLI: validate a deterministic video split before distributing workers.

    Reads a split directory produced by ``split_keyframe_extraction_shards`` and
    verifies it is internally consistent: ``split_manifest.json`` exists, every
    worker ``videos.json`` exists and parses, the video/frame totals match the
    recorded manifest, and no video is duplicated or missing across workers.

    This is a thin CLI; the verification logic lives in
    ``aic2026.assets.keyframe_extraction_split.verify_video_split``.
    """


    import argparse
    import sys

    from aic2026.assets.keyframe_extraction_split import verify_video_split


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="verify_video_split",
            description=(
                "Validate that a video split directory is consistent: total "
                "videos match, no duplicates, no missing videos, frame counts "
                "match split_manifest.json."
            ),
        )
        parser.add_argument(
            "--split-dir",
            required=True,
            help="directory containing split_manifest.json and worker_*/videos.json",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code (0 valid, 1 invalid)."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            report = verify_video_split(args.split_dir)
        except (FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        for worker in report.workers:
            print(f"worker_{worker.worker_id:03d}:")
            print(f"  videos: {worker.num_videos}")
            print(f"  frames: {worker.num_frames}")

        print("Summary:")
        print(f"Total videos: {report.total_videos}")
        print(f"Unique videos: {report.unique_videos}")
        print(f"Duplicates: {report.duplicates}")
        print(f"Total frames: {report.total_frames}")

        if not report.valid:
            print("Validation failed:")
            for issue in report.issues:
                if issue.worker_id is not None:
                    print(f"  worker_{issue.worker_id:03d}: {issue.message}")
                else:
                    print(f"  {issue.message}")
            return 1

        print("OK: split is valid")
        return 0
    return main(argv)


_DISPATCH = {
    "manifest": _cmd_manifest,
    "split": _cmd_split,
    "extract": _cmd_extract,
    "merge": _cmd_merge,
    "verify": _cmd_verify,
    "materialize": _cmd_materialize,
    "verify-materialize": _cmd_verify_materialize,
    "verify-split": _cmd_verify_split,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to one asset-builder stage."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        avail = ", ".join(sorted(_DISPATCH))
        print(f"usage: {sys.argv[0]} <{avail}> [stage options...]\n"
              "pass '-h' after a stage for its options", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    handler = _DISPATCH.get(command)
    if handler is None:
        print(f"error: unknown stage {command!r}; expected one of: "
              + ", ".join(sorted(_DISPATCH)), file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
