"""CLI entrypoint for the AIC 2026 project.

Commands:
    inspect          - inspect the raw dataset and print a concise report.

CLI logic stays thin; the real work lives in reusable modules
(``aic2026.inspection``, ``aic2026.config``).
"""

import argparse
import logging
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.inspection import DatasetReport, inspect_dataset

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Expected input:
        - None.

    Returns:
        - The configured :class:`argparse.ArgumentParser`.

    Raises:
        - No exceptions.
    """
    parser = argparse.ArgumentParser(
        prog="aic2026",
        description="AIC 2026 dataset inspection and validation tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the raw dataset and print a report.",
    )
    inspect_parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Path to the raw dataset root (overrides config).",
    )
    inspect_parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Only inspect at most MAX_VIDEOS videos (ordered by video id).",
    )
    inspect_parser.add_argument(
        "--sample-objects",
        type=int,
        default=3,
        help="Number of object JSON files to schema-check per video (0 to skip).",
    )
    inspect_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def print_report(report: DatasetReport) -> None:
    """Print a human-readable report to stdout.

    Expected input:
        - report: the report produced by :func:`inspect_dataset`.

    Returns:
        - None (prints to stdout).

    Raises:
        - No exceptions.
    """
    print(f"Raw dataset root: {report.raw_root}")
    print(f"Unique videos: {len(report.video_ids)}")
    print("Collections:")
    for name in ("clip-features-32", "keyframes", "map-keyframes", "media-info", "objects"):
        print(f"  {name}: {report.collection_counts.get(name, 0)}")
    print(f"Total keyframes: {report.total_keyframes}")
    print(f"Total objects: {report.total_objects}")
    print(f"Total map-keyframes rows: {report.map_rows_total}")
    print("Embedding shapes:")
    for shape, count in report.embedding_shapes.most_common():
        print(f"  {shape}: {count}")
    print("Embedding dtypes:")
    for dtype, count in report.embedding_dtypes.most_common():
        print(f"  {dtype}: {count}")
    print("FPS distribution (videos):")
    for fps, count in report.fps_histogram.most_common():
        print(f"  {fps}: {count}")

    if report.object_payload_problems:
        print("Object payload schema issues (sampled):")
        for vid, problems in report.object_payload_problems.items():
            print(f"  {vid}:")
            for problem in problems[:5]:
                print(f"    - {problem}")

    if report.inconsistent_videos:
        print("Videos with consistency problems:")
        for vid, problems in report.inconsistent_videos.items():
            print(f"  {vid}:")
            for problem in problems[:5]:
                print(f"    - {problem}")
    else:
        print("Videos with consistency problems: none")

    if report.videos_with_media_issues:
        print("Videos with media-info issues:")
        for vid, problems in report.videos_with_media_issues.items():
            print(f"  {vid}: {problems}")

    print(f"Missing/unreadable expected files: {report.missing_files_total}")

    print("Observations:")
    for observation in report.observations:
        print(f"  - {observation}")


def configure_logging(verbose: bool) -> None:
    """Configure root logging level.

    Expected input:
        - verbose: whether to enable DEBUG logging.

    Returns:
        - None.

    Raises:
        - No exceptions.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Expected input:
        - argv: optional list of arguments (defaults to ``sys.argv[1:]``).

    Returns:
        - Exit code (0 on success, 2 on argument errors).

    Raises:
        - No exceptions (failures are reported via the exit code).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "inspect":
        config = load_config()
        raw_root = args.raw_root or config.data_raw_dir
        try:
            report = inspect_dataset(
                raw_root=raw_root,
                max_videos=args.max_videos,
                sample_objects=args.sample_objects,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print_report(report)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())