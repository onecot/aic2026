"""CLI entrypoint for the AIC 2026 project.

Commands:
    inspect          - inspect the raw dataset and print a concise report.
    predict-kis      - generate ranked KIS predictions from an index artifact.
    temporal-expand-kis - apply temporal window expansion to a KIS prediction
                          artifact and save an expanded prediction artifact.

CLI logic stays thin; the real work lives in reusable modules
(``aic2026.inspection``, ``aic2026.config``, ``aic2026.kis_prediction_runner``,
``aic2026.temporal_window_expansion``).
"""

import argparse
import logging
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.inspection import DatasetReport, inspect_dataset
from aic2026.kis_prediction_runner import KisPredictionRunnerConfig, run_kis_prediction
from aic2026.model_backends.openclip import OpenClipEncoder
from aic2026.temporal_window_expansion import (
    TemporalWindowConfig,
    write_expanded_prediction_artifact,
)

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

    predict_kis_parser = subparsers.add_parser(
        "predict-kis",
        help="Run KIS prediction generation and write a JSON artifact.",
    )
    predict_kis_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    predict_kis_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    predict_kis_parser.add_argument(
        "--index-artifact",
        type=Path,
        required=True,
        help="Path to the index artifact directory.",
    )
    predict_kis_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output prediction artifact JSON.",
    )
    predict_kis_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked retrieval candidates per query.",
    )
    predict_kis_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_vi",
        help="Query text field to encode.",
    )
    predict_kis_parser.add_argument(
        "--query-id",
        dest="query_ids",
        action="append",
        default=None,
        help="Optional query id filter; repeat to select an explicit subset.",
    )
    predict_kis_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id recorded in the output artifact.",
    )
    predict_kis_parser.add_argument(
        "--allow-embedding-space-mismatch",
        action="store_true",
        help=(
            "Explicitly allow query/index embedding-space mismatch for diagnostic experiments. "
            "Disabled by default."
        ),
    )
    predict_kis_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    expand_parser = subparsers.add_parser(
        "temporal-expand-kis",
        help="Apply temporal window expansion to a KIS prediction artifact.",
    )
    expand_parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Path to the source KIS prediction artifact JSON.",
    )
    expand_parser.add_argument(
        "--index-artifact",
        type=Path,
        default=None,
        help="Path to the index artifact directory (defaults to the artifact's recorded index_artifact_path).",
    )
    expand_parser.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="Maximum absolute frame offset for generated candidates.",
    )
    expand_parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Spacing between candidate offsets (must be a positive integer).",
    )
    expand_parser.add_argument(
        "--decay",
        type=float,
        required=True,
        help="Score multiplier for propagated candidates, in (0, 1].",
    )
    expand_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output expanded prediction artifact JSON.",
    )
    expand_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id recorded in the output artifact.",
    )
    expand_parser.add_argument(
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

    if args.command == "predict-kis":
        config = load_config(args.config)
        try:
            encoder = OpenClipEncoder(config.openclip)
            artifact = run_kis_prediction(
                KisPredictionRunnerConfig(
                    evaluation_set_path=args.evaluation_set,
                    index_artifact_dir=args.index_artifact,
                    prediction_output_path=args.output,
                    model_config={
                        "backend": config.openclip.backend,
                        "model_name": config.openclip.model_name,
                        "pretrained": config.openclip.pretrained,
                        "device": config.openclip.device,
                        "precision": config.openclip.precision,
                        "normalization": config.openclip.normalization,
                        "preprocessing_id": config.openclip.preprocessing_id,
                        "tokenizer_id": config.openclip.tokenizer_id,
                    },
                    top_k=args.top_k,
                    query_text_field=args.query_text_field,
                    query_subset_ids=tuple(args.query_ids) if args.query_ids else None,
                    allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                    run_id=args.run_id,
                ),
                encoder,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Prediction artifact: {args.output}")
        print(f"Processed queries: {artifact.processed_query_count}/{artifact.evaluation_set_query_count}")
        print(f"Index artifact id: {artifact.index_manifest['artifact_id']}")
        return 0

    if args.command == "temporal-expand-kis":
        try:
            config = TemporalWindowConfig(
                window_size=args.window_size,
                step=args.step,
                decay=args.decay,
            )
            artifact = write_expanded_prediction_artifact(
                output_path=args.output,
                source_path=args.prediction,
                config=config,
                index_artifact_dir=args.index_artifact,
                run_id=args.run_id,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Expanded prediction artifact: {args.output}")
        print(f"Processed queries: {artifact.processed_query_count}/{artifact.evaluation_set_query_count}")
        print(f"Run id: {artifact.run_id}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())