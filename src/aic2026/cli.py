"""CLI entrypoint for the AIC 2026 project.

Commands:
    inspect          - inspect the raw dataset and print a concise report.
    build-index      - build a retrieval index artifact from an embedding artifact.
    predict-kis      - generate ranked KIS predictions from an index artifact.
    predict-qa      - generate QA retrieval evidence (and optional VLM answers) from an index artifact.
    predict-trake   - generate per-event TRAKE retrieval evidence from an index artifact.
    export-qa-submission - export a QA artifact to the provisional BTC CSV.

CLI logic stays thin; the real work lives in reusable modules
(``aic2026.inspection``, ``aic2026.config``, ``aic2026.kis_prediction_runner``,
``aic2026.qa.qa_prediction_runner``, ``aic2026.trake``).
"""

import argparse
import logging
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.dataset import DatasetError
from aic2026.embedding_dataset import EmbeddingArtifactDataset
from aic2026.index_artifact import validate_index_files
from aic2026.index_builder import build_index
from aic2026.inspection import DatasetReport, inspect_dataset
from aic2026.image_embedder import create_text_encoder
from aic2026.kis_prediction_runner import (
    KisPredictionRunnerConfig,
    run_kis_prediction,
)
from aic2026.qa.qa_prediction_runner import QaPredictionRunnerConfig, read_qa_artifact, run_qa_prediction
from aic2026.qa.qa_submission import write_qa_submission_csv
from aic2026.trake import TrakePredictionRunnerConfig, run_trake_prediction

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

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build a retrieval index artifact from a verified embedding artifact.",
    )
    build_index_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="path to a verified embedding artifact directory (manifest.json + embeddings.npy + frames.ndjson)",
    )
    build_index_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path for the output index artifact directory (created if absent)",
    )
    build_index_parser.add_argument(
        "--artifact-id",
        required=True,
        help="unique run identity recorded in the index manifest",
    )
    build_index_parser.add_argument(
        "--batch-flush",
        type=int,
        default=1000,
        help="rows between memmap flush and checkpoint write (default: 1000)",
    )
    build_index_parser.add_argument(
        "--verify",
        action="store_true",
        help="run validate_index_files on the completed artifact and fail if inconsistent",
    )
    build_index_parser.add_argument(
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

    predict_qa_parser = subparsers.add_parser(
        "predict-qa",
        help="Run QA retrieval evidence generation and write a QA artifact directory.",
    )
    predict_qa_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    predict_qa_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    predict_qa_parser.add_argument(
        "--index-artifact",
        type=Path,
        required=True,
        help="Path to the index artifact directory.",
    )
    predict_qa_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output QA artifact directory (manifest.json + predictions.jsonl).",
    )
    predict_qa_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked frame retrieval candidates per query.",
    )
    predict_qa_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_vi",
        help="Query text field to encode.",
    )
    predict_qa_parser.add_argument(
        "--video-aggregation",
        choices=("none", "max", "mean", "mean_top_n"),
        default="none",
        help="Video-level ranking strategy (none = frame-level ranking).",
    )
    predict_qa_parser.add_argument(
        "--video-aggregation-top-n",
        type=int,
        default=None,
        help="top_n parameter for mean_top_n video aggregation.",
    )
    predict_qa_parser.add_argument(
        "--window-max-gap",
        type=int,
        default=None,
        help="Optional temporal window grouping gap (positive integer).",
    )
    predict_qa_parser.add_argument(
        "--query-id",
        dest="query_ids",
        action="append",
        default=None,
        help="Optional query id filter; repeat to select an explicit subset.",
    )
    predict_qa_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id recorded in the output artifact.",
    )
    predict_qa_parser.add_argument(
        "--allow-embedding-space-mismatch",
        action="store_true",
        help=(
            "Explicitly allow query/index embedding-space mismatch for diagnostic experiments. "
            "Disabled by default."
        ),
    )
    predict_qa_parser.add_argument(
        "--generate-answer",
        action="store_true",
        default=False,
        help="Enable Jina VLM answer generation over the retrieved evidence frames.",
    )
    predict_qa_parser.add_argument(
        "--answer-top-n",
        type=int,
        default=5,
        help="Number of diverse evidence frames passed to the Jina VLM per query (default: 5).",
    )
    predict_qa_parser.add_argument(
        "--answer-device",
        choices=("auto", "mps"),
        default="mps",
        help="Device for the Jina VLM answer generator (auto or mps).",
    )
    predict_qa_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    predict_trake_parser = subparsers.add_parser(
        "predict-trake",
        help="Run per-event TRAKE retrieval evidence generation and write a TRAKE artifact directory.",
    )
    predict_trake_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    predict_trake_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    predict_trake_parser.add_argument(
        "--index-artifact",
        type=Path,
        required=True,
        help="Path to the index artifact directory.",
    )
    predict_trake_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output TRAKE artifact directory (manifest.json + predictions.jsonl).",
    )
    predict_trake_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of ranked frame retrieval candidates per event.",
    )
    predict_trake_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_vi",
        help="Query text field to encode.",
    )
    predict_trake_parser.add_argument(
        "--window-max-gap",
        type=int,
        default=60,
        help="Temporal window grouping gap (positive integer; default: 60).",
    )
    predict_trake_parser.add_argument(
        "--query-id",
        dest="query_ids",
        action="append",
        default=None,
        help="Optional query id filter; repeat to select an explicit subset.",
    )
    predict_trake_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id recorded in the output artifact.",
    )
    predict_trake_parser.add_argument(
        "--allow-embedding-space-mismatch",
        action="store_true",
        help=(
            "Explicitly allow query/index embedding-space mismatch for diagnostic experiments. "
            "Disabled by default."
        ),
    )
    predict_trake_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    export_qa_parser = subparsers.add_parser(
        "export-qa-submission",
        help="Export a QA artifact directory to the provisional BTC submission CSV.",
    )
    export_qa_parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Path to a QA artifact directory (manifest.json + predictions.jsonl).",
    )
    export_qa_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output submission CSV.",
    )
    export_qa_parser.add_argument(
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

    if args.command == "build-index":
        try:
            source = EmbeddingArtifactDataset(args.source)
            manifest = build_index(
                source,
                args.output,
                artifact_id=args.artifact_id,
                batch_flush=args.batch_flush,
                embedding_space=source.embedding_space,
            )
        except (FileNotFoundError, ValueError, OSError, DatasetError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Index artifact: {args.output}")
        print(f"Artifact id: {manifest.artifact_id}")
        print(f"Rows: {manifest.num_rows}  Videos: {manifest.num_videos}")
        if args.verify:
            problems = validate_index_files(manifest, Path(args.output))
            if problems:
                for problem in problems:
                    print(f"validation: {problem}", file=sys.stderr)
                return 1
            print("Verification: OK")
        return 0

    if args.command == "predict-kis":
        config = load_config(args.config)
        try:
            encoder = create_text_encoder(config)
            artifact = run_kis_prediction(
                KisPredictionRunnerConfig(
                    evaluation_set_path=args.evaluation_set,
                    index_artifact_dir=args.index_artifact,
                    prediction_output_path=args.output,
                    model_config=config.model_config,
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

    if args.command == "predict-qa":
        config = load_config(args.config)
        try:
            encoder = create_text_encoder(config)
            artifact = run_qa_prediction(
                QaPredictionRunnerConfig(
                    evaluation_set_path=args.evaluation_set,
                    index_artifact_dir=args.index_artifact,
                    prediction_output_dir=args.output,
                    model_config=config.model_config,
                    top_k=args.top_k,
                    query_text_field=args.query_text_field,
                    video_aggregation=None if args.video_aggregation == "none" else args.video_aggregation,
                    video_aggregation_top_n=args.video_aggregation_top_n,
                    window_max_gap=args.window_max_gap,
                    query_subset_ids=tuple(args.query_ids) if args.query_ids else None,
                    allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                    run_id=args.run_id,
                    generate_answer=args.generate_answer,
                    answer_top_n=args.answer_top_n,
                    answer_device=args.answer_device,
                    keyframes_dir=config.data_keyframes_dir if args.generate_answer else None,
                ),
                encoder,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"QA artifact: {args.output}")
        print(f"Processed queries: {artifact.manifest.processed_query_count}/{artifact.manifest.evaluation_set_query_count}")
        print(f"Index artifact id: {artifact.manifest.index_manifest['artifact_id']}")
        answer_generation = artifact.manifest.answer_generation
        if answer_generation is not None:
            print(
                f"Answer generation: model={answer_generation['model']} "
                f"device={answer_generation['device']} top_n={answer_generation['top_n']} "
                f"evidence_queries={answer_generation['evidence_query_count']} "
                f"failed={answer_generation['failed_query_count']}"
            )
        return 0

    if args.command == "predict-trake":
        config = load_config(args.config)
        try:
            encoder = create_text_encoder(config)
            artifact = run_trake_prediction(
                TrakePredictionRunnerConfig(
                    evaluation_set_path=args.evaluation_set,
                    index_artifact_dir=args.index_artifact,
                    prediction_output_dir=args.output,
                    model_config=config.model_config,
                    top_k=args.top_k,
                    query_text_field=args.query_text_field,
                    window_max_gap=args.window_max_gap,
                    query_subset_ids=tuple(args.query_ids) if args.query_ids else None,
                    allow_embedding_space_mismatch=args.allow_embedding_space_mismatch,
                    run_id=args.run_id,
                ),
                encoder,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"TRAKE artifact: {args.output}")
        print(f"Processed queries: {artifact.manifest.processed_query_count}/{artifact.manifest.evaluation_set_query_count}")
        print(f"Index artifact id: {artifact.manifest.index_manifest['artifact_id']}")
        return 0

    if args.command == "export-qa-submission":
        try:
            artifact = read_qa_artifact(args.artifact)
            write_qa_submission_csv(artifact, args.output)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"QA submission CSV: {args.output}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())