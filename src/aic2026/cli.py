"""CLI entrypoint for the AIC 2026 project.

Commands:
    inspect          - inspect the raw dataset and print a concise report.
    predict-kis      - generate ranked KIS predictions from an index artifact.
    temporal-expand-kis - apply temporal window expansion to a KIS prediction
                          artifact and save an expanded prediction artifact.
    media-info-search - rank videos by BM25 relevance over media-info text.
    media-info-evaluate - measure video-level retrieval of media-info text.
    media-info-report - categorize queries by media-info / CLIP GT-video retrieval.
    objects-analyze - analyze BTC object detections for KIS retrieval usefulness.
    object-rerank-kis - rerank a KIS prediction artifact with object detections.
    predict-qa      - generate QA retrieval evidence from an index artifact.
    export-qa-submission - export a QA artifact to the provisional BTC CSV.

CLI logic stays thin; the real work lives in reusable modules
(``aic2026.inspection``, ``aic2026.config``, ``aic2026.kis_prediction_runner``,
``aic2026.temporal_window_expansion``, ``aic2026.media_info_retrieval``,
``aic2026.media_info_evaluation``).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from aic2026.config import load_config
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.inspection import DatasetReport, inspect_dataset
from aic2026.kis_prediction_runner import (
    KisPredictionRunnerConfig,
    load_kis_predictions_for_scoring,
    run_kis_prediction,
)
from aic2026.media_info_evaluation import (
    category_report_to_dict,
    compare_to_clip_baseline,
    evaluate_media_info_video_retrieval,
    format_category_report,
    format_report,
)
from aic2026.media_info_retrieval import (
    MediaInfoIndex,
    MediaInfoRetrievalConfig,
)
from aic2026.model_backends.openclip import OpenClipEncoder
from aic2026.object_analysis import format_report as format_object_report
from aic2026.object_analysis import run_object_analysis
from aic2026.qa.qa_prediction_runner import QaPredictionRunnerConfig, read_qa_artifact, run_qa_prediction
from aic2026.qa.qa_submission import write_qa_submission_csv
from aic2026.object_reranking import (
    FrameObjectSource,
    ObjectRerankingConfig,
    write_reranked_prediction_artifact,
)
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

    rerank_parser = subparsers.add_parser(
        "object-rerank-kis",
        help=(
            "Rerank a KIS prediction artifact by combining CLIP scores with an "
            "object-score bonus from BTC object detections."
        ),
    )
    rerank_parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Path to the source KIS prediction artifact JSON.",
    )
    rerank_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the output re-ranked prediction artifact JSON.",
    )
    rerank_parser.add_argument(
        "--objects-dir",
        type=Path,
        default=None,
        help="Path to the raw objects collection (defaults to config raw_dir/objects).",
    )
    rerank_parser.add_argument(
        "--map-dir",
        type=Path,
        default=None,
        help=(
            "Path to the raw map-keyframes collection (defaults to the sibling "
            "map-keyframes directory of --objects-dir)."
        ),
    )
    rerank_parser.add_argument(
        "--lambda",
        dest="lambda_weight",
        type=float,
        default=1.0,
        help="Weight of the object score in the combined score (default: 1.0).",
    )
    rerank_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum object detection score to count (default: 0.3).",
    )
    rerank_parser.add_argument(
        "--normalization-method",
        choices=("none", "lower", "stem"),
        default="lower",
        help="Query/entity token normalization (default: lower).",
    )
    rerank_parser.add_argument(
        "--object-score-method",
        choices=("max", "sum"),
        default="max",
        help="Object-score aggregation over matched entities (default: max).",
    )
    rerank_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id recorded in the output artifact.",
    )
    rerank_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    media_info_parser = subparsers.add_parser(
        "media-info-search",
        help="Rank videos by BM25 relevance over media-info text.",
    )
    media_info_parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=None,
        help="Path to the raw media-info directory (defaults to config raw_dir/media-info).",
    )
    media_info_parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        default=None,
        help="Query text to rank videos for; repeat to run multiple queries.",
    )
    media_info_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked hits to return per query (default: 10).",
    )
    media_info_parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="BM25 term-frequency saturation (default: 1.5).",
    )
    media_info_parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="BM25 length normalization (default: 0.75).",
    )
    media_info_parser.add_argument(
        "--title-weight",
        type=float,
        default=None,
        help="BM25 weight for the title field (default: 3.0).",
    )
    media_info_parser.add_argument(
        "--keywords-weight",
        type=float,
        default=None,
        help="BM25 weight for the keywords field (default: 2.0).",
    )
    media_info_parser.add_argument(
        "--description-weight",
        type=float,
        default=None,
        help="BM25 weight for the description field (default: 1.0).",
    )
    media_info_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the ranked hits to.",
    )
    media_info_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    evaluate_parser = subparsers.add_parser(
        "media-info-evaluate",
        help="Measure video-level retrieval of media-info text against KIS ground truth.",
    )
    evaluate_parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=None,
        help="Path to the raw media-info directory (defaults to config raw_dir/media-info).",
    )
    evaluate_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    evaluate_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_en",
        help="Query text field to retrieve with (default: query_text_en).",
    )
    evaluate_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Media-info retrieval cutoff (default: 100).",
    )
    evaluate_parser.add_argument(
        "--clip-predictions",
        type=Path,
        default=None,
        help=(
            "Optional CLIP baseline prediction artifact for the error-analysis "
            "comparison (default: data/interim/predictions/kis-l14-laion2b-full-en.json "
            "when it exists)."
        ),
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the evaluation metrics to.",
    )
    evaluate_parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="BM25 term-frequency saturation (default: 1.5).",
    )
    evaluate_parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="BM25 length normalization (default: 0.75).",
    )
    evaluate_parser.add_argument(
        "--title-weight",
        type=float,
        default=None,
        help="BM25 weight for the title field (default: 3.0).",
    )
    evaluate_parser.add_argument(
        "--keywords-weight",
        type=float,
        default=None,
        help="BM25 weight for the keywords field (default: 2.0).",
    )
    evaluate_parser.add_argument(
        "--description-weight",
        type=float,
        default=None,
        help="BM25 weight for the description field (default: 1.0).",
    )
    evaluate_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    report_parser = subparsers.add_parser(
        "media-info-report",
        help=(
            "Categorize KIS queries by whether media-info and the CLIP baseline "
            "retrieve the GT video (MEDIA_ONLY / CLIP_ONLY / BOTH / NEITHER)."
        ),
    )
    report_parser.add_argument(
        "--media-info-dir",
        type=Path,
        default=None,
        help="Path to the raw media-info directory (defaults to config raw_dir/media-info).",
    )
    report_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    report_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_en",
        help="Query text field to retrieve with (default: query_text_en).",
    )
    report_parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Media-info retrieval cutoff (default: 100).",
    )
    report_parser.add_argument(
        "--clip-predictions",
        type=Path,
        default=None,
        help=(
            "Optional CLIP baseline prediction artifact for the categorization "
            "(default: data/interim/predictions/kis-l14-laion2b-full-en.json "
            "when it exists)."
        ),
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the categorized report to.",
    )
    report_parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="BM25 term-frequency saturation (default: 1.5).",
    )
    report_parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="BM25 length normalization (default: 0.75).",
    )
    report_parser.add_argument(
        "--title-weight",
        type=float,
        default=None,
        help="BM25 weight for the title field (default: 3.0).",
    )
    report_parser.add_argument(
        "--keywords-weight",
        type=float,
        default=None,
        help="BM25 weight for the keywords field (default: 2.0).",
    )
    report_parser.add_argument(
        "--description-weight",
        type=float,
        default=None,
        help="BM25 weight for the description field (default: 1.0).",
    )
    report_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    objects_parser = subparsers.add_parser(
        "objects-analyze",
        help="Analyze BTC object detections for KIS retrieval usefulness.",
    )
    objects_parser.add_argument(
        "--objects-dir",
        type=Path,
        default=None,
        help="Path to the raw objects collection (defaults to config raw_dir/objects).",
    )
    objects_parser.add_argument(
        "--map-dir",
        type=Path,
        default=None,
        help="Path to the raw map-keyframes collection (defaults to config raw_dir/map-keyframes).",
    )
    objects_parser.add_argument(
        "--evaluation-set",
        type=Path,
        default=Path("data/eval/eval_set.csv"),
        help="Path to evaluation CSV (default: data/eval/eval_set.csv).",
    )
    objects_parser.add_argument(
        "--clip-predictions",
        type=Path,
        default=None,
        help=(
            "Path to a CLIP KIS prediction artifact for the retrieval-potential "
            "section (default: data/interim/predictions/"
            "kis-l14-laion2b-full-en-top1000.json when it exists)."
        ),
    )
    objects_parser.add_argument(
        "--query-text-field",
        choices=("query_text_vi", "query_text_en"),
        default="query_text_en",
        help="Query text field used for object-word extraction (default: query_text_en).",
    )
    objects_parser.add_argument(
        "--confidence-thresholds",
        type=str,
        default="0.1,0.3,0.5,0.7",
        help="Comma-separated confidence thresholds for vocabulary retention effects.",
    )
    objects_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum detection score for GT / predicted object sets (default: 0.3).",
    )
    objects_parser.add_argument(
        "--top-classes",
        type=int,
        default=20,
        help="Number of top object classes reported (default: 20).",
    )
    objects_parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of CLIP predictions analyzed per query (default: 1000).",
    )
    objects_parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on videos scanned for vocabulary statistics.",
    )
    objects_parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of matched/unmatched coverage examples reported (default: 5).",
    )
    objects_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the analysis report to.",
    )
    objects_parser.add_argument(
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


def _parse_confidence_thresholds(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated confidence threshold string.

    Expected input:
        - raw: e.g. ``"0.1,0.3,0.5,0.7"``.

    Returns:
        - Tuple of float thresholds.

    Raises:
        - ValueError: if any element is not a float in [0, 1].
    """
    values = tuple(float(part) for part in raw.split(","))
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("confidence thresholds must be in [0, 1]")
    return values


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


def _run_media_info_evaluation(
    args: argparse.Namespace,
    config: object,
) -> tuple[
    tuple,
    object,
    object | None,
    object | None,
    dict[str, object],
]:
    """Build the media-info index, evaluate, and compare with CLIP baseline.

    Shared by ``media-info-evaluate`` and ``media-info-report``.

    Expected input:
        - args: parsed CLI arguments (media-info flags).
        - config: loaded project config exposing ``data_raw_dir``.

    Returns:
        - ``(results, summary, comparison, clip_path, retrieval_config)``:
          per-query results, aggregate summary, optional CLIP baseline
          comparison, the resolved CLIP artifact path (or ``None``), and the
          effective BM25 retrieval config.

    Raises:
        - FileNotFoundError / ValueError / OSError: for missing inputs or
          invalid configuration; these are reported by the caller.
    """
    media_info_dir = args.media_info_dir or config.data_raw_dir / "media-info"
    clip_path = args.clip_predictions
    if clip_path is None:
        default_clip = Path("data/interim/predictions/kis-l14-laion2b-full-en.json")
        clip_path = default_clip if default_clip.is_file() else None
    retrieval_config = MediaInfoRetrievalConfig(
        k1=args.k1 if args.k1 is not None else 1.5,
        b=args.b if args.b is not None else 0.75,
        title_weight=args.title_weight if args.title_weight is not None else 3.0,
        keywords_weight=args.keywords_weight if args.keywords_weight is not None else 2.0,
        description_weight=args.description_weight if args.description_weight is not None else 1.0,
    )
    index = MediaInfoIndex.build_from_dir(
        media_info_dir,
        config=retrieval_config,
    )
    eval_set = load_eval_set(args.evaluation_set)
    queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))
    if not queries:
        raise ValueError("evaluation set contains no KIS queries")
    results, summary = evaluate_media_info_video_retrieval(
        queries,
        index,
        query_text_field=args.query_text_field,
        top_k=args.top_k,
    )
    comparison = None
    clip_predictions = None
    if clip_path is not None:
        clip_predictions = load_kis_predictions_for_scoring(clip_path)
    if clip_predictions is not None:
        results_by_query = {result.query_id: result for result in results}
        try:
            comparison = compare_to_clip_baseline(
                queries,
                results_by_query,
                clip_predictions,
                cutoff=args.top_k,
            )
        except ValueError as exc:
            if args.clip_predictions is not None:
                raise
            logger.warning(
                "CLIP baseline comparison skipped (%s); it does not cover this "
                "evaluation set's queries",
                exc,
            )
            comparison = None
    return results, summary, comparison, clip_path, retrieval_config


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

    if args.command == "predict-qa":
        config = load_config(args.config)
        try:
            encoder = OpenClipEncoder(config.openclip)
            artifact = run_qa_prediction(
                QaPredictionRunnerConfig(
                    evaluation_set_path=args.evaluation_set,
                    index_artifact_dir=args.index_artifact,
                    prediction_output_dir=args.output,
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
                    video_aggregation=None if args.video_aggregation == "none" else args.video_aggregation,
                    video_aggregation_top_n=args.video_aggregation_top_n,
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
        print(f"QA artifact: {args.output}")
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

    if args.command == "object-rerank-kis":
        config = load_config()
        objects_dir = args.objects_dir or config.data_raw_dir / "objects"
        map_dir = args.map_dir or (objects_dir.parent / "map-keyframes")
        try:
            object_source = FrameObjectSource(objects_dir, map_dir)
            rerank_config = ObjectRerankingConfig(
                confidence_threshold=args.confidence_threshold,
                lambda_weight=args.lambda_weight,
                normalization_method=args.normalization_method,
                object_score_method=args.object_score_method,
            )
            artifact = write_reranked_prediction_artifact(
                output_path=args.output,
                source_path=args.prediction,
                object_source=object_source,
                config=rerank_config,
                run_id=args.run_id,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Reranked prediction artifact: {args.output}")
        print(f"Processed queries: {artifact.processed_query_count}/{artifact.evaluation_set_query_count}")
        print(f"Run id: {artifact.run_id}")
        return 0

    if args.command == "media-info-search":
        config = load_config()
        media_info_dir = args.media_info_dir or config.data_raw_dir / "media-info"
        try:
            retrieval_config = MediaInfoRetrievalConfig(
                k1=args.k1 if args.k1 is not None else 1.5,
                b=args.b if args.b is not None else 0.75,
                title_weight=args.title_weight if args.title_weight is not None else 3.0,
                keywords_weight=args.keywords_weight if args.keywords_weight is not None else 2.0,
                description_weight=args.description_weight if args.description_weight is not None else 1.0,
            )
            index = MediaInfoIndex.build_from_dir(
                media_info_dir,
                config=retrieval_config,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not args.queries:
            print(f"error: at least one --query is required", file=sys.stderr)
            return 1
        results = {query: index.query(query, top_k=args.top_k) for query in args.queries}
        if args.output is not None:
            payload = {
                "schema": {
                    "num_documents": index.num_documents,
                    "top_k": args.top_k,
                    "config": retrieval_config.__dict__,
                },
                "queries": {
                    query: [
                        {"rank": hit.rank, "video_id": hit.video_id, "score": hit.score}
                        for hit in hits
                    ]
                    for query, hits in results.items()
                },
            }
            try:
                with args.output.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            except OSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"Ranked hits written to: {args.output}")
        for query, hits in results.items():
            print(f"Query: {query}")
            for hit in hits:
                print(f"  rank={hit.rank} video_id={hit.video_id} score={hit.score:.6f}")
        return 0

    if args.command == "media-info-evaluate":
        config = load_config()
        try:
            results, summary, comparison, _clip_path, _retrieval_config = (
                _run_media_info_evaluation(args, config)
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_report(summary, comparison=comparison), end="")
        if args.output is not None:
            payload = {
                "metrics": summary.to_dict(),
                "results": [
                    {
                        "query_id": result.query_id,
                        "query_text": result.query_text,
                        "gt_video_id": result.gt_video_id,
                        "gt_rank": result.gt_rank,
                        "hits": [
                            {"rank": hit.rank, "video_id": hit.video_id, "score": hit.score}
                            for hit in result.hits
                        ],
                    }
                    for result in results
                ],
                "comparison": comparison.to_dict() if comparison is not None else None,
            }
            try:
                with args.output.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            except OSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"Metrics written to: {args.output}")
        return 0

    if args.command == "media-info-report":
        config = load_config()
        try:
            _results, _summary, comparison, _clip_path, _retrieval_config = (
                _run_media_info_evaluation(args, config)
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if comparison is None:
            print(
                "error: categorization requires a CLIP baseline; supply "
                "--clip-predictions or ensure the default artifact exists",
                file=sys.stderr,
            )
            return 1
        print(format_category_report(comparison), end="")
        if args.output is not None:
            payload = category_report_to_dict(comparison)
            try:
                with args.output.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            except OSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"Categorized report written to: {args.output}")
        return 0

    if args.command == "objects-analyze":
        config = load_config()
        objects_dir = args.objects_dir or config.data_raw_dir / "objects"
        map_dir = args.map_dir or config.data_raw_dir / "map-keyframes"
        clip_path = args.clip_predictions
        if clip_path is None:
            default_clip = Path("data/interim/predictions/kis-l14-laion2b-full-en-top1000.json")
            clip_path = default_clip if default_clip.is_file() else None
        if clip_path is None:
            print(
                "error: a CLIP prediction artifact is required; supply "
                "--clip-predictions or ensure the default top-1000 artifact exists",
                file=sys.stderr,
            )
            return 1
        try:
            thresholds = _parse_confidence_thresholds(args.confidence_thresholds)
            report = run_object_analysis(
                objects_dir=objects_dir,
                map_dir=map_dir,
                evaluation_set_path=args.evaluation_set,
                clip_predictions_path=clip_path,
                query_text_field=args.query_text_field,
                thresholds=thresholds,
                top_classes=args.top_classes,
                min_confidence=args.min_confidence,
                prediction_top_k=args.top_k,
                example_limit=args.examples,
                max_videos=args.max_videos,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_object_report(report), end="")
        if args.output is not None:
            try:
                with args.output.open("w", encoding="utf-8") as handle:
                    json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            except OSError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"Object analysis written to: {args.output}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())