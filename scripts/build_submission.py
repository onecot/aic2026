"""CLI wrapper around :mod:`aic2026.submission`.

All export/matching/validation/packaging logic lives in the library module;
this script only parses arguments, runs the build, and prints the
human-readable report from the returned summary.

Usage::

    python scripts/build_submission.py \\
        --query-dir data/sotuyen/thunghiem \\
        --kis-artifact data/interim/btc/kis-predictions.json \\
        --qa-artifact data/interim/btc/qa-predictions \\
        --trake-artifact data/interim/btc/trake-predictions \\
        --output-dir submission

No Modal runs. No GPU runs. No changes to core prediction pipelines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.btc_queries import enumerate_query_files, read_query_text
from aic2026.submission import build_submission


def print_mapping_report(query_dir: Path) -> None:
    """Print the query files discovered in a query directory."""
    try:
        query_files = enumerate_query_files(query_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("query_file | task | csv_name")
    for info in query_files:
        print(f"{info.path.name} | {info.task} | {info.csv_name}")
    print("")


def main(argv: list[str] | None = None) -> int:
    """Run the BTC submission builder CLI.

    Expected input:
        - argv: optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        - Exit code (0 = success).

    Raises:
        - SystemExit: on argument parsing errors.
    """
    parser = argparse.ArgumentParser(
        description="Build BTC submission CSV files from prediction artifacts.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query-*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for submission/ and submission.zip.",
    )
    parser.add_argument(
        "--kis-artifact",
        type=Path,
        default=None,
        help="Path to KIS prediction artifact (JSON file or directory).",
    )
    parser.add_argument(
        "--qa-artifact",
        type=Path,
        default=None,
        help="Path to QA prediction artifact directory.",
    )
    parser.add_argument(
        "--trake-artifact",
        type=Path,
        default=None,
        help="Path to TRAKE prediction artifact directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = build_submission(
            query_dir=args.query_dir,
            output_dir=args.output_dir,
            kis_artifact=args.kis_artifact,
            qa_artifact=args.qa_artifact,
            trake_artifact=args.trake_artifact,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("query_file | task | matched_prediction_id | match_method")
    for row in result["mapping"]:
        print(
            f"{row['query_file']} | {row['task']} | {row['matched_id']} | "
            f"{row['method']}"
        )
    print("")
    print("Matching summary:")
    for task in ("kis", "qa", "trake"):
        print(f"{task.upper()}: {result['counts'][task]}/{result['totals'][task]} matched")
    print("")
    print("Per-file validation:")
    print("QUERY FILE | TASK | MATCHED | CSV | ROWS | VALID")
    validations = {v["filename"]: v for v in result["validations"]}
    generated_by_csv = {item["csv_name"]: item for item in result["generated"]}
    for csv_name, item in generated_by_csv.items():
        validation = validations[csv_name]
        print(
            f"{item['query_file']} | {item['task']} | {item['matched_id']} | "
            f"{csv_name} | {validation['row_count']} | "
            f"{'VALID' if validation['valid'] else 'INVALID'}"
        )
        if item["task"] == "trake":
            print(
                f"  {csv_name}: events={item['expected_columns'] - 1} "
                f"expected_columns={item['expected_columns']} "
                f"actual_columns={validation['column_counts']} "
                f"invalid_rows={validation['invalid_rows']}"
            )
    print("")
    print("submission.zip contents:")
    for entry in result["zip_info"]["entries"]:
        print(f"  {entry}")
    print("")
    print(f"KIS queries:  {result['counts']['kis']}")
    print(f"QA queries:   {result['counts']['qa']}")
    print(f"TRAKE queries: {result['counts']['trake']}")
    print(f"Total CSVs:   {len(result['generated'])}")
    print(f"Submission:   {result['zip_path']}")
    print("")
    print("SUBMISSION VALID: yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
