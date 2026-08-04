"""Run English known-item-search queries against the CLIP index.

Query schema: a CSV with ``query_id`` and ``text`` columns, or a JSONL file with one
object per line containing those fields.

Usage example:
    python -m scripts.run_kis --config configs/default.yaml --queries queries.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.data.loader import load_index
from src.model.text_encoder import TextEncoder
from src.retrieve import search
from src.tasks.kis import format_kis, write_submission
from src.utils.config import load_config

QUERY_ID_FIELD = "query_id"
QUERY_TEXT_FIELD = "text"


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a known-item-search run.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed config, query, output, retrieval, device, and limit options.
    Raises: SystemExit if CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(description="Run known-item-search retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(arguments)


def load_queries(query_path: Path) -> list[dict[str, Any]]:
    """Load query records from the supported CSV or JSONL schema.

    Expected input: ``query_path`` has a ``.csv`` suffix with ``query_id`` and ``text``
    columns, or a ``.jsonl`` suffix with one object per line containing those fields.
    Returns: Query dictionaries in their source-file order.
    Raises: FileNotFoundError if the query file is missing; ValueError for unsupported
        suffixes or records missing required fields; json.JSONDecodeError for invalid JSONL.
    """
    if query_path.suffix.lower() == ".csv":
        with query_path.open("r", encoding="utf-8", newline="") as input_file:
            queries = list(csv.DictReader(input_file))
    elif query_path.suffix.lower() == ".jsonl":
        with query_path.open("r", encoding="utf-8") as input_file:
            queries = [json.loads(line) for line in input_file if line.strip()]
    else:
        raise ValueError("Queries must be provided as a .csv or .jsonl file.")

    for query in queries:
        if not isinstance(query, dict) or (
            QUERY_ID_FIELD not in query or QUERY_TEXT_FIELD not in query
        ):
            raise ValueError(
                f"Each query must contain '{QUERY_ID_FIELD}' and '{QUERY_TEXT_FIELD}'."
            )
        if not isinstance(query[QUERY_TEXT_FIELD], str):
            raise ValueError(f"'{QUERY_TEXT_FIELD}' must be a string.")
    return queries


def main(arguments: Sequence[str] | None = None) -> None:
    """Encode configured queries, retrieve keyframes, and write a KIS submission.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``;
    config and query files follow the documented schemas.
    Returns: None.
    Raises: FileNotFoundError if configured inputs are missing; ValueError if inputs are
        invalid; OSError if the submission cannot be written.
    """
    args = parse_args(arguments)
    start_time = perf_counter()
    cfg = load_config(args.config)
    emb, vids, frame_idx = load_index(cfg["index"]["path"])
    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    texts = [q[QUERY_TEXT_FIELD] for q in queries]
    # TODO(translate): if config.translate, texts = translator.translate(texts)
    encoder = TextEncoder(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        device=args.device or cfg["model"]["device"],
    )
    query_vecs = encoder.encode(texts)

    top_k = args.top_k if args.top_k is not None else cfg["retrieve"]["top_k"]
    results_per_query = search(query_vecs, emb, vids, frame_idx, top_k=top_k)
    query_ids = [str(query[QUERY_ID_FIELD]) for query in queries]
    predictions = format_kis(
        results_per_query,
        query_ids,
        max_results=cfg["kis"]["max_results"],
    )
    out_path = args.out or cfg["kis"]["output_path"]
    write_submission(predictions, out_path)

    elapsed_seconds = perf_counter() - start_time
    print(f"#queries: {len(queries)}")
    print(f"top_k: {top_k}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print(f"output_path: {out_path}")


if __name__ == "__main__":
    main()
