"""Build a retrieval index artifact from an embedding artifact.

Thin CLI over :func:`aic2026.index_builder.build_index`; no retrieval or
ranking logic lives here.

Usage::

    python scripts/build_index.py \
        --source outputs/embeddings_jina_v2 \
        --output data/interim/indexes/jina-v2-full \
        --artifact-id jina-v2-full --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aic2026.dataset import DatasetError
from aic2026.embedding_dataset import EmbeddingArtifactDataset
from aic2026.index_artifact import validate_index_files
from aic2026.index_builder import build_index


def main(argv: list[str] | None = None) -> int:
    """Run the index builder CLI.

    Expected input:
        - argv: optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        - Exit code (0 on success).

    Raises:
        - SystemExit: on argument parsing errors.
    """
    parser = argparse.ArgumentParser(
        prog="build-index",
        description="Build a retrieval index artifact from a verified embedding artifact.",
    )
    parser.add_argument("--source", type=Path, required=True, help="Embedding artifact directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output index artifact directory.")
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="Stable identifier recorded in the index manifest.",
    )
    parser.add_argument(
        "--batch-flush",
        type=int,
        default=2048,
        help="Flush the growing index to disk every N rows (default: 2048).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate the written artifact against its manifest after building.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

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


if __name__ == "__main__":
    raise SystemExit(main())
