"""Offline asset builder: canonical keyframes -> embedding artifacts.

Consolidates the former per-stage scripts into one entrypoint; each
subcommand wraps the same assets-module implementation and flag surface as
the previous standalone script:

    split    partition frames.ndjson into balanced worker shards
    embed    embed one shard (Jina CLIP v2 by default, from configs/default.yaml)
    merge    merge completed shard artifacts into one embedding artifact
    verify   validate an embedding or shard artifact

Typical rebuild flow:

    python scripts/build_embeddings.py split --frames outputs/keyframes_full/frames.ndjson --output-dir data/embedding-splits --shards 4
    python scripts/build_embeddings.py embed --frames <shard dir> --output-dir outputs/embeddings_jina_v2 --config configs/default.yaml
    python scripts/build_embeddings.py merge --shard-dir outputs/... --output-dir data/interim/embeddings/jina-v2-full
    python scripts/build_embeddings.py verify --artifact-dir data/interim/embeddings/jina-v2-full
"""

from __future__ import annotations

import sys



def _cmd_split(argv: list[str] | None = None) -> int:
    """partition a frames.ndjson into balanced shard inputs (formerly scripts/split_embedding_shards)."""
    #!/usr/bin/env python3
    """CLI: deterministically split a canonical keyframe set into worker inputs.

    Reads a canonical artifact ``frames.ndjson`` and partitions it into ``N``
    worker input files (``worker_000/frames.ndjson``, ...), keeping each video
    intact and balancing the number of frames per worker. The split is
    deterministic: the same input and ``--num-workers`` always produce identical
    worker files. Writes a ``split_manifest.json`` describing the split.

    This is a thin CLI; the splitting logic lives in ``aic2026.assets.embedding_split``.
    """


    import argparse
    import sys

    from aic2026.assets.embedding_split import split_frames


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="split_embedding_shards",
            description=(
                "Partition a canonical frames.ndjson into balanced, deterministic "
                "worker input files, keeping videos intact."
            ),
        )
        parser.add_argument(
            "--frames",
            required=True,
            help="path to a canonical artifact frames.ndjson to split",
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
            help="directory to write worker_*/frames.ndjson and split_manifest.json",
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
            manifest = split_frames(
                args.frames,
                args.num_workers,
                args.output_dir,
                overwrite=args.overwrite,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Split complete: {sum(w.num_frames for w in manifest.workers)} frames "
            f"across {manifest.num_workers} workers"
        )
        for worker in manifest.workers:
            print(
                f"  {worker.worker_id:03d}: {worker.num_frames} frames "
                f"({worker.num_videos} videos)"
            )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_embed(argv: list[str] | None = None) -> int:
    """embed one shard of keyframes with the active model backend (formerly scripts/embed_shard)."""
    #!/usr/bin/env python3
    """CLI: embed a shard of canonical keyframe records into a resumable artifact.

    Reads a canonical artifact ``frames.ndjson``, encodes the first ``--max-frames``
    records (or all when omitted) through a selected model, and writes a resumable
    shard directory containing ``checkpoint.json``, ``embeddings.npy``,
    ``frames.ndjson``, and ``manifest.json``. Re-running the same command after an
    interruption resumes from the checkpoint and skips already-embedded frames.

    This is a thin CLI; the pipeline lives in ``aic2026.assets.embedding_shard`` and the
    embedder interface in ``aic2026.image_embedder``.
    """


    import argparse
    import logging
    import sys
    from pathlib import Path

    from aic2026.config import load_config
    from aic2026.assets.embedding_shard import embed_shard
    from aic2026.image_embedder import DEVICE_CHOICES, create_image_embedder
    from aic2026.progress import PROGRESS_MODES, create_progress_reporter

    DEFAULT_FRAMES = "data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson"


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="embed_shard",
            description=(
                "Encode a bounded shard of keyframe records into a resumable "
                "embedding artifact on the selected hardware."
            ),
        )
        parser.add_argument(
            "--frames",
            type=Path,
            required=True,
            help=f"path to a canonical artifact frames.ndjson (e.g. {DEFAULT_FRAMES})",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="directory to write checkpoint.json, embeddings.npy, frames.ndjson, manifest.json",
        )
        parser.add_argument(
            "--model",
            default=None,
            help=(
                "model spec '[backend:]model[:pretrained]', e.g. "
                "'jina-clip:jina-clip-v2:jinaai/jina-clip-v2' (default: the "
                "config's active model)"
            ),
        )
        parser.add_argument(
            "--device",
            choices=DEVICE_CHOICES,
            default="auto",
            help="device selection: auto (CUDA > MPS > CPU), cuda, mps, or cpu (default: auto)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="number of keyframe images encoded per model invocation (default: 32)",
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="only embed the first MAX_FRAMES records (default: all records)",
        )
        parser.add_argument(
            "--worker-id",
            type=int,
            default=0,
            help="worker machine identifier recorded in the manifest (default: 0)",
        )
        parser.add_argument(
            "--keyframes-dir",
            type=Path,
            default=None,
            help=(
                "root directory of the keyframe image collection used to resolve "
                "image paths: the materialized adaptive_sbd_v1 images/ layout, "
                "falling back to a legacy keyframes/ collection (default: config "
                "data.raw_dir)"
            ),
        )
        parser.add_argument(
            "--precision",
            choices=("fp32", "fp16"),
            default="fp32",
            help="output precision for supported backends (default: fp32)",
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

        config = load_config()
        model = args.model or f"{config.model.backend}:{config.model.model_name}:{config.model.pretrained}"
        keyframes_root = args.keyframes_dir or config.data_raw_dir

        reporter = create_progress_reporter(args.progress)
        reporter.begin()
        try:
            with reporter.stages.time("load_model"):
                embedder = create_image_embedder(
                    model,
                    device=args.device,
                    precision=args.precision,
                    config=config.model,
                )
            print(f"Selected device: {embedder.device}")
            print(
                f"Model: {embedder.model_name} "
                f"(dim={embedder.embedding_dim}, dtype={embedder.embedding_space.output_dtype})"
            )
            manifest = embed_shard(
                frames_path=args.frames,
                output_dir=args.output_dir,
                embedder=embedder,
                batch_size=args.batch_size,
                max_frames=args.max_frames,
                worker_id=args.worker_id,
                precision=args.precision,
                keyframes_root=keyframes_root,
                progress=reporter,
            )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Shard complete: {manifest.num_embeddings} embeddings "
            f"for {manifest.num_videos} videos (worker_id={manifest.worker_id})"
        )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_merge(argv: list[str] | None = None) -> int:
    """merge completed embedding shards into one artifact (formerly scripts/merge_embedding_shards)."""
    #!/usr/bin/env python3
    """CLI: merge completed embedding shards into a single embedding artifact.

    Reads completed shard directories (each with ``manifest.json``,
    ``embeddings.npy``, and ``frames.ndjson``), validates that they share the same
    model, embedding dimension, and embedding space, rejects any duplicate
    keyframes, and concatenates them in deterministic worker order into a merged
    ``embeddings.npy``, ``frames.ndjson``, and ``manifest.json``.

    This is a thin CLI; the merge logic lives in ``aic2026.assets.embedding_merge``.
    """


    import argparse
    import sys
    from pathlib import Path

    from aic2026.assets.embedding_merge import merge_shards


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="merge_embedding_shards",
            description=(
                "Merge completed embedding shards into one embedding artifact, "
                "validating model/space consistency and rejecting duplicate frames."
            ),
        )
        parser.add_argument(
            "--shards",
            nargs="+",
            type=Path,
            required=True,
            help="one or more completed shard directories to merge",
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory to write merged embeddings.npy, frames.ndjson, manifest.json",
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
            f"Merge complete: {manifest.total_embeddings} embeddings from "
            f"{len(manifest.source_shards)} shards (model={manifest.model})"
        )
        print(f"Output: {args.output_dir}")
        return 0
    return main(argv)


def _cmd_verify(argv: list[str] | None = None) -> int:
    """verify an embedding/shard artifact against expectations (formerly scripts/verify_embedding)."""
    #!/usr/bin/env python3
    """CLI: verify a completed embedding artifact.

    Reads an embedding artifact (``embeddings.npy`` + ``frames.ndjson`` +
    ``manifest.json``) and verifies it is internally consistent: artifact
    existence, manifest parsing and model metadata, frame record validity, and
    the embedding matrix (2-D floating, row count matches frames and manifest,
    dimension and dtype match the manifest when recorded, no empty or NaN/Inf
    vectors).

    This is a thin CLI; the verification logic lives in
    ``aic2026.assets.embedding_verify.verify_embedding``.
    """


    import argparse
    import sys

    from aic2026.assets.embedding_verify import verify_embedding


    def build_argument_parser() -> argparse.ArgumentParser:
        """Return the CLI argument parser."""
        parser = argparse.ArgumentParser(
            prog="verify_embedding",
            description=(
                "Validate a completed embedding artifact (embeddings.npy + "
                "frames.ndjson + manifest.json): manifest, model metadata, frame "
                "records, matrix shape/dtype, and vector health."
            ),
        )
        parser.add_argument(
            "--output-dir",
            required=True,
            help="directory containing embeddings.npy, frames.ndjson, and manifest.json",
        )
        return parser


    def main(argv: list[str] | None = None) -> int:
        """Run the CLI and return an exit code (0 valid, 1 invalid)."""
        parser = build_argument_parser()
        args = parser.parse_args(argv)

        try:
            report = verify_embedding(args.output_dir)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        status = "PASS" if report.valid else "FAIL"
        print(f"Embedding verification: {status}")
        print()
        print("Artifact:")
        print(f"  model: {report.model_id}")
        print(f"  dim: {report.dim if report.dim is not None else 'unavailable'}")
        print(f"  dtype: {report.dtype if report.dtype is not None else 'unavailable'}")
        print()
        print("Rows:")
        print(f"  embeddings: {report.rows}")
        print(f"  frames: {report.frames}")
        print()
        print("Vectors:")
        print(f"  empty: {report.empty_vectors}")
        print(f"  nan/inf: {report.nan_vectors}")

        if not report.valid:
            print()
            print("Validation failed:")
            for issue in report.issues:
                print(f"  [{issue.section}] {issue.message}")
            return 1

        print()
        print("OK: embedding artifact is valid")
        return 0
    return main(argv)


_DISPATCH = {
    "split": _cmd_split,
    "embed": _cmd_embed,
    "merge": _cmd_merge,
    "verify": _cmd_verify,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to one embedding-asset stage."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(
            f"usage: {sys.argv[0]} <" + ", ".join(sorted(_DISPATCH))
            + "> [stage options...]\npass '-h' after a stage for its options",
            file=sys.stderr,
        )
        return 2
    command, rest = argv[0], argv[1:]
    handler = _DISPATCH.get(command)
    if handler is None:
        print(
            f"error: unknown stage {command!r}; expected one of: "
            + ", ".join(sorted(_DISPATCH)),
            file=sys.stderr,
        )
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
