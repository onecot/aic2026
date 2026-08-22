#!/usr/bin/env python3
"""CLI: build the representation-v2 OCR experiment artifact (Phase 5B).

Transforms existing OCR chunks/records into cleaned chunks + salient units
(see ``aic2026.ocr_representation``), embeds CLEANED_CONCAT texts and unique
unit texts with the local Jina CLIP text encoder, and writes a separate
artifact directory (originals are never modified):

    representation-v2/
        manifest.json
        cleaned-chunks.ndjson   chunk rows with cleaned text + unit counts
        units.ndjson            unit rows (chunk_id, unit_index, text, row)
        embeddings.npy          rows = cleaned chunks ++ unique units
        diagnostics.json

Row layout of ``embeddings.npy``:
    rows [0, num_chunks)                -> CLEANED_CONCAT chunk vectors
    rows [num_chunks, num_chunks + U)   -> unique unit vectors (first-seen
                                           normalized-text order)

Deterministic: identical inputs produce byte-identical ndjson outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aic2026.config import load_config
from aic2026.ocr_chunks import read_ocr_chunks
from aic2026.ocr_generation import normalize_ocr_text, read_ocr_artifact
from aic2026.ocr_representation import (
    RepresentationConfig,
    build_cleaned_representation,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="build_ocr_representation_v2",
        description="Build cleaned/unit OCR representation embeddings (experiment).",
    )
    parser.add_argument(
        "--chunks-artifact",
        type=Path,
        default=Path("data/interim/ocr/chunks"),
        help="source OCR chunk artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--merged-ocr",
        type=Path,
        default=Path("data/interim/ocr/merged"),
        help="merged OCR record artifact, read-only (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/ocr/representation-v2"),
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/jina-clip-v2.yaml"),
        help="experiment config selecting the embedding backend (default: %(default)s)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=None, help="override encoder device"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if config.jina is None:
            raise ValueError(f"config {args.config} has no models.jina section")

        from aic2026.model_backends.jina import JinaClipEncoder

        jina_config = dataclasses.replace(config.jina, device=args.device) if args.device else config.jina
        encoder = JinaClipEncoder(jina_config)

        _, chunks = read_ocr_chunks(args.chunks_artifact)
        _, records = read_ocr_artifact(args.merged_ocr)
        rep_config = RepresentationConfig()
        cleaned, diagnostics = build_cleaned_representation(chunks, records, rep_config)

        # Unique unit texts for shared embeddings: rows are allocated in
        # first-appearance order of the normalized text; the first-seen raw
        # form is the text that gets embedded.
        unique_row_by_key: dict[str, int] = {}
        unique_texts: list[str] = []
        unit_rows: list[dict] = []
        for cleaned_chunk in cleaned:
            for unit_index, text in enumerate(cleaned_chunk.units):
                dedup_key = normalize_ocr_text(text)
                row = unique_row_by_key.get(dedup_key)
                if row is None:
                    row = len(unique_texts)
                    unique_row_by_key[dedup_key] = row
                    unique_texts.append(text)
                unit_rows.append(
                    {
                        "chunk_id": cleaned_chunk.chunk_id,
                        "video_id": cleaned_chunk.video_id,
                        "unit_index": unit_index,
                        "text": text,
                        "unique_row": len(cleaned) + row,
                    }
                )

        all_texts = [c.text for c in cleaned] + unique_texts
        vectors: list[np.ndarray] = []
        batch_size = max(1, args.batch_size)
        empty_ok = 0
        for start in range(0, len(all_texts), batch_size):
            batch = all_texts[start : start + batch_size]
            safe_batch = [text if text.strip() else " " for text in batch]
            empty_ok += sum(1 for text in batch if not text.strip())
            encoded = encoder.encode_text_batch(safe_batch)
            vectors.extend(e.vector.astype(np.float32, copy=False) for e in encoded)
            if (start // batch_size) % 100 == 0:
                print(f"encoded {min(start + batch_size, len(all_texts))}/{len(all_texts)}", flush=True)
        matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros(
            (0, encoder.embedding_space.embedding_dim), dtype=np.float32
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.0",
            "artifact_id": "representation-v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_chunks_artifact": str(args.chunks_artifact.resolve()),
            "source_merged_ocr": str(args.merged_ocr.resolve()),
            "embedding_space": encoder.embedding_space.to_dict(),
            "params": dataclasses.asdict(rep_config),
            "num_chunks": len(cleaned),
            "num_unique_unit_texts": len(unique_texts),
            "num_unit_rows": len(unit_rows),
            "empty_chunk_texts_encoded_as_blank": empty_ok,
            "diagnostics": diagnostics,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (args.output_dir / "cleaned-chunks.ndjson").open("w", encoding="utf-8") as handle:
            for cleaned_chunk in cleaned:
                handle.write(json.dumps(dataclasses.asdict(cleaned_chunk), ensure_ascii=False) + "\n")
        with (args.output_dir / "units.ndjson").open("w", encoding="utf-8") as handle:
            for row in unit_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(args.output_dir / "embeddings.npy", matrix, allow_pickle=False)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Built representation-v2: {len(cleaned)} chunks "
        f"({sum(1 for c in cleaned if c.text)} non-empty), "
        f"{len(unit_rows)} unit rows, {len(unique_texts)} unique unit texts, "
        f"embeddings {matrix.shape}"
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
