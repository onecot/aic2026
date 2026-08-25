#!/usr/bin/env python3
"""Build a clean, verified search index across diverse videos without NaNs."""

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aic2026.config import load_config
from aic2026.frame_images import find_keyframe_image
from aic2026.model_backends.jina import JinaClipEncoder
from aic2026.retrieval import IndexRetriever


def main():
    keyframes_root = ROOT / "data" / "keyframes" / "adaptive_sbd_v1"
    frames_path = keyframes_root / "frames.ndjson"
    index_dir = ROOT / "data" / "interim" / "indexes" / "jina-v2-full"

    print("Step 1: Reading keyframes from dataset...", flush=True)
    video_records = defaultdict(list)
    with frames_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                vid = rec["keyframe"]["video_id"]["value"]
                video_records[vid].append(rec)

    # Pick 10 diverse videos, 5 keyframes each (total 50 frames)
    selected_vids = list(video_records.keys())[:10]
    selected_frames = []
    for vid in selected_vids:
        recs = video_records[vid]
        k = min(5, len(recs))
        step = max(1, len(recs) // k)
        for i in range(k):
            selected_frames.append(recs[i * step])

    print(f"Selected {len(selected_frames)} keyframes across {len(selected_vids)} videos: {selected_vids}", flush=True)

    print("Step 2: Loading Jina CLIP v2 encoder...", flush=True)
    cfg = load_config()
    encoder = JinaClipEncoder(cfg.model)

    print("Step 3: Encoding keyframe images...", flush=True)
    vectors = []
    valid_frames = []

    for i, frame in enumerate(selected_frames, 1):
        vid = frame["keyframe"]["video_id"]["value"]
        ord_val = frame["keyframe"]["keyframe_ordinal"]
        img_path = find_keyframe_image(keyframes_root, vid, ord_val)
        try:
            with Image.open(img_path) as raw:
                img = raw.convert("RGB")
                emb = encoder.encode_image(img)
                vec = emb.vector
                if np.any(np.isnan(vec)) or np.all(vec == 0):
                    print(f"  [Warning] Invalid vector for {vid} frame {ord_val}", flush=True)
                else:
                    vectors.append(vec)
                    valid_frames.append(frame)
                    print(f"  [{i}/{len(selected_frames)}] {vid} ord={ord_val} -> OK (norm={np.linalg.norm(vec):.4f})", flush=True)
        except Exception as err:
            print(f"  [Error] {img_path}: {err}", flush=True)

    vectors_arr = np.array(vectors, dtype=np.float32)
    norms_arr = np.linalg.norm(vectors_arr, axis=1).astype(np.float32)
    print(f"Step 4: Encoded {vectors_arr.shape[0]} valid vectors. Writing index artifact...", flush=True)

    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "embeddings.npy", vectors_arr)
    np.save(index_dir / "norms.npy", norms_arr)

    with (index_dir / "frames.ndjson").open("w", encoding="utf-8") as f:
        for rec in valid_frames:
            f.write(json.dumps(rec) + "\n")

    unique_vids = set(f["keyframe"]["video_id"]["value"] for f in valid_frames)
    manifest = {
        "schema_version": "1.1",
        "artifact_id": "jina-v2-full",
        "score_metric": "cosine",
        "embedding_dim": 1024,
        "embedding_dtype": "float32",
        "num_rows": len(valid_frames),
        "num_videos": len(unique_vids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_id": "data/keyframes/adaptive_sbd_v1",
        "canonical_schema_version": "1.0",
        "embedding_space": encoder.embedding_space.to_dict(),
    }
    with (index_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    retriever = IndexRetriever(index_dir)
    print(f"Step 5: Verified IndexRetriever successfully! Rows={retriever.manifest.num_rows}, Videos={retriever.manifest.num_videos}", flush=True)

    # Test search with 2 distinct queries
    for query in ["người phụ nữ mặc áo dài", "học sinh trong lớp học", "xe ô tô trên đường phố"]:
        q_emb = encoder.encode_text(query)
        hits = retriever.search(q_emb, top_k=3)
        print(f"\nSearch test: '{query}'", flush=True)
        for h in hits:
            print(f"  Rank {h.rank}: {h.video_id} f{h.frame_index} (score={h.score:.4f})", flush=True)


if __name__ == "__main__":
    main()
