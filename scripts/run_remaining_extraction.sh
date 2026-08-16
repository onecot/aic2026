#!/bin/bash

set -e

mkdir -p logs

for id in 001 002 003 004
do
    echo "========================================"
    echo "Starting worker_${id}"
    echo "========================================"

    LOG="logs/worker_${id}.log"

    python scripts/extract_keyframes.py \
      --videos data/keyframe-splits/worker_${id}/videos.json \
      --video-root data/raw/videos \
      --output-dir outputs/keyframes_worker_${id} \
      --extractor omnishotcut_adaptive \
      --detector omnishotcut \
      --sampler adaptive_v1 \
      --device auto \
      --progress on \
      2>&1 | tee "$LOG"

    python scripts/verify_keyframe_extraction.py \
      --output-dir outputs/keyframes_worker_${id} \
      --videos data/keyframe-splits/worker_${id}/videos.json \
      2>&1 | tee -a "$LOG"

    echo "worker_${id} completed successfully"
    echo ""

done

echo "All remaining workers completed."
