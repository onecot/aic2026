# AIC 2026 Data Contracts

This document records the contracts of the BTC-provided raw dataset as
derived from the actual files under `data/raw/aic2026_batch1/`. These
contracts are the stable interface that later pipeline stages rely on.

Schema version: 1.0 (derived 2026-08 from batch 1 files).

## Overview

The raw dataset is organised as five per-video collections sharing a
common video identifier format: `L<lane>_V<number>` (e.g. `L21_V001`).

| Collection      | Per-video entry                           | Content                                  |
|-----------------|-------------------------------------------|------------------------------------------|
| `clip-features-32` | `<video_id>.npy`                       | CLIP feature matrix, shape `(N, 512)`    |
| `keyframes`     | `<video_id>/NNN.jpg`                      | keyframe images (3-digit index)          |
| `objects`       | `<video_id>/NNN.json`                     | per-keyframe object detections           |
| `map-keyframes` | `<video_id>.csv`                          | keyframe -> original-frame map           |
| `media-info`    | `<video_id>.json`                         | per-video YouTube metadata               |

`N` is the number of keyframes for that video and varies per video
(observed range: 24..632 across 873 videos).

## Video identifier

- Format: `L<lane>_V<number>`, e.g. `L21_V001`, `L26_V499`.
- Regex (`contracts.py`): `^L\d+_V\d+$`.
- Non-video files (`.DS_Store`, etc.) are not video entries.

## clip-features-32

- One `.npy` file per video.
- Matrix shape `(num_keyframes, embedding_dim)` with
  `embedding_dim == 512` for all observed videos.
- Dtype: `float16`.
- Row order maps to keyframe indices `001..N` in ascending order.
- Contract type: `ClipFeatures` (metadata) in `entities.py`.

## keyframes

- One directory per video containing `<index>.jpg` where `<index>` is a
  three-digit zero-padded keyframe index starting at `001`.
- Keyframe index is 1-based; filename pattern `^(\d{3})\.(jpg|jpeg|png)$`.

## objects

- One directory per video containing `<index>.json` matching the
  keyframe index space exactly (same set of indices as `keyframes/`).
- Each JSON has five parallel lists of equal length (`100` detections
  for all observed frames); keys:
  `detection_scores`, `detection_class_names`, `detection_class_entities`,
  `detection_boxes`, `detection_class_labels`.
- Boxes are lists of four floats in `[0, 1]` (normalised coordinates).
- Contract type: `ObjectDetectionFrame` in `entities.py`; required keys
  and length checks in `validate_object_payload`.

## map-keyframes

- One CSV per video with header `n, pts_time, fps, frame_idx`.
- `n`: 1-based keyframe sequence number (== keyframe index space).
- `pts_time`: presentation timestamp in seconds.
- `fps`: frame rate of the original video (per-video; varies).
- `frame_idx`: 0-based index of the keyframe within the original video.
- Contract type: `KeyframeMapEntry` in `entities.py`; parsed by
  `parse_keyframe_map`.

## media-info

- One JSON per video with top-level keys:
  `author`, `channel_id`, `channel_url`, `description`, `keywords`,
  `length`, `publish_date`, `thumbnail_url`, `title`, `watch_url`.
- `length` is the video duration in seconds.

## Invariants (validated)

- Keyframe index set == object index set == map `n` set (per video).
- `num_keyframes` (files) == CLIP matrix rows == CSV row count.
- `frame_idx` increases monotonically with `n`.
- `frame_idx` is consistent with `pts_time * fps` (verified across all
  rows of the full batch: no mismatch).