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

## Raw → canonical flow

The raw BTC layout is never consumed directly by downstream stages.
Data moves through three layers:

```text
raw BTC directory/file layout          (data/raw/aic2026_batch1/)
        |
        |  aic2026.entities     parse CSV/JSON/NPY -> typed raw records
        v
raw typed records               (KeyframeMapEntry, ObjectDetectionFrame, ...)
        |
        |  aic2026.converters   validate + map types -> canonical entities
        v
canonical records               (aic2026.canonical)
```

- `aic2026.contracts` defines the raw layout vocabulary (collection
  names, patterns, required keys); it has no I/O.
- `aic2026.entities` parses raw files into typed raw records.
- `aic2026.converters` converts raw records into canonical entities.
- `aic2026.canonical` defines the stable, storage-agnostic entity model
  downstream stages depend on (see entity docs above).

## Dataset loading layer (`aic2026.dataset`)

`Dataset` is the read-only accessor that downstream code should use. It
hides every raw-layout detail (folder names, filename patterns,
zero-padding, CSV/JSON structure, and how CLIP embeddings are stored) and
returns canonical entities from `aic2026.canonical`.

### Loader responsibility

- Discover videos and iterate them as `VideoId` records.
- Provide per-video metadata, timings, and keyframe iteration.
- Stream embeddings and detections lazily (`iter_embeddings`,
  `iter_detections`) so the full dataset is never resident in memory.
- Fail loudly on missing or corrupt records instead of skipping them.
- Never write, rename, copy, or generate files under `data/raw/`, and
  keep no cache/materialized copy of the dataset.

### Public API

```python
Dataset(raw_root: Path | str)  # validate root; FileNotFoundError if absent

raw_root            -> Path                   # the raw dataset root

# discovery
video_ids()         -> list[str]              # sorted, union of collections
iter_video_ids()    -> Iterator[str]
__iter__()          -> Iterator[VideoId]
__len__()           -> int
__contains__(id)    -> bool
has_video(id)       -> bool

# video-level (materialized; small per video)
metadata(id)        -> VideoMetadata
timings(id)         -> list[KeyframeTiming]   # ordered keyframe set per video
keyframe_ordinals(id) -> list[int]
iter_keyframes(id)  -> Iterator[KeyframeKey]

# lazy streaming
iter_embeddings(id) -> Iterator[ClipEmbedding]        # one row at a time (mmap)
iter_detections(id) -> Iterator[KeyframeDetections]   # one keyframe JSON at a time
```

### Errors

- `DatasetError` — base class for all loader errors.
- `MissingRecordError(DatasetError)` — an expected raw file/directory is
  absent (e.g. missing `media-info/*.json` or an object JSON).
- `CorruptRecordError(DatasetError)` — a raw file exists but is
  malformed or inconsistent (e.g. invalid JSON, non-2-D `.npy`).

Records are never silently skipped. A malformed `video_id` raises
`ValueError` (via the canonical `VideoId` contract).

### What downstream stages are allowed to depend on

- Canonical entities from `aic2026.canonical`.
- The `Dataset` accessor and its error types from `aic2026.dataset`.
- Reusable helpers (e.g. `config.load_config`) where appropriate.
- Raw parsers/converters only when a non-Dataset integration point is
  genuinely required; prefer `Dataset` by default.

Downstream code must not depend on collection names, file patterns,
zero-padding, or CSV/JSON/npy internals from the raw layout.