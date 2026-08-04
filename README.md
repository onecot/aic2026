# AIC 2026 Retrieval Foundation

## Overview

This project builds a searchable index over organizer-provided CLIP keyframe features,
encodes English text queries into the same ViT-B/32 embedding space, retrieves ranked
keyframes, and writes known-item-search (KIS) submissions.

## Data layout

From the `aic2026/` code directory, organizer data is located under `../../data/`:

- `clip-features-32/`: one `(n_keyframes, 512)` NumPy feature array per video.
- `map-keyframes/`: matching CSV mappings with real source-video `frame_idx` values.
- `keyframes/`: keyframe images.
- `media-info/`: media metadata.
- `objects/`: object annotations.

These files are read-only and must not be committed.

## Setup

Use Python 3.11 and install the pinned project dependencies:

```bash
pip install -r requirements.txt
```

Run all commands below from `aic2026/` as Python modules so `src.` imports resolve.

## Build index

Build the normalized keyframe index defined by the config:

```bash
python -m scripts.build_index --config configs/default.yaml
```

For a fast check, add `--limit 10`. The generated NPZ index is intentionally ignored by Git.

## Run KIS

Run known-item search against a built index:

```bash
python -m scripts.run_kis --config configs/default.yaml --queries queries.csv
```

Queries may be CSV with `query_id,text` columns or JSONL with one object per line containing
`query_id` and `text`. The default output is a CSV with `query_id,video_id,frame_idx` rows.

## Evaluate

Score a local KIS submission before submission:

```bash
python -m eval.score --submission outputs/kis_submission.csv --ground-truth ground_truth.csv
```

Ground truth is assumed to be CSV or JSONL records with `query_id`, `video_id`, `s`, and `e`;
`[s, e]` is the inclusive correct source-frame range.

## Config

`configs/default.yaml` controls all data and output paths, index normalization, matching CLIP
model settings, retrieval depth, and KIS output limits. Paths are resolved relative to the
config file. The text encoder's `model.name` and `model.pretrained` must match the ViT-B/32
image-feature embedding space.

## Known limitations

- OpenAI CLIP text encoding is English-oriented. Vietnamese queries need translation; this is
  planned through the marked translation seam in `scripts/run_kis.py`.
- ViT-B/32 cosine-score magnitudes can be low. Relative ranking, rather than an absolute score
  threshold, is what matters for retrieval.
