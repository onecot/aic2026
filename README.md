# AIC 2026

Project foundation for the AIC 2026 research work: a reproducible, modular
pipeline covering the dataset layer, image-embedding stage, index build,
retrieval, KIS prediction, and KIS evaluation.

For the full developer guide (per-stage Python API and CLI usage), see
[`docs/usage.md`](docs/usage.md). Artifact schemas and data contracts are
documented in [`docs/data-contracts.md`](docs/data-contracts.md). The
minimal HTTP API for retrieval and frame navigation is documented in
[`docs/api.md`](docs/api.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # activate on POSIX
# .venv\Scripts\activate         # activate on Windows

pip install -e ".[dev]"
```

### Keyframe-extraction dependency: ffmpeg

Real-video keyframe extraction (`scripts/extract_keyframes.py --frame-reader ffmpeg`)
decodes with `ffmpeg` and probes with `ffprobe`; both must be installed and on
`PATH` **before** extraction starts (a preflight check fails fast otherwise).
`ffmpeg >= 4.4` is required (the extractor uses `-fps_mode passthrough`).
Install via your package manager, e.g. `brew install ffmpeg` (macOS),
`apt install ffmpeg` (Debian/Ubuntu), or `conda install -c conda-forge ffmpeg`.

## Run tests

```bash
pytest
```

## Configuration

Default configuration lives in `configs/default.yaml`. Experiment-specific
overrides and CLI flags will be layered on top of it as the pipeline evolves.

## Encoder backends (Jina CLIP v2 is the production default)

The active model section in `configs/default.yaml` is `models.jina`
(`jina-clip-v2`), matching the production retrieval stack and the
`data/interim/indexes/jina-v2-full` index. The simplest way to build an
encoder for the active model is:

```python
from aic2026.config import load_config
from aic2026.image_embedder import create_text_encoder

encoder = create_text_encoder(load_config())
text_embedding = encoder.encode_text("example query")
```

The optional OpenCLIP backend remains available as a baseline; install its
runtime dependencies with `pip install -e ".[openclip]"` and instantiate it
explicitly via `aic2026.model_backends.openclip.OpenClipEncoder`. Both
backends implement the shared `EmbeddingEncoder` contract and record full
`EmbeddingSpace` metadata, so artifacts built with different models are never
silently mixed.

## Dataset inspection

The raw BTC-provided dataset lives under `data/raw/aic2026_batch1/`
(configured as `data.raw_dir` in `configs/default.yaml`). To inspect
and validate it:

```bash
aic2026 inspect
```

The command is read-only and prints a concise report: number of videos
and keyframes/embeddings/object records, embedding dimensionality and
dtype, fps distribution, mapping consistency, missing/orphan records,
and schema observations.

Useful flags:

```bash
aic2026 inspect --max-videos 20        # quick check of a subset
aic2026 inspect --sample-objects 1     # schema-check fewer object files
aic2026 inspect --raw-root /other/path # override the configured root
```

## Data contracts

The schema of the raw dataset (video ids, keyframes, CLIP features,
map-keyframes, objects, media-info) is documented in
`docs/data-contracts.md`. Parsers and validated models live in
`src/aic2026/contracts.py` and `src/aic2026/entities.py`.

## KIS prediction

Generate ranked KIS predictions from an index artifact:

```bash
aic2026 predict-kis \
  --index-artifact data/interim/indexes/btc-batch1-full \
  --output data/interim/predictions/kis.json \
  --top-k 100 \
  --query-text-field query_text_vi
```

See `aic2026 predict-kis --help` and `docs/usage.md` for all flags.