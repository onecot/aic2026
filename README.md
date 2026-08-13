# AIC 2026

Project foundation for the AIC 2026 research work: a reproducible, modular
pipeline covering the dataset layer, image-embedding stage, index build,
retrieval, KIS prediction, and KIS evaluation.

For the full developer guide (per-stage Python API and CLI usage), see
[`docs/usage.md`](docs/usage.md). Artifact schemas and data contracts are
documented in [`docs/data-contracts.md`](docs/data-contracts.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # activate on POSIX
# .venv\Scripts\activate         # activate on Windows

pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Configuration

Default configuration lives in `configs/default.yaml`. Experiment-specific
overrides and CLI flags will be layered on top of it as the pipeline evolves.

## OpenCLIP encoder backend

The optional OpenCLIP backend implements the shared `EmbeddingEncoder`
contract without making the core package depend on model libraries. Install
its runtime dependencies when encoding is needed:

```bash
pip install -e ".[openclip]"
```

`configs/default.yaml` provides the `models.openclip` configuration block:
model name, pretrained checkpoint, `auto`/`cpu`/`mps`/`cuda` device,
`fp32`/`fp16` output precision, normalization, and explicit preprocessing /
tokenizer revision identifiers. Instantiate it from application code with:

```python
from aic2026.config import load_config
from aic2026.model_backends.openclip import OpenClipEncoder

encoder = OpenClipEncoder(load_config().openclip)
text_embedding = encoder.encode_text("example query")
image_embedding = encoder.encode_image(image)
```

`encode_text_batch` and `encode_image_batch` support batches. Both image and
text methods use the same configured normalization convention and return
compatible `EmbeddingSpace` metadata. Model weights may be downloaded by
OpenCLIP on first use; normal tests do not load a model or access the network.

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