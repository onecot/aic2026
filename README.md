# AIC 2026

Project foundation for the AIC 2026 research work. The repository is
being built as a reproducible, modular pipeline; the extract / embed /
index / retrieval / evaluation stages are not implemented yet.

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