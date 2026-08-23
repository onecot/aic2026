# AIC 2026

Video retrieval system for the AIC 2026 competition: a small production
runtime (Jina CLIP v2 retrieval, KIS/QA/TRAKE tasks, BTC submission
exporter), an offline asset toolchain that can rebuild the full chain from
raw videos, and a minimal HTTP API for candidate inspection.

## Layout

```text
src/aic2026/
    core layer        config, canonical entities, raw dataset, evaluation sets
    retrieval         embeddings contract, backends, index artifacts,
                      SearchEngine, frame catalog + image resolution
    tasks             KIS / QA / TRAKE runners (+ shared task foundation),
                      BTC adapter, VLM answer generation
    api               minimal FastAPI surface (docs/api.md)
    submission        BTC CSV/zip exporters (library)
    assets/           offline rebuild toolchain (keyframes, embeddings)
scripts/              six entrypoints: build_keyframes, build_embeddings,
                      build_index, run_btc, build_submission, qa_smoke
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -e ".[jina,api,dev]"
```

## Documentation

- `docs/usage.md` — configuration, asset rebuild chain, runtime API.
- `docs/api.md` — HTTP endpoints, request/response examples.
- `docs/data-contracts.md` — artifact schemas.

## Configuration

`configs/default.yaml` is canonical; the production model is Jina CLIP v2
against `data/interim/indexes/jina-v2-full`. Override by passing an
alternative same-schema YAML to `load_config(path)` or script `--config`.

## Tests

```bash
pytest
```
