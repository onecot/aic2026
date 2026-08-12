# AIC 2026

Project foundation for the AIC 2026 research work. The repository is being
built as a reproducible, modular pipeline; the extract / embed / index /
retrieval / evaluation stages are not implemented yet.

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