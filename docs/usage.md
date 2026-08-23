# Developer guide

The repository has four layers:

```text
raw BTC dataset
      ↓  scripts/build_keyframes.py     (offline assets)
canonical keyframes (adaptive_sbd_v1)
      ↓  scripts/build_embeddings.py    (offline assets)
embedding artifact
      ↓  scripts/build_index.py         (offline assets)
index artifact (jina-v2-full)
      ↓  aic2026.search / tasks / api   (runtime, in-memory)
typed results → submission exporters
```

Runtime modules never import `aic2026.assets.*`; asset builders are driven
exclusively through `scripts/`.

## Configuration

`configs/default.yaml` is the single base configuration. The active model is
Jina CLIP v2 (`models.backend: jina-clip`). The only override mechanism is an
alternative YAML with the same schema passed explicitly:

```python
from aic2026.config import load_config
config = load_config("path/to/override.yaml")
```

## Asset chain (offline)

Each builder exposes subcommands wrapping the same implementations as the
historical per-stage scripts. Run any stage with `-h` for its full options.

```bash
# raw videos -> videos.json manifest
python scripts/build_keyframes.py manifest --video-dir data/raw/aic2026_batch1 \
    --output data/interim/videos/videos.json

# split -> extract -> merge -> verify -> materialize (adaptive_sbd_v1)
python scripts/build_keyframes.py split --videos data/interim/videos/videos.json \
    --output-dir data/keyframe-splits --num-workers 5
python scripts/build_keyframes.py extract --videos <worker>/videos.json \
    --output-dir outputs/keyframes_worker_000
python scripts/build_keyframes.py merge --shard-dir outputs/keyframes_worker_* \
    --output-dir outputs/keyframes_full
python scripts/build_keyframes.py verify --artifact-dir outputs/keyframes_full
python scripts/build_keyframes.py materialize --videos data/interim/videos/videos.json \
    --frames outputs/keyframes_full/frames.ndjson --output-dir data/keyframes/adaptive_sbd_v1

# keyframes -> embeddings (Jina CLIP v2 by default) -> index
python scripts/build_embeddings.py split --frames outputs/keyframes_full/frames.ndjson \
    --output-dir data/embedding-splits --shards 4
python scripts/build_embeddings.py embed --frames <shard frames.ndjson> \
    --output-dir outputs/embeddings_jina_v2_shard000   # uses configs/default.yaml
python scripts/build_embeddings.py merge --shard-dir outputs/embeddings_jina_v2_* \
    --output-dir data/interim/embeddings/jina-v2-src
python scripts/build_index.py --source data/interim/embeddings/jina-v2-src \
    --output data/interim/indexes/jina-v2-full --artifact-id jina-v2-full --verify
```

Artifact schemas and validation rules: see `docs/data-contracts.md`.

## Retrieval runtime (Python API)

```python
from pathlib import Path
from aic2026.config import load_config
from aic2026.search import SearchEngine
from aic2026.kis_prediction_runner import search_kis, compute_kis_artifact
from aic2026.qa.qa_prediction_runner import search_qa
from aic2026.trake import search_trake
from aic2026.text_query import TextQuery

config = load_config()
engine = SearchEngine.from_config(config, Path("data/interim/indexes/jina-v2-full"), top_k=100)

kis = search_kis(engine, "một người đang nấu ăn trong bếp")          # QueryPredictionResult
qa = search_qa(engine, TextQuery(query_id="q", query_text="..."))    # QaQueryResult (+answer via answerer=)
tr = search_trake(engine, "(1) sự kiện một (2) sự kiện hai")         # TrakeQueryResult
```

Batch runs over query sources (eval CSV or BTC txt files):

- `run_kis_prediction_for_queries` / `run_qa_prediction_for_queries` /
  `run_trake_prediction_for_queries` — compute + write standard artifacts.
- `scripts/run_btc.py --query-dir data/sotuyen/thunghiem
  --index-artifact data/interim/indexes/jina-v2-full --output-dir <dir>`
  runs all three tasks over BTC query files.

Artifact schemas (`schema_version="1.0"`) are unchanged; writers are atomic.

## QA answer generation

QA answering uses the shared Jina VLM MLX wrapper (`aic2026.vlm_jina`,
install extra `vlm-mlx`) with evidence frames resolved from
`data.keyframes_dir`. Per-query failures degrade to `answer_error`; the
prompt/model/device are recorded in the run manifest.

## Submission

```bash
python scripts/run_btc.py ...            # produce prediction artifacts
python scripts/build_submission.py \
    --query-dir data/sotuyen/thunghiem \
    --kis-artifact <out>/kis-predictions.json \
    --qa-artifact <out>/qa-predictions \
    --trake-artifact <out>/trake-predictions \
    --output-dir submission
```

Matching is strict (query_id → exact text → normalized text); ambiguous or
missing matches abort the build. Library entry point:
`aic2026.submission.build_submission(...)` also accepts typed in-memory
results directly.

## HTTP API

See `docs/api.md`. Four endpoints over `SearchEngine` + `FrameCatalog`;
services load once at startup.

## Tests

```bash
pip install -e ".[dev,api]"
pytest
```
