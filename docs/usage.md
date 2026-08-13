# AIC 2026 Pipeline Usage

Developer guide for the current retrieval pipeline. Every stage is documented
with its Python API and, where a CLI command exists, its CLI usage. This
document describes the current implementation; see
`docs/data-contracts.md` for the artifact schemas.

```bash
source .venv/bin/activate      # all examples assume the project venv
```

## 1. Pipeline overview

```text
Dataset        aic2026.dataset.Dataset          read-only access to raw BTC Batch 1
   |            raw -> canonical entities, no raw layout leaks
   v
Embedding      aic2026.embedding_runner.build_embeddings
   |            keyframes -> (embedding artifact) manifest.json + embeddings.npy
   |            + frames.ndjson
   v
Index          aic2026.index_builder.build_index
   |            dataset embeddings -> (index artifact) embeddings + norms + frames
   v
Retrieval      aic2026.retrieval.IndexRetriever     exact cosine search
   v
KIS Prediction aic2026.kis_prediction_runner.run_kis_prediction  (CLI: aic2026 predict-kis)
   v
Evaluation     aic2026.kis_scoring.evaluate_kis     R@1..R@100 + Final Score
```

Artifact purposes:

- **Embedding artifact** (`manifest.json`, `embeddings.npy`, `frames.ndjson`):
  model-specific keyframe vectors for one run. Row `i` of `embeddings.npy`
  corresponds to line `i` of `frames.ndjson`.
- **Index artifact** (`manifest.json`, `embeddings.npy`, `norms.npy`,
  `frames.ndjson`, reserved `index.bin`): retrieval-ready vectors plus stored
  L2 norms so cosine scores are computed without touching the raw dataset.
- **Prediction artifact** (JSON): per-query ranked candidates written by
  `predict-kis`, directly consumable by `evaluate_kis`.

Only two CLI commands exist (`aic2026 inspect`, `aic2026 predict-kis`). The
embedding and index stages are Python-API only and are composed in scripts.

## 2. Dataset stage

### Initialization

```python
from aic2026.config import load_config
from aic2026.dataset import Dataset

config = load_config()                       # configs/default.yaml
dataset = Dataset(config.data_raw_dir)       # data/raw/aic2026_batch1
```

`Dataset(raw_root)` raises `FileNotFoundError` if the root is absent. It is
strictly read-only with respect to `data/raw/`.

### Iteration APIs

```python
dataset.video_ids()        # list[str], sorted; union across collections
list(dataset)              # Iterator[VideoId]
len(dataset)               # number of videos
"L21_V001" in dataset      # has_video(id)  -> bool
dataset.metadata("L21_V001")            # VideoMetadata (media-info)
dataset.timings("L21_V001")             # list[KeyframeTiming], canonical order
dataset.keyframe_ordinals("L21_V001")   # [1, 2, ..., N]
```

### Keyframe / image access

```python
from aic2026.dataset import MissingRecordError, CorruptRecordError

dataset.iter_keyframes("L21_V001")               # Iterator[KeyframeKey]
image_path = dataset.keyframe_image_path("L21_V001", 1)   # Path, layout hidden

# lazy streaming (never materializes the full dataset)
for emb in dataset.iter_embeddings("L21_V001"):   # Iterator[ClipEmbedding]
    pass
for det in dataset.iter_detections("L21_V001"):   # Iterator[KeyframeDetections]
    pass
```

Missing records raise `MissingRecordError`; malformed records raise
`CorruptRecordError`. Malformed video ids raise `ValueError`.

### Expected raw dataset layout

```text
data/raw/aic2026_batch1/
    clip-features-32/<video_id>.npy   (N, 512) float16 CLIP matrix
    keyframes/<video_id>/NNN.jpg      keyframe images, 3-digit 1-based index
    objects/<video_id>/NNN.json       per-keyframe object detections
    map-keyframes/<video_id>.csv      n, pts_time, fps, frame_idx
    media-info/<video_id>.json        video metadata
```

Downstream code must never read this layout directly; use `Dataset`.

### Subset iteration

```python
vids = ["L30_V041", "L21_V001"]
missing = [v for v in vids if v not in dataset]
for vid in dataset.video_ids():
    if vid in vids:
        print(vid, len(dataset.timings(vid)))
```

## 3. Model / encoder stage

### Available implementations

- `aic2026.model_backends.openclip.OpenClipEncoder` — the only current
  encoder. Implements the `EmbeddingEncoder` protocol from
  `aic2026.embeddings`. Requires the optional `openclip` dependency group:
  `pip install -e ".[openclip]"`.
- Any object satisfying the `EmbeddingEncoder` protocol
  (`embedding_space`, `encode_image`, `encode_text`) is a valid encoder;
  `encode_image_batch` / `encode_text_batch` are used when present.

### OpenClipEncoder configuration

```python
from aic2026.config import load_config, OpenClipConfig
from aic2026.model_backends.openclip import OpenClipEncoder

config = load_config()                      # defaults: ViT-B-32, laion2b_s34b_b79k
encoder = OpenClipEncoder(config.openclip)

# to override the checkpoint without editing configs/default.yaml,
# construct an OpenClipConfig explicitly:
encoder = OpenClipEncoder(
    OpenClipConfig(
        backend="openclip",
        model_name="ViT-B-32",
        pretrained="openai",
        device=config.openclip.device,          # auto | cpu | mps | cuda
        precision=config.openclip.precision,    # fp32 | fp16
        normalization=config.openclip.normalization,  # l2 | none
        preprocessing_id=config.openclip.preprocessing_id,
        tokenizer_id=config.openclip.tokenizer_id,
    )
)
```

Supported model/checkpoint combinations are whatever OpenCLIP provides
(`model_name` + `pretrained`, e.g. `ViT-B-32` / `openai` or
`laion2b_s34b_b79k`). Weights are downloaded by OpenCLIP on first use.

### Experiment configurations

Model/checkpoint selection is file-driven so experiments are reproducible
without editing `configs/default.yaml`. Shipped example configs live in
`configs/experiments/` (each is a complete standalone config):

| Config file                          | `model_name` | `pretrained`        |
|--------------------------------------|--------------|---------------------|
| `configs/experiments/vit-b-32-openai.yaml`   | `ViT-B-32`   | `openai`            |
| `configs/experiments/vit-l-14-openai.yaml`   | `ViT-L-14`   | `openai`            |
| `configs/experiments/vit-l-14-laion2b.yaml`  | `ViT-L-14`   | `laion2b_s32b_b82k` |

Load an experiment config anywhere a config is expected (Python API):

```python
config = load_config("configs/experiments/vit-l-14-openai.yaml")
encoder = OpenClipEncoder(config.openclip)
```

Each experiment records its full embedding space in the artifacts it
produces, so artifacts built from different configs are never silently mixed
(embedding dim, model, and checkpoint are all part of `space_id`). Note that
`ViT-L-14` produces 768-dimensional embeddings vs. 512 for `ViT-B-32`; an
index built from one cannot be queried with the other.

### EmbeddingSpace

`EmbeddingSpace` is the exact identity of a vector space: backend, model_id,
checkpoint_id, `space_id`, embedding_dim, output_dtype, normalization,
preprocessing_id, model_version. Matching dimension alone never establishes
compatibility — the complete space must be equal.

```python
space = encoder.embedding_space
print(space.space_id)          # e.g. openclip:ViT-B-32:openai:fp32:l2:...
print(space.embedding_dim)     # 512
```

### Image/text compatibility

`encode_image` and `encode_text` both return an `Embedding` with `kind`
`"image"` or `"text"`, sharing one `EmbeddingSpace`, so image and text
embeddings are directly comparable when produced by the same encoder config.

```python
text_emb = encoder.encode_text("a truck")
img_emb = encoder.encode_image_batch([image1, image2])   # batch form
```

## 4. Embedding stage

### API

```python
from aic2026.embedding_runner import build_embeddings

manifest = build_embeddings(
    dataset,             # Dataset accessor
    encoder,             # EmbeddingEncoder (e.g. OpenClipEncoder)
    artifact_dir,        # output dir; created if absent
    artifact_id,         # unique run identity (non-empty string)
    batch_size=16,       # keyframes per model invocation / checkpoint flush
    resume=True,         # continue a partial build; validate a complete one
    stop_after_rows=None,# testing hook; no final manifest written
    video_ids=None,      # optional subset; videos stay in dataset order
    image_loader=None,   # callable path -> image (default Pillow, RGB)
)
```

There is **no CLI command** for this stage; call the Python API from a script.

Arguments:

- `artifact_dir` — parent directory; the artifact is written to
  `<artifact_dir>/<artifact_id>/`.
- `artifact_id` — run identity recorded in the manifest; used to detect
  mismatches on resume.
- `batch_size` — images per model call; also the checkpoint flush granularity.
- `resume` — resume from `.checkpoint.json`; an already-complete artifact is
  validated and returned without recomputation.
- `video_ids` — restrict to a subset; videos are always processed in the
  dataset's sorted `video_ids()` order, so row order is deterministic.

### Output artifact

```text
<artifact_dir>/<artifact_id>/
    manifest.json        # EmbeddingManifest (schema "1.0")
    embeddings.npy       # (N, D) matrix; row i <-> frames[i]
    frames.ndjson        # one canonical KeyframeTiming per line
```

`EmbeddingManifest` records `artifact_id`, full `embedding_space`,
`embedding_dim`, `embedding_dtype`, `num_rows`, `num_videos`,
`source_dataset_id`, `canonical_schema_version`, `created_at`.

### Resume behavior

- Completed rows are never re-encoded.
- A resumed build whose encoder space / dim / dtype / `artifact_id` differs
  from `.build_meta.json` is rejected.
- The artifact is complete only when `manifest.json` exists (written
  atomically after validation); its absence means the build is incomplete.
- `resume=False` on an already-complete artifact is rejected to prevent
  silent overwrites.

### Validation

```python
from aic2026.embedding_artifact import (
    read_manifest, validate_embedding_artifact, iter_frames,
)

manifest = read_manifest("data/interim/embeddings/<id>/manifest.json")
problems = validate_embedding_artifact(manifest, "<artifact_dir>/<id>")
assert not problems, problems
```

### Tiny subset example

```python
from pathlib import Path
from aic2026.config import load_config
from aic2026.dataset import Dataset
from aic2026.model_backends.openclip import OpenClipEncoder
from aic2026.embedding_runner import build_embeddings

config = load_config("configs/experiments/vit-b-32-openai.yaml")  # experiment config
dataset = Dataset(config.data_raw_dir)
encoder = OpenClipEncoder(config.openclip)     # ViT-B-32 / openai

manifest = build_embeddings(
    dataset, encoder,
    artifact_dir=Path("data/interim/embeddings"),
    artifact_id="openclip-openai-mini-l30v041-l21v001",
    batch_size=8,
    video_ids=["L30_V041", "L21_V001"],        # 24 + 307 = 331 rows
)
print(manifest.num_rows, manifest.num_videos, manifest.embedding_space.space_id)
```

There is no embedding CLI, so "running an experiment" means loading its config
with `load_config("<configs/experiments/....yaml>")` and passing the resulting
`encoder` (and `config.openclip` model metadata) through the Python API, for
both `build_embeddings` here and `run_kis_prediction` (Section 7). Pick
`artifact_id` values that encode the model/checkpoint so artifacts stay
distinguishable across experiments.

### Full dataset example

```python
manifest = build_embeddings(
    dataset, encoder,
    artifact_dir=Path("data/interim/embeddings"),
    artifact_id="openclip-batch1-full",        # 873 videos, 177321 keyframes
    batch_size=128,
    resume=True,
)
```

Run a tiny subset first (see Section 9) before launching the full build.

## 5. Index stage

### API

```python
from aic2026.index_builder import build_index
from aic2026.embeddings import EmbeddingSpace

manifest = build_index(
    dataset,              # Dataset accessor (reads dataset embeddings)
    artifact_dir,         # output dir; created if absent
    artifact_id,          # unique run identity
    score_metric="cosine",
    batch_flush=1000,     # rows between memmap flush + checkpoint
    resume=True,
    stop_after_rows=None, # testing hook; no final manifest written
    video_ids=None,       # optional subset
    embedding_space=None, # EmbeddingSpace identity; None => legacy/unknown
)
```

There is **no CLI command** for this stage.

`build_index` currently consumes the canonical `Dataset` (the BTC
`clip-features-32` matrices). Pass `embedding_space=encoder.embedding_space`
when the indexed vectors are model-produced so the artifact records that
identity; pass `None` to index raw BTC features as an explicitly
`legacy/unknown` space (what `data/interim/indexes/btc-batch1-full` is). The
embedding and index artifacts share the same row-order/identity contract, so
an embedding artifact is index-ready; the current builder reads the dataset
directly.

### Output artifact

```text
<artifact_dir>/<artifact_id>/
    manifest.json        # IndexManifest (schema "1.1")
    embeddings.npy       # (N, D) matrix; row p <-> frames[p]
    norms.npy            # (N,) float32 L2 norms
    frames.ndjson        # canonical KeyframeTiming per line
    index.bin            # RESERVED for a future ANN payload (not produced)
```

### Validation

```python
from aic2026.index_artifact import read_manifest, validate_index_files

manifest = read_manifest("<index_dir>/manifest.json")
problems = validate_index_files(manifest, "<index_dir>")
assert not problems, problems
```

## 6. Retrieval stage

```python
from aic2026.retrieval import IndexRetriever

retriever = IndexRetriever("data/interim/indexes/btc-batch1-full")
print(retriever.manifest.artifact_id)          # validated index metadata
```

`search` accepts either an `Embedding` (required for model-identified
indexes, so the space is validated) or a raw 1-D vector (only for explicit
`legacy/unknown` indexes):

```python
results = retriever.search(query_embedding, top_k=10)
for candidate in results:
    print(candidate.rank, candidate.video_id, candidate.frame_index, candidate.score)
```

`RetrievalCandidate` fields: `rank`, `video_id`, `keyframe_ordinal`,
`frame_index`, `pts_time`, `score`. Rank is 1-based; `score` is cosine
similarity computed from stored norms. Mismatched embedding spaces, wrong
dimensions, zero-norm queries, or a non-positive `top_k` raise `ValueError`.

## 7. KIS evaluation stage

### CLI: `aic2026 predict-kis`

```bash
aic2026 predict-kis \
  --config configs/experiments/vit-b-32-openai.yaml \
  --evaluation-set data/eval/eval_set.csv \
  --index-artifact data/interim/indexes/btc-batch1-full \
  --output data/interim/predictions/kis.json \
  --top-k 100 \
  --query-text-field query_text_vi \
  --run-id my-run
```

The `--config` file selects the OpenCLIP model/checkpoint used to encode the
query text. Use `configs/experiments/vit-l-14-openai.yaml` or
`configs/experiments/vit-l-14-laion2b.yaml` to run another model; the index
must be in that model's embedding space (build it from a matching embedding
artifact, Section 4/5), or the run fails unless
`--allow-embedding-space-mismatch` is explicitly passed for the `legacy/unknown`
BTC baseline index.

Flags:

- `--config` — YAML config (defaults to `configs/default.yaml`); supplies the
  OpenCLIP model/checkpoint.
- `--evaluation-set` — evaluation CSV (default `data/eval/eval_set.csv`).
- `--index-artifact` — index artifact directory (required).
- `--output` — prediction artifact JSON path (required).
- `--top-k` — ranked candidates per query (default 100).
- `--query-text-field` — `query_text_vi` (default) or `query_text_en`.
- `--query-id` — repeat to restrict to an explicit subset of query ids.
- `--run-id` — stable run id recorded in the artifact.
- `--allow-embedding-space-mismatch` — diagnostic override; disabled by
  default. See below.

### Python API

```python
from pathlib import Path
from aic2026.config import load_config
from aic2026.kis_prediction_runner import (
    KisPredictionRunnerConfig, run_kis_prediction,
)
from aic2026.model_backends.openclip import OpenClipEncoder

config = load_config()
artifact = run_kis_prediction(
    KisPredictionRunnerConfig(
        evaluation_set_path=Path("data/eval/eval_set.csv"),
        index_artifact_dir=Path("data/interim/indexes/btc-batch1-full"),
        prediction_output_path=Path("data/interim/predictions/kis.json"),
        model_config=config.openclip.__dict__,   # provenance only
        top_k=100,
        query_text_field="query_text_en",
        allow_embedding_space_mismatch=False,
        run_id="my-run",
    ),
    OpenClipEncoder(config.openclip),
)
print(artifact.processed_query_count)
```

### query_text_vi vs query_text_en

`query_text_field` selects which field of each evaluation query is encoded:
`query_text_vi` (Vietnamese, always present) or `query_text_en` (English,
optional per row). Requesting `query_text_en` when a query has none raises
`ValueError`.

### Embedding-space mismatch behavior

A query must share the index's exact `EmbeddingSpace`. Mismatches fail fast
(`ValueError`) by default. `--allow-embedding-space-mismatch` permits the
diagnostic case where the index is an explicit `legacy/unknown` space (such as
`btc-batch1-full`) and the encoder is model-identified; it is refused for
model-identified indexes. The artifact records
`embedding_space_compatible` / `embedding_space_override_used`.

### Scoring (no CLI; Python API)

```python
from aic2026.evaluation import load_eval_set
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring
from aic2026.kis_scoring import evaluate_kis, KIS_TOP_K

evaluation_set = load_eval_set("data/eval/eval_set.csv")
predictions = load_kis_predictions_for_scoring("data/interim/predictions/kis.json")
result = evaluate_kis(evaluation_set, predictions, cutoffs=KIS_TOP_K)

for metric in result.top_k_scores:          # R@1, R@5, R@20, R@50, R@100
    print(metric.k, metric.score)
print(result.final_score)                   # mean(R@1..R@100)
```

`evaluate_kis` requires predictions for exactly the KIS query ids in the
evaluation set, so an incomplete experiment cannot silently produce a partial
score. `KisPrediction(rank, video_id, frame_index)` carries the original-video
frame identity used for scoring.

## 8. Baseline reproduction (reference experiment)

The verified reference baseline uses **BTC-provided features** (not a
model-built index) with an OpenCLIP query encoder. It is a **reference
experiment only**, not the current best configuration.

| Setting                    | Value                                  |
|----------------------------|----------------------------------------|
| Model                      | `ViT-B-32`                             |
| Checkpoint                 | `openai`                               |
| Query text field           | `query_text_en`                        |
| Index                      | `data/interim/indexes/btc-batch1-full` (BTC features, `legacy/unknown` space) |
| Evaluation set             | `data/eval/eval_set.csv` (101 KIS rows) |
| R@1 / R@5 / R@20           | `0.059406` / `0.138614` / `0.247525`   |
| R@50 / R@100               | `0.326733` / `0.425743`                |
| **Final score**            | **`0.239604`**                         |

Verified against `data/interim/predictions/kis-openai-en-vs-btc-batch1-full.json`.

Reproduce it with the shipped experiment config that selects the `openai`
checkpoint (`configs/default.yaml` defaults to `laion2b_s34b_b79k`):
`configs/experiments/vit-b-32-openai.yaml` (ViT-B-32 / openai).

```bash
aic2026 predict-kis \
  --config configs/experiments/vit-b-32-openai.yaml \
  --evaluation-set data/eval/eval_set.csv \
  --index-artifact data/interim/indexes/btc-batch1-full \
  --output data/interim/predictions/kis-openai-en-vs-btc-batch1-full.json \
  --top-k 100 \
  --query-text-field query_text_en \
  --allow-embedding-space-mismatch \
  --run-id baseline-openai-en
```

`--allow-embedding-space-mismatch` is required here because the BTC index is
an explicit `legacy/unknown` space that differs from the model-identified
OpenCLIP space; the baseline artifact recorded
`embedding_space_compatible=False`, `embedding_space_override_used=True`.

Then score with the Section 7 snippet; `result.final_score` must equal
`0.239604` (R@1 `0.059406`, R@5 `0.138614`, R@20 `0.247525`,
R@50 `0.326733`, R@100 `0.425743`).

## 9. Experiment workflow

1. **Subset smoke test** — build embeddings for 1–2 videos
   (`video_ids=["L30_V041", "L21_V001"]`, small `batch_size`) with a
   `stop_after_rows` limit on the first attempt if desired.
2. **Build the embedding artifact** — full run with `resume=True`.
3. **Validate the artifact** — `validate_embedding_artifact`; confirm shape,
   dtype, row count, and mmap readability.
4. **Build the index** — `build_index` with `embedding_space=encoder.embedding_space`
   for model-built vectors, or `None` for BTC features.
5. **Run retrieval** — `IndexRetriever.search` on a few queries; inspect
   `RetrievalCandidate` fields.
6. **Evaluate KIS** — `predict-kis` (or `run_kis_prediction`), then
   `evaluate_kis`.
7. **Compare experiments** — keep the baseline (Section 8) fixed; any
   improvement must be measured against the same evaluation procedure and
   recorded with the run's config and artifact ids.
