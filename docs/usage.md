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

CLI commands exist (`aic2026 inspect`, `aic2026 predict-kis`,
`aic2026 temporal-expand-kis`, `aic2026 media-info-search`,
`aic2026 media-info-evaluate`, `aic2026 media-info-report`,
`aic2026 objects-analyze`, `aic2026 object-rerank-kis`). The embedding and
index stages are Python-API only and are composed in scripts. Analysis-only
CLIs live under `scripts/` (e.g. `scripts/analyze_kis_errors.py`,
`scripts/analyze_keyframes.py`, `scripts/embed_shard.py`,
`scripts/split_embedding_shards.py`, `scripts/merge_embedding_shards.py`,
`scripts/create_experiment_manifest.py`; Sections 15-18).

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

## 10. Media-info retrieval branch (lexical)

An independent retrieval *branch* over the per-video `media-info` text
(`title`, `keywords`, `description`). It ranks videos by BM25 relevance for a
text query and does not touch CLIP retrieval, the scorer, evaluation, or the
prediction-artifact schema. Use it to study video-level text signals before
any fusion with the CLIP index.

Source: `aic2026.media_info_retrieval` (stdlib-only; reads the raw
`media-info` files directly).

### Python API

```python
from aic2026.media_info_retrieval import MediaInfoIndex

# From the raw media-info directory:
index = MediaInfoIndex.build_from_dir("data/raw/aic2026_batch1/media-info")
print(index.num_documents)      # 873 (one document per video)

# Or build from parsed documents:
from aic2026.media_info_retrieval import MediaInfoDocument, MediaInfoRetrievalConfig
index = MediaInfoIndex.build(
    [MediaInfoDocument("L21_V001", "60 Giây Sáng tin tức", ("tin tức", "HTV"), "Bản tin thời sự")],
    config=MediaInfoRetrievalConfig(title_weight=3.0, keywords_weight=2.0, description_weight=1.0),
)

# Query: ranked video ids with BM25 scores.
hits = index.query("tin tức mới nhất hôm nay", top_k=10)
for hit in hits:
    print(hit.rank, hit.video_id, hit.score)   # MediaInfoHit(rank, video_id, score)
```

### CLI

```bash
aic2026 media-info-search \
  --media-info-dir data/raw/aic2026_batch1/media-info \
  --query "một chiếc thuyền chở nhiều du khách" \
  --top-k 10 \
  --output data/interim/media-info-search/hits.json
```

Expected stdout:

```text
Ranked hits written to: data/interim/media-info-search/hits.json
Query: một chiếc thuyền chở nhiều du khách
  rank=1 video_id=L21_V001 score=4.223456
```

`--query` is repeatable (multiple queries per run). Optional BM25 overrides:
`--k1`, `--b`, `--title-weight`, `--keywords-weight`, `--description-weight`.
Without `--output`, hits are printed to stdout only.

### Determinism and schema

- Documents are indexed in sorted video-id order; ties in BM25 score are
  resolved by video id, so identical inputs always produce identical output.
- `MediaInfoIndex.save(path)` writes a self-describing JSON artifact
  (schema version, config, source documents); `MediaInfoIndex.load(path)`
  reproduces the same rankings without re-reading the raw dataset.
- Loading is strict by default (`build_from_dir(strict=True)` raises on any
  unreadable/malformed file). Pass `strict=False` to skip problem files;
  skipped files are recorded in `index.problems`.

### Reproducing the experiment

```python
from aic2026.media_info_retrieval import MediaInfoIndex

index = MediaInfoIndex.build_from_dir("data/raw/aic2026_batch1/media-info")
index.save("data/interim/media-info-index.json")

# Later / elsewhere: reload and re-query without the raw dataset.
reloaded = MediaInfoIndex.load("data/interim/media-info-index.json")
print(reloaded.query("thuyền chở du khách", top_k=5))
```

## 11. Media-info video-level evaluation

Measures how often the media-info lexical branch retrieves the ground-truth
*video* for each KIS query, and compares it against the CLIP baseline at the
video level. This is a video-retrieval study only: it reports the rank of the
GT video in the media-info result list and does not score temporal
localization. Use it to decide whether a media-info prefilter is worth fusing
with the CLIP index.

Source: `aic2026.media_info_evaluation` (consumes a `MediaInfoIndex`, the
evaluation set, and optionally a CLIP prediction artifact).

### Python API

```python
from aic2026.evaluation import EvaluationTask, load_eval_set
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring
from aic2026.media_info_evaluation import (
    evaluate_media_info_video_retrieval,
    compare_to_clip_baseline,
    format_report,
)
from aic2026.media_info_retrieval import MediaInfoIndex

index = MediaInfoIndex.build_from_dir("data/raw/aic2026_batch1/media-info")
eval_set = load_eval_set("data/eval/eval_set.csv")
queries = tuple(eval_set.iter_queries(EvaluationTask.KIS))

results, summary = evaluate_media_info_video_retrieval(
    queries, index, query_text_field="query_text_en", top_k=100
)
print(summary.recall_at)       # ((1, ...), (5, ...), ..., (100, ...))
print(summary.avg_gt_rank)
print(summary.not_retrieved)   # query ids whose GT video was not retrieved

# Compare against the CLIP baseline at the video level (optional).
clip_predictions = load_kis_predictions_for_scoring(
    "data/interim/predictions/kis-l14-laion2b-full-en.json"
)
comparison = compare_to_clip_baseline(
    queries, {r.query_id: r for r in results}, clip_predictions, cutoff=100
)
print(comparison.media_info_only)   # media-info succeeds, CLIP fails
print(comparison.clip_only)         # CLIP succeeds, media-info fails

print(format_report(summary, results=results, comparison=comparison))
```

`evaluate_media_info_video_retrieval` returns a per-query list of
`VideoRetrievalQueryResult(query_id, query_text, gt_video_id, gt_rank, hits)`
and a `VideoRetrievalSummary(total, recall_at, avg_gt_rank, not_retrieved)`.
`gt_rank` is `None` when the GT video is not in the media-info result list;
`not_retrieved` collects those query ids. `recall_at` covers
`DEFAULT_TOP_K_VALUES = (1, 5, 10, 20, 50, 100)`; pass `cutoffs=` to override,
with `top_k >= max(cutoffs)`.

A "succeeds" outcome in `compare_to_clip_baseline` means the GT video rank is
`<= cutoff` (top-100 by default). Each record is grouped into exactly one of
`media_info_only`, `clip_only`, `both`, `neither`.

### CLI

```bash
aic2026 media-info-evaluate \
  --media-info-dir data/raw/aic2026_batch1/media-info \
  --evaluation-set data/eval/eval_set.csv \
  --query-text-field query_text_en \
  --top-k 100 \
  --output data/interim/media-info-evaluate/en-metrics.json
```

Expected stdout (abridged):

```text
Media-info video-level retrieval evaluation
================================================================================
Queries: 101
  Video Recall@1: 0.0198
  Video Recall@5: 0.0594
  Video Recall@100: 0.2079
  average GT video rank: 51.86
  queries where GT video not retrieved: 80
Comparison vs CLIP baseline (cutoff top-100):
  media-info succeeds, CLIP fails: 1
  CLIP succeeds, media-info fails: 70
  both succeed: 20
  neither succeeds: 10
Metrics written to: data/interim/media-info-evaluate/en-metrics.json
```

The JSON output contains `metrics` (aggregate summary), `results` (per-query
details), and `comparison` (present only when a CLIP baseline was used).

Key flags:

- `--query-text-field` `query_text_en` (default) or `query_text_vi`; run both
  to compare the two query languages.
- `--top-k` number of videos retrieved per query (default 100; must be `>=`
  the largest recall cutoff).
- `--clip-predictions` path to the CLIP prediction artifact. When omitted, the
  default `data/interim/predictions/kis-l14-laion2b-full-en.json` is used if it
  exists and covers the evaluation set; otherwise the comparison is skipped.
  Passing the flag explicitly makes a non-covering artifact an error.
- BM25 overrides (`--k1`, `--b`, `--title-weight`, `--keywords-weight`,
  `--description-weight`) apply to the built index.

### Reproducing the experiment

The command above reproduces the video-level media-info metrics and the
CLIP-baseline comparison for the English queries. Both the `query_text_en` and
`query_text_vi` runs use the same `MediaInfoIndex` (deterministic, built from
the raw media-info directory) and the same evaluation set and CLIP baseline, so
their outputs are directly comparable. Record the command, artifact paths, and
index config with any results reported.

## 12. Media-info query categorization report

Categorizes every KIS query by whether media-info and the CLIP baseline
retrieve the ground-truth *video* (rank `<= top-k`) at the video level:

- **MEDIA_ONLY** — media-info retrieves the GT video, CLIP does not.
- **CLIP_ONLY** — CLIP retrieves the GT video, media-info does not.
- **BOTH** — both retrieve the GT video.
- **NEITHER** — neither retrieves the GT video.

This is a pure reporting step over the existing evaluation outputs; it performs
no retrieval or fusion and does not modify any contract.

Source: `aic2026.media_info_evaluation` (`category_report_to_dict`,
`format_category_report`).

### Python API

```python
from aic2026.media_info_evaluation import (
    category_report_to_dict,
    compare_to_clip_baseline,
    format_category_report,
)
# comparison from compare_to_clip_baseline(...) (see Section 11)

print(format_category_report(comparison))     # human-readable report
payload = category_report_to_dict(comparison)  # JSON-safe dict
print(payload["counts"])   # {'MEDIA_ONLY': ..., 'CLIP_ONLY': ..., ...}
```

The JSON payload contains `cutoff`, `counts` (per-category query counts),
`total`, and `categories` — for each category a list of
`{query_id, query_text_en, query_text_vi, gt_video_id, media_info_gt_rank,
clip_gt_rank}` records.

### CLI

```bash
aic2026 media-info-report \
  --media-info-dir data/raw/aic2026_batch1/media-info \
  --evaluation-set data/eval/eval_set.csv \
  --clip-predictions data/interim/predictions/kis-l14-laion2b-full-en.json \
  --output data/interim/media-info-report/categories.json
```

Expected stdout (abridged):

```text
Media-info vs CLIP baseline query categorization
================================================================================
Cutoff: top-100
MEDIA_ONLY (1): media-info retrieves GT video, CLIP does not
  kis_112 | EN: Find a news video about rising pork prices ... | VI: Tìm một
  video bản tin về thịt heo tăng giá ... | GT: L22_V022 | media-info rank: 58
  | CLIP rank: —
CLIP_ONLY (70): CLIP retrieves GT video, media-info does not
  ...
Summary: MEDIA_ONLY=1, CLIP_ONLY=70, BOTH=20, NEITHER=10 | total=101
Categorized report written to: data/interim/media-info-report/categories.json
```

Each line lists the query id, English and Vietnamese texts, GT video id,
media-info rank (`—` when not retrieved), and CLIP first GT-video rank (`—`
when not in the baseline). `--output` writes the JSON export; without it the
report is printed to stdout only. A CLIP baseline is required (the default
artifact is used when it exists and covers the evaluation set; otherwise the
command fails with an explanatory message). `--query-text-field`, `--top-k`,
and the BM25 overrides behave as in Section 11.

### Reproducing the report

Re-running the command above with the same media-info directory, evaluation
set, CLIP baseline, and `--top-k` reproduces the same categorization because
the media-info index and evaluation are deterministic. Record the command and
artifact paths with any reported categorization.

## 13. BTC object-detection analysis for KIS

Analyzes whether the per-keyframe BTC object detections
(`data/raw/aic2026_batch1/objects/<VIDEO_ID>/<NNN>.json`) can support KIS
retrieval. This is a read-only analysis step: it never runs or modifies
retrieval, the scorer, evaluation, or any prediction artifact, and it
implements no fusion.

Source: `aic2026.object_analysis` (`run_object_analysis`,
`analyze_object_vocabulary`, `analyze_query_object_coverage`,
`analyze_gt_frames`, `analyze_retrieval_potential`, `format_report`).

The report has four sections:

1. **Object vocabulary** — total frames/detections scanned, unique entities,
   confidence percentiles, retention effects at configurable confidence
   thresholds, and the top object classes.
2. **Query-object coverage** — for each query, whether words in the query text
   (English or Vietnamese) match entities detected in the query's
   ground-truth frames. `MATCH` when at least one GT entity shares a token
   with the query; otherwise `NO_MATCH`.
3. **Ground-truth frame analysis** — the keyframe ordinals, object list,
   and detection confidence statistics inside each query's annotated frame
   range (mapped from original-video frame indices via `map-keyframes`).
4. **Retrieval potential** — against the CLIP top-`k` predictions, the fraction
   of GT-frame predictions whose frame overlaps a GT object entity
   (`gt_overlap_rate`) minus the same fraction over false-positive frames
   (`false_positive_overlap_rate`). Positive separation means objects can help
   separate GT frames from false positives.

### Python API

```python
from aic2026.object_analysis import format_report, run_object_analysis

report = run_object_analysis(
    objects_dir="data/raw/aic2026_batch1/objects",
    map_dir="data/raw/aic2026_batch1/map-keyframes",
    evaluation_set_path="data/eval/eval_set.csv",
    clip_predictions_path="data/interim/predictions/kis-l14-laion2b-full-en-top1000.json",
    query_text_field="query_text_en",   # or "query_text_vi"
    thresholds=(0.1, 0.3, 0.5, 0.7),    # confidence retention thresholds
    top_classes=20,
    min_confidence=0.3,                 # min score for GT / predicted object sets
    prediction_top_k=1000,              # CLIP predictions analyzed per query
    example_limit=5,
)
print(format_report(report))
payload = report.to_dict()   # JSON-safe dict (use --output to persist it)
print(payload["coverage"]["matched_count"])
print(payload["retrieval_potential"]["summary"])
```

`run_object_analysis` requires the objects collection, the map-keyframes
collection, the evaluation CSV, and a CLIP prediction artifact. It raises
`FileNotFoundError` when a required input is absent and `ValueError` when the
inputs are invalid (for example, no KIS queries). Frame indices in the
prediction artifact and the evaluation CSV are original-video frame indices;
they are mapped to keyframe ordinals through `map-keyframes` before object
files are located.

### CLI

```bash
aic2026 objects-analyze \
  --objects-dir data/raw/aic2026_batch1/objects \
  --map-dir data/raw/aic2026_batch1/map-keyframes \
  --evaluation-set data/eval/eval_set.csv \
  --clip-predictions data/interim/predictions/kis-l14-laion2b-full-en-top1000.json \
  --output data/analysis/objects/kis-analyze-l14-laion2b-top1000.json
```

Available flags: `--objects-dir`, `--map-dir`, `--evaluation-set`,
`--clip-predictions` (defaults to the top-1000 artifact when present),
`--query-text-field`, `--confidence-thresholds` (comma-separated),
`--min-confidence`, `--top-classes`, `--top-k`, `--max-videos`,
`--examples`, and `--output` (JSON export). Without `--output` the report is
printed to stdout only.

Expected stdout (abridged):

```text
BTC object detection analysis for KIS retrieval
================================================================================
1. Object vocabulary
  total object frames: 177321
  total detections: 17732100
  unique object entities: 584
  ...
2. Query-object coverage
  queries with object coverage (MATCH): 18/101
  queries without object coverage (NO_MATCH): 83/101
  ...
3. Ground-truth frame analysis
  kis_100 (L21_V001):
    keyframes in range: (238,)
    objects (6): Boat:1, Clothing:1, Human face:1, Man:1, Person:1, Woman:1
  ...
4. Retrieval potential (CLIP top-k)
  queries with GT-frame predictions: 78/101
  queries where object overlap separates (positive separation): 76/78
  average separation: 0.1785
  mean GT-frame overlap rate: 0.9968
  mean false-positive overlap rate: 0.7722
```

### Reference results (en, top-1000, min confidence 0.3)

Recorded from the full evaluation set for later comparison:

- Vocabulary: 177,321 frames, 17,732,100 detections, 584 unique entities;
  confidence median ≈ 0.017; retention at `>= 0.3` is 1,122,384 detections
  (514 entities, 6.33/frame).
- Coverage: 18/101 queries MATCH, 83/101 NO_MATCH.
- Retrieval potential: 78/101 queries have GT-frame predictions; object
  overlap gives positive separation in 76/78 of those (mean separation 0.18,
  mean GT-frame overlap 0.997, mean false-positive overlap 0.77).

### Reproducing the analysis

Re-running the command above with the same collection directories, evaluation
set, CLIP baseline, and analysis parameters reproduces the same numbers; the
scan order and aggregation are deterministic. Record the command, thresholds,
and artifact paths with any reported analysis.

### CLI command reference

```text
aic2026 inspect          - summarize the raw dataset (media-info, objects, keyframes).
aic2026 build-index      - build the CLIP index artifact.
aic2026 predict-kis      - produce KIS prediction artifacts from an index.
aic2026 temporal-expand-kis - expand KIS predictions with temporal window decay.
aic2026 media-info-search - rank videos by BM25 relevance over media-info text.
aic2026 media-info-evaluate - measure video-level retrieval of media-info text.
aic2026 media-info-report - categorize queries by media-info / CLIP GT-video retrieval.
aic2026 objects-analyze  - analyze BTC object detections for KIS retrieval usefulness.
aic2026 object-rerank-kis - rerank a KIS prediction artifact with object detections.
scripts/analyze_kis_errors.py - categorize KIS retrieval failures (HIT / NEAR_VIDEO / FAIL).
scripts/analyze_keyframes.py - analyze BTC keyframe extraction quality for KIS (Section 15).
scripts/deduplicate_keyframes.py - deduplicate keyframes before split/embed (Section 17).
scripts/embed_shard.py      - resumable, hardware-aware shard embedding (Section 16).
scripts/split_embedding_shards.py - deterministic balanced split into worker inputs (Section 18).
scripts/merge_embedding_shards.py - validate + merge completed shards (Section 18).
scripts/split_keyframe_extraction_shards.py - split a video manifest into worker inputs (Section 20).
scripts/extract_keyframes.py - resumable DAKE-inspired keyframe extraction (Section 20).
scripts/merge_keyframe_shards.py - validate + merge extraction shards (Section 20).
scripts/build_video_manifest.py - build the canonical videos.json inventory (Section 19).
scripts/create_experiment_manifest.py - record experiment metadata (Section 21).
```

## 14. Object-based reranking of KIS predictions

Re-ranks an existing ``predict-kis`` prediction artifact using BTC OWL-ViT
object detections, testing whether objects can improve CLIP retrieval ranking.
For each prediction the combined score is

```text
final_score = clip_score + lambda_weight * object_score
```

where the object score is derived from the entities detected in the predicted
frame that match the query. Predictions are re-ranked by ``final_score`` in
descending order with fresh strictly increasing ranks; ties keep the original
ranking order (stable sort).

This is a post-processing stage over prediction artifacts only. It does not
modify retrieval, the scorer, evaluation, the embedding/index pipeline, or the
prediction-artifact schema. The output is a standard prediction artifact that
remains loadable by ``read_kis_prediction_artifact`` /
``load_kis_predictions_for_scoring`` and consumable by ``evaluate_kis``.

Source: `aic2026.object_reranking` (`ObjectRerankingConfig`,
`FrameObjectSource`, `build_reranked_artifact`,
`rerank_prediction_artifact`, `write_reranked_prediction_artifact`).

### Matching method

1. The query text is tokenized into word tokens.
2. Tokens are normalized by `normalization_method`:
   - `none` — raw tokens (case-sensitive).
   - `lower` — lowercase only.
   - `stem` — lowercase plus naive suffix stripping (`ing`/`ed`/`es`/`s`).
3. The frame-level object representation keeps only detections with
   `confidence >= confidence_threshold`, merging duplicate entities by maximum
   confidence.
4. An entity matches the query when its normalized entity-name tokens overlap
   the normalized query tokens.
5. The object score is the maximum (`max`, default) or sum (`sum`) of the
   confidences of matched entities; `0.0` when nothing matches.

### Input/output schema

- **Input (prediction)**: any `predict-kis` prediction artifact (schema
  version `1.0`); predictions reference original-video `frame_index` values.
- **Input (objects)**: `data/raw/aic2026_batch1/objects/<VIDEO_ID>/<NNN>.json`
  (keyed by keyframe ordinal) plus `map-keyframes/<VIDEO_ID>.csv`, used to
  resolve each predicted `frame_index` to a keyframe ordinal.
- **Output**: a new prediction artifact (same schema version `1.0`) with the
  same top-level metadata (evaluation set, index reference, model metadata,
  query text field) but updated `score` (combined score) and `rank` per
  prediction. `run_id` defaults to a deterministic id derived from the source
  run id and the configuration; `created_at` is the run timestamp.

### Python API

```python
from aic2026.object_reranking import (
    FrameObjectSource,
    ObjectRerankingConfig,
    build_reranked_artifact,
    rerank_prediction_artifact,
    write_reranked_prediction_artifact,
)

source = FrameObjectSource(
    objects_dir="data/raw/aic2026_batch1/objects",
    map_dir="data/raw/aic2026_batch1/map-keyframes",
)
config = ObjectRerankingConfig(
    confidence_threshold=0.3,
    lambda_weight=1.0,
    normalization_method="lower",   # "none" | "lower" | "stem"
    object_score_method="max",      # "max" | "sum"
)

# From an in-memory artifact or a path; object_source may also be a directory.
reranked = rerank_prediction_artifact(
    prediction_artifact="data/interim/predictions/kis-l14-laion2b-full-en-top1000.json",
    object_source=source,
    config=config,
)

# Persist with the standard atomic artifact writer.
write_reranked_prediction_artifact(
    output_path="data/interim/predictions/kis-obj-rerank-l1-c0.3.json",
    source_path="data/interim/predictions/kis-l14-laion2b-full-en-top1000.json",
    object_source=source,
    config=config,
    run_id="obj-rerank-1",
)
```

`rerank_prediction_artifact` accepts a `KisPredictionArtifact` or a path for
`prediction_artifact`, and a `FrameObjectSource` or an objects directory for
`object_source`. It raises `FileNotFoundError` when a collection or artifact
path is absent and `ValueError` for invalid configuration or payloads.

### CLI

```bash
aic2026 object-rerank-kis \
  --prediction data/interim/predictions/kis-l14-laion2b-full-en-top1000.json \
  --output data/interim/predictions/kis-obj-rerank-l1-c0.3.json \
  --objects-dir data/raw/aic2026_batch1/objects \
  --lambda 1.0 \
  --confidence-threshold 0.3
```

Available flags: `--prediction` (required), `--output` (required),
`--objects-dir` (defaults to `config raw_dir/objects`), `--map-dir` (defaults
to the sibling `map-keyframes` of `--objects-dir`), `--lambda`,
`--confidence-threshold`, `--normalization-method` (`none`/`lower`/`stem`),
`--object-score-method` (`max`/`sum`), and `--run-id`.

Expected stdout:

```text
Reranked prediction artifact: data/interim/predictions/kis-obj-rerank-l1-c0.3.json
Processed queries: 101/101
Run id: object-rerank-kis-run-20260814T050833Z-l1-c0.3-nlower-mmax
```

### Reproduction steps

1. Produce (or reuse) a base prediction artifact:
   `aic2026 predict-kis ...` (Section 4/5) or use an existing artifact such as
   `kis-l14-laion2b-full-en-top1000.json`.
2. Re-run the CLI command above with fixed `--lambda`,
   `--confidence-threshold`, and normalization/aggregation choices. Record the
   full command and both artifact paths.
3. The reranking is deterministic: the same source artifact, object/map
   collections, and configuration reproduce identical `score`, `rank`, and
   `run_id` (except `created_at`).
4. Evaluate the reranked artifact with the standard scorer and compare against
   the base artifact using the same evaluation procedure.

The full `evaluate_kis` metrics shown in earlier sections are reproduced by
feeding the reranked artifact to `load_kis_predictions_for_scoring` +
`evaluate_kis` exactly like any other prediction artifact.

---

## 15. BTC keyframe extraction analysis for KIS

Measures how well the current BTC-provided keyframes cover the KIS
ground-truth events, before implementing any new keyframe extractor. This is a
read-only analysis step: it never runs or modifies retrieval, the embedding
pipeline, the scorer, or evaluation.

Source: `aic2026.keyframe_analysis` (`run_keyframe_analysis`,
`load_keyframes`, `analyze_keyframe_statistics`, `analyze_gt_coverage`,
`build_error_report`, `analyze_clip_comparison`, `format_report`,
`write_report`). The CLI is `scripts/analyze_keyframes.py`.

### Inputs

- **Keyframe set:** a canonical artifact `frames.ndjson` listing the current
  keyframes with their original-video `frame_index`
  (e.g. `data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson`).
- **Evaluation set:** `data/eval/eval_set.csv` (KIS queries).
- **CLIP baseline:** a `predict-kis` prediction artifact
  (e.g. `data/interim/predictions/kis-l14-laion2b-full-en-top1000.json`).

### Report sections

1. **Keyframe statistics** — total videos, total keyframes (unique original
   frames), raw `frames.ndjson` record count and duplicate-record count, mean
   unique keyframes per video, and the per-video distribution (min/max,
   p50/p90/p95). When a `frames.ndjson` records the same original frame more
   than once, the frame is counted once for coverage and per-video statistics;
   the raw record and duplicate counts are reported so the statistics reconcile
   with the artifact row count.
2. **GT frame coverage** — for each query, whether a keyframe falls inside the
   GT frame interval (exact hit) and the minimum absolute frame distance from
   any keyframe of the GT video to the interval (relaxed hit). Summaries report
   the exact-hit count/percentage and relaxed-hit percentages at thresholds
   `<=10`, `<=30`, `<=60`, `<=150`, `<=300` frames (configurable). Queries
   whose GT video has no keyframes have no defined distance.
3. **Per-query error report** — every miss (no exact hit), with query id, GT
   video, GT range, closest keyframe `frame_index`, distance, and number of
   keyframes in the GT video, sorted worst misses first (no-keyframe videos
   first, then by distance descending).
4. **CLIP baseline comparison** — for each query, whether the GT video / GT
   frame appears in the CLIP top-k predictions and whether missing keyframe
   coverage can explain a frame failure. Exclusive per-query categories:

   - `GT event covered by keyframes` — a keyframe lies inside the GT interval,
     so the correct frame is extractable and any CLIP failure is a ranking
     problem, not a coverage problem.
   - `GT event not covered by keyframes` — the GT interval has no keyframe.
   - `GT video retrieved but frame coverage issue` — the most specific case:
     CLIP retrieved the GT video but not a GT frame, and the GT event is not
     covered by a keyframe, so the coverage gap can explain the frame failure.

### Python API

```python
from aic2026.keyframe_analysis import run_keyframe_analysis, write_report

report = run_keyframe_analysis(
    evaluation_set_path="data/eval/eval_set.csv",
    frames_path="data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson",
    clip_predictions_path="data/interim/predictions/kis-l14-laion2b-full-en-top1000.json",
    relaxed_thresholds=(10, 30, 60, 150, 300),
    prediction_top_k=1000,
)
write_report("data/analysis/keyframes/current_keyframe_analysis.json", report)
```

`run_keyframe_analysis` raises `FileNotFoundError` when an input path is absent
and `ValueError` for invalid queries, thresholds, or prediction payloads.

### CLI

```bash
python scripts/analyze_keyframes.py \
  --eval-csv data/eval/eval_set.csv \
  --frames data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson \
  --clip-predictions data/interim/predictions/kis-l14-laion2b-full-en-top1000.json \
  --output data/analysis/keyframes/current_keyframe_analysis.json
```

Available flags: `--eval-csv`, `--frames`, `--clip-predictions`,
`--relaxed-thresholds` (comma-separated distances, default
`10,30,60,150,300`), `--top-k` (default 1000), and `--output` (default
`data/analysis/keyframes/current_keyframe_analysis.json`). The human-readable
report is printed to stdout and the full report is written as structured JSON
(atomic write) to `--output`.

## 16. Resumable, hardware-aware shard embedding

Builds a bounded embedding "shard" from a canonical artifact ``frames.ndjson``
using a reusable :class:`ImageEmbedder`, designed so the full keyframe set can
later be embedded in parallel across multiple laptops and merged into a full
embedding artifact. It is distinct from the in-repo dataset embedding runner
(Section 4): it reads a ``frames.ndjson`` directly, is device-aware, and
records a shard manifest.

Only the first ``--max-frames`` records of ``frames.ndjson`` are embedded (or
all when omitted), so the CLI doubles as a tiny-subset validation tool without
running the full dataset.

### Interface

`aic2026.image_embedder` defines the reusable interface and hardware handling:

- `ImageEmbedder` — protocol with `encode_images(batch_images) -> (B, D)`
  array, plus `model_name`, `embedding_dim`, `device`, and `embedding_space`.
- `resolve_device(requested, ...)` — auto-selection priority **CUDA > MPS >
  CPU**; accepts `auto|cuda|mps|cpu` and raises on an unavailable requested
  accelerator.
- `create_image_embedder(model, device, precision, config)` — builds an
  embedder from a spec `[backend:]model[:pretrained]` (default backend
  `openclip`). Register new backends (Jina CLIP, SigLIP, MobileCLIP, ...) with
  `@register_embedder("name")`.
- `OpenClipImageEmbedder` (in `aic2026.model_backends.openclip`) adapts the
  existing OpenCLIP encoder to this interface.

`aic2026.embedding_shard.embed_shard(...)` performs the resumable build and
returns a `ShardManifest`.

### How to run

```bash
python scripts/embed_shard.py \
  --frames data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson \
  --output-dir data/test_embedding/shard1 \
  --model openclip:ViT-B-32:laion2b_s34b_b79k \
  --max-frames 100 \
  --device auto \
  --batch-size 32
```

Arguments: `--frames` (canonical frames.ndjson), `--output-dir`, `--model`
(default: the config's OpenCLIP model), `--device auto|cuda|mps|cpu` (default
`auto`), `--batch-size` (default 32), `--max-frames` (testing/subset),
`--worker-id` (default 0), `--keyframes-dir` (default: config `data.raw_dir`),
`--precision fp32|fp16`, and `--progress auto|on|off` (default `auto`). The CLI
prints the resolved device and model identity before embedding.

### Progress reporting

While embedding, the CLI reports throttled progress (at most once every few
seconds or per batch, never per frame) with the current/elapsed frame counts,
percentage, a progress bar, elapsed time, throughput (frames/s), and the
estimated time remaining, then a stage-timing summary:

```text
[=======>........] 42.5% | 75200/177321 frames  elapsed=01:35:20 speed=13.1 frames/s ETA=02:12:40

Stage timing:
load_frames: 2.1s
load_model: 45.3s
encode: 03:21:10
write_outputs: 8.4s
```

`--progress` behavior:

- `auto` (default): single-line interactive updates on a terminal; throttled
  `INFO` log lines (visible with `-v`) when output is piped to a file or log.
- `on`: force interactive updates even when not attached to a terminal.
- `off`: disable all progress reporting (no effect on outputs).

Reporting is metadata only: it never changes embeddings, frames, checkpoints,
or manifests. The reusable logic lives in `aic2026.progress`
(`ProgressReporter`, `StageTimer`, `format_duration`, `estimate_eta`).

### Output contract

Each shard directory contains:

```text
checkpoint.json   resume state (see below)
embeddings.npy    dense (N, D) matrix; row i <-> frames[i]
frames.ndjson     the shard's keyframe records, in row order
manifest.json     shard metadata (written only on successful completion)
```

`embeddings[i]` always corresponds exactly to `frames.ndjson[i]`. The manifest
records `model`, `embedding_dim`, `dtype`, `device`, `num_embeddings`,
`num_videos`, `source_frames`, the full `embedding_space`, and `created_at`.

### Resume behavior

Progress is checkpointed atomically after each batch into `checkpoint.json`,
recording `completed_frame_count`, `completed_video_ids`, `model`, `device`,
`timestamp`, `total_frames`, `worker_id`, and `source_frames`.

- An interrupted run leaves no `manifest.json` (partial outputs are never
  exposed as valid); re-running the same command resumes from the checkpoint
  and skips already-embedded frames.
- A resumed run must match the original `model`, `device`, `worker_id`,
  `source_frames`, and `total_frames`; otherwise it fails with an explicit
  error instead of silently producing a mixed artifact.
- Embeddings are stored via memory-mapped files, and the checkpoint and
  manifest are written atomically, so a crash does not corrupt partial output.

For a deterministic stub-embedder the resume output is byte-identical to a
clean run; for real models only batch-boundary float noise (~1e-8) may differ.

### Distributed usage example

For a laptop cluster, use the deterministic split/embed/merge workflow in
Section 18: split the keyframe list into balanced worker inputs, embed one
worker per machine, then merge the shards back into a single artifact.

Each worker's shard is independently resumable, and a worker may resume on a
different machine (device changes are allowed). The merged artifact records
the shared `embedding_space` identity before building an index.

## 17. Deduplicated keyframe artifact

The BTC-provided `frames.ndjson` may contain duplicate records: the same
`(video_id, frame_index)` key (the 0-based position of the keyframe in the
original encoded video) can appear more than once. For example the first
5000 records of the raw artifact contain duplicates such as
`('L21_V006', 0)`, `('L21_V007', 0)`, ... (about 614 duplicate records across
the full 177,321-record artifact).

The distributed splitter (Section 18) intentionally rejects duplicate
keyframes: splitting duplicates into different workers would make the merge
reject the artifact after wasting embedding time. Deduplication is therefore a
separate preprocessing step that produces a distinct, reproducible artifact,
and downstream experiments should reference the deduplicated artifact rather
than the raw BTC one.

`scripts/deduplicate_keyframes.py` implements Phase 0 (logic in
`aic2026.keyframe_dedup`, `deduplicate_keyframes`):

```bash
python scripts/deduplicate_keyframes.py \
  --input data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson \
  --output data/interim/keyframes/btc-v1-dedup/frames.ndjson \
  --report data/interim/keyframes/btc-v1-dedup/dedup_report.json
```

Behavior:

- Key identity is `(video_id, frame_index)`.
- **Keep-first strategy**: the first occurrence of each key is kept; later
  occurrences are dropped. `strategy` in the report is always `"keep_first"`.
- The original ordering of the remaining records is preserved.
- The input artifact is never modified.
- The output is written atomically (same-directory temp file + rename), so a
  failed or interrupted run never leaves a partial artifact that looks valid;
  on any error no output or report is produced.
- When `--report` is omitted, the report is written as `dedup_report.json`
  next to `--output`.

The JSON report (`DedupReport`, schema version 1.0) records `source_path`,
`output_path`, `original_records`, `unique_records`, `removed_duplicates`,
`duplicate_examples` (the first duplicate keys, first-seen order), and
`strategy`:

```text
Original records:   177321
Unique records:     176707
Removed duplicates: 614
Strategy:           keep_first
Output:             data/interim/keyframes/btc-v1-dedup/frames.ndjson
Report:             data/interim/keyframes/btc-v1-dedup/dedup_report.json
```

After deduplication the output is a canonical `frames.ndjson` that the
splitter accepts. Experiments that embed the full keyframe set must start from
this deduplicated artifact so every `(video_id, frame_index)` appears exactly
once in the final embedding artifact.

## 18. Distributed shard workflow (split -> embed -> merge -> index)

Section 16 built a single resumable shard. For a laptop cluster, split the
full keyframe set into balanced, deterministic worker inputs, embed one worker
per machine, merge the completed shards back into one artifact, then build the
index. All three steps are reproducible and resumable.

```text
1. split      scripts/split_embedding_shards.py   (once, on any machine)
2. distribute worker_*/frames.ndjson to the workers
3. embed      scripts/embed_shard.py              (one worker per machine)
4. merge      scripts/merge_embedding_shards.py   (once, on any machine)
5. index      build index from the merged artifact (Section 4)
```

### Step 1: split (deterministic and balanced)

```bash
python scripts/split_embedding_shards.py \
  --frames data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson \
  --num-workers 5 \
  --output-dir data/splits/
```

Produces:

```text
data/splits/
    split_manifest.json
    worker_000/frames.ndjson
    ...
    worker_004/frames.ndjson
```

Splitting is workload-based, not a plain row cut: each video is kept intact and
assigned whole to the least-loaded worker, so frame counts stay balanced while
videos are never split across machines. It is deterministic (same input +
same `--num-workers` => identical splits), preserves frame order inside each
worker, and never duplicates or drops frames. `split_manifest.json` records
`source_frames`, `num_workers`, and per-worker `worker_id`, `num_frames`,
`num_videos`, `video_ids`. Re-running into an existing split fails unless
`--overwrite` is given.

**Duplicate keyframes are rejected.** Before assigning frames, the splitter
checks for duplicate keyframes keyed by `(video_id, frame_index)` (the 0-based
position in the original video). If the input contains duplicates, it fails
fast with an error reporting the total duplicate count and a few example keys,
e.g.:

```text
error: input frames file ... contains 7 duplicate keyframes (keyed by
(video_id, frame_index)); duplicate keys e.g. ('L21_V006', 0),
('L21_V007', 0), ... Splitting would spread duplicates across workers and the
merge would reject them; create a deduplicated frames artifact first ...
```

The splitter deliberately does **not** deduplicate: a split must preserve the
experiment input exactly, and duplicates distributed across workers would later
make the merge reject the artifact after wasting embedding time. If your
source `frames.ndjson` contains duplicate keyframes, create a deduplicated
frames artifact first (keeping one record per `(video_id, frame_index)`, without
modifying the original data) and split that instead.

### Steps 2-3: distribute and embed (one worker per machine)

Copy each `worker_<id>/frames.ndjson` to its machine (or run against a shared
filesystem). Embed each worker input into a shard directory:

```bash
# machine A (worker 0, GPU):
python scripts/embed_shard.py --frames worker_000/frames.ndjson \
  --output-dir shards/worker_000 --worker-id 0 \
  --model openclip:ViT-L-14:laion2b_s32b_b82k --device auto

# machine B (worker 1, CPU-only):
python scripts/embed_shard.py --frames worker_001/frames.ndjson \
  --output-dir shards/worker_001 --worker-id 1 \
  --model openclip:ViT-L-14:laion2b_s32b_b82k --device cpu
```

Each shard is independently resumable. Resuming is strict about the model, the
embedding space, the source frames, the shard size, and the worker id; a
changed device or precision is allowed and logged as a warning, so a worker can
be resumed on a different machine (e.g. a GPU shard continued on CPU). The
`embedding_space` identity must be identical across all workers because it is
part of the merge validation.

### Step 4: merge

```bash
python scripts/merge_embedding_shards.py \
  --shards shards/worker_000 shards/worker_001 shards/worker_002 \
  --output-dir data/interim/embeddings/merged/
```

Produces `embeddings.npy`, `frames.ndjson`, and `manifest.json`
(`model`, `embedding_dim`, `total_embeddings`, `source_shards`,
`embedding_space`, `created_at`). The merge:

- rejects incomplete shards (no `manifest.json`);
- rejects mixed models, dimensions, or embedding spaces;
- rejects any duplicate keyframe across shards;
- concatenates in deterministic worker_id order (directory name tie-break);
- guarantees `embeddings[i]` <-> `frames.ndjson[i]`.

Re-running into an existing merged output fails unless `--overwrite` is given.

### Step 5: build the index

Feed the merged artifact (Section 4 conventions) into the index builder.
Because every worker shard recorded the same `embedding_space`, the merged
matrix and its `embedding_space` identity are directly consumable by the
existing embedding/index pipeline.

### Exact smoke-test (small subset, no full run)

```bash
# take the first 570 records (3 videos) of the real keyframe set
head -570 data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson > /tmp/frames570.ndjson

python scripts/split_embedding_shards.py --frames /tmp/frames570.ndjson \
  --num-workers 3 --output-dir /tmp/splits

python scripts/embed_shard.py --frames /tmp/splits/worker_000/frames.ndjson \
  --output-dir /tmp/shards/worker_000 --worker-id 0 \
  --model openclip:ViT-B-32:laion2b_s34b_b79k --device auto
# ... repeat for worker_001, worker_002 ...

python scripts/merge_embedding_shards.py \
  --shards /tmp/shards/worker_000 /tmp/shards/worker_001 /tmp/shards/worker_002 \
  --output-dir /tmp/merged
# -> Merge complete: 570 embeddings from 3 shards
```

This exercises input discovery, splitting, embedding, and merging end to end on
a tiny subset without running the full 873-video set.

## 19. Video source manifest

The keyframe-extraction workflow starts from a canonical video inventory: a
single `videos.json` listing every `.mp4` under a raw video directory, with
per-video metadata used by the splitter for balanced assignment. The builder
(`scripts/build_video_manifest.py`, core logic in
`aic2026.video_manifest`) is deterministic and independent of the extractor:

```text
videos/
    L21_V001.mp4
    L21_V002.mp4
    ...
```

becomes:

```bash
python scripts/build_video_manifest.py \
  --video-dir data/raw/videos \
  --output data/interim/videos/videos.json
```

```json
{
  "schema_version": "1.0",
  "source_dir": ".../data/raw/videos",
  "created_at": "2026-08-14T20:14:20+00:00",
  "videos": [
    {
      "video_id": "L21_V001",
      "path": ".../data/raw/videos/L21_V001.mp4",
      "file_size_bytes": 15402,
      "duration_seconds": 2.0,
      "fps": 25.0,
      "num_frames": 50
    }
  ]
}
```

Each `video_id` is the filename stem, only `.mp4` files are accepted (other
extensions are ignored), entries are ordered deterministically by `video_id`,
and duplicate ids or empty (zero-byte) files are rejected. When metadata
probing is enabled (default), `ffprobe` supplies `fps`, `duration_seconds`,
and `num_frames` (falling back to `round(duration * fps)` when `nb_frames` is
absent); a probing failure aborts the build rather than producing an invalid
manifest. `--no-metadata` skips probing, leaving those fields `null`. The
manifest is written atomically (same-directory temp file + rename).

Schema compatibility: the serialized format is additive over the existing
`aic2026.keyframe_extraction` video-manifest schema (`schema_version` stays
`1.0`; the extra keys are ignored by existing consumers), so the generated
file is consumed unchanged by:

```text
video split (Section 20)
  -> keyframe extraction (Section 20)
  -> embedding (Section 16)
```

## 20. DAKE-inspired keyframe extraction

The BTC-provided keyframes are a closed data source. To make the pipeline
self-sufficient, this repository ships a lightweight, DAKE-inspired extractor
(derived from our own implementation, now at V2) that derives the canonical
keyframe artifact from raw video files, producing the same `frames.ndjson`
records and `map-keyframes/<VIDEO_ID>.csv` files the embedding/retrieval
pipeline already consumes. It is a drop-in replacement for the BTC keyframe
source: no retrieval, scoring, evaluation, index, or embedding code needs to
change.

```text
1. split    scripts/split_keyframe_extraction_shards.py   (once, any machine)
2. extract  scripts/extract_keyframes.py                  (one worker/machine)
3. merge    scripts/merge_keyframe_shards.py              (once, any machine)
4. embed    scripts/embed_shard.py                        (unchanged, Section 16)
```

### Input: video manifest (`videos.json`)

Extraction consumes a video manifest listing `video_id` + `path` entries
(`VideoManifest` in `aic2026.keyframe_extraction`, schema version 1.0).
`num_frames` is an optional weight used only by the splitter for balancing:

```json
{
  "schema_version": "1.0",
  "videos": [
    {"video_id": "L21_V001", "path": "/data/videos/L21_V001.mp4", "num_frames": 11482},
    {"video_id": "L21_V002", "path": "/data/videos/L21_V002.mp4"}
  ]
}
```

### Step 1: split (deterministic and balanced)

```bash
python scripts/split_keyframe_extraction_shards.py \
  --videos data/interim/videos/videos.json \
  --num-workers 5 \
  --output-dir data/keyframe-splits/
```

Produces `split_manifest.json` and one `worker_<id>/videos.json` per worker.
Each video is kept intact and assigned whole to the least-loaded worker
(greedy longest-processing-time by `num_frames`, falling back to weight 1 when
absent). The split is deterministic (same input + same `--num-workers` =>
identical splits) and never duplicates or drops videos. Re-running into an
existing split fails unless `--overwrite` is given.

### Step 2: extract (resumable)

Each worker runs the extractor over its manifest:

```bash
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_000/videos.json \
  --output-dir data/keyframes/shard_000 \
  --target-keyframes-per-video 60 \
  --frame-reader ffmpeg \
  --progress auto
```

Use `--sampling-ratio 0.02` instead of `--target-keyframes-per-video` to
select a fixed fraction of each video's frames (exactly one of the two is
required). `--frame-reader` selects the frame source: `ffmpeg` (real videos,
decoded to downscaled grayscale) or `synthetic` (deterministic JSON test
descriptors). `--scorer` selects the change-scoring method:

- `frame_difference` (V1): cheap mean-absolute-difference over downscaled
  grayscale frames.
- `jpeg_size` (V2): DAKE-inspired JPEG-compressed-size change. Each frame's
  downscaled grayscale image is JPEG-encoded in memory and the score is the
  relative change in compressed size between consecutive frames (a percentage
  of the larger size). `--jpeg-quality` (default 70, range 1-100) controls the
  encoding. Deterministic, model-free, and GPU-free.

`--min-temporal-distance` (default 30) and `--peak-threshold` (default 8.0)
control selection: candidate peaks are local maxima above the threshold, and
selected keyframes are never closer than the minimum temporal distance.

### Hybrid temporal anchors (V2)

The first and last frames of every video are always kept as anchors. With
`--anchor-interval-seconds`, additional uniform anchors are added every N
seconds (rounded to whole frames using the video fps) between them. Candidate
peaks must also stay at least `--min-temporal-distance` frames away from
anchors, and when `--target-keyframes-per-video` or `--sampling-ratio` is
given the total (anchors + peaks) does not exceed the target (anchors alone
are always kept even if they reach the target). This guarantees content
coverage across long videos without suppressing peaks inside uniform
segments.

The pipeline (`aic2026.keyframe_extraction.extract_keyframes`) writes:

```text
data/keyframes/shard_000/
    frames.ndjson             canonical KeyframeTiming records
    map-keyframes/L21_V001.csv   n,pts_time,fps,frame_idx  (one per video)
    ...
    manifest.json             versioned extraction metadata
    checkpoint.json           resume state (present during processing)
```

Guarantees:

- **Resumable**: `checkpoint.json` records completed videos; an interrupted
  run skips them on resume. A resume with a different configuration is
  rejected (`config_hash`, sha256 of the canonicalized config, must match) so
  artifacts are never mixed.
- **Atomic**: per-video outputs are staged (`.staging/`) and the final
  `frames.ndjson`, `map-keyframes/*.csv`, and `manifest.json` are swapped
  into place only after all videos complete. A crash never leaves a partial
  artifact that looks valid.
- **Validated**: the finished artifact is re-read and checked for duplicate
  `(video_id, frame_index)` keys and complete per-video coverage before it is
  marked done.
- **Deterministic**: same manifest + same config => identical records.
- **Canonical**: `frames.ndjson` lines match `KeyframeTiming.to_dict()` and
  the CSV files match the `parse_keyframe_map` column contract, so the output
  feeds the unchanged split -> embed -> merge -> index workflow (Section 18).

`manifest.json` records the extractor/scorer/frame reader, the effective
parameters, `config_hash`, `source_videos`, counts, and aggregate statistics
(frames scanned, keyframes per video), so every artifact is self-describing.

### Step 3: merge

```bash
python scripts/merge_keyframe_shards.py \
  --shards data/keyframes/shard_000 data/keyframes/shard_001 ... \
  --output-dir data/keyframes/merged
```

Merge (`aic2026.keyframe_extraction_merge.merge_shards`) validates that all
shards share the same extractor, scorer, frame reader, and configuration,
rejects any keyframe or video appearing in more than one shard, and rebuilds
`map-keyframes/*.csv` deterministically from the merged records so the maps
are always consistent with `frames.ndjson`.

### Extensibility

The extractor is deliberately modular (`aic2026.keyframe_extractor`): the
`FrameReader` protocol (how frames are decoded) and the `ChangeScorer`
protocol (how change is measured) are pluggable, and selection is pure and
deterministic. A stronger scorer (e.g. perceptual or model-based) can be added
without touching the pipeline, split, merge, or downstream stages. When you do,
bump `extractor_version` so `config_hash` changes and old artifacts are not
confused with new ones. The `jpeg_size` scorer and `anchor_interval_seconds`
option are V2 additions; `config_hash` covers both, so a resume with a
different scorer, `jpeg_quality`, or anchor interval is rejected.

## 21. Experiment tracking

Now that experiments flow through split -> embed -> merge -> index -> evaluate,
distinct model runs (OpenCLIP, Jina, SigLIP, MobileCLIP, ...) must not be
confused. Each experiment gets a small, self-describing manifest
(`manifest.json`) recording its identity, provenance, and parameters. This is
metadata/documentation infrastructure only: it does not modify retrieval,
scoring, evaluation, embedding, or artifact-schema logic.

The manifest lives in `aic2026.experiment_manifest` (`ExperimentManifest`,
`save_manifest`, `load_manifest`). Required fields are `experiment_id`,
`experiment_type` (`embedding`, `retrieval`, `reranking`, `fusion`, or any
custom string), `created_at`, and `description`. Optional fields cover the
embedding side (`model_name`, `embedding_space`, `embedding_dim`,
`frames_source`, `split_manifest`, `workers`, `hardware`) and the retrieval
side (`index_path`, `prediction_artifact`, `evaluation_set`, `metrics`), plus a
free-form `parameters` dict. JSON output is deterministic (dataclass field
order, sorted nested dicts) and written atomically.

### Recommended layout

```text
experiments/
    openclip-baseline/
        manifest.json
    jina-v1/
        manifest.json
    siglip-v1/
        manifest.json
```

### Example

```bash
python scripts/create_experiment_manifest.py \
  --output experiments/jina-v1/manifest.json \
  --experiment-id jina-v1 \
  --type embedding \
  --model jina-clip-v2 \
  --frames-source btc-keyframes-v1 \
  --description "Jina CLIP embedding experiment" \
  --parameter batch_size=64 \
  --parameter seed=42
```

This writes:

```json
{
  "schema_version": "1.0",
  "experiment_id": "jina-v1",
  "experiment_type": "embedding",
  "created_at": "2026-08-14T...",
  "description": "Jina CLIP embedding experiment",
  "model_name": "jina-clip-v2",
  "frames_source": "btc-keyframes-v1",
  "parameters": {"batch_size": 64, "seed": 42}
}
```

`--parameter` / `--metric` accept repeated `key=value` pairs whose values are
typed (bool, int, float, else string), e.g. `--parameter fp16=true`. For a
distributed run, also record the split and merge inputs:

```bash
python scripts/create_experiment_manifest.py \
  --output experiments/openclip-baseline/manifest.json \
  --experiment-id openclip-baseline \
  --type embedding \
  --model openclip:ViT-L-14:laion2b_s32b_b82k \
  --frames-source data/interim/embeddings/openclip-l14-laion2b-full/frames.ndjson \
  --split-manifest data/splits/split_manifest.json \
  --workers 5 \
  --hardware "3x mps, 2x cpu" \
  --description "OpenCLIP L-14 distributed baseline" \
  --parameter batch_size=64
```

Manifests can also be created and read from Python:

```python
from aic2026.experiment_manifest import load_manifest, new_manifest, save_manifest
m = new_manifest(
    experiment_id="siglip-v1",
    experiment_type="retrieval",
    description="SigLIP retrieval run",
    model_name="siglip2",
    index_path="data/interim/index.npy",
    evaluation_set="data/eval/eval_set.csv",
    metrics={"ndcg": 0.42},
)
save_manifest("experiments/siglip-v1/manifest.json", m)
loaded = load_manifest("experiments/siglip-v1/manifest.json")
```

Keep one manifest per experiment directory, name it `manifest.json`, and treat
`experiment_id` as globally unique so results from different models and runs
can be compared without ambiguity.
