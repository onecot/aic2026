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
`aic2026 objects-analyze`). The embedding and
index stages are Python-API only and are composed in scripts.

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
```
