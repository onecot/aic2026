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
   |            frame candidates (RetrievalCandidate)
   v
Candidates     aic2026.retrieval_candidates         grouping / aggregation /
   |            frame / video / window candidates   ranking (task-independent)
   v
KIS Prediction aic2026.kis_prediction_runner.run_kis_prediction  (CLI: aic2026 predict-kis)
   v
Evaluation     aic2026.kis_scoring.evaluate_kis     R@1..R@100 + Final Score
   v
QA Prediction  aic2026.qa.qa_prediction_runner.run_qa_prediction  (CLI: aic2026 predict-qa)
   |            reuses Candidates foundation (frame / video / window evidence)
   v
QA Submission  aic2026.qa.qa_submission.write_qa_submission_csv   (CLI: aic2026 export-qa-submission)
                provisional BTC CSV format; QA scoring not implemented
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
`aic2026 predict-qa`, `aic2026 export-qa-submission`). The embedding and
index stages are Python-API only and are composed in scripts. Asset-chain
CLIs live under `scripts/` (e.g. `scripts/embed_shard.py`,
`scripts/split_embedding_shards.py`, `scripts/merge_embedding_shards.py`;
Sections 10-13).

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

- `aic2026.model_backends.jina.JinaClipEncoder` — the active production
  encoder (`jina-clip-v2`); selected automatically by
  `aic2026.image_embedder.create_text_encoder` when `models.jina` is
  present in the config.
- `aic2026.model_backends.openclip.OpenClipEncoder` — optional baseline
  backend. Implements the `EmbeddingEncoder` protocol from
  `aic2026.embeddings`. Requires the optional `openclip` dependency group:
  `pip install -e ".[openclip]"`.
- Any object satisfying the `EmbeddingEncoder` protocol
  (`embedding_space`, `encode_image`, `encode_text`) is a valid encoder;
  `encode_image_batch` / `encode_text_batch` are used when present.

### Active model selection

```python
from aic2026.config import load_config
from aic2026.image_embedder import create_text_encoder

config = load_config()                  # models.jina (jina-clip-v2) is active by default
encoder = create_text_encoder(config)   # JinaClipEncoder for the default config
```

### OpenClipEncoder configuration

```python
from aic2026.config import load_config, OpenClipConfig
from aic2026.model_backends.openclip import OpenClipEncoder

config = load_config()
encoder = OpenClipEncoder(config.openclip)   # optional baseline backend

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

### CLI: `aic2026 build-index`

Indexing a stored embedding artifact is done through the main CLI:

```bash
source .venv/bin/activate
aic2026 build-index \
  --source outputs/embeddings_jina_v2 \
  --output data/interim/indexes/jina-v2-full \
  --artifact-id jina-v2-full \
  --verify
```

Options:

- `--source <dir>` (required): a verified embedding artifact directory. Any of
  the three embedding manifest schemas is accepted — the runner's canonical
  `EmbeddingManifest`, a per-worker `ShardManifest` (e.g. a single Jina worker
  output), or a merged `MergeManifest` (`scripts/merge_embedding_shards.py`).
- `--output <dir>` (required): index artifact directory (created if absent).
- `--artifact-id <id>` (required): unique run identity recorded in the index
  manifest.
- `--batch-flush <n>`: rows between memmap flush and checkpoint write
  (default 1000).
- `--verify`: run `validate_index_files` on the completed artifact and fail
  (exit 1) if it is inconsistent.

The command wraps `EmbeddingArtifactDataset` -> `build_index` and records the
source artifact's `embedding_space` in the index manifest, so a Jina/OpenCLIP
index stays space-identified for downstream `predict-kis` / `predict-qa`.

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

### Retrieval candidate foundation (frame / video / window)

The retrieval core returns *frame-level* candidates (`RetrievalCandidate`).
For video-level or window-level retrieval (future QA and TRAKE tasks), the
task-independent foundation in `aic2026.retrieval_candidates` provides pure
primitives that consume those candidates:

- `group_candidates_by_video(candidates)` — grouping by `video_id`,
  preserving first-appearance video order and per-video input order.
- `aggregate_video_score(frame_scores, method="max", top_n=None)` — video
  aggregation strategies: `"max"`, `"mean"`, and `"mean_top_n"`.
- `build_video_candidates(frame_candidates, method="max", top_n=None)` —
  grouped + aggregated + ranked `VideoCandidate` values.
- `sort_by_score(candidates)` — stable (deterministic) score ranking.
- `group_windows(frame_indices, max_gap)` — temporal window grouping for
  future window-level candidates (`WindowCandidate`).

```python
from aic2026.retrieval_candidates import build_video_candidates, group_candidates_by_video

results = retriever.search(query_embedding, top_k=100)   # frame candidates
groups = group_candidates_by_video(results)              # by video
videos = build_video_candidates(results, method="max")   # VideoCandidate list
```

The foundation never runs retrieval, encodes queries, or applies
task-specific scoring; QA/TRAKE layers compose these primitives.

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

- `--config` — YAML config (defaults to `configs/default.yaml`); supplies
  the active model section (`models.jina` / jina-clip-v2 by default).
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
        model_config=config.model_config,   # provenance only
        top_k=100,
        query_text_field="query_text_en",
        allow_embedding_space_mismatch=False,
        run_id="my-run",
    ),
    create_text_encoder(config),   # active model section (jina-clip-v2 by default)
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

## 10. Resumable, hardware-aware shard embedding

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
`--worker-id` (default 0), `--keyframes-dir` (default: config `data.raw_dir`;
resolves images via the `images/` layout or the legacy `keyframes/` fallback),
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
Section 11: split the keyframe list into balanced worker inputs, embed one
worker per machine, then merge the shards back into a single artifact.

Each worker's shard is independently resumable, and a worker may resume on a
different machine (device changes are allowed). The merged artifact records
the shared `embedding_space` identity before building an index.

## 11. Distributed shard workflow (split -> embed -> merge -> index)

Section 10 built a single resumable shard. For a laptop cluster, split the
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
modifying the original data) and split that instead. The current extraction
pipeline guarantees unique keys, so this situation no longer arises for
artifacts produced by `scripts/extract_keyframes.py`.

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

Verify the merged embedding artifact before feeding it to the index builder
(independent of embedding internals; checks file existence, the manifest and
model metadata, frame records, matrix shape/dtype, and vector health):

```bash
python scripts/verify_embedding.py \
  --output-dir data/interim/embeddings/merged
```

Passing run:

```text
Embedding verification: PASS

Artifact:
  model: openclip:ViT-L-14:laion2b_s32b_b82k
  dim: 768
  dtype: float16

Rows:
  embeddings: 65000
  frames: 65000

Vectors:
  empty: 0
  nan/inf: 0

OK: embedding artifact is valid
```

A failing run prints each failed check as `[section] message` and exits with
code 1 (exit 0 on success). The verifier accepts all three embedding manifest
schemas — the per-shard `ShardManifest` (written by `embed_shard.py`), the
merge-output `MergeManifest`, and the runner's `EmbeddingManifest` — and can
be run on any machine without the model backend or media files. It can
therefore validate any artifact from the pipeline: a Jina/OpenCLIP worker
shard, a merged collection, or a `build_embeddings` output.

### Step 5: build the index

Feed the merged artifact (Section 4 conventions) into the index builder.
Because every worker shard recorded the same `embedding_space`, the merged
matrix and its `embedding_space` identity are directly consumable by the
existing embedding/index pipeline.

```bash
aic2026 build-index \
  --source data/interim/embeddings/merged \
  --output data/interim/indexes/openclip-l14-full \
  --artifact-id openclip-l14-full \
  --verify
```

`--source` accepts any of the three embedding manifest schemas — a merged
`MergeManifest`, a single completed worker `ShardManifest`, or the runner's
canonical `EmbeddingManifest` — so a single Jina/OpenCLIP worker shard can be
indexed directly (Section 5).

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

## 12. Video source manifest

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
      "path": "L21_V001.mp4",
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

Video paths are stored **relative to `source_dir`**, so the manifest is
relocatable: the same `videos.json` can be copied to another machine or shared
with distributed workers. Consumers resolve relative paths against their own
`--video-root` override (falling back to the recorded `source_dir`), so
machine-specific absolute paths never need to be baked into the artifact.

Schema compatibility: the serialized format is additive over the existing
`aic2026.keyframe_extraction` video-manifest schema (`schema_version` stays
`1.0`; the extra keys are ignored by existing consumers), so the generated
file is consumed unchanged by:

```text
video split (Section 13)
  -> keyframe extraction (Section 13)
  -> embedding (Section 10)
```

## 13. DAKE-inspired keyframe extraction

The BTC-provided keyframes are a closed data source. To make the pipeline
self-sufficient, this repository ships a lightweight, DAKE-inspired extractor
(derived from our own implementation, now at V2) that derives the canonical
keyframe artifact from raw video files, producing the same `frames.ndjson`
records and `map-keyframes/<VIDEO_ID>.csv` files the embedding/retrieval
pipeline already consumes. A subsequent materialization stage decodes the
selected frames into our own image collection, so the embedding step no longer
depends on the BTC-provided keyframe images. It is a drop-in replacement for
the BTC keyframe source: no retrieval, scoring, evaluation, index, or embedding
code needs to change.

```text
1. split       scripts/split_keyframe_extraction_shards.py   (once, any machine)
2. extract     scripts/extract_keyframes.py                  (one worker/machine)
3. merge       scripts/merge_keyframe_shards.py              (once, any machine)
4. materialize scripts/materialize_keyframes.py              (once, any machine)
5. embed       scripts/embed_shard.py                        (unchanged, Section 10)
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

Paths may be absolute (legacy manifests) or relative; relative paths are
resolved against the manifest's `source_dir` (when present) or the
`--video-root` override (Section 13, Step 2).

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

### Step 1b: verify the split (before distributing workers)

Validate that a split directory is internally consistent before copying
`worker_<id>` inputs to machines (the source `videos.json` is not required):

```bash
python scripts/verify_video_split.py --split-dir data/keyframe-splits/
```

Prints per-worker video/frame counts and a summary (`Total videos`, `Unique
videos`, `Duplicates`, `Total frames`). Exit code is 0 when valid and 1 when
any check fails. Checks: `split_manifest.json` exists and parses; every
recorded `worker_<id>/videos.json` exists and parses; total videos and frames
match `split_manifest.json`; no `video_id` is duplicated or missing across
workers. Verification logic lives in
`aic2026.keyframe_extraction_split.verify_video_split` and only reads the
split directory.

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

**Dependencies (Step 2 and Step 1b only).** Real-video extraction decodes with
`ffmpeg` and probes with `ffprobe`; both must be installed and on `PATH` before
extraction starts, otherwise the run fails up front (a preflight check runs
before any video is processed). ffmpeg >= 4.4 is required (the extractor uses
`-fps_mode passthrough`). Install via your package manager, e.g.
`brew install ffmpeg` (macOS), `apt install ffmpeg` (Debian/Ubuntu), or
`conda install -c conda-forge ffmpeg`. The splitter, merge step, and the
embedding pipeline do not call ffmpeg.

**Relocatable manifests and `--video-root`.** Video paths in a `videos.json`
may be relative to the manifest's `source_dir`. When a worker runs on a
machine where the videos live somewhere else, point the extractor at the new
location without editing the manifest:

```bash
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_000/videos.json \
  --video-root /mnt/videos \
  --output-dir data/keyframes/shard_000 \
  --target-keyframes-per-video 60 \
  --frame-reader ffmpeg
```

Without `--video-root`, relative paths resolve against the manifest's
`source_dir`; absolute paths (legacy manifests) are used unchanged.

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
anchors.

`--target-keyframes-per-video` is a **hard cap on the total number of
keyframes per video, including anchors**. If the anchors alone would meet or
exceed the target, they are deterministically downsampled to the target
(always preserving frame 0 and the last frame) and no peaks are added;
otherwise the best peaks are kept so the total never exceeds the target. The
result always satisfies `len(selected) <= target` and is deterministic for a
given input. This guarantees content coverage across long videos while keeping
the per-video keyframe budget bounded.

`--sampling-ratio` behaves the same way: it derives a per-video target from
the frame count, and that target is then applied as the same hard total cap.

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
  feeds the unchanged split -> embed -> merge -> index workflow (Section 11).

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

Verify the merged artifact before using it downstream (the verifier detects it
as `merged` via the `source_shards` field and checks the shard list, counts,
uniqueness, ordering, and maps):

```bash
python scripts/verify_keyframe_extraction.py \
  --output-dir data/keyframes/merged
```

### Step 3b: materialize the keyframe images (replaces the BTC keyframe collection)

Extraction and merge write only timing records. Before embedding, the selected
frames must exist as actual image files. `materialize_keyframes.py` decodes
each selected frame from its original mp4 and writes the `adaptive_sbd_v1`
image artifact:

```bash
python scripts/materialize_keyframes.py \
  --videos data/interim/videos/videos.json \
  --frames data/keyframes/merged/frames.ndjson \
  --output-dir data/keyframes-images/merged \
  --video-root /path/to/videos
```

`--video-root` is required whenever the manifest's video paths are relative and
the videos are not at the recorded `source_dir` (same rule as Step 2). This
output replaces the BTC-provided keyframe image collection as the embedding
step's image source.

Output (`data/keyframes-images/merged/`):

- `images/<video_id>/000001.jpg` — one JPEG per record, named by
  `keyframe_ordinal` (zero-padded to 6 digits). The record with ordinal `N`
  maps to `images/<video_id>/<N:06d>.jpg`;
- `frames.ndjson` — a byte copy of the validated input records;
- `manifest.json` — versioned metadata (`artifact_id` is
  `adaptive_sbd_v1`, source inputs, counts, image format, `jpeg_quality`).

The artifact is assembled in a sibling `.staging` directory and moved into
place only after an internal verification passes: every `frames.ndjson` record
has an image, no image is missing, and the image count matches the records. A
partial or failed run never looks like a valid artifact, and re-running the
same command against a complete, matching output is a no-op. `--overwrite`
rebuilds from scratch.

Verify the delivered artifact before using it as the embedding image source
(checks the manifest, canonical frame records, and image coverage):

```bash
python scripts/verify_keyframe_materialization.py \
  --output-dir data/keyframes-images/merged
```

Passing run:

```text
Keyframe materialization verification: PASS

Artifact:
  type: adaptive_sbd_v1

Keyframes:
  records: 148322
  images: 148322
  missing: 0
  unexpected: 0

OK: materialized keyframe artifact is valid
```

A failing run prints each failed check as `[section] message` and exits with
code 1 (exit 0 on success). This mirrors how Step 2/merge results are verified
and can be run on any machine without ffmpeg or media files.

### Adaptive extractor (OmniShotCut shots + per-shot sampling)

The DAKE extractor above applies one fixed budget per video. As an
alternative, the pipeline supports `--extractor omnishotcut_adaptive`, which
detects shots with the OmniShotCut model and samples keyframes per shot:

```bash
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_000/videos.json \
  --output-dir data/keyframes/shard_000 \
  --extractor omnishotcut_adaptive \
  --detector omnishotcut \
  --sampler adaptive_v1 \
  --frame-reader ffmpeg \
  --progress auto
```

For `omnishotcut_adaptive`, `--target-keyframes-per-video` and
`--sampling-ratio` are not allowed (the DAKE budget flags apply only to the
`dake` extractor). The output artifact is byte-compatible with the DAKE one
(`frames.ndjson`, `map-keyframes/*.csv`, `manifest.json`), so downstream
embed/merge/index stages are unchanged.

The adaptive pipeline is split into two pluggable stages:

- **Shot detection** (`aic2026.shot_detection`): the `ShotDetector` protocol
  returns inclusive `Shot(start_frame, end_frame)` ranges for a video.
  - `--detector omnishotcut`: the OmniShotCut model (default). The checkpoint
    (~164MB) is downloaded from HuggingFace on first use, and the model is
    loaded once per run and reused across videos. `--device` selects
    `auto|cpu|mps|cuda`; `--omnishotcut-mode clean_shot|default` and
    `--omnishotcut-overlap` (default 20) control inference.
  - `--detector synthetic`: reads a deterministic JSON descriptor (used by
    tests; no model).
- **Per-shot sampling** (`aic2026.keyframe_sampling`): the `Sampler` protocol
  maps one shot to a list of frame indices.
  - `--sampler adaptive_v1` (default): shots shorter than
    `--short-shot-seconds` (default 5s) keep only their middle frame; shots up
    to `--long-shot-seconds` (default 60s) add a uniform sample every
    `--uniform-interval-seconds` (default 10s); longer shots additionally add
    high-change frames (via the same `candidate_peaks`/`temporal_suppression`
    machinery as DAKE, using `--min-temporal-distance` and `--peak-threshold`).

Change scores for the long-shot high-change path are computed only when at
least one shot is "long", reusing the same `FrameReader`/`ChangeScorer`
components as DAKE. The detector, sampler, extractor name/version, and all
effective parameters are recorded in `manifest.json`, and `config_hash` covers
them so a resume or merge across differing adaptive configurations is
rejected.

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

The adaptive extractor (`--extractor omnishotcut_adaptive`) adds two further
pluggable stages — shot detection (`aic2026.shot_detection`) and per-shot
sampling (`aic2026.keyframe_sampling`) — while keeping the canonical artifact
contract identical, so downstream stages never change. Add a new detector or
sampler by implementing its protocol and registering it in the
`SHOT_DETECTOR_NAMES` / `SAMPLER_NAMES` tuples; `config_hash` covers both so a
resume or merge across different detectors/samplers is rejected.

## 14. Distributed worker guide: extract keyframes and embed

This section is the end-to-end runbook for **one assigned worker** in the
distributed keyframe-extraction + embedding workflow. A new team member should
be able to follow it from start to finish without the coordinator or the
split/merge machinery. It is documentation only; it does not modify any
pipeline code.

### 1. Overview

The full workflow is owned by the coordinator, but each worker executes a
self-contained slice:

```text
video split
    |
    v
worker_N/videos.json
    |
    v
OmniShotCut adaptive keyframe extraction
    |
    v
frames.ndjson + map-keyframes
    |
    v
embedding with OpenCLIP/Jina/etc.
    |
    v
embeddings.npy + manifest
    |
    v
send artifacts back to coordinator
```

Key properties:

- **Each worker processes only its assigned videos.** Your input
  `worker_<id>/videos.json` already contains exactly the videos you are
  responsible for; never add or remove videos manually.
- **Workers are independent.** You do not communicate with other workers, and
  your run does not depend on theirs. If yours fails, only your shard is
  missing and it can be re-run alone.
- **Different devices are allowed.** Workers may run on any supported device:
  Apple Silicon **MPS**, NVIDIA **CUDA**, or plain **CPU**. `--device auto`
  picks CUDA > MPS > CPU in that priority order, so a single command works on
  every machine.

#### Hardware and runtime expectations

- **CPU execution is supported but slow.** OmniShotCut shot detection and
  embedding on CPU are dramatically slower than on an accelerator, so budget
  accordingly if your assigned worker is CPU-only.
- **GPU (CUDA) or MPS is recommended for large shards.** Larger keyframe sets
  benefit most from an accelerator; `--device auto` selects it for you when
  available.
- **Do not run multiple workers on one machine without enough resources.** Each
  worker loads its own model and decodes videos; running several on one machine
  can exhaust memory or CPU before any of them completes. Prefer one worker per
  machine unless the hardware demonstrably fits more.

### 2. Expected folder structure

Before running, the coordinator provides you with a split directory. You only
touch your own worker folder:

```text
project/
├── data/
│   ├── raw/
│   │   ├── aic2026_batch1/
│   │   │   └── videos/                      # raw videos (extraction input)
│   │   │       ├── L21_V001.mp4
│   │   │       └── ...
│   │   │
│   │   └── ...
│   │
│   └── keyframe-splits/
│       ├── split_manifest.json
│       └── worker_000/
│           └── videos.json
│
├── scripts/
└── outputs/
```

Extraction writes timing records only; the keyframe images the embedding step
consumes are produced by the materialization stage (Section 13, Step 3b), not
taken from the BTC raw data.

After you run worker_000, the extraction and embedding commands (Sections 5-6)
produce:

```text
outputs/
├── keyframes_worker_000/
│   ├── frames.ndjson
│   ├── map-keyframes/
│   └── manifest.json
│
└── embeddings_worker_000/
    ├── embeddings.npy
    ├── frames.ndjson
    ├── manifest.json
    └── checkpoint.json
```

Note the checkpoint files' differing lifecycles: `keyframes_worker_000/`
contains no `checkpoint.json` after success (extraction deletes it on
completion), while `embeddings_worker_000/checkpoint.json` is kept by the
embedder even after success (see Sections 6 and 8).

### 3. Worker input

You need three things before starting:

- **`worker_<id>/videos.json`** — your assigned video manifest (video ids +
  paths). It may also record a `source_dir` that the paths are relative to.
- **The corresponding video files** — every video listed in your manifest,
  available on your machine (copied locally or on a shared/mounted
  filesystem).
- **Installed dependencies** — the Python environment plus ffmpeg/ffprobe
  (Section 4).

#### The `video_root` concept

Your `videos.json` stores paths that are usually relative to where the videos
were discovered, e.g.:

```json
{"path": "L21_V001.mp4", ...}
```

Every machine has the videos at a different absolute location, so you pass the
actual location with `--video-root` and the pipeline resolves the relative
paths for you. For example, on a Mac the videos might live under:

```text
/Users/alice/datasets/videos
```

and on Windows under:

```text
D:\datasets\videos
```

Both workers run the same command except for `--video-root`:

```bash
# Mac
--video-root /Users/alice/datasets/videos

# Windows
--video-root D:\datasets\videos
```

Pass `--video-root` to both the extraction and analysis entrypoints whenever
your videos do not sit at the location recorded in the manifest.

If your split directory is available, sanity-check it before extracting:

```bash
python scripts/verify_video_split.py --split-dir data/keyframe-splits
```

This confirms `split_manifest.json` exists, every `worker_*/videos.json`
parses, no video is duplicated or missing across workers, and the totals match
the recorded manifest.

### 4. Environment setup

Clone the repository and create the Python environment, then install ffmpeg.

#### macOS (Apple Silicon)

```bash
brew install ffmpeg

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,openclip]"
```

This single install covers both the extraction and the embedding commands
(`.[dev]` adds pytest; `.[openclip]` adds the model dependencies). See the
notes below for what each group provides and the extra `omnishotcut` install.

With `--device auto`, Apple Silicon machines automatically use the **MPS**
backend (CUDA is detected first, then MPS, then CPU). The CLI prints the
selected device when a model loads, e.g. `Selected device: mps`.

#### Windows (PowerShell)

```powershell
winget install Gyan.FFmpeg

python -m venv .venv

.venv\Scripts\activate

pip install -e ".[dev,openclip]"
```

Notes for all platforms:

- **The `.[openclip]` group is required even for extraction.** The documented
  extraction command uses the OmniShotCut detector, which loads the model
  through torch; installing only the base `.[dev]` group leaves extraction
  failing with `ModuleNotFoundError: No module named 'torch'`. The combined
  install above provides everything for both commands.
- **The `omnishotcut` package is installed separately.** The OmniShotCut
  detector imports the upstream `omnishotcut` library, which is not part of
  this repository's extras. Install it from its repository as the project does:

  ```bash
  pip install "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git"
  ```

  Without it, extraction fails with `ModuleNotFoundError: No module named
  'omnishotcut'` when the detector loads.
- **CUDA users** must install a PyTorch build that matches their CUDA version
  (the `pip install -e ".[openclip]"` wheels are CPU-compatible and may not
  include CUDA). Follow the torch install command printed at
  <https://pytorch.org/get-started/locally/> for your driver, then verify with
  `python -c "import torch; print(torch.cuda.is_available())"`.
- **ffmpeg and ffprobe must be available in `PATH`.** The extraction command
  preflights both and fails fast with a clear error if either is missing
  (Section 9).

### 5. Extraction command

Extract keyframes from your assigned videos:

```bash
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_000/videos.json \
  --video-root /path/to/videos \
  --output-dir outputs/keyframes_worker_000 \
  --extractor omnishotcut_adaptive \
  --detector omnishotcut \
  --sampler adaptive_v1 \
  --device auto \
  --progress on
```

Only `--video-root` differs per machine:

```bash
# Mac example:
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_000/videos.json \
  --video-root /Users/alice/datasets/videos \
  --output-dir outputs/keyframes_worker_000 \
  --extractor omnishotcut_adaptive --detector omnishotcut \
  --sampler adaptive_v1 --device auto --progress on
```

```powershell
# Windows example:
python scripts/extract_keyframes.py `
  --videos data/keyframe-splits/worker_000/videos.json `
  --video-root D:\datasets\videos `
  --output-dir outputs/keyframes_worker_000 `
  --extractor omnishotcut_adaptive --detector omnishotcut `
  --sampler adaptive_v1 --device auto --progress on
```

**Output.** On success you get, in `outputs/keyframes_worker_000/`:

- `frames.ndjson` — one canonical keyframe record per line (video id,
  ordinal, timestamp, fps, frame index);
- `map-keyframes/` — per-video `n, pts_time, fps, frame_idx` maps;
- `manifest.json` — versioned metadata (extractor, detector, sampler, all
  effective parameters, config hash, source videos manifest path, keyframe
  count).

The run is resumable: an interrupted run is restarted with the same command and
skips already-completed videos. On completion, `checkpoint.json` is removed and
the artifact is validated (unique `(video_id, frame_index)` keys, every video
accounted for) before being marked done.

### 6. Embedding command

Embed the extracted keyframes into a resumable shard:

```bash
python scripts/embed_shard.py \
  --frames outputs/keyframes_worker_000/frames.ndjson \
  --output-dir outputs/embeddings_worker_000 \
  --model openclip:ViT-L-14:laion2b_s32b_b82k \
  --worker-id 0 \
  --device auto \
  --progress on
```

Set `--worker-id` to your worker number (0 for `worker_000`). It is recorded in
the shard manifest and used by the coordinator's merge to order shards
deterministically; it must be fixed for a given worker so a resumed run matches.

Each keyframe record in `frames.ndjson` carries a 1-based `keyframe_ordinal`
(its position in the extraction's new selection). The embedder resolves the
image for that record as `<keyframes-root>/images/<video_id>/<NNN:06d>.jpg`
(the materialized `adaptive_sbd_v1` layout, Section 13 Step 3b). When the
`images/` directory is absent, it falls back to the legacy BTC-style
`<keyframes-root>/keyframes/<video_id>/NNN.jpg` layout (trying `.jpg`, then
`.jpeg`, then `.png`), where `NNN` is the record's ordinal. The default
`<keyframes-root>` is the config's `data.raw_dir`
(`data/raw/aic2026_batch1`); override it with
`--keyframes-dir /path/to/root` when the images live elsewhere.

Two consequences of this contract:

- **Materialize before embedding.** `extract_keyframes.py` writes only timing
  records (`frames.ndjson`, `map-keyframes/*.csv`, `manifest.json`); it never
  materializes JPEGs. Run `scripts/materialize_keyframes.py` (Section 13,
  Step 3b) first to produce the `images/<video_id>/<NNN:06d>.jpg` collection
  your `frames.ndjson` records resolve against. The legacy `keyframes/` layout
  is only used as a fallback when no `images/` directory exists.
- **`--keyframes-dir` points at the root that *contains* the collection**, not
  at an `images/` or `keyframes/` folder itself. If the materialized collection
  is at `/data/kf/images/L21_V001/000001.jpg`, pass `--keyframes-dir /data/kf`.

What you get, in `outputs/embeddings_worker_000/`:

- `embeddings.npy` — dense `(N, D)` matrix;
- `frames.ndjson` — the shard's keyframe records, in row order;
- `manifest.json` — model, embedding space, dimension, dtype, device, worker
  id, counts (written only on successful completion);
- `checkpoint.json` — resume state.

Two contracts to rely on:

- **`embeddings.npy` row `i` matches `frames.ndjson` row `i`.** The matrix rows
  always correspond exactly to the record lines, in order.
- **The checkpoint allows resume.** If the run is interrupted, re-running the
  *same* command resumes from `checkpoint.json` and skips already-embedded
  frames. The resume is strict about the model, embedding space, source frames,
  worker id, and total count, so always reuse the exact command from the
  interrupted run. A changed device or precision is allowed (logged as a
  warning), so a shard started on one machine can be resumed on another.

Checkpoint lifecycle: unlike extraction (which deletes its checkpoint on
success), `embed_shard` **keeps** `checkpoint.json` after a successful run. It
is harmless and can stay on disk, but it is not part of the deliverable —
exclude it when uploading (Section 8).

The `--model` spec is `[backend:]model[:pretrained]`; the `openclip` backend
ships with the repo, and other backends (Jina CLIP, SigLIP, MobileCLIP, ...)
can be attached via `register_embedder` (`aic2026.image_embedder`).

### 7. Verification before upload

Before sending anything back to the coordinator, verify both artifacts. All
checks below use plain `python` (your activated virtual environment) and work
on macOS, Linux, and Windows PowerShell.

#### Keyframes (`outputs/keyframes_worker_000/`)

Verify the extraction artifact with the dedicated verifier (mirrors the
`verify_video_split` check you ran on the input):

```bash
python scripts/verify_keyframe_extraction.py \
  --output-dir outputs/keyframes_worker_000 \
  --videos data/keyframe-splits/worker_000/videos.json
```

`--videos` is optional; pass the worker's assigned `videos.json` to also check
coverage (every assigned video present, no unexpected videos, totals match).
The verifier supports both **worker** artifacts (manifest records
`source_videos`) and **merged** artifacts produced by
`scripts/merge_keyframe_shards.py` (manifest records `source_shards`, checked
non-empty). It reports the artifact type, manifest consistency
(extractor/detector/sampler, `num_videos`, `num_keyframes`, schema version),
frame-record validity (valid JSON, canonical fields, no duplicated
`(video_id, frame_index)` keys, deterministic ordering), and per-video
`map-keyframes/*.csv` presence and parseability. Exit code is 0 when valid, 1
otherwise. A complete worker report looks like:

```text
Keyframe extraction verification: PASS

Artifact type: worker

Videos:
  expected: 174
  found: 174

Keyframes:
  total: 65000
  duplicates: 0

Manifest:
  extractor: omnishotcut_adaptive
  detector: omnishotcut
  sampler: adaptive_v1
```

For a merged artifact (e.g. `outputs/keyframes_full/`), run the same command
without `--videos`; `Artifact type: merged` is printed and the manifest's
`source_shards` is validated.

(`verify_keyframe_extraction` lives in
`aic2026.keyframe_extraction_verify`.)

#### Embeddings (`outputs/embeddings_worker_000/`)

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

d = Path("outputs/embeddings_worker_000")
assert (d / "embeddings.npy").is_file(), "embeddings.npy missing"
assert (d / "manifest.json").is_file(), "manifest.json missing"

emb = np.load(d / "embeddings.npy")
n_frames = sum(1 for _ in (d / "frames.ndjson").open())
assert emb.shape[0] == n_frames, f"rows {emb.shape[0]} != frames {n_frames}"
print(f"embeddings OK: {emb.shape} matches {n_frames} frames, manifest present")
PY
```

Notes:

- The heredoc form `python - <<'PY'` works in bash/zsh (macOS, Linux) and in
  PowerShell (which also supports `@'...'@ | python -`). If your shell cannot
  feed stdin, save either block to `verify.py` and run `python verify.py`.
- The keyframe check keys on `(video_id, frame_index)` inside the record's
  `keyframe` object — that is the exact uniqueness contract the extraction
  validates before declaring the artifact complete.
- Both `manifest.json` files are required for the coordinator's merge to accept
  your shards (an embedding shard without `manifest.json` is treated as
  incomplete and rejected).

### 8. What to send back

Send the two artifacts back to the coordinator, bundled as `worker_000_result/`:

```text
worker_000_result/

├── keyframes/
│   ├── frames.ndjson
│   ├── map-keyframes/
│   └── manifest.json
│
└── embeddings/
    ├── embeddings.npy
    ├── frames.ndjson
    └── manifest.json
```

Do **not** include `checkpoint.json` from either directory. Extraction removes
its checkpoint automatically on success; embedding keeps its `checkpoint.json`
on disk but it is resume state only and not a deliverable. Include the folder
name matching your worker id (e.g. `worker_001_result/`) so the merge can order
shards deterministically.

### 9. Troubleshooting

**`ffmpeg` not found**

```text
required media tools not found on PATH: ffmpeg, ffprobe. Install ffmpeg >= 4.4 ...
```

The preflight ran before extraction and stopped the run. Install ffmpeg
(Section 4), confirm `ffmpeg -version` and `ffprobe -version` work in your
shell, and re-run the same command.

**CUDA unavailable**

`--device auto` fell back to a non-CUDA device, or `torch.cuda.is_available()`
is `False`. Your PyTorch build does not match your CUDA driver. Reinstall the
CUDA build for your driver version (Section 4), or explicitly choose a
supported device with `--device mps` (Apple Silicon) / `--device cpu` and
proceed — the outputs are device-independent.

**MPS fallback**

On Apple Silicon, `--device auto` uses MPS automatically. If MPS is
unavailable or misbehaves, `--device cpu` is a safe fallback. The embedding
space identity is unchanged across devices, so a shard embedded on MPS and one
on CPU can still be merged.

**Resume after interruption**

Re-run the exact same extraction or embedding command. Extraction skips
completed videos; embedding resumes from `checkpoint.json` and skips
already-embedded frames. If you changed any parameter, the run refuses to mix
configurations and tells you what differs — reuse the original command rather
than editing it.

**Wrong `--video-root`**

If videos cannot be found, the extractor reports unresolved or missing paths.
Confirm the path resolves: your manifest stores relative paths (e.g.
`L21_V001.mp4`) and `--video-root` must be the directory that contains them.
Verify by listing the directory and comparing with the manifest's recorded
`source_dir`.

### Checklist

**Before running:**

- [ ] pulled latest code
- [ ] ffmpeg installed (ffmpeg and ffprobe on `PATH`)
- [ ] worker videos downloaded
- [ ] `verify_video_split` passed

**After running:**

- [ ] keyframe artifact generated (`frames.ndjson`, `manifest.json`)
- [ ] embedding artifact generated (`embeddings.npy`, `manifest.json`)
- [ ] manifests included in the folder you send back

## 15. QA task foundation

The QA task foundation (`aic2026.qa`) produces ranked retrieval evidence for
QA queries using the same index retriever and candidate foundation as KIS. It
implements retrieval infrastructure only: no reranking, no LLM/VLM answer
generation, and no QA scoring. It reuses `IndexRetriever`/`RetrievalCandidate`
(`aic2026.retrieval`) and the frame/video/window candidate primitives
(`aic2026.retrieval_candidates`) rather than re-implementing grouping,
aggregation, or ranking.

Module layout (`src/aic2026/qa/`):

- `qa_types.py` — typed records and the artifact contract.
- `qa_dataset.py` — QA query selection and query-text resolution.
- `qa_prediction_runner.py` — wires `EvaluationSet -> encoder ->
  IndexRetriever -> candidates -> artifact`.
- `qa_submission.py` — isolated BTC submission export.
- `qa_scoring.py` — placeholder for the future QA scorer.

### Python API

```python
from pathlib import Path
from aic2026.config import load_config
from aic2026.model_backends.openclip import OpenClipEncoder
from aic2026.qa import QaPredictionRunnerConfig, run_qa_prediction

config = load_config()
artifact = run_qa_prediction(
    QaPredictionRunnerConfig(
        evaluation_set_path=Path("data/eval/eval_set.csv"),
        index_artifact_dir=Path("data/interim/indexes/btc-batch1-full"),
        prediction_output_dir=Path("data/interim/predictions/qa"),
        model_config=config.model_config,   # provenance only
        top_k=100,
        query_text_field="query_text_vi",
        video_aggregation=None,      # None = frame-level ranking
        window_max_gap=None,         # optional temporal windows
        allow_embedding_space_mismatch=False,
        run_id="my-qa-run",
    ),
    OpenClipEncoder(config.openclip),
)
print(artifact.manifest.processed_query_count)
```

The runner operates only on `task == qa` rows of the evaluation CSV. It fails
fast (`ValueError`) when the evaluation set has no QA queries. Query text
selection and embedding-space mismatch behavior match the KIS runner
(Section 7).

### Artifact layout

The output is a directory:

- `manifest.json` — run/experiment metadata: `schema_version`, `run_id`,
  `created_at`, `evaluation_set_path`, `evaluation_set_query_count`,
  `processed_query_count`, `query_text_field`, `top_k`, `video_aggregation`,
  `video_aggregation_top_n`, `window_max_gap`, embedding-space diagnostics,
  and index/model provenance.
- `predictions.jsonl` — one `QaQueryResult` per line with `query_id`,
  `query_text`, `answer`, ranked `predictions` (frame evidence), ranked
  `video_candidates` (video-level aggregates), and optional ranked
  `window_candidates`.

The `answer` field is left empty by the retrieval foundation; a future
answer-generation stage fills it before submission. See
`docs/data-contracts.md` for the full schema.

### Experiment controls

- `top_k` — frame retrieval cutoff per query.
- `query_text_field` — `query_text_vi` (default) or `query_text_en`.
- `video_aggregation` — `None` (frame-level ranking) or one of `max`,
  `mean`, `mean_top_n` (rank evidence at video level). `mean_top_n` requires
  `video_aggregation_top_n`.
- `window_max_gap` — positive integer to group each video's frames into
  temporal windows (`WindowCandidate`); `None` disables windows.
- `query_subset_ids` / CLI `--query-id` — restrict to explicit query ids.

### CLI

```bash
aic2026 predict-qa \
  --config configs/experiments/vit-b-32-openai.yaml \
  --evaluation-set data/eval/eval_set.csv \
  --index-artifact data/interim/indexes/btc-batch1-full \
  --output data/interim/predictions/qa \
  --top-k 100 \
  --video-aggregation max \
  --window-max-gap 5 \
  --run-id my-qa-run
```

The query encoder is selected from the active model section of `--config`:
an optional `models.jina` section selects the Jina CLIP backend; otherwise
the OpenCLIP section is used. For Jina CLIP v2 against the `jina-v2-full`
index built from `outputs/embeddings_jina_v2`:

```bash
aic2026 predict-qa \
  --config configs/experiments/jina-clip-v2.yaml \
  --evaluation-set data/eval/eval_set-qa_trake.csv \
  --index-artifact data/interim/indexes/jina-v2-full \
  --output data/interim/qa/qa-jina-real-test \
  --top-k 100 \
  --query-text-field query_text_en \
  --run-id qa-jina-real-test
```

The Jina config must fix `pretrained: jinaai/jina-clip-v2`, `precision: fp32`,
`normalization: l2`, and the `jina-clip-v2-preprocess-v1` /
`jina-clip-v2-tokenizer-v1` revision labels so the encoder reports the exact
`jina-clip:jina-clip-v2:jinaai/jina-clip-v2:fp32:l2:...` space recorded in the
index. Note that compatibility ignores the build-environment `model_version`
(installed Transformers/OpenCLIP patch revision); the space identity is the
model/checkpoint/preprocessing convention, not the runtime patch.

### Submission export (provisional)

```bash
aic2026 export-qa-submission \
  --artifact data/interim/predictions/qa \
  --output data/interim/predictions/qa_submission.csv
```

The AIC 2026 BTC submission format is not yet confirmed for QA, so the
formatter in `qa_submission.py` owns the entire provisional format
(`provisional-v1`: `query_id,rank,video_id,frame_index,answer`, one row per
ranked candidate). When the official format is published, only
`qa_submission.py` and its tests need to change. The formatter refuses to
export when a query has no answer (answers come from a later stage) or no
evidence.

### QA scoring

Not implemented. `aic2026.qa.qa_scoring` documents the intended contract and
always raises `NotImplementedError`; see `docs/data-contracts.md`.

## 16. QA smoke test (synthetic, end-to-end)

A reproducible end-to-end QA pipeline smoke test on a synthetic QA query set.
It validates the whole chain — synthetic input → QA dataset loader →
retrieval → QA prediction artifact → submission export → artifact
verification — without any VQA model. It is a testing/demo layer only; it
does not modify the retrieval core or KIS behavior.

### How to run

```bash
source .venv/bin/activate
python scripts/qa_smoke.py --work-dir data/interim/qa-smoke
```

Optional flags: `--evaluation-set` (default
`data/eval/qa_smoke_eval.csv`), `--top-k` (default 50), `--seed` (default
42), `--report-json <path>` to also write the outcome as JSON.

The synthetic dataset `data/eval/qa_smoke_eval.csv` contains 15 BTC-style QA
queries (events/actions, objects/persons, scenes) with Vietnamese and English
questions, expected-answer placeholders, and GT video/frame metadata.

### Expected output

Running the script prints a report and writes demo artifacts under
`--work-dir`:

- `index/` — a small deterministic synthetic index (6 videos × 10 frames).
- `qa_predictions/` — the QA prediction artifact (`manifest.json` +
  `predictions.jsonl`).
- `qa_predictions_determinism/` — a second identical run used for the
  determinism check.
- `qa_submission.csv` — the provisional BTC submission CSV
  (`query_id,rank,video_id,frame_index,answer`).

Example report:

```text
QA smoke run complete
- evaluation set: .../data/eval/qa_smoke_eval.csv (15 QA queries)
- index artifact: .../index
- prediction artifact: .../qa_predictions
- determinism artifact: .../qa_predictions_determinism
- predictions: 750 across 15 queries
- determinism verified: True
- placeholder answers filled: 15
- submission CSV: .../qa_submission.csv
```

### What is tested

- Synthetic QA dataset loading (15 QA queries, GT video/frame/answer
  present, video ids match the synthetic index).
- QA runner execution on the small index: every query produces `top_k`
  predictions, ranks are `1..top_k`.
- Prediction artifact validation: `video_id` exists, `frame_index` is valid
  for that video, scores are finite, ranks strictly increase, every query has
  predictions.
- Determinism: two runs produce identical query results.
- Artifact serialization round-trip (`read_qa_artifact`).
- Submission export: CSV columns, row count, non-empty answers, and
  rejection of queries without evidence.

The logic lives in `aic2026.qa.qa_smoke` (reusable, tested in
`tests/test_qa_smoke.py`); `scripts/qa_smoke.py` is a thin CLI.

### What is intentionally not tested

- VQA / answer generation: the smoke pipeline fills each submission answer
  with the query's expected answer as a placeholder (`qa_submission` refuses
  to export empty answers, by design). Real answering is a future model stage.
- Retrieval quality (no GT-retrieval assertions — the goal is pipeline
  mechanics, not ranking accuracy).
- The final BTC submission format, which is not yet confirmed (see the
  provisional `provisional-v1` note in `docs/data-contracts.md`).

## 17. OCR generation (PaddleOCR PP-OCRv5)

The OCR stage augments canonical keyframes with extracted text as a separate
retrieval modality, producing a resumable OCR artifact under
`data/interim/ocr/<artifact_id>/`. It reads the canonical keyframe artifact
(`frames.ndjson` + `images/`) and never modifies it. The keyframe artifact,
the CLIP embedding/index pipeline, and existing index artifacts are untouched.

OCR is deterministic text extraction: it never translates. Vietnamese and
English text is kept as-is; `ocr_text_normalized` is a lowercase,
whitespace-normalized copy with duplicated symbols removed (Vietnamese
accents are preserved). Retrieval fusion (OCR-text embedding / BM25 fused
with CLIP) is a separate, future stage.

### Prerequisites

```bash
source .venv/bin/activate
pip install -e '.[ocr]'   # paddleocr>=3.0, paddlepaddle>=3.0, numpy, Pillow
```

For GPU inference on a Modal instance, install the GPU build of PaddlePaddle
per the PaddlePaddle install guide (e.g. `paddlepaddle-gpu`) in addition to
`paddleocr`; the pipeline then resolves `--device cuda` to PaddleOCR's
`gpu:0`. PaddleOCR is imported lazily only when the CLI runs; the core
package and its unit tests do not require the runtime.

### How to run (full keyframe set)

```bash
source .venv/bin/activate
python scripts/generate_ocr.py \
    --keyframes-dir data/keyframes/adaptive_sbd_v1 \
    --output-dir data/interim/ocr/paddleocr-v5 \
    --backend paddleocr \
    --languages vi,en \
    --device cuda \
    --batch-size 16 \
    --resume
```

### Options

- `--backend paddleocr`: OCR backend (default: `paddleocr`).
- `--keyframes-dir <dir>`: canonical keyframe artifact directory
  (default: `data/keyframes/adaptive_sbd_v1`).
- `--output-dir <dir>`: OCR artifact output directory.
- `--model <name>`: PaddleOCR model family (default: `PP-OCRv5_mobile`).
- `--languages vi,en`: comma-separated OCR languages (default: `vi,en`);
  Vietnamese and English are both recognized and kept as-is.
- `--device auto|cuda|cpu|mps`: device selection (auto: CUDA > CPU;
  PaddlePaddle has no MPS/Metal backend, so `mps` is mapped to `cpu`).
- `--batch-size <n>`: keyframe images passed to the OCR engine per call
  (default: 16).
- `--max-records <n>`: only process the first `n` records (small-scale
  validation before the full run).
- `--resume` / `--no-resume`: resume an interrupted build (default resume
  on); an already-complete artifact is validated and returned without
  recompute.
- `--upscale-factor <f>`: preprocess upscale for small text (default `1.0`,
  off; e.g. `2.0`).
- `--contrast <f>`: preprocess contrast enhancement (default `1.0`,
  unchanged).
- `--sharpen`: preprocess sharpening (off by default).
- `--progress auto|on|off`: progress reporting (default auto).
- `-v`: verbose logging.

### Output

The artifact directory contains:

- `manifest.json` — versioned metadata (backend, model, languages, device,
  `batch_size`, record/video/failed counts, source, timestamps);
- `ocr.ndjson` — one `KeyframeOcr` per keyframe, in source row order, each
  with `video_id`, `keyframe_ordinal`, `frame_index`, `ocr_text`,
  `ocr_text_normalized`, `ocr_lines` (text + confidence + bbox per line),
  `status`, and optional `error`;
- `checkpoint.json` — resume state, present only while the build is
  incomplete (deleted on completion).

Row `i` of `ocr.ndjson` always corresponds to line `i` of the source
`frames.ndjson`. A missing/corrupted image yields an `error` record (empty
text) instead of aborting the run, so row alignment is preserved.

### Small-scale validation

Before the full run, validate the workflow on a small subset:

```bash
python scripts/generate_ocr.py \
    --keyframes-dir data/keyframes/adaptive_sbd_v1 \
    --output-dir /tmp/ocr-smoke \
    --max-records 100 \
    --batch-size 16
```

A small successful run verifies input discovery, model loading, output
schema, checkpoint behavior, and downstream compatibility.

### Reproducibility notes

- Input order is deterministic (source file order); PaddleOCR extraction is
  deterministic per image.
- The exact `backend`, `model`, `languages`, and preprocessing options are
  recorded in `manifest.json`; resuming with a changed backend, model,
  languages, source, or record count is rejected.

The pipeline lives in `aic2026.ocr_generation` (reusable, tested in
`tests/test_ocr_generation.py`); `scripts/generate_ocr.py` is a thin CLI.
OCR-text embedding / retrieval fusion is not implemented yet.
