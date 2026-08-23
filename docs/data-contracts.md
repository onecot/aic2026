# AIC 2026 Data Contracts

This document records the contracts of the BTC-provided raw dataset as
derived from the actual files under `data/raw/aic2026_batch1/`. These
contracts are the stable interface that later pipeline stages rely on.

Schema version: 1.0 (derived 2026-08 from batch 1 files).

## Overview

The raw dataset is organised as five per-video collections sharing a
common video identifier format: `L<lane>_V<number>` (e.g. `L21_V001`).

| Collection      | Per-video entry                           | Content                                  |
|-----------------|-------------------------------------------|------------------------------------------|
| `clip-features-32` | `<video_id>.npy`                       | CLIP feature matrix, shape `(N, 512)`    |
| `keyframes`     | `<video_id>/NNN.jpg`                      | keyframe images (3-digit index)          |
| `objects`       | `<video_id>/NNN.json`                     | per-keyframe object detections           |
| `map-keyframes` | `<video_id>.csv`                          | keyframe -> original-frame map           |
| `media-info`    | `<video_id>.json`                         | per-video YouTube metadata               |

`N` is the number of keyframes for that video and varies per video
(observed range: 24..632 across 873 videos).

## Video identifier

- Format: `L<lane>_V<number>`, e.g. `L21_V001`, `L26_V499`.
- Regex (`contracts.py`): `^L\d+_V\d+$`.
- Non-video files (`.DS_Store`, etc.) are not video entries.

## clip-features-32

- One `.npy` file per video.
- Matrix shape `(num_keyframes, embedding_dim)` with
  `embedding_dim == 512` for all observed videos.
- Dtype: `float16`.
- Row order maps to keyframe indices `001..N` in ascending order.
- Contract type: `ClipFeatures` (metadata) in `entities.py`.

## keyframes

- One directory per video containing `<index>.jpg` where `<index>` is a
  three-digit zero-padded keyframe index starting at `001`.
- Keyframe index is 1-based; filename pattern `^(\d{3})\.(jpg|jpeg|png)$`.

## objects

- One directory per video containing `<index>.json` matching the
  keyframe index space exactly (same set of indices as `keyframes/`).
- Each JSON has five parallel lists of equal length (`100` detections
  for all observed frames); keys:
  `detection_scores`, `detection_class_names`, `detection_class_entities`,
  `detection_boxes`, `detection_class_labels`.
- Boxes are lists of four floats in `[0, 1]` (normalised coordinates).
- Contract type: `ObjectDetectionFrame` in `entities.py`; required keys
  and length checks in `validate_object_payload`.

## map-keyframes

- One CSV per video with header `n, pts_time, fps, frame_idx`.
- `n`: 1-based keyframe sequence number (== keyframe index space).
- `pts_time`: presentation timestamp in seconds.
- `fps`: frame rate of the original video (per-video; varies).
- `frame_idx`: 0-based index of the keyframe within the original video.
- Contract type: `KeyframeMapEntry` in `entities.py`; parsed by
  `parse_keyframe_map`.

## media-info

- One JSON per video with top-level keys:
  `author`, `channel_id`, `channel_url`, `description`, `keywords`,
  `length`, `publish_date`, `thumbnail_url`, `title`, `watch_url`.
- `length` is the video duration in seconds.

## Invariants (validated)

- Keyframe index set == object index set == map `n` set (per video).
- `num_keyframes` (files) == CLIP matrix rows == CSV row count.
- `frame_idx` increases monotonically with `n`.
- `frame_idx` is consistent with `pts_time * fps` (verified across all
  rows of the full batch: no mismatch).

## Raw → canonical flow

The raw BTC layout is never consumed directly by downstream stages.
Data moves through three layers:

```text
raw BTC directory/file layout          (data/raw/aic2026_batch1/)
        |
        |  aic2026.entities     parse CSV/JSON/NPY -> typed raw records
        v
raw typed records               (KeyframeMapEntry, ObjectDetectionFrame, ...)
        |
        |  aic2026.converters   validate + map types -> canonical entities
        v
canonical records               (aic2026.canonical)
```

- `aic2026.contracts` defines the raw layout vocabulary (collection
  names, patterns, required keys); it has no I/O.
- `aic2026.entities` parses raw files into typed raw records.
- `aic2026.converters` converts raw records into canonical entities.
- `aic2026.canonical` defines the stable, storage-agnostic entity model
  downstream stages depend on (see entity docs above).

## Dataset loading layer (`aic2026.dataset`)

`Dataset` is the read-only accessor that downstream code should use. It
hides every raw-layout detail (folder names, filename patterns,
zero-padding, CSV/JSON structure, and how CLIP embeddings are stored) and
returns canonical entities from `aic2026.canonical`.

### Loader responsibility

- Discover videos and iterate them as `VideoId` records.
- Provide per-video metadata, timings, and keyframe iteration.
- Stream embeddings and detections lazily (`iter_embeddings`,
  `iter_detections`) so the full dataset is never resident in memory.
- Fail loudly on missing or corrupt records instead of skipping them.
- Never write, rename, copy, or generate files under `data/raw/`, and
  keep no cache/materialized copy of the dataset.

### Public API

```python
Dataset(raw_root: Path | str)  # validate root; FileNotFoundError if absent

raw_root            -> Path                   # the raw dataset root

# discovery
video_ids()         -> list[str]              # sorted, union of collections
iter_video_ids()    -> Iterator[str]
__iter__()          -> Iterator[VideoId]
__len__()           -> int
__contains__(id)    -> bool
has_video(id)       -> bool

# video-level (materialized; small per video)
metadata(id)        -> VideoMetadata
timings(id)         -> list[KeyframeTiming]   # ordered keyframe set per video
keyframe_ordinals(id) -> list[int]
keyframe_image_path(id, ordinal) -> Path      # raw image path (layout hidden)
iter_keyframes(id)  -> Iterator[KeyframeKey]

# lazy streaming
iter_embeddings(id) -> Iterator[ClipEmbedding]        # one row at a time (mmap)
iter_detections(id) -> Iterator[KeyframeDetections]   # one keyframe JSON at a time
```

### Errors

- `DatasetError` — base class for all loader errors.
- `MissingRecordError(DatasetError)` — an expected raw file/directory is
  absent (e.g. missing `media-info/*.json` or an object JSON).
- `CorruptRecordError(DatasetError)` — a raw file exists but is
  malformed or inconsistent (e.g. invalid JSON, non-2-D `.npy`).

Records are never silently skipped. A malformed `video_id` raises
`ValueError` (via the canonical `VideoId` contract).

### What downstream stages are allowed to depend on

- Canonical entities from `aic2026.canonical`.
- The `Dataset` accessor and its error types from `aic2026.dataset`.
- Reusable helpers (e.g. `config.load_config`) where appropriate.
- Raw parsers/converters only when a non-Dataset integration point is
  genuinely required; prefer `Dataset` by default.

Downstream code must not depend on collection names, file patterns,
zero-padding, or CSV/JSON/npy internals from the raw layout.

## Index artifact contract (`aic2026.index_artifact`)

This section defines the on-disk contract for the artifact produced by the
indexing stage. It is the interface between the indexing stage and the
retrieval/evaluation stages, and is intentionally independent of any ANN
algorithm and of the raw BTC directory layout.

### Purpose and flow

```text
Dataset (canonical)  --index builder (future)-->  index artifact files
    ----------------------------------------------------------->
                 retrieval / evaluation read ONLY the artifact
```

Retrieval and evaluation must consume the index artifact, not the raw
dataset, so:

- `video_id` and `keyframe_ordinal` (keyframe identity) come from the
  artifact's frame records;
- the original frame identity (`frame_index`, `pts_time`, `fps`) comes
  from the same frame records;
- embedding vectors come from the artifact's matrix;
- score computation uses the artifact's stored norms.

### Storage location

Artifacts live under the configured indexes root, one subdirectory per
artifact:

```text
<data.indexes_dir>/<artifact_id>/
    manifest.json        # versioned, self-describing metadata
    frames.ndjson        # one canonical KeyframeTiming per line (row p)
    embeddings.npy       # dense (N, D) float16 matrix; row p <-> frames[p]
    norms.npy            # (N,) float32 L2 norms;      row p <-> frames[p]
    index.bin            # RESERVED slot for ANN payload (not produced yet)
```

`data.indexes_dir` defaults to `data/interim/indexes` (set in
`configs/default.yaml`). The artifact files themselves are git-ignored
under `data/`.

### File layout and row-order contract

Every row index `p` in `[0, N)` refers to the same indexed keyframe
across all three row files. This single position is the stable lookup
key for retrieval:

- `frames.ndjson[p]` → canonical `KeyframeTiming`; provides
  `keyframe.video_id` + `keyframe.keyframe_ordinal` (identity) and
  `frame_index` + `pts_time` + `fps` (original frame mapping), so a
  retrieval result can be traced back to the original frame.
- `embeddings.npy[p]` → the embedding vector row for that keyframe.
- `norms.npy[p]` → the L2 norm of that row, enabling cosine score
  computation without reading the raw dataset.

### Manifest and embedding identity

`IndexManifest` is a frozen dataclass with `to_dict`/`from_dict` (JSON-
safe). `INDEX_SCHEMA_VERSION = "1.1"` adds `embedding_space`, so every new
artifact records the identity required to validate a query before cosine
comparison:

- `schema_version` — current artifacts use `"1.1"`.
- `artifact_id` — unique run identity for the artifact.
- `score_metric` — supported similarity metric (currently `cosine`).
- `embedding_dim` (`D`) and `embedding_dtype` — shape/type of each stored
  vector.
- `embedding_space` — backend, model identifier, optional checkpoint,
  stable embedding-space identifier, dimension, output dtype, normalization
  convention, preprocessing/tokenization revision, and optional model
  implementation version.
- `num_rows` (`N`) and `num_videos` — sizes of the indexed set.
- `created_at` — ISO-8601 timestamp (run identity).
- `source_dataset_id` — identifier of the source dataset (e.g.
  `aic2026_batch1`), storage-layout independent.
- `canonical_schema_version` — canonical schema the `frames.ndjson`
  records conform to.

The model-independent `aic2026.embeddings` boundary defines
`EmbeddingSpace`, `Embedding` (with `kind` `image` or `text`), and the
`EmbeddingEncoder` protocol (`encode_image`, `encode_text`). It imports no
model backend libraries. A query embedding may be compared with an index only
when its complete `EmbeddingSpace` exactly equals the manifest's space;
matching dimension alone is insufficient.

For practical compatibility, a version 1.0 manifest without
`embedding_space` is readable as an explicit `legacy/unknown` space derived
from its dimension and dtype. Raw vectors are accepted only for this legacy
case. Version 1.1 model-identified indexes require an `Embedding` query and
fail clearly on a different model/space.

### Serialization helpers

- `write_manifest(path, manifest)` / `read_manifest(path)` — JSON.
- `write_frames(path, frames)` / `iter_frames(path)` — NDJSON of
  `KeyframeTiming` records in row order.
- `validate_index_files(manifest, artifact_dir)` — checks that the four
  required files exist and that their shapes/counts agree with the
  manifest (rows, dimensions, norms length).

### Errors

- `ValueError` — unsupported schema version, unsupported score metric, or
  otherwise invalid manifest fields.
- `json.JSONDecodeError` / `FileNotFoundError` — corrupt/missing files are
  surfaced, never silently skipped.

### Versioning policy

`INDEX_SCHEMA_VERSION` uses the same convention as the canonical schema:
bump the minor number for additive/backward-compatible changes and the
major number for breaking changes. `read_manifest` and
`IndexManifest.from_dict` accept versions 1.0 and 1.1 during the 1.1
migration; unknown versions are rejected.

### What retrieval/evaluation may depend on

- `IndexManifest` and the file-layout constants from `aic2026.index_artifact`.
- The four artifact files described above.
- Canonical `KeyframeTiming` records (via canonical schema).

Retrieval/evaluation must not depend on the raw BTC collection names,
filenames, zero-padding, or npy layout. Currently the artifact's
`embeddings.npy` is the lookup source; `index.bin` is reserved so a future
ANN payload can replace or supplement it without changing the row-order or
identity contract.

## Image embedding artifact (`aic2026.embedding_artifact`, `aic2026.assets.embedding_shard`)

The image embedding stage converts canonical dataset keyframes into
model-specific embedding vectors produced by an injected image encoder. It
sits between the `Dataset` loading layer and the index stage:

```text
Dataset (canonical) --embedding runner--> embedding artifact files
     ------------------------------------------------------------>
                       index builder reads the embedding artifact rows
     ------------------------------------------------------------>
                       retrieval / evaluation read only the index artifact
```

The runner is model/backend-agnostic: it accepts any object satisfying the
`EmbeddingEncoder` protocol from `aic2026.embeddings` (`embedding_space`,
`encode_image`, `encode_text`), and uses `encode_image_batch` when the
encoder exposes it (falling back to per-image `encode_image`). Model or
backend-specific logic (preprocessing, device, normalization) stays inside
the encoder, never in the runner.

### Public API

```python
from aic2026.assets.embedding_shard import embed_shard

manifest = embed_shard(
    frames_path,      # canonical frames.ndjson input shard
    artifact_dir,     # output directory (created if absent)
    encoder,          # EmbeddingEncoder / ImageEmbedder implementation
    batch_size=16,    # keyframes per model invocation / checkpoint flush
    resume=True,      # continue or validate a partial/complete build
    stop_after_rows=None,  # testing hook: no final manifest is written
    video_ids=None,   # optional subset; videos stay in dataset order
    image_loader=None,    # callable path -> backend-native image (default Pillow)
) -> EmbeddingManifest
```

Keyframe images are resolved through `Dataset.keyframe_image_path(id,
ordinal)` and loaded by an injected `image_loader`, so neither the runner
nor the encoder depends on the raw `keyframes/` filenames or folder layout.
Videos are always processed in the dataset's sorted `video_ids()` order and
keyframes in the canonical `timings()` order, making every build
deterministic.

### Artifact layout and row-order contract

Artifacts live under one directory per `artifact_id`:

```text
<artifact_dir>/<artifact_id>/
    manifest.json        # versioned, self-describing metadata
    embeddings.npy       # dense (N, D) matrix; row i <-> frames[i]
    frames.ndjson        # one canonical KeyframeTiming per line (row i)
```

Every row index `i` in `[0, N)` refers to the same keyframe across
`embeddings.npy[i]` and `frames.ndjson[i]`. Each frame record preserves
`video_id`, `keyframe_ordinal`, `frame_index`, `pts_time`, and `fps`, so an
embedding can always be mapped back to its original frame. The matrix is
written through a memory-mapped file and is efficiently loadable with
`np.load(path, mmap_mode='r')`.

`EmbeddingManifest` is a frozen dataclass with `to_dict`/`from_dict`
(JSON-safe). `EMBEDDING_ARTIFACT_SCHEMA_VERSION = "1.0"`. The manifest
records at minimum:

- `schema_version` — `"1.0"`.
- `artifact_id` — unique run identity.
- `embedding_space` — full model/backend identity from the encoder
  (backend, model id, checkpoint, dimension, output dtype, normalization,
  preprocessing), which is what makes the artifact comparable with index
  artifacts from the same model.
- `embedding_dim` (`D`) and `embedding_dtype` — shape/type of each vector.
- `num_rows` (`N`) and `num_videos` — sizes of the encoded set.
- `source_dataset_id` — identifier of the source dataset.
- `canonical_schema_version` — canonical schema the `frames.ndjson`
  records conform to.
- `created_at` — ISO-8601 creation timestamp.

### Build behavior

- Two passes: (1) inspect the dataset for per-video keyframe counts and
  total rows; (2) stream keyframe images through the encoder in batches and
  write rows incrementally. No keyframe set is materialized in memory.
- Resumable via `.checkpoint.json` and `.build_meta.json`. The artifact is
  complete only when `manifest.json` is written atomically; its absence
  means the build is incomplete. A resume never re-encodes completed rows
  and rejects a build whose encoder embedding space (or dimension/dtype)
  differs from the recorded build metadata.
- The final manifest is written only after the artifact passes
  `validate_embedding_artifact(manifest, artifact_dir)`, which checks row
  counts, dimensionality, dtype, and that every frame record parses.
- Building over an already-complete artifact with `resume=True` validates
  and returns the existing manifest without recomputation; `resume=False`
  with an existing final manifest is rejected to avoid silent overwrites.

### Serialization helpers

- `write_manifest(path, manifest)` / `read_manifest(path)` — JSON.
- `write_frames(path, frames)` / `iter_frames(path)` — NDJSON of
  `KeyframeTiming` records in row order.
- `validate_embedding_artifact(manifest, artifact_dir)` — checks the two
  required files and their consistency with the manifest.

### Relationship to the index stage

The index builder consumes the embedding artifact's rows: `embeddings.npy[i]`
provides the vector and `frames.ndjson[i]` the canonical `KeyframeTiming`
identity for indexed keyframe `i`. The index artifact therefore reuses the
same row-order and identity contract as the embedding artifact, so an
embedding build can be indexed directly without re-reading the raw dataset.

## Team evaluation dataset (`aic2026.evaluation`)

`data/eval/eval_set.csv` is a separate, team-created annotation dataset. It
is read-only input: evaluation logic consumes canonical queries from
`aic2026.evaluation`, not CSV dictionaries or raw column names.

### CSV schema

The current file has this exact ordered header:

```text
query_id,task,query_text_vi,query_text_en,gt_video_id,gt_frame_start,
gt_frame_end,gt_answer,gt_events,annotator,note
```

- `query_id` is a unique non-empty identifier.
- `task` is case-insensitively normalized to one of `kis`, `qa`, or `trake`.
- `query_text_vi` is required; `query_text_en` is optional.
- `gt_video_id` is normalized for surrounding CSV whitespace and validated as
  canonical `VideoId` (`L<lane>_V<number>`).
- `gt_frame_start` and `gt_frame_end` are required inclusive, non-negative
  original-video frame indices; `start <= end`.
- `gt_answer` is required only for QA.
- `gt_events` is required only for TRAKE and is retained as one typed,
  non-empty `EventAnnotation` text value.
- `annotator` is required annotation provenance; `note` is optional.

KIS must not contain QA answers or TRAKE events. QA must not contain a TRAKE
event, and TRAKE must not contain a QA answer. These task-specific invariants
make invalid annotations fail at load time rather than in a future evaluator.

### Current observed data

The checked-in team CSV contains 101 valid KIS rows, all with Vietnamese and
English query text, one inclusive ground-truth frame range, and annotation
metadata. `gt_answer` and `gt_events` are blank for every current row. Seven
raw `gt_video_id` cells have a trailing carriage return; the loader strips
surrounding whitespace during canonicalization without modifying the CSV.

No non-empty TRAKE event payload is currently available, so no unobserved
list/delimiter/nested-event syntax is assumed. A future structured event
format must be introduced as an explicit contract version change.

### CSV → canonical flow and public API

```text
data/eval/eval_set.csv
        |
        | RawEvaluationRow (all CSV cells are strings)
        v
load_eval_set(path)
        |
        v
EvaluationSet[EvaluationQuery]
```

```python
from aic2026.evaluation import load_eval_set

evaluation_set = load_eval_set("data/eval/eval_set.csv")
query = evaluation_set.get_query("kis_100")
for query in evaluation_set.iter_queries("kis"):
    pass
```

`EvaluationQuery` carries a validated task, Vietnamese/optional English text,
canonical `VideoId`, `FrameRange`, task-specific `answer` / `event`, and
`AnnotationMetadata`. The loader is deterministic, preserves CSV row order,
and never writes or modifies the evaluation file. It does not score
predictions or invoke retrieval.

## KIS scoring (`aic2026.kis_scoring`)

The KIS scorer consumes canonical `EvaluationQuery` values and explicit
`KisPrediction` values. It does not invoke retrieval, encode a query, or
inspect raw BTC files. A prediction carries only the ranked identity needed
for KIS: `rank`, `video_id`, and original-video `frame_index`.

A candidate is correct exactly when both conditions hold:

```text
candidate.video_id == query.ground_truth_video_id
and
query.ground_truth_frame_range.start <= candidate.frame_index
    <= query.ground_truth_frame_range.end
```

The comparison deliberately uses `frame_index` only. `keyframe_ordinal` and
`pts_time` are distinct canonical identifiers and must not be used for the
organizer's inclusive original-frame ground-truth range.

For each query and cutoff `k` in `{1, 5, 20, 50, 100}`:

```text
R@k = max(correct(candidate) for candidates whose rank <= k)
```

A query therefore receives binary R-Scores. Empty prediction lists and missing
ranks never create a correct result. Fewer than `k` returned candidates are
valid; the maximum is taken over the candidates that exist.

For a complete KIS query set, aggregate each cutoff over all queries, then
compute:

```text
Final Score = mean(R@1, R@5, R@20, R@50, R@100)
```

Public reusable functions are:

```python
from aic2026.kis_scoring import KisPrediction, evaluate_kis

result = evaluate_kis(
    kis_queries,
    {"kis_100": [KisPrediction(rank=1, video_id="L21_V001", frame_index=29292)]},
)
print(result.score_at(1), result.final_score)
```

`evaluate_kis` requires prediction entries for exactly the supplied query ids,
so an incomplete experiment cannot silently produce a partial benchmark.
`KisEvaluationResult` retains per-query scores, aggregate `TopKScore` values,
and `final_score`. QA and TRAKE scoring are intentionally not implemented.

## QA prediction artifact contract (`aic2026.qa`)

The QA task foundation (`src/aic2026/qa/`) produces ranked retrieval evidence
for `task == qa` evaluation rows. It reuses the shared retriever and the
task-independent candidates from `aic2026.retrieval_candidates`
(`VideoCandidate`, `WindowCandidate`) and defines its own typed records in
`aic2026.qa.qa_types`.

### Purpose and flow

```text
data/eval/eval_set.csv (qa rows)
        |
        v
EvaluationQuery (task=qa, answer required)
        |
        v
run_qa_prediction  ->  index retriever (frame candidates)
        |              group / aggregate / rank (retrieval_candidates)
        v
QA artifact directory: manifest.json + predictions.jsonl
        |
        v
qa_submission  ->  provisional BTC submission CSV (isolated)
```

### Artifact layout

One directory produced by `run_qa_prediction`:

- `manifest.json`: one JSON object with the run/experiment metadata below.
- `predictions.jsonl`: one JSON object per processed query, in query order.

Both files are written atomically (temp file then rename); an interrupted run
never exposes a partially written artifact.

### `manifest.json` fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | must equal `QA_PREDICTION_SCHEMA_VERSION = "1.0"` |
| `run_id` / `created_at` | string | run provenance (ISO-8601 UTC with `Z` suffix) |
| `evaluation_set_path` | string | portable query-source identifier (repo-relative path when inside the repository, otherwise the final path component; never an absolute machine path) |
| `evaluation_set_query_count` | int | number of QA rows in the set (`>= 1`) |
| `processed_query_count` | int | number of query results (`== len(query_results)`) |
| `query_text_field` | string | `query_text_vi` or `query_text_en` |
| `top_k` | int | frame retrieval cutoff per query (`>= 1`) |
| `video_aggregation` | string \| null | `max` / `mean` / `mean_top_n`, or `null` for frame-level ranking |
| `video_aggregation_top_n` | int \| null | `mean_top_n` parameter; only valid for that method |
| `window_max_gap` | int \| null | positive temporal-window grouping gap, or `null` |
| `embedding_space_compatible` | bool | whether encoder/index spaces matched |
| `embedding_space_override_used` | bool | override used (implies `embedding_space_compatible == false`) |
| `index_artifact_path` | string | portable index artifact location (same rules as `evaluation_set_path`) |
| `index_manifest` | object | the referenced index artifact's manifest |
| `model_embedding_space` | object | encoder embedding space |
| `model_config` | object \| null | JSON-serializable model configuration (provenance) |
| `selected_query_ids` | array \| null | explicit processed subset, or `null` |

### `predictions.jsonl` per-line schema

Each line is one `QaQueryResult`:

- `query_id`: canonical query identifier.
- `query_text`: the resolved query text that was encoded.
- `answer`: per-query textual answer. Retrieval-only runs write `""`;
  answer generation (`aic2026.qa.answer_generation`, Jina VLM via
  `aic2026.vlm_jina`) fills it before submission.
- `predictions`: ranked frame evidence, each `{rank, video_id, frame_index,
  score}` (1-based strictly increasing ranks; empty when retrieval returned
  nothing).
- `video_candidates`: video-level aggregates, each `{video_id, score, rank,
  frame_count, top_frame_index}`.
- `window_candidates`: optional temporal windows, each `{video_id,
  frame_start, frame_end, score, rank}`; empty when no window grouping was
  configured.

Ranks within each list are strictly increasing. The artifact loader
(`read_qa_artifact`) re-validates the whole artifact on load.

### Provisional BTC submission format

The AIC 2026 BTC QA submission format is not yet confirmed, so the *only*
module that knows the on-disk submission layout is `aic2026.qa.qa_submission`
(provisional `provisional-v1`):

```text
query_id,rank,video_id,frame_index,answer
```

One CSV row per ranked candidate; the per-query answer is repeated on every
row. The formatter raises `ValueError` when any processed query has an empty
answer or no evidence. When the official format is published, only
`qa_submission.py` and its tests change; the artifact contract above is
independent of the submission layout.


---

## KIS prediction artifact contract (`aic2026.kis_prediction_runner`)

One JSON file written atomically. `schema_version` is `"1.0"`.

Top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | must equal `"1.0"` |
| `run_id` / `created_at` | string | run provenance (ISO-8601 UTC `Z`) |
| `evaluation_set_path` | string | portable query-source identifier (repo-relative when possible, else final component) |
| `evaluation_set_query_count` | int | total queries at the source (`>= 1`) |
| `processed_query_count` | int | `== len(query_results)` |
| `query_text_field` | string | `query_text_vi` or `query_text_en` |
| `top_k` | int | retrieval cutoff per query |
| `embedding_space_compatible` | bool | encoder/index space match |
| `embedding_space_override_used` | bool | legacy-index override applied |
| `index_artifact_path` | string | portable index location |
| `index_manifest` | object | referenced index manifest |
| `model_embedding_space` | object | encoder embedding space |
| `model_config` | object \| null | model provenance |
| `selected_query_ids` | array \| null | explicit subset, or `null` |
| `query_results` | array | one result per processed query |

Each `query_results[i]`: `{query_id, query_text,
predictions: [{rank, video_id, frame_index, score}]}` with strictly
increasing ranks. Readers re-validate on load
(`read_kis_prediction_artifact`).

## TRAKE prediction artifact contract (`aic2026.trake`)

One directory: `manifest.json` + `predictions.jsonl`, both atomic;
`schema_version = "1.0"`.

Manifest fields match QA minus QA-specific options, plus:

- `window_max_gap`: positive gap for per-event temporal windows (`null`
  disables windows).
- No `video_aggregation*` or `answer_generation` fields.

Each `predictions.jsonl` line (`TrakeQueryResult`):

```text
{query_id, query_text, events: [{event_id, event_text,
    predictions: [{rank, video_id, keyframe_ordinal, frame_index, pts_time, score}],
    windows: [{video_id, frame_start, frame_end, score, rank}]}]}
```

Events appear in declared order; event predictions are rank-ordered; event
windows are score-ranked. Event lists come from the query text only
(`E<number>: text` lines or `(n) event ...` numbered lists).

## Submission export contract (`aic2026.submission`)

Per-query CSV files named after the query file (`query-<name>-<task>.csv`),
no header, UTF-8, comma-delimited:

| Task | Columns | Limits |
| --- | --- | --- |
| KIS | `video_id,frame_index` | ≤ 100 rows |
| QA | `video_id,frame_index,answer` | answer non-empty, ≤ 100 chars |
| TRAKE | `video_id,frame_E1,...,frame_EN` | frames strictly increasing; complete sequences only |

Packaging produces `submission.zip` containing the whole `submission/`
directory; zip contents are re-inspected and every CSV re-validated after
generation. Query matching is strict and fail-loud:
`btc-<name>-<task>` query-id equality (with normalized-text check), then
exact text, then whitespace-normalized text; ambiguity aborts the build.
The library entry point also accepts typed in-memory results instead of
artifact paths.
