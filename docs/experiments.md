# AIC 2026 KIS Retrieval Experiments

Tracking record for completed retrieval experiments on the current KIS
project. Each experiment is scored with the **same evaluation procedure** on
the **same evaluation set**, so cross-experiment comparisons are meaningful
within the limits stated at the end of this document.

This document records experiments as they were run; it does not claim
statistical significance or robustness beyond the measured numbers.

## Shared methodology

- **Evaluation set:** `data/eval/eval_set.csv` — 101 KIS queries.
- **Query field:** `query_text_en` (all experiments below).
- **Scoring:** `aic2026.kis_scoring.evaluate_kis` with cutoffs `R@1`, `R@5`,
  `R@20`, `R@50`, `R@100`; **Final score** = mean of the five R@k values.
- **Retrieval:** exact cosine search over the full index
  (`IndexRetriever`), `top_k = 100`.
- **Indexes:** full-dataset, 177,321 rows / 873 videos, except where noted.
- **Environment:** this repository (venv), OpenCLIP optional dependency
  installed; runtime `open_clip` package version recorded in artifacts as
  `3.3.0`. OpenCLIP runs use `device: auto`, `precision: fp32`,
  `normalization: l2`.

### Standard evaluation pipeline

```bash
# 1. (model experiments) build image embeddings, then index them
# 2. rank queries (CLI; model config comes from --config)
aic2026 predict-kis \
  --config <configs/experiments/....yaml> \
  --evaluation-set data/eval/eval_set.csv \
  --index-artifact <index_dir> \
  --output <prediction.json> \
  --top-k 100 \
  --query-text-field query_text_en \
  --run-id <run_id>
# 3. score
python - <<'PY'
from aic2026.evaluation import load_eval_set
from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring
from aic2026.kis_scoring import evaluate_kis, KIS_TOP_K

evaluation_set = load_eval_set("data/eval/eval_set.csv")
predictions = load_kis_predictions_for_scoring("<prediction.json>")
result = evaluate_kis(evaluation_set, predictions, cutoffs=KIS_TOP_K)
print(result.final_score)
PY
```

## Results overview

| Exp | Image embeddings | Text model / checkpoint | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|-----|------------------|-------------------------|---------|---------|---------|---------|---------|---------|
| 0 | BTC provided `clip-features-32` | `ViT-B-32` / `openai` | 0.059406 | 0.138614 | 0.247525 | 0.326733 | 0.425743 | **0.239604** |
| 1 | OpenCLIP `ViT-B-32` / `openai` | `ViT-B-32` / `openai` | 0.059406 | 0.128713 | 0.267327 | 0.326733 | 0.386139 | **0.233663** |
| 2 | OpenCLIP `ViT-L-14` / `openai` | `ViT-L-14` / `openai` | 0.108911 | 0.227723 | 0.356436 | 0.435644 | 0.514851 | **0.328713** |
| 3 | OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` | `ViT-L-14` / `laion2b_s32b_b82k` | 0.108911 | 0.257426 | 0.366337 | 0.455446 | 0.534653 | **0.344554** |

---

## Experiment 0 — BTC provided CLIP features reproduction (reference baseline)

1. **Objective:** Reproduce the reference baseline by indexing the
   BTC-provided CLIP features and scoring OpenCLIP text queries against them.
   Establishes the fixed comparison point used by every later experiment.
2. **Environment / config:** repo `aic2026-v2` + venv; OpenCLIP installed
   (`open_clip` 3.3.0). Text encoder configured by
   `configs/experiments/vit-b-32-openai.yaml`. Index built from the raw
   `Dataset` with `embedding_space=None`, so the index records an explicit
   `legacy/unknown` space.
3. **Image embedding model:** none — BTC-provided `clip-features-32` matrices
   (`float16`, dim 512).
4. **Text embedding model:** OpenCLIP `ViT-B-32`, checkpoint `openai`
   (`fp32`, l2-normalized).
5. **Query field:** `query_text_en`.
6. **Index artifact:** `data/interim/indexes/btc-batch1-full`
   (177,321 rows / 873 videos; `legacy-unknown-d512-float16`). Prediction:
   `data/interim/predictions/kis-openai-en-vs-btc-batch1-full.json`
   (`run_id: kis-run-20260813T100634Z`).
7. **Evaluation command / pipeline:** build index (BTC features,
   `legacy/unknown` space) → `predict-kis` with the `vit-b-32-openai` config
   and `--allow-embedding-space-mismatch` (the model-identified query encoder
   differs from the legacy index; recorded `embedding_space_compatible=False`,
   `embedding_space_override_used=True`) → `evaluate_kis` on the shared eval
   set.
8. **Metrics:**

   | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
   |---------|---------|---------|---------|---------|---------|
   | 0.059406 | 0.138614 | 0.247525 | 0.326733 | 0.425743 | 0.239604 |

9. **Observation:** Reproduces the previously recorded reference baseline
   (Final `0.239604`) exactly, confirming the evaluation procedure. Because
   the query side is model-identified and the index is `legacy/unknown`, the
   run required the explicit mismatch override; this cross-space comparison is
   the reference point, not a model-built pipeline.

---

## Experiment 1 — OpenCLIP ViT-B-32 / openai re-embedding

1. **Objective:** Replace the provided BTC image features with
   self-computed OpenCLIP `ViT-B-32` (`openai`) image embeddings, producing an
   end-to-end model-identified image + text pipeline, and measure the effect
   of the re-embedding on retrieval quality.
2. **Environment / config:** same environment; image and text encoding both
   configured by `configs/experiments/vit-b-32-openai.yaml` (`fp32`, l2).
3. **Image embedding model:** OpenCLIP `ViT-B-32` / `openai` →
   `data/interim/embeddings/openclip-b32-openai-full`
   (177,321 rows / 873 videos, `float32`, dim 512).
4. **Text embedding model:** OpenCLIP `ViT-B-32` / `openai` (same model and
   checkpoint as the image side).
5. **Query field:** `query_text_en`.
6. **Index artifact:** `data/interim/indexes/openclip-b32-openai-full`
   (built from the corresponding embedding artifact;
   `openclip:ViT-B-32:openai:fp32:l2:openclip-default-v1:openclip-default-v1`).
   Prediction: `data/interim/predictions/kis-openclip-b32-openai-full-en.json`
   (`run_id: kis-run-20260813T130217Z`; `embedding_space_compatible=True`).
7. **Evaluation command / pipeline:** `build_embeddings` (image side) →
   `build_index` with the recorded `EmbeddingSpace` → `predict-kis` with the
   matching config/index (compatible spaces, no mismatch override) →
   `evaluate_kis` on the shared eval set.
8. **Metrics:**

   | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
   |---------|---------|---------|---------|---------|---------|
   | 0.059406 | 0.128713 | 0.267327 | 0.326733 | 0.386139 | 0.233663 |

9. **Observation:** Overall close to the baseline (Final `0.233663` vs
   `0.239604`), with identical R@1 (`0.059406`) and R@50 (`0.326733`). The
   re-embedding scored higher at R@20 (`0.267327`) but lower at R@5 and R@100.
   Direction differs by cutoff, so the re-embedded pipeline neither clearly
   outperformed nor underperformed the provided features on this evaluation
   set.

---

## Experiment 2 — OpenCLIP ViT-L-14 / openai

1. **Objective:** Test whether a larger encoder (`ViT-L-14`, 768-dim) improves
   retrieval over the `ViT-B-32` pipeline.
2. **Environment / config:** same environment; image and text encoding
   configured by `configs/experiments/vit-l-14-openai.yaml` (`fp32`, l2).
3. **Image embedding model:** OpenCLIP `ViT-L-14` / `openai` →
   `data/interim/embeddings/openclip-l14-openai-full`
   (177,321 rows / 873 videos, `float32`, dim 768).
4. **Text embedding model:** OpenCLIP `ViT-L-14` / `openai` (same model and
   checkpoint as the image side).
5. **Query field:** `query_text_en`.
6. **Index artifact:** `data/interim/indexes/openclip-l14-openai-full`
   (built from the corresponding embedding artifact;
   `openclip:ViT-L-14:openai:fp32:l2:openclip-default-v1:openclip-default-v1`).
   Prediction: `data/interim/predictions/kis-l14-openai-full-en.json`
   (`run_id: kis-run-20260813T182944Z`; `embedding_space_compatible=True`).
7. **Evaluation command / pipeline:** `build_embeddings` → `build_index` with
   the recorded `EmbeddingSpace` → `predict-kis` with the matching config/index
   → `evaluate_kis` on the shared eval set.
8. **Metrics:**

   | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
   |---------|---------|---------|---------|---------|---------|
   | 0.108911 | 0.227723 | 0.356436 | 0.435644 | 0.514851 | 0.328713 |

9. **Observation:** Higher at every cutoff than both the baseline (Exp 0) and
   the `ViT-B-32`/`openai` re-embedding (Exp 1) on the same evaluation set
   (Final `0.328713` vs `0.239604` / `0.233663`). Note `ViT-L-14` produces a
   different embedding space (768-dim) than `ViT-B-32` (512-dim), so these
   runs are separate pipelines compared only through their scores.

---

## Experiment 3 — OpenCLIP ViT-L-14 / laion2b_s32b_b82k

1. **Objective:** Compare the LAION-2B pretrained `ViT-L-14` checkpoint
   (`laion2b_s32b_b82k`) with the OpenAI-pretrained `ViT-L-14` (Exp 2) to see
   whether checkpoint choice affects KIS retrieval quality.
2. **Environment / config:** same environment; image and text encoding
   configured by `configs/experiments/vit-l-14-laion2b.yaml` (`fp32`, l2).
3. **Image embedding model:** OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` →
   `data/interim/embeddings/openclip-l14-laion2b-full`
   (177,321 rows / 873 videos, `float32`, dim 768).
4. **Text embedding model:** OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` (same
   model and checkpoint as the image side).
5. **Query field:** `query_text_en`.
6. **Index artifact:** `data/interim/indexes/openclip-l14-laion2b-full`
   (built from the corresponding embedding artifact;
   `openclip:ViT-L-14:laion2b_s32b_b82k:fp32:l2:openclip-default-v1:openclip-default-v1`).
   Prediction: `data/interim/predictions/kis-l14-laion2b-full-en.json`
   (`run_id: kis-run-20260814T041506Z`; `embedding_space_compatible=True`).
7. **Evaluation command / pipeline:** `build_embeddings` → `build_index` with
   the recorded `EmbeddingSpace` → `predict-kis` with the matching config/index
   → `evaluate_kis` on the shared eval set.
8. **Metrics:**

   | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
   |---------|---------|---------|---------|---------|---------|
   | 0.108911 | 0.257426 | 0.366337 | 0.455446 | 0.534653 | 0.344554 |

9. **Observation:** Recorded the highest Final score among the completed
   experiments (`0.344554`), with the same R@1 as Exp 2 (`0.108911`) but
   higher R@5, R@20, R@50, and R@100. This reflects one checkpoint swap
   measured on one evaluation set; no claim is made that this generalizes
   beyond the measured runs.

---

## Summary observations

- Across the recorded runs, the larger encoder (`ViT-L-14`, 768-dim) improved
  every cutoff over the `ViT-B-32` (512-dim) runs on the same evaluation set.
- For `ViT-L-14`, the LAION-2B checkpoint (`laion2b_s32b_b82k`, Exp 3)
  recorded the best measured Final score and the best R@5–R@100; R@1 was equal
  to the OpenAI checkpoint (Exp 2) at `0.108911`.
- Re-embedding images with `ViT-B-32`/`openai` (Exp 1) produced a score very
  close to, but mixed in direction against, the provided BTC features (Exp 0)
  across cutoffs.

## Scope and caveats

- Single evaluation set (`data/eval/eval_set.csv`, 101 queries) and a single
  run per configuration; no significance testing or repeated runs.
- Only the `query_text_en` query field was evaluated.
- Final score is the mean of the five R@k cutoffs; comparing Final scores
  alone can hide differing behavior at individual cutoffs.
- `ViT-B-32` (512-dim) and `ViT-L-14` (768-dim) results are from separate
  pipelines in distinct embedding spaces and are compared only via their
  scores on the shared evaluation set.

---

## Temporal Aggregation Experiments

Post-processing study on top of the strongest recorded retrieval pipeline:
whether re-ranking the raw retrieval predictions with temporal aggregation
improves KIS scoring.

**Context**
- **Backbone:** OpenCLIP `ViT-L-14`
- **Checkpoint:** `laion2b_s32b_b82k`
- **Query field:** `query_text_en`
- **Index:** `data/interim/indexes/openclip-l14-laion2b-full`
- **Evaluation:** KIS scoring (`evaluate_kis`, cutoffs R@1/R@5/R@20/R@50/R@100;
  Final = mean of the five R@k values) on the shared eval set
  (`data/eval/eval_set.csv`, 101 queries)

**Pipeline:** `predict-kis` (retrieval, `top_k=1000`) → sliding-window mean
aggregation (`aic2026.temporal_aggregation`) → `evaluate_kis`. The aggregation
groups each query's predictions by `video_id`, sorts frames by `frame_index`,
and replaces every prediction's score with the mean score over frames within
`window_size` frames on each side **of the same video**, then re-ranks by the
aggregated score.

| Config | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline (`top_k=1000`, no aggregation) | 0.108911 | 0.257426 | 0.366337 | 0.455446 | 0.534653 | **0.344554** |
| Mean, `window_size=3` | 0.108911 | 0.257426 | 0.366337 | 0.435644 | 0.524752 | **0.338614** |
| Mean, `window_size=5` | 0.108911 | 0.257426 | 0.356436 | 0.425743 | 0.524752 | **0.334653** |
| Mean, `window_size=10` | 0.108911 | 0.257426 | 0.356436 | 0.425743 | 0.514851 | **0.332673** |

The baseline matches Experiment 3 (same model, checkpoint, and evaluation set);
it is recorded here with `top_k=1000` as the untouched retrieval output that the
aggregation stage consumes. R@1 and R@5 are unchanged across all temporal
configs, and every degradation appears at R@20 and above.

### Conclusion

- Naive sliding-window mean aggregation reduced KIS performance on this
  evaluation set relative to the untouched baseline.
- Larger windows caused larger degradation (Final fell monotonically as
  `window_size` increased from 3 to 10).
- Temporal smoothing may over-reward dense but incorrect frame clusters and
  hurt precise frame localization, which is what KIS scoring rewards.
- Keep the baseline retrieval output unchanged.
- This records one specific aggregation method; it does not claim temporal
  methods are ineffective in general.

---

## Temporal Bonus Reranking Experiments

Post-processing study on the same retrieval pipeline as the temporal
aggregation experiments, testing whether re-ranking the raw retrieval
predictions with a per-frame temporal bonus improves KIS scoring.

**Context**
- **Backbone:** OpenCLIP `ViT-L-14`
- **Checkpoint:** `laion2b_s32b_b82k`
- **Query field:** `query_text_en`
- **Index artifact:** `data/interim/indexes/openclip-l14-laion2b-full`
- **Prediction artifact:** `kis-l14-laion2b-full-en-top1000.json`
- **Evaluation:** KIS scoring (`evaluate_kis`, cutoffs R@1/R@5/R@20/R@50/R@100;
  Final = mean of the five R@k values) on the shared eval set
  (`data/eval/eval_set.csv`, 101 queries)

**Pipeline:** `predict-kis` (retrieval, `top_k=1000`) → temporal bonus
reranking (`aic2026.temporal_reranking`) → `evaluate_kis`. The reranking groups
each query's predictions by `video_id`, sorts frames by `frame_index`, and
boosts every prediction by

```text
new_score = original_score + lambda_weight * temporal_support
```

where `temporal_support` counts how many predictions from the same video are
within `window_size` frames (the current frame itself excluded), then re-ranks
by `new_score` in descending order.

### Baseline (no temporal reranking)

- `top_k`: 1000
- R@1: 0.108911
- R@5: 0.257426
- R@20: 0.366337
- R@50: 0.455446
- R@100: 0.534653
- **Final: 0.344554**

This matches the temporal aggregation baseline and Experiment 3 (same model,
checkpoint, index, and evaluation set), recorded here as the untouched
retrieval output the reranking stage consumes.

### Temporal bonus

Configuration:
- `window_size`: 5

| `lambda_weight` | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|-----------------|---------|---------|---------|---------|---------|---------|
| 0.01 | 0.079208 | 0.227723 | 0.316832 | 0.405941 | 0.504950 | **0.306931** |
| 0.05 | 0.019802 | 0.069307 | 0.178218 | 0.326733 | 0.396040 | **0.198020** |
| 0.1 | 0.000000 | 0.029703 | 0.079208 | 0.217822 | 0.356436 | **0.136634** |

All three temporal bonus settings degraded every cutoff relative to the
baseline, and Final fell as `lambda_weight` increased. R@1 was hit hardest:
from `0.108911` at baseline down to `0.079208` at `lambda_weight=0.01` and
`0.000000` at `lambda_weight=0.1`.

### Conclusion

- Temporal bonus reranking consistently reduced KIS performance on this
  evaluation set relative to the untouched baseline.
- Even a small temporal weight (`lambda_weight=0.01`) degraded the baseline.
- Larger `lambda_weight` values caused severe ranking degradation.
- The current temporal support heuristic likely rewards dense frame clusters
  without enough awareness of exact temporal localization.
- The experiment suggests that raw CLIP similarity ranking is a stronger
  signal for KIS than this naive temporal consistency heuristic.
- This records one specific temporal bonus reranking strategy; it does not
  claim temporal methods in general are ineffective.

---

## Video-level Reranking Experiments

Post-processing study on the same retrieval pipeline as the temporal
experiments, testing whether reranking the raw retrieval predictions at the
video level improves KIS scoring.

**Context**
- **Backbone:** OpenCLIP `ViT-L-14`
- **Checkpoint:** `laion2b_s32b_b82k`
- **Query field:** `query_text_en`
- **Prediction artifact:** `kis-l14-laion2b-full-en-top1000.json`
- **Evaluation:** KIS scoring (`evaluate_kis`, cutoffs R@1/R@5/R@20/R@50/R@100;
  Final = mean of the five R@k values) on the shared eval set
  (`data/eval/eval_set.csv`, 101 queries)
- **Baseline Final: 0.344554** (untouched retrieval, `top_k=1000`)

**Pipeline:** `predict-kis` (retrieval, `top_k=1000`) → video-level reranking
(`aic2026.video_reranking`) → `evaluate_kis`. The reranking groups each query's
predictions by `video_id`, computes a per-video score from its frame candidates
(`"max"` of the frame scores, or the mean of the top `top_n` frame scores),
ranks videos by video score, and reconstructs the frame ranking as videos
ordered by video score with each video's frames ordered by original CLIP score.

| Config | R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline (no reranking, `top_k=1000`) | 0.108911 | 0.257426 | 0.366337 | 0.455446 | 0.534653 | **0.344554** |
| `max`, `top_n=5` | 0.108911 | 0.188119 | 0.277228 | 0.435644 | 0.475248 | **0.297030** |
| `max`, `top_n=10` | 0.108911 | 0.188119 | 0.277228 | 0.435644 | 0.475248 | **0.297030** |
| `mean_top_n`, `top_n=3` | 0.099010 | 0.188119 | 0.277228 | 0.386139 | 0.455446 | **0.281188** |
| `mean_top_n`, `top_n=10` | 0.089109 | 0.178218 | 0.227723 | 0.346535 | 0.445545 | **0.257426** |

The `max` aggregation with `top_n=5` and `top_n=10` produced identical results
(`top_n` only affects the `mean_top_n` aggregation). All video-level settings
degraded Final relative to the baseline; `mean_top_n` degraded further as
`top_n` grew.

### Conclusion

- Video-level aggregation reduced KIS performance on this evaluation set
  relative to the untouched baseline.
- Grouping frames by video loses temporal localization precision, which is
  exactly what KIS scoring rewards.
- Frame-level CLIP ranking remains the strongest signal among the post-
  processing strategies recorded so far.
- Do not use this naive video reranking for KIS.
- This records one specific video-level reranking approach; it does not claim
  video reranking methods in general are ineffective.

---

## KIS Error Analysis

Read-only categorization of the baseline retrieval results to understand where
and why KIS retrieval fails, using `scripts/analyze_kis_errors.py`.

### Setup

- **Baseline analyzed:** OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k`
- **Prediction artifact:** `kis-l14-laion2b-full-en`
- **Evaluation:** 101 KIS queries (`data/eval/eval_set.csv`), top-100/top-k
  analysis of each query's ranked predictions
- **Categories:** `HIT` = correct video with a frame inside the inclusive
  ground-truth range; `NEAR_VIDEO` = correct video but no in-range frame;
  `FAIL` = correct video absent from the top-k predictions

### Results

Category breakdown:

| Category | Count | Share |
|----------|-------|-------|
| HIT | 54 / 101 | 53.5% |
| NEAR_VIDEO | 36 / 101 | 35.6% |
| FAIL | 11 / 101 | 10.9% |

Video retrieval:

- Ground-truth video appears in top-100: 89.1%
- Average first ground-truth video rank: 17.98

So the correct video is retrieved for the large majority of queries; failure
at the video level is rare.

### Temporal observations

- Many `NEAR_VIDEO` cases are very close to the ground-truth frame boundaries:
  - distances as small as 4 frames
  - common distances in the 22–40 frame range
- Some cases show large temporal offsets, where CLIP retrieves the correct
  semantic video but the wrong moment in it.

These point to a frame-selection rather than a video-selection problem.

### Failure observations

Some `FAIL` cases reflect fine-grained visual/text confusion:

- similar food preparation scenes
- educational / slides / lecture videos
- text-heavy or OCR-dependent queries

### Conclusion

- The main bottleneck is not video-level retrieval.
- The current CLIP backbone already retrieves the correct video for most
  queries.
- Future improvements should focus on temporal localization and frame
  candidate expansion rather than only changing the backbone.
