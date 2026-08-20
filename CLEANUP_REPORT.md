# Cleanup report

## Scope and preserved behavior

This cleanup keeps the competition path intact: keyframe extraction and
materialization, OCR and caption artifact formats, embedding/index artifacts,
retrieval outputs, and evaluation commands. No NDJSON field, manifest schema,
or artifact directory layout was changed.

## Measured reduction

| Measure | Before | After | Change |
| --- | ---: | ---: | ---: |
| Source Python files | 66 | 65 | -1 |
| Source LOC | 54,653 | 53,667 | -986 |
| Pipeline/Modal CLI LOC | 3,293 | 3,079 | -214 |
| Test files | 56 | 54 | -2 |
| Test LOC | 21,201 | 20,687 | -514 |
| Classes | 190 | 186 | -4 |
| CLI options | 257 | 255 | -2 |

The separate one-off shell launcher was also deleted. Existing uncommitted
work in the repository is included in both measurements.

## Repository tree change

```text
before                                  after
modal_ocr.py                     ->     (deleted; use distributed runner with one worker)
modal_ocr_distributed.py          ->     modal_ocr_distributed.py
scripts/run_remaining_extraction.sh ->   (deleted)
src/aic2026/keyframe_materialize_verify.py -> (deleted)
src/aic2026/keyframe_materialize.py ->     src/aic2026/keyframe_materialize.py
tests/test_modal_ocr.py           ->     (deleted)
tests/test_keyframe_materialize_verify.py -> (deleted)
```

## Removed

- `modal_ocr.py`: the single-GPU runner duplicated the distributed runner's
  one-shard mode. Use `modal_ocr_distributed.py --num-workers 1`.
- `tests/test_modal_ocr.py`: tests coupled to that deleted duplicate runner.
- `src/aic2026/keyframe_materialize_verify.py`: duplicated materialization
  verification implementation and report model.
- `tests/test_keyframe_materialize_verify.py`: coverage for the deleted
  duplicate verifier.
- `scripts/run_remaining_extraction.sh`: a hard-coded, machine-specific
  one-off worker launcher.
- OCR/caption CLI factory flags and their test hooks:
  `--ocr-factory` and `--captioner-factory`.
- `OcrEngine` and `Captioner` protocols: they added public-looking extension
  points used only for test doubles; tests now inject their fakes directly
  through the core functions.
- Environment-dependent OCR CLI test that expected PaddleOCR not to be
  installed.

## Simplified interfaces

- Caption generation now uses `--backend`, matching OCR. The former
  `--caption-backend` option was removed.
- Modal OCR has one runner. `--num-workers 1` is the single-GPU command;
  `--validate` remains isolated from production shard output.
- Materialization verification has one implementation:
  `aic2026.keyframe_materialize.verify_materialization`. The verification CLI
  calls it directly.

## Commands

```bash
# Keyframes
python scripts/extract_keyframes.py --videos data/videos.json --output-dir data/keyframes/run

# OCR
python scripts/generate_ocr.py --backend paddleocr --keyframes-dir data/keyframes/run --output-dir data/interim/ocr/paddleocr-v5

# Captions
python scripts/generate_captions.py --backend qwen2.5-vl --keyframes-dir data/keyframes/run --output-dir data/interim/captions/qwen2.5-vl-3b

# Distributed OCR (or a single GPU with --num-workers 1)
modal run modal_ocr_distributed.py --num-workers 8

# Build index and retrieve
aic2026 build-index --source data/interim/embeddings/run --output data/interim/indexes/run --artifact-id run
aic2026 predict-kis --index-artifact data/interim/indexes/run --output data/interim/predictions/kis.json

# Evaluate (the project exposes evaluation as a Python API)
python -c 'from aic2026.evaluation import load_eval_set; from aic2026.kis_prediction_runner import load_kis_predictions_for_scoring; from aic2026.kis_scoring import evaluate_kis; print(evaluate_kis(load_eval_set("data/eval/eval_set.csv"), load_kis_predictions_for_scoring("data/interim/predictions/kis.json")).final_score)'
```

## Remaining technical debt

- OCR and caption generation still share substantial resumable-artifact code.
  It was intentionally left explicit rather than introducing a generic
  framework that would obscure their distinct artifact contracts.
- Keyframe extraction retains advanced extractor-specific tuning flags because
  they affect existing experiment configuration and artifact provenance.
- The main CLI still contains experimental retrieval analysis commands; those
  have documented experiment usage and were not removed without usage evidence.

## Validation

- Syntax/import/CLI help checks passed for `src`, `scripts`,
  `modal_ocr_distributed.py`, `aic2026`, OCR, captions, and materialization
  verification.
- Targeted cleanup tests passed:
  `tests/test_ocr_generation.py`, `tests/test_caption_generation.py`,
  `tests/test_modal_ocr_distributed.py`, and
  `tests/test_keyframe_materialize.py`.
- Full suite passed: `PYTHONPATH=src:. pytest -q`.
