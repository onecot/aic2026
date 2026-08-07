# AGENTS.md — Rules for coding agents on this project

## Project layout
- `aic2026/` — repo root; agents run here and all source lives here.
  - `data/` — organizer-provided data (features, mapping, keyframes, ...). READ-ONLY.
    Never commit; it is gitignored.
- Config paths resolve relative to the config file's location (`configs/`), so the data
  folder (`aic2026/data/`) is reached via `../data`.

## Core principles
- **Config-driven, no hard-coded paths.** Every path/param comes from a YAML config
  passed via `--config`. Nothing machine-specific in code. New path keys must be added to
  `_PATH_KEYS` in `src/utils/config.py` so they get resolved.
- **Cross-platform (multi-OS).** Must run on macOS, Linux, Windows:
  - Always use `pathlib.Path`; never string-concatenate paths or hard-code `/` or `\`.
  - No OS-specific shell calls in core logic. No absolute paths in code or config.
  - Read/write text files with `encoding="utf-8"` explicitly.
- **Simple over clever.** This is a foundation to extend, not to over-engineer.

## Reproducibility (required)
- **Deterministic.** Same input → same output, every run. No hidden randomness.
  - ML inference: always `model.eval()` + `torch.no_grad()`; no dropout at inference.
  - Any tie-break (e.g. equal similarity scores in ranking) must be stable and explicit,
    so ordering never changes between runs.
  - If randomness is ever unavoidable, expose a `seed` in config and set it explicitly.
- **Environment pinned.** Dependencies pinned in `requirements.txt`; all params live in the
  YAML config. A run must be reproducible from (repo commit + config) alone.
- **Resumable — only for long-running stages.** Do NOT add resume logic to fast steps
  (e.g. index building runs in ~1s — rerunning is simpler than resuming). Add resumability
  ONLY to stages that take many minutes/hours (e.g. embedding 100k+ images ourselves):
  write per-item outputs with a "done" marker so an interrupted run can skip finished work.
  Prefer rerunning from scratch when a stage is cheap.

## Code style
- Python 3.11. Type hints on all function signatures.
- Every function has a docstring with: 1-line summary, **Expected input** (shapes/dtypes/
  layout), **Returns**, **Raises**. Comments in English.
- Add inline comments where logic is non-obvious (index alignment, off-by-one, normalization).
- Any place needing a human decision or a TODO: mark with `# TODO(clarify): ...`.

## Pipeline scripts
- Every runnable script exposes CLI arguments (argparse): a `--config` flag plus
  overridable knobs (paths, top_k, limits, device). Make functions testable by accepting
  an optional args list.
- Scripts print basic stats/timing so runs are verifiable.
- Run scripts as modules from the repo root: `python -m scripts.<name> --config ...`
  (so `src.` imports resolve).

## Data contract (established)
- `src/data/loader.py` provides:
  `build_index(features_dir, mapping_dir, normalize=True, limit=0) -> (emb, vids, frame_idx)`,
  `save_index(...)`, `load_index(...) -> (emb, vids, frame_idx)`.
  - `emb` `(N,512)` float32, L2-normalized when `normalize=True`.
  - `vids` `(N,)` str — video_id per row. `frame_idx` `(N,)` int — REAL source-video frame
    per row (this is what gets submitted; NOT the keyframe counter `n`).
  - Return order is always `(emb, vids, frame_idx)` — keep it consistent everywhere.
- CLIP features are ViT-B/32; any text encoder MUST match this variant (same embedding space).
- Data folders under `data/`: `clip-features-32/` (.npy per video), `map-keyframes/`
  (.csv per video: columns n, pts_time, fps, frame_idx), `keyframes/`, `media-info/`, `objects/`.

## Do NOT
- Do not modify files under `data/` (gitignored).
- Do not commit data, model weights, `*.npz` artifacts, or `AGENTS.md` (see `.gitignore`).
- Do not introduce new dependencies without adding them to `requirements.txt`.
