# Distributed Worker Quickstart

## 1. Workflow overview

```text
worker_N/videos.json
        |
        v
verify split
        |
        v
OmniShotCut adaptive keyframe extraction
        |
        v
keyframes artifact
        |
        v
embedding
        |
        v
embedding artifact
        |
        v
send back to coordinator
```

## 2. Required inputs

- **Worker manifest**
  `data/keyframe-splits/worker_XXX/videos.json`
  Lists exactly the videos assigned to you. Do not add or remove videos.

- **Raw videos**
  `/path/to/videos`
  The video files listed in your manifest, available on your machine.

- **Keyframe images for embedding**
  `/path/to/aic2026_batch1/keyframes/`
  A pre-existing collection of JPEGs, one per keyframe.

> **Important:** Extraction does NOT create JPG images. Embedding requires
> existing images at `keyframes/<video_id>/<ordinal>.jpg`.

## 3. Environment setup

**macOS:**

```bash
brew install ffmpeg

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,openclip]"
pip install "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git"
```

**Windows PowerShell:**

```powershell
winget install Gyan.FFmpeg

python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev,openclip]"
pip install "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git"
```

Notes:

- **CUDA users:** install a PyTorch build that matches your CUDA driver
  version. The default wheels may not include CUDA.
- **Apple Silicon:** MPS is used automatically by `--device auto`.

## 4. Worker id convention

Replace `XXX` with your assigned worker id.

Examples of worker ids:

- `worker_000`
- `worker_001`
- `worker_002`

Use the same id everywhere: in paths (`worker_XXX`) and as the embedding
flag (`--worker-id XXX`). Never use a concrete worker number in commands.

## 5. Step 1: Verify assigned split

```bash
python scripts/verify_video_split.py \
  --split-dir data/keyframe-splits/
```

Expected: exit code `0`.

This confirms the split manifest exists, every worker manifest parses, no
video is duplicated or missing, and the totals match.

## 6. Step 2: Extract keyframes

```bash
python scripts/extract_keyframes.py \
  --videos data/keyframe-splits/worker_XXX/videos.json \
  --video-root /PATH/TO/videos \
  --output-dir outputs/keyframes_worker_XXX \
  --extractor omnishotcut_adaptive \
  --detector omnishotcut \
  --sampler adaptive_v1 \
  --device auto \
  --progress on
```

The only part that differs per machine is `--video-root`, the directory
that contains your video files.

**Mac example:**

```bash
--video-root /Users/<user>/datasets/videos
```

**Windows PowerShell example:**

```powershell
--video-root D:\datasets\videos
```

**Output** — in `outputs/keyframes_worker_XXX/`:

```text
frames.ndjson
map-keyframes/
manifest.json
```

The run is resumable: if interrupted, re-run the same command and it skips
already-completed videos.

## 7. Step 3: Verify extracted keyframes

```bash
python - <<'PY'
import json
from pathlib import Path

d = Path("outputs/keyframes_worker_XXX")
frames = d / "frames.ndjson"
manifest = d / "manifest.json"

assert frames.is_file() and frames.stat().st_size > 0, "frames.ndjson missing or empty"
assert manifest.is_file(), "manifest.json missing"

seen = {}
for line in frames.open():
    r = json.loads(line)
    key = (r["keyframe"]["video_id"]["value"], r["frame_index"])
    if key in seen:
        raise SystemExit(f"DUPLICATE keyframe {key} (was line {seen[key]})")
    seen[key] = line

print(f"keyframes OK: {len(seen)} unique keyframes, manifest present")
PY
```

## 8. Step 4: Embed keyframes

```bash
python scripts/embed_shard.py \
  --frames outputs/keyframes_worker_XXX/frames.ndjson \
  --output-dir outputs/embeddings_worker_XXX \
  --model openclip:ViT-L-14:laion2b_s32b_b82k \
  --worker-id XXX \
  --keyframes-dir /PATH/TO/aic2026_batch1 \
  --device auto \
  --progress on
```

**Mac example:**

```bash
--keyframes-dir /Users/<user>/datasets/aic2026_batch1
```

**Windows PowerShell example:**

```powershell
--keyframes-dir D:\datasets\aic2026_batch1
```

> **Important:** `--keyframes-dir` points to the parent directory containing
> `keyframes/<video_id>/<ordinal>.jpg` — NOT the `keyframes` folder itself.

The run is resumable: if interrupted, re-run the exact same command and it
resumes from `checkpoint.json`, skipping already-embedded frames.

## 9. Step 5: Verify embeddings

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

d = Path("outputs/embeddings_worker_XXX")
assert (d / "embeddings.npy").is_file(), "embeddings.npy missing"
assert (d / "manifest.json").is_file(), "manifest.json missing"

emb = np.load(d / "embeddings.npy")
n_frames = sum(1 for _ in (d / "frames.ndjson").open())
assert emb.shape[0] == n_frames, f"rows {emb.shape[0]} != frames {n_frames}"
print(f"embeddings OK: {emb.shape} matches {n_frames} frames, manifest present")
PY
```

## 10. Send back artifacts

Send one folder back to the coordinator, named `worker_XXX_result/`:

```text
worker_XXX_result/

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

**Do not send:**

- checkpoint files
- raw videos
- environment folders
