# Distributed Worker Quickstart

## 1. Workflow

```text
worker_XXX/videos.json
        |
        v
verify split
        |
        v
OmniShotCut adaptive keyframe extraction
        |
        v
keyframe artifact
        |
        v
send back to coordinator
```

Your only job is to extract keyframes from the videos assigned to you.

> **Important:**
> - Extraction does NOT create JPEG images.
> - It only creates metadata (`frames.ndjson`, `map-keyframes/`) describing
>   which frames are keyframes.
> - The existing BTC keyframe image collection is still required later, when
>   embeddings are computed centrally.
> - You only need access to the assigned raw videos.

## 2. Required inputs

- **Worker manifest**
  `data/keyframe-splits/worker_XXX/videos.json`
  Lists exactly the videos assigned to you. Do not add or remove videos.

- **Raw videos**
  `/path/to/videos`
  The video files listed in your manifest, available on your machine.

## 3. Environment setup

**macOS:**

```bash
brew install ffmpeg

python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git"
```

**Windows PowerShell:**

```powershell
winget install Gyan.FFmpeg

python -m venv .venv
.venv\Scripts\activate

pip install -e .
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

Use the same id everywhere in paths: the worker's assigned manifest
(`data/keyframe-splits/worker_XXX/videos.json`) and the extraction output
directory (`outputs/keyframes_worker_XXX`). Never use a concrete worker number
in commands.

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

The run is resumable: if interrupted, re-run the same command and it skips
already-completed videos.

## 7. Output artifact

In `outputs/keyframes_worker_XXX/`:

```text
frames.ndjson
map-keyframes/
manifest.json
```

## 8. Step 3: Verify extracted keyframes

Verify the artifact before sending it back (exit code 0 = valid, 1 = invalid):

```bash
python scripts/verify_keyframe_extraction.py \
  --output-dir outputs/keyframes_worker_XXX \
  --videos data/keyframe-splits/worker_XXX/videos.json
```

`--videos` is optional; pass the worker's assigned `videos.json` to also check
coverage (every assigned video present, no unexpected videos, totals match).
The verifier checks the manifest, frame-record validity and ordering, and the
per-video `map-keyframes/` files:

```text
Keyframe extraction verification: PASS

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

## 9. Send back artifacts

Send one folder back to the coordinator, named `worker_XXX_result/`:

```text
worker_XXX_result/

└── keyframes/
    ├── frames.ndjson
    ├── map-keyframes/
    └── manifest.json
```

**Do not send:**

- checkpoint files
- raw videos
- environment folders

## 10. Coordinator next steps

Once all workers deliver their `worker_XXX_result/keyframes/` folders:

1. **Merge keyframe shards** into one canonical `frames.ndjson` /
   `map-keyframes/` artifact.
2. **Prepare the embedding environment** on the target GPU machine
   (Kaggle/Modal): install the `jina` dependency group and the Jina CLIP v2
   checkpoint.
3. **Run Jina CLIP v2 embeddings** on the merged keyframes.
4. **Merge embeddings** into the final embedding artifact.
