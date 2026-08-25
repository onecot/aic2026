# Retrieval API (minimal HTTP surface)

A small FastAPI application exposing the Phase-2 Python core for a future
human-inspection UI. The HTTP layer is thin: every endpoint calls the plain
Python services (`SearchEngine`, `FrameCatalog`, `RetrievalService`)
directly. No subprocesses, no prediction artifacts written on request, no
filesystem paths in responses.

## Install

```bash
pip install -e ".[api]"
```

## Start the server

```bash
# production defaults: index data/interim/indexes/jina-v2-full,
# keyframes from configs/default.yaml (adaptive_sbd_v1)
uvicorn aic2026.server:app --host 127.0.0.1 --port 8000

# explicit overrides
AIC_INDEX_DIR=data/interim/indexes/jina-v2-full \
AIC_KEYFRAMES_DIR=data/keyframes/adaptive_sbd_v1 \
uvicorn aic2026.server:app --host 127.0.0.1 --port 8000
```

Programmatic use (tests, notebooks, non-web consumers):

```python
from fastapi.testclient import TestClient
from aic2026.api import create_app
from aic2026.config import load_config

app = create_app(config=load_config(), index_artifact="data/interim/indexes/jina-v2-full")
client = TestClient(app)
```

The model, index, and frame metadata are loaded **once** at application
startup (`create_app`); requests never reload them.

## Endpoints

### `GET /api/health`

```json
{"status": "ready", "model": true, "index": true, "frame_catalog": true}
```

- HTTP 200 when the runtime is fully initialized (Jina encoder, index, and
  FrameCatalog load exactly once at server startup).
- HTTP 503 with `{"status": "starting", ...}` while initialization is in
  flight or after a failed startup. Retrieval endpoints return
  `503 {"detail": "Backend is not ready"}` until ready; they never construct
  resources lazily per request.

### `POST /api/search`

Optional `task`:

- `"kis"` (default): general frame retrieval.
- `"trake"`: event-centric search. The response additionally contains
  `"events"`, each `{event_id, event_text, predictions: [hit...],
  windows: [{video_id, frame_start, frame_end, score, rank}]}` — the same
  shape as TRAKE prediction artifacts (`results` is empty for this task).



```json
{"query": "một người đang nấu ăn trong bếp", "limit": 100}
```

`task` is optional (`"kis"` is currently the only value and the default).

Response:

```json
{
  "query": "một người đang nấu ăn trong bếp",
  "results": [
    {
      "rank": 1,
      "video_id": "L26_V435",
      "frame_index": 2151,
      "keyframe_ordinal": 73,
      "pts_time": 71.2,
      "score": 0.482
    }
  ]
}
```

- `score` is a ranking score, **not** a calibrated probability.
- Deterministic for identical inputs.

### `GET /api/videos/{video_id}`

```json
{"video_id": "L26_V435", "frame_count": 172, "first_frame_index": 20, "last_frame_index": 28491}
```

Metadata is derived from indexed keyframes only; nothing is invented.
Unknown video → `404`.

### `GET /api/videos/{video_id}/frames`

Two modes; do not combine:

- paging: `?offset=0&limit=100`
- around-center: `?center=2151&radius=1000&limit=50` (`radius` optional)

```json
{
  "video_id": "L26_V435",
  "frames": [{"keyframe_ordinal": 70, "frame_index": 2050, "pts_time": 68.1, "fps": 30.0}],
  "total": 172
}
```

Results are always in temporal (frame_index) order and contain only actual
indexed keyframes — missing frame indices are never synthesized.

### `GET /api/frames/{video_id}/{frame_index}`

Resolves `video_id + frame_index` through the catalog to a
`keyframe_ordinal`, then to the canonical image. Returns the image bytes
(`image/jpeg`) directly — no JSON wrapper. Unknown video/frame or missing
image file → `404`.

## Frame identity semantics

- A frame's logical identity is `video_id + frame_index`.
- Each indexed keyframe also has a stable `keyframe_ordinal`
  (`images/<video_id>/<ordinal:06d>.jpg` on disk).
- `frame_index` positions are sparse: only indexed keyframes exist; the
  catalog never fabricates intermediate frames.

## Error behavior

| Status | Meaning |
| ------ | ------- |
| 400 | invalid query parameters (pagination/radius) |
| 503 | backend starting or startup failed (health: `status="starting"`) |
| 404 | unknown video / unknown frame / unavailable image |
| 422 | invalid request body |
| 500 | unexpected internal failure |

Responses never include filesystem paths, embeddings, scores beyond the
ranking score, or internal artifact locations.
