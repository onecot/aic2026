#!/usr/bin/env python3
"""Run a lightweight local development API server with synthetic fixtures.

Instantly ready without waiting for large model downloads or full dataset indexing.
"""

from __future__ import annotations

import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aic2026.api import create_app
from aic2026.frame_catalog import FrameCatalog
from aic2026.qa.qa_smoke import QaPlaceholderEncoder, smoke_embedding_space
from aic2026.search import SearchEngine


def create_dev_app():
    root = Path(__file__).resolve().parents[1]
    real_keyframes = root / "data" / "keyframes" / "adaptive_sbd_v1"
    fixtures_dir = root / "tests" / "_ui_backend_fixtures"
    index_dir = fixtures_dir / "index"
    keyframes_dir = real_keyframes if real_keyframes.is_dir() else (fixtures_dir / "keyframes")

    encoder = QaPlaceholderEncoder(smoke_embedding_space(), 8)
    engine = SearchEngine.from_parts(encoder, index_dir, top_k=50)
    catalog = FrameCatalog.from_keyframes_root(keyframes_dir)
    return create_app(engine=engine, catalog=catalog, keyframes_root=keyframes_dir)


if __name__ == "__main__":
    app = create_dev_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
