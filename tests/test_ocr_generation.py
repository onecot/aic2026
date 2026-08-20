"""Tests for the resumable keyframe OCR extraction pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.canonical import KeyframeTiming
from aic2026.ocr_generation import (
    CHECKPOINT_FILENAME,
    DEFAULT_BACKEND,
    DEFAULT_LANGUAGES,
    DEFAULT_OCR_MODEL,
    MANIFEST_FILENAME,
    OCR_BACKENDS,
    OCR_FILENAME,
    KeyframeOcr,
    OcrCheckpoint,
    OcrGenerationError,
    OcrLine,
    OcrManifest,
    PaddleOcrEngine,
    PaddleOcrRuntimeError,
    create_ocr_engine,
    generate_ocr,
    iter_ocr_records,
    normalize_ocr_text,
    preprocess_image,
    read_ocr_artifact,
    validate_ocr_artifact,
)

#: video_id -> (ordinals) in the synthetic input order.
LAYOUT = [("L21_V001", [1, 2, 3]), ("L21_V002", [1, 2]), ("L22_V001", [1])]
VIDEO_INDEX = {vid: idx for idx, (vid, _) in enumerate(LAYOUT, start=1)}


def _write_frames(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for vid, ordinals in LAYOUT:
            for ordinal in ordinals:
                record = {
                    "keyframe": {"video_id": {"value": vid}, "keyframe_ordinal": ordinal},
                    "pts_time": (ordinal - 1) * 0.4,
                    "fps": 25.0,
                    "frame_index": (ordinal - 1) * 10,
                }
                handle.write(json.dumps(record) + "\n")


def _make_keyframes(tmp_path: Path) -> Path:
    """Build a materialized keyframes artifact (frames.ndjson + images/)."""
    root = tmp_path / "keyframes"
    (root / "images").mkdir(parents=True)
    _write_frames(root / "frames.ndjson")
    for vid, ordinals in LAYOUT:
        vid_dir = root / "images" / vid
        vid_dir.mkdir(parents=True)
        for ordinal in ordinals:
            (vid_dir / f"{ordinal:06d}.jpg").write_bytes(
                bytes([VIDEO_INDEX[vid], ordinal, 0])
            )
    return root


class _StubOcrEngine:
    """Deterministic OCR engine deriving text from image bytes."""

    device = "cpu"
    model_name = "stub:ocr"
    languages = DEFAULT_LANGUAGES

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect_text(self, images: list[object]) -> list[list[OcrLine]]:
        self.batch_sizes.append(len(images))
        out: list[list[OcrLine]] = []
        for image in images:
            data = bytes(image)  # type: ignore[arg-type]
            text = f"TEXT-v{data[0]}-k{data[1]}"
            out.append(
                [
                    OcrLine(
                        text=text,
                        confidence=0.96,
                        bbox=(data[0], data[1], data[0] + 100.0, data[1] + 20.0),
                    )
                ]
            )
        return out


def _image_loader(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture
def keyframes(tmp_path: Path) -> Path:
    return _make_keyframes(tmp_path)


def _run(
    keyframes: Path,
    out_dir: Path,
    **kwargs: object,
) -> tuple[_StubOcrEngine, OcrManifest]:
    engine = _StubOcrEngine()
    kwargs.setdefault("image_loader", _image_loader)
    manifest = generate_ocr(
        keyframes,
        out_dir,
        ocr_engine=engine,
        **kwargs,
    )
    return engine, manifest


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------------ text processing


def test_normalize_ocr_text_lowercases_keeps_accents() -> None:
    assert normalize_ocr_text("PHƯƠNG TRÌNH DAO ĐỘNG") == "phương trình dao động"
    assert normalize_ocr_text("Hà Nội - Việt Nam") == "hà nội - việt nam"


def test_normalize_ocr_text_whitespace_and_symbols() -> None:
    assert normalize_ocr_text("  Hello    World  ") == "hello world"
    assert normalize_ocr_text("Wait!!! Really???") == "wait! really?"


def test_normalize_ocr_text_requires_string() -> None:
    with pytest.raises(ValueError, match="text must be a string"):
        normalize_ocr_text(123)  # type: ignore[arg-type]


# ------------------------------------------------------------------ records


def test_ocr_line_validates() -> None:
    line = OcrLine(text="PHƯƠNG TRÌNH", confidence=0.96, bbox=(1, 2, 30, 40))
    assert line.to_dict() == {"text": "PHƯƠNG TRÌNH", "confidence": 0.96, "bbox": [1.0, 2.0, 30.0, 40.0]}
    assert OcrLine.from_dict(line.to_dict()) == line
    with pytest.raises(ValueError, match="confidence"):
        OcrLine(text="x", confidence=1.5, bbox=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="bbox"):
        OcrLine(text="x", confidence=0.5, bbox=(5, 5, 1, 1))


def test_keyframe_ocr_round_trip() -> None:
    record = KeyframeOcr(
        video_id=KeyframeTiming.from_dict(
            {
                "keyframe": {"video_id": {"value": "L25_V085"}, "keyframe_ordinal": 63},
                "pts_time": 1.0,
                "fps": 25.0,
                "frame_index": 3578,
            }
        ).keyframe.video_id,
        keyframe_ordinal=63,
        frame_index=3578,
        ocr_text="PHƯƠNG TRÌNH DAO ĐỘNG",
        ocr_text_normalized="phương trình dao động",
        ocr_lines=(OcrLine(text="PHƯƠNG TRÌNH DAO ĐỘNG", confidence=0.96, bbox=(1, 2, 30, 40)),),
        status="ok",
    )
    assert KeyframeOcr.from_dict(record.to_dict()) == record
    error = KeyframeOcr(
        video_id=record.video_id,
        keyframe_ordinal=63,
        frame_index=3578,
        ocr_text="",
        ocr_text_normalized="",
        ocr_lines=(),
        status="error",
        error="boom",
    )
    with pytest.raises(ValueError, match="error"):
        KeyframeOcr.from_dict({**error.to_dict(), "error": None})


# ------------------------------------------------------------------ pipeline


def test_generate_ocr_creates_valid_artifact(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr"
    _, manifest = _run(keyframes, out)

    for name in (MANIFEST_FILENAME, OCR_FILENAME):
        assert (out / name).is_file()
    assert not (out / CHECKPOINT_FILENAME).exists()

    assert manifest.num_records == 6
    assert manifest.num_videos == 3
    assert manifest.num_failed == 0
    assert manifest.backend == DEFAULT_BACKEND
    assert manifest.model == "stub:ocr"
    assert manifest.device == "cpu"
    assert manifest.batch_size == 16
    assert manifest.languages == DEFAULT_LANGUAGES
    assert manifest.created_at != ""
    assert manifest.artifact_id == "ocr"

    records = iter_ocr_records(out / OCR_FILENAME)
    assert len(records) == 6
    assert all(record.status == "ok" for record in records)
    assert validate_ocr_artifact(manifest, out) == []


def test_manifest_records_required_fields(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-fields"
    _run(keyframes, out)
    payload = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for key in (
        "backend", "model", "languages", "device", "batch_size", "num_records",
        "num_videos", "num_failed", "canonical_schema_version", "created_at",
    ):
        assert key in payload
    assert payload["schema_version"] == "1.0"


def test_records_preserve_identity_and_alignment(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-align"
    _run(keyframes, out)
    rows = _load_rows(out / OCR_FILENAME)
    frames = list(KeyframeTiming.from_dict(line) for line in _load_rows(
        keyframes / "frames.ndjson"
    ))

    assert len(rows) == len(frames)
    for i, (row, frame) in enumerate(zip(rows, frames)):
        assert row["video_id"]["value"] == frame.keyframe.video_id.value
        assert row["keyframe_ordinal"] == frame.keyframe.keyframe_ordinal
        assert row["frame_index"] == frame.frame_index
        assert row["status"] == "ok"
        assert row["error"] is None
        assert row["ocr_lines"][0]["text"] == f"TEXT-v{VIDEO_INDEX[frame.keyframe.video_id.value]}-k{frame.keyframe.keyframe_ordinal}"
        assert row["ocr_text_normalized"] == normalize_ocr_text(row["ocr_text"])


def test_deterministic_output(keyframes: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run(keyframes, out_a)
    _run(keyframes, out_b)
    assert _load_rows(out_a / OCR_FILENAME) == _load_rows(out_b / OCR_FILENAME)


def test_interrupt_leaves_no_manifest(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-interrupt"
    with pytest.raises(OcrGenerationError, match="interrupted OCR build"):
        _run(keyframes, out, batch_size=1, stop_after_records=2)

    assert not (out / MANIFEST_FILENAME).exists()
    assert (out / CHECKPOINT_FILENAME).is_file()
    assert (out / OCR_FILENAME).is_file()
    checkpoint = json.loads((out / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    assert checkpoint["completed_record_count"] == 2
    assert set(checkpoint.keys()) >= {
        "completed_record_count", "completed_video_ids", "backend", "model",
        "languages", "device", "batch_size", "source_frames", "total_records",
        "timestamp",
    }
    assert len(_load_rows(out / OCR_FILENAME)) == 2


def test_resume_skips_existing_records(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-resume"
    engine = _StubOcrEngine()
    with pytest.raises(OcrGenerationError):
        generate_ocr(
            keyframes, out, ocr_engine=engine, image_loader=_image_loader,
            batch_size=1, stop_after_records=2,
        )
    calls_before_resume = sum(engine.batch_sizes)

    manifest = generate_ocr(
        keyframes, out, ocr_engine=engine, image_loader=_image_loader,
        batch_size=1,
    )
    assert manifest.num_records == 6
    assert (out / MANIFEST_FILENAME).is_file()
    assert sum(engine.batch_sizes) - calls_before_resume == 4


def test_resume_matches_clean_run(keyframes: Path, tmp_path: Path) -> None:
    clean_out = tmp_path / "clean"
    _, clean_manifest = _run(keyframes, clean_out)

    resumed_out = tmp_path / "resumed"
    with pytest.raises(OcrGenerationError):
        _run(keyframes, resumed_out, batch_size=2, stop_after_records=4)
    resume_manifest = _run(keyframes, resumed_out, batch_size=2)[1]

    assert clean_manifest.num_records == resume_manifest.num_records == 6
    assert _load_rows(clean_out / OCR_FILENAME) == _load_rows(
        resumed_out / OCR_FILENAME
    )


def test_resume_rejects_languages_mismatch(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-languages"
    with pytest.raises(OcrGenerationError):
        _run(keyframes, out, batch_size=1, stop_after_records=2)
    with pytest.raises(ValueError, match="languages"):
        _run(keyframes, out, languages=("en",))


def test_complete_fast_path_returns_existing(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-fast"
    engine = _StubOcrEngine()
    first = generate_ocr(
        keyframes, out, ocr_engine=engine, image_loader=_image_loader,
    )
    calls = sum(engine.batch_sizes)
    second = generate_ocr(
        keyframes, out, ocr_engine=engine, image_loader=_image_loader,
    )
    assert first == second
    assert sum(engine.batch_sizes) == calls


def test_max_records_limits_artifact(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-limited"
    _, manifest = _run(keyframes, out, max_records=3)
    assert manifest.num_records == 3
    assert manifest.num_videos == 1
    rows = _load_rows(out / OCR_FILENAME)
    assert len(rows) == 3
    assert all(row["video_id"]["value"] == "L21_V001" for row in rows)


def test_missing_image_recorded_as_error(keyframes: Path, tmp_path: Path) -> None:
    missing = keyframes / "images" / "L21_V002" / "000002.jpg"
    missing.unlink()

    out = tmp_path / "ocr-missing"
    _, manifest = _run(keyframes, out, batch_size=2)

    assert manifest.num_records == 6
    assert manifest.num_failed == 1
    records = iter_ocr_records(out / OCR_FILENAME)
    assert len(records) == 6
    error_records = [r for r in records if r.status == "error"]
    assert len(error_records) == 1
    assert error_records[0].ocr_text == ""
    assert error_records[0].ocr_lines == ()
    assert error_records[0].error is not None
    assert error_records[0].video_id.value == "L21_V002"
    assert error_records[0].keyframe_ordinal == 2
    assert validate_ocr_artifact(manifest, out) == []


def test_corrupt_image_does_not_fail_batch(keyframes: Path, tmp_path: Path) -> None:
    corrupt = keyframes / "images" / "L21_V001" / "000002.jpg"
    corrupt.write_bytes(b"not-an-image")

    def flaky_loader(path: Path) -> bytes:
        data = path.read_bytes()
        if data == b"not-an-image":
            raise OSError("cannot decode image")
        return data

    out = tmp_path / "ocr-corrupt"
    _, manifest = _run(keyframes, out, image_loader=flaky_loader, batch_size=3)

    assert manifest.num_records == 6
    assert manifest.num_failed == 1
    records = iter_ocr_records(out / OCR_FILENAME)
    assert sum(1 for r in records if r.status == "ok") == 5
    assert validate_ocr_artifact(manifest, out) == []


def test_engine_batch_failure_falls_back_per_image(keyframes: Path, tmp_path: Path) -> None:
    class _FailingBatch(_StubOcrEngine):
        def detect_text(self, images: list[object]) -> list[list[OcrLine]]:
            if len(images) > 1:
                raise RuntimeError("OOM")
            return super().detect_text(images)

    out = tmp_path / "ocr-fallback"
    engine = _FailingBatch()
    manifest = generate_ocr(
        keyframes, out, ocr_engine=engine, image_loader=_image_loader, batch_size=3,
    )
    assert manifest.num_failed == 0
    assert _load_rows(out / OCR_FILENAME) == [row for row in _load_rows(out / OCR_FILENAME)]
    assert validate_ocr_artifact(manifest, out) == []


# ------------------------------------------------------------------ validation


def test_validate_ocr_artifact_detects_mismatch(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-validate"
    _, manifest = _run(keyframes, out)

    with (out / OCR_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_load_rows(out / OCR_FILENAME)[0]) + "\n")
    problems = validate_ocr_artifact(manifest, out)
    assert any("records" in problem for problem in problems)


def test_read_ocr_artifact_round_trip(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-read"
    _, manifest = _run(keyframes, out)
    read_manifest, records = read_ocr_artifact(out)
    assert read_manifest == manifest
    assert len(records) == manifest.num_records
    assert read_ocr_artifact(out) == read_ocr_artifact(out)


# ------------------------------------------------------------------ backend selection


def test_ocr_backends_registry() -> None:
    assert OCR_BACKENDS == ("paddleocr",)
    assert DEFAULT_BACKEND == "paddleocr"
    assert DEFAULT_OCR_MODEL == "PP-OCRv5_mobile"


def test_create_ocr_engine_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakePaddleOcr:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    def fake_paddleocr_load() -> object:
        return type("FakePaddleOCRModule", (), {"PaddleOCR": _FakePaddleOcr})

    def fake_import(name: str) -> object:
        if name == "paddleocr":
            return fake_paddleocr_load()
        if name == "paddle":
            return type(
                "FakePaddle",
                (),
                {
                    "device": type(
                        "FakeDevice",
                        (),
                        {"is_compiled_with_cuda": lambda: False, "cuda": type("C", (), {"device_count": lambda: 0})},
                    )
                },
            )
        return type("FakeModule", (), {})

    monkeypatch.setattr("aic2026.ocr_generation.import_module", fake_import)
    engine = create_ocr_engine("paddleocr", device="cpu", languages=("vi", "en"))
    assert isinstance(engine, PaddleOcrEngine)
    assert engine.device == "cpu"
    assert engine.languages == ("vi", "en")
    assert calls == [
        {
            "lang": "vi",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
    ]


def test_create_ocr_engine_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported OCR backend"):
        create_ocr_engine("bogus-backend")


def test_generate_ocr_rejects_unknown_backend(keyframes: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported OCR backend"):
        generate_ocr(
            keyframes,
            tmp_path / "out",
            ocr_engine=_StubOcrEngine(),
            backend="bogus-backend",
        )


def test_paddle_engine_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr("aic2026.ocr_generation.import_module", fake_import)
    with pytest.raises(PaddleOcrRuntimeError, match="missing packages"):
        PaddleOcrEngine._load_dependencies()


def test_paddle_engine_coerce_result() -> None:
    class _FakeResult(dict):
        pass

    result = _FakeResult(
        rec_texts=["PHƯƠNG TRÌNH", "", "DAO ĐỘNG"],
        rec_scores=[0.98, 0.9, 0.96],
        rec_boxes=[[1, 2, 30, 40], [0, 0, 0, 0], [50, 60, 200, 220]],
    )
    lines = PaddleOcrEngine._coerce_result(result)
    assert [line.text for line in lines] == ["PHƯƠNG TRÌNH", "DAO ĐỘNG"]
    assert lines[0].confidence == 0.98
    assert lines[0].bbox == (1.0, 2.0, 30.0, 40.0)
    assert PaddleOcrEngine._coerce_result(None) == []
    assert PaddleOcrEngine._coerce_result({"rec_texts": []}) == []


def test_paddle_engine_coerce_bbox_from_polys() -> None:
    class _FakeResult(dict):
        pass

    result = _FakeResult(
        rec_texts=["x"],
        rec_scores=[0.5],
        rec_polys=[[[1, 2], [10, 2], [10, 40], [1, 40]]],
    )
    lines = PaddleOcrEngine._coerce_result(result)
    assert lines[0].bbox == (1.0, 2.0, 10.0, 40.0)


# ------------------------------------------------------------------ preprocessing


def test_preprocess_image_rgb_and_options(tmp_path: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (100, 50), color=(10, 20, 30))
    rgb = preprocess_image(img)
    assert rgb.mode == "RGB"
    assert rgb.size == (100, 50)
    upscaled = preprocess_image(img, upscale_factor=2.0)
    assert upscaled.size == (200, 100)
    contrasted = preprocess_image(img, contrast=1.5)
    assert contrasted.size == (100, 50)
    sharpened = preprocess_image(img, sharpen=True)
    assert sharpened.size == (100, 50)


def test_preprocess_image_requires_positive_factors(tmp_path: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="upscale_factor"):
        preprocess_image(img, upscale_factor=0.0)
    with pytest.raises(ValueError, match="contrast"):
        preprocess_image(img, contrast=0.0)


# ------------------------------------------------------------------ CLI


def test_cli_rejects_missing_frames(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "cli-out"

    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_ocr.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--keyframes-dir",
            str(empty),
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "error" in result.stderr.lower()


def test_parse_languages_cli(tmp_path: Path) -> None:
    from scripts.generate_ocr import _parse_languages

    assert _parse_languages("vi,en") == ("vi", "en")
    assert _parse_languages(" en ") == ("en",)
    with pytest.raises(ValueError, match="at least one language"):
        _parse_languages(" , ")  # type: ignore[arg-type]


def test_checkpoint_round_trip() -> None:
    ckpt = OcrCheckpoint(
        completed_record_count=3,
        completed_video_ids=("L21_V001",),
        backend="paddleocr",
        model="paddleocr:PP-OCRv5_mobile",
        languages=("vi", "en"),
        device="cpu",
        batch_size=16,
        source_frames="/tmp/frames.ndjson",
        total_records=6,
        timestamp="2026-08-20T00:00:00+00:00",
    )
    assert OcrCheckpoint.from_dict(ckpt.to_dict()) == ckpt


def test_manifest_round_trip(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "ocr-manifest"
    _run(keyframes, out)
    manifest = OcrManifest.from_dict(
        json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    )
    assert manifest.to_dict() == json.loads(
        (out / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
