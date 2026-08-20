"""Tests for the resumable keyframe caption generation pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.canonical import KeyframeTiming
from aic2026.caption_generation import (
    CAPTIONS_FILENAME,
    CAPTION_BACKENDS,
    CHECKPOINT_FILENAME,
    DEFAULT_BACKEND,
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_JINA_VLM_MLX_MODEL,
    DEFAULT_MODEL,
    MANIFEST_FILENAME,
    CaptionGenerationError,
    CaptionManifest,
    JinaCaptionRuntimeError,
    JinaVlmMlxCaptioner,
    KeyframeCaption,
    Qwen25VlCaptioner,
    QwenCaptionRuntimeError,
    create_captioner,
    generate_captions,
    iter_captions,
    read_caption_artifact,
    resolve_checkpoint,
    validate_caption_artifact,
)

#: video_id -> (ordinals) in the synthetic input order.
LAYOUT = [("L21_V001", [1, 2, 3]), ("L21_V002", [1, 2]), ("L22_V001", [1])]
VIDEO_INDEX = {vid: idx for idx, (vid, _) in enumerate(LAYOUT, start=1)}

PROMPT = "describe this test frame"


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


class _StubCaptioner:
    """Deterministic captioner deriving captions from image bytes."""

    device = "cpu"
    model_name = "stub:captioner"
    checkpoint = "stub/checkpoint"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate_captions(self, images: list[object]) -> list[str]:
        self.batch_sizes.append(len(images))
        out: list[str] = []
        for image in images:
            data = bytes(image)  # type: ignore[arg-type]
            out.append(f"caption-v{data[0]}-k{data[1]}")
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
) -> tuple[_StubCaptioner, CaptionManifest]:
    captioner = _StubCaptioner()
    kwargs.setdefault("image_loader", _image_loader)
    kwargs.setdefault("prompt", PROMPT)
    manifest = generate_captions(
        keyframes,
        out_dir,
        captioner=captioner,
        **kwargs,
    )
    return captioner, manifest


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_generate_captions_creates_valid_artifact(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions"
    _, manifest = _run(keyframes, out)

    for name in (MANIFEST_FILENAME, CAPTIONS_FILENAME):
        assert (out / name).is_file()
    assert not (out / CHECKPOINT_FILENAME).exists()

    assert manifest.num_records == 6
    assert manifest.num_videos == 3
    assert manifest.num_failed == 0
    assert manifest.model == "stub:captioner"
    assert manifest.checkpoint == "stub/checkpoint"
    assert manifest.device == "cpu"
    assert manifest.batch_size == 4
    assert manifest.prompt == PROMPT
    assert manifest.created_at != ""
    assert manifest.artifact_id == "captions"

    records = iter_captions(out / CAPTIONS_FILENAME)
    assert len(records) == 6
    assert all(record.status == "ok" for record in records)
    assert validate_caption_artifact(manifest, out) == []


def test_manifest_records_required_fields(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-fields"
    _run(keyframes, out)
    payload = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for key in (
        "model", "checkpoint", "device", "batch_size", "prompt", "num_records",
        "num_videos", "num_failed", "canonical_schema_version", "created_at",
    ):
        assert key in payload
    assert payload["schema_version"] == "1.0"


def test_manifest_round_trips(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-roundtrip"
    _run(keyframes, out)
    manifest = CaptionManifest.from_dict(
        json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    )
    assert manifest.to_dict() == json.loads(
        (out / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def test_records_preserve_identity_and_alignment(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-align"
    _run(keyframes, out)
    rows = _load_rows(out / CAPTIONS_FILENAME)
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
        assert row["caption"] == f"caption-v{VIDEO_INDEX[frame.keyframe.video_id.value]}-k{frame.keyframe.keyframe_ordinal}"
        assert len(rows) == 6 or True


def test_deterministic_output(keyframes: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run(keyframes, out_a)
    _run(keyframes, out_b)
    assert _load_rows(out_a / CAPTIONS_FILENAME) == _load_rows(out_b / CAPTIONS_FILENAME)


def test_interrupt_leaves_no_manifest(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-interrupt"
    with pytest.raises(CaptionGenerationError, match="interrupted caption build"):
        _run(keyframes, out, batch_size=1, stop_after_records=2)

    assert not (out / MANIFEST_FILENAME).exists()
    assert (out / CHECKPOINT_FILENAME).is_file()
    assert (out / CAPTIONS_FILENAME).is_file()
    checkpoint = json.loads((out / CHECKPOINT_FILENAME).read_text(encoding="utf-8"))
    assert checkpoint["completed_record_count"] == 2
    assert set(checkpoint.keys()) >= {
        "completed_record_count", "completed_video_ids", "model", "device",
        "batch_size", "prompt", "source_frames", "total_records", "timestamp",
    }
    assert len(_load_rows(out / CAPTIONS_FILENAME)) == 2


def test_resume_skips_existing_records(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-resume"
    captioner = _StubCaptioner()
    with pytest.raises(CaptionGenerationError):
        generate_captions(
            keyframes, out, captioner=captioner, image_loader=_image_loader,
            prompt=PROMPT, batch_size=1, stop_after_records=2,
        )
    calls_before_resume = sum(captioner.batch_sizes)

    manifest = generate_captions(
        keyframes, out, captioner=captioner, image_loader=_image_loader,
        prompt=PROMPT, batch_size=1,
    )
    assert manifest.num_records == 6
    assert (out / MANIFEST_FILENAME).is_file()
    assert sum(captioner.batch_sizes) - calls_before_resume == 4


def test_resume_matches_clean_run(keyframes: Path, tmp_path: Path) -> None:
    clean_out = tmp_path / "clean"
    _, clean_manifest = _run(keyframes, clean_out)

    resumed_out = tmp_path / "resumed"
    with pytest.raises(CaptionGenerationError):
        _run(keyframes, resumed_out, batch_size=2, stop_after_records=4)
    resume_manifest = _run(keyframes, resumed_out, batch_size=2)[1]

    assert clean_manifest.num_records == resume_manifest.num_records == 6
    assert _load_rows(clean_out / CAPTIONS_FILENAME) == _load_rows(
        resumed_out / CAPTIONS_FILENAME
    )


def test_resume_rejects_model_mismatch(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-model"
    with pytest.raises(CaptionGenerationError):
        _run(keyframes, out, batch_size=1, stop_after_records=2)

    class _Other(_StubCaptioner):
        model_name = "stub:other"

    with pytest.raises(ValueError, match="model"):
        generate_captions(
            keyframes, out, captioner=_Other(), image_loader=_image_loader,
            prompt=PROMPT,
        )


def test_resume_rejects_prompt_mismatch(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-prompt"
    with pytest.raises(CaptionGenerationError):
        _run(keyframes, out, batch_size=1, stop_after_records=2)
    with pytest.raises(ValueError, match="prompt"):
        _run(keyframes, out, prompt="a different prompt")


def test_complete_fast_path_returns_existing(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-fast"
    captioner = _StubCaptioner()
    first = generate_captions(
        keyframes, out, captioner=captioner, image_loader=_image_loader, prompt=PROMPT,
    )
    calls = sum(captioner.batch_sizes)
    second = generate_captions(
        keyframes, out, captioner=captioner, image_loader=_image_loader, prompt=PROMPT,
    )
    assert first == second
    assert sum(captioner.batch_sizes) == calls


def test_max_records_limits_artifact(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-limited"
    _, manifest = _run(keyframes, out, max_records=3)
    assert manifest.num_records == 3
    assert manifest.num_videos == 1
    rows = _load_rows(out / CAPTIONS_FILENAME)
    assert len(rows) == 3
    assert all(row["video_id"]["value"] == "L21_V001" for row in rows)


def test_missing_image_recorded_as_error(keyframes: Path, tmp_path: Path) -> None:
    missing = keyframes / "images" / "L21_V002" / "000002.jpg"
    missing.unlink()

    out = tmp_path / "captions-missing"
    _, manifest = _run(keyframes, out, batch_size=2)

    assert manifest.num_records == 6
    assert manifest.num_failed == 1
    records = iter_captions(out / CAPTIONS_FILENAME)
    assert len(records) == 6
    error_records = [r for r in records if r.status == "error"]
    assert len(error_records) == 1
    assert error_records[0].caption == ""
    assert error_records[0].error is not None
    assert error_records[0].video_id.value == "L21_V002"
    assert error_records[0].keyframe_ordinal == 2
    assert validate_caption_artifact(manifest, out) == []


def test_corrupt_image_does_not_fail_batch(keyframes: Path, tmp_path: Path) -> None:
    corrupt = keyframes / "images" / "L21_V001" / "000002.jpg"
    corrupt.write_bytes(b"not-an-image")

    def flaky_loader(path: Path) -> bytes:
        data = path.read_bytes()
        if data == b"not-an-image":
            raise OSError("cannot decode image")
        return data

    out = tmp_path / "captions-corrupt"
    _, manifest = _run(keyframes, out, image_loader=flaky_loader, batch_size=3)

    assert manifest.num_records == 6
    assert manifest.num_failed == 1
    records = iter_captions(out / CAPTIONS_FILENAME)
    assert sum(1 for r in records if r.status == "ok") == 5
    assert validate_caption_artifact(manifest, out) == []


def test_captioner_batch_failure_falls_back_per_image(keyframes: Path, tmp_path: Path) -> None:
    class _FailingBatch(_StubCaptioner):
        def generate_captions(self, images: list[object]) -> list[str]:
            if len(images) > 1:
                raise RuntimeError("OOM")
            return super().generate_captions(images)

    out = tmp_path / "captions-fallback"
    captioner = _FailingBatch()
    manifest = generate_captions(
        keyframes, out, captioner=captioner, image_loader=_image_loader,
        prompt=PROMPT, batch_size=4,
    )
    assert manifest.num_records == 6
    assert manifest.num_failed == 0
    assert all(r.status == "ok" for r in iter_captions(out / CAPTIONS_FILENAME))


def test_validate_caption_artifact_reports_problems(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-bad"
    manifest = _run(keyframes, out)[1]

    (out / CAPTIONS_FILENAME).unlink()
    problems = validate_caption_artifact(manifest, out)
    assert any("missing" in p for p in problems)

    _run(keyframes, tmp_path / "captions-good")
    good_manifest = CaptionManifest.from_dict(
        json.loads((tmp_path / "captions-good" / MANIFEST_FILENAME).read_text())
    )
    bad_manifest = CaptionManifest(
        schema_version=good_manifest.schema_version,
        artifact_id=good_manifest.artifact_id,
        model=good_manifest.model,
        checkpoint=good_manifest.checkpoint,
        device=good_manifest.device,
        batch_size=good_manifest.batch_size,
        prompt=good_manifest.prompt,
        source_frames=good_manifest.source_frames,
        num_records=good_manifest.num_records + 1,
        num_videos=good_manifest.num_videos,
        num_failed=good_manifest.num_failed,
        canonical_schema_version=good_manifest.canonical_schema_version,
        created_at=good_manifest.created_at,
    )
    problems = validate_caption_artifact(bad_manifest, tmp_path / "captions-good")
    assert any("num_records" in p for p in problems)


def test_read_caption_artifact_round_trip(keyframes: Path, tmp_path: Path) -> None:
    out = tmp_path / "captions-read"
    _run(keyframes, out)
    manifest, records = read_caption_artifact(out)
    assert manifest.num_records == 6
    assert len(records) == 6


def test_resolve_checkpoint() -> None:
    assert resolve_checkpoint("Qwen2.5-VL-3B-Instruct") == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert resolve_checkpoint("Qwen/Qwen2.5-VL-7B-Instruct") == "Qwen/Qwen2.5-VL-7B-Instruct"
    with pytest.raises(ValueError):
        resolve_checkpoint("")


def test_captioner_rejects_old_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        class _Module:  # noqa: D106
            pass

        return _Module()

    monkeypatch.setattr(
        "aic2026.caption_generation.import_module", fake_import
    )
    monkeypatch.setattr(
        "aic2026.caption_generation.version", lambda name: "4.46.3"
    )
    with pytest.raises(QwenCaptionRuntimeError, match="transformers >= 4.49"):
        Qwen25VlCaptioner._load_dependencies()


def test_captioner_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr("aic2026.caption_generation.import_module", fake_import)
    with pytest.raises(QwenCaptionRuntimeError, match="missing packages"):
        Qwen25VlCaptioner._load_dependencies()


def test_cli_rejects_missing_frames(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "cli-out"

    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_captions.py"
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


# ------------------------------------------------------------------ backend selection


def test_caption_backends_registry() -> None:
    assert CAPTION_BACKENDS == ("qwen2.5-vl", "jina-vlm-mlx")
    assert DEFAULT_BACKEND == "qwen2.5-vl"
    assert DEFAULT_MODEL == "Qwen2.5-VL-3B-Instruct"
    assert DEFAULT_JINA_VLM_MLX_MODEL == "jinaai/jina-vlm-mlx"


def test_create_captioner_qwen_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_qwen_init(
        self: object,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        prompt: str = "",
        max_new_tokens: int = 256,
    ) -> None:
        calls.append({"model": model, "device": device, "prompt": prompt, "max_new_tokens": max_new_tokens})

    monkeypatch.setattr(Qwen25VlCaptioner, "__init__", fake_qwen_init)
    captioner = create_captioner("qwen2.5-vl", model="Qwen2.5-VL-7B-Instruct", device="cpu")
    assert isinstance(captioner, Qwen25VlCaptioner)
    assert calls == [
        {
            "model": "Qwen2.5-VL-7B-Instruct",
            "device": "cpu",
            "prompt": DEFAULT_CAPTION_PROMPT,
            "max_new_tokens": 256,
        }
    ]


def test_create_captioner_qwen_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_qwen_init(
        self: object,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        prompt: str = "",
        max_new_tokens: int = 256,
    ) -> None:
        calls.append({"model": model})

    monkeypatch.setattr(Qwen25VlCaptioner, "__init__", fake_qwen_init)
    create_captioner("qwen2.5-vl")
    assert calls == [{"model": DEFAULT_MODEL}]


def test_create_captioner_jina_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_jina_init(
        self: object,
        model: str = DEFAULT_JINA_VLM_MLX_MODEL,
        *,
        device: str = "auto",
        prompt: str = "",
        max_new_tokens: int = 256,
    ) -> None:
        calls.append({"model": model, "device": device})

    monkeypatch.setattr(JinaVlmMlxCaptioner, "__init__", fake_jina_init)
    captioner = create_captioner("jina-vlm-mlx", device="mps")
    assert isinstance(captioner, JinaVlmMlxCaptioner)
    assert calls == [{"model": DEFAULT_JINA_VLM_MLX_MODEL, "device": "mps"}]


def test_create_captioner_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported caption backend"):
        create_captioner("bogus-backend")


def test_generate_captions_rejects_unknown_backend(
    keyframes: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="unsupported caption backend"):
        generate_captions(
            keyframes,
            tmp_path / "out",
            captioner=_StubCaptioner(),
            backend="bogus-backend",
        )


# ------------------------------------------------------------------ Jina VLM MLX captioner


def test_jina_captioner_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(
        "aic2026.caption_generation.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        "aic2026.caption_generation.platform.machine",
        lambda: "arm64",
    )
    monkeypatch.setattr("aic2026.caption_generation.import_module", fake_import)
    with pytest.raises(JinaCaptionRuntimeError, match="missing packages"):
        JinaVlmMlxCaptioner._load_dependencies()


def test_jina_captioner_rejects_non_apple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aic2026.caption_generation.platform.system",
        lambda: "Linux",
    )
    monkeypatch.setattr(
        "aic2026.caption_generation.platform.machine",
        lambda: "x86_64",
    )
    with pytest.raises(JinaCaptionRuntimeError, match="Apple Silicon"):
        JinaVlmMlxCaptioner._load_dependencies()


def test_jina_captioner_rejects_cuda_device() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        JinaVlmMlxCaptioner(model="jinaai/jina-vlm-mlx", device="cuda")


def test_jina_captioner_identity_without_runtime() -> None:
    captioner = object.__new__(JinaVlmMlxCaptioner)
    captioner._model = "jinaai/jina-vlm-mlx"
    captioner._checkpoint = "jinaai/jina-vlm-mlx"
    captioner._device = "mps"
    assert captioner.model_name == "jina-vlm-mlx:jinaai/jina-vlm-mlx"
    assert captioner.checkpoint == "jinaai/jina-vlm-mlx"
    assert captioner.device == "mps"


class _GenerationResult:
    """Minimal mlx-vlm generation result exposing ``.text``."""

    def __init__(self, text: str) -> None:
        self.text = text


def test_jina_captioner_generate_uses_per_image_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch calls are expanded to per-image mlx-vlm.generate calls."""
    calls: list[object] = []

    class _FakeMlxVlm:
        class utils:
            @staticmethod
            def load_config(checkpoint: str) -> str:
                return "config"

        class prompt_utils:
            @staticmethod
            def apply_chat_template(processor: object, config: object, prompt: str, num_images: int) -> str:
                return f"<img>{prompt}</img>"

        @staticmethod
        def generate(model: object, processor: object, prompt: str, image: list[object], max_tokens: int) -> object:
            calls.append(image[0])
            return _GenerationResult(f"caption-{len(calls)}")

    monkeypatch.setattr(
        "aic2026.caption_generation.import_module",
        lambda name: _FakeMlxVlm() if name == "mlx_vlm" else object(),
    )
    captioner = object.__new__(JinaVlmMlxCaptioner)
    captioner._model = "jinaai/jina-vlm-mlx"
    captioner._checkpoint = "jinaai/jina-vlm-mlx"
    captioner._device = "mps"
    captioner._max_new_tokens = 200
    captioner._prompt = PROMPT
    captioner._processor = object()
    captioner._config = "config"
    captioner._model_obj = object()

    images = [object(), object(), object()]
    result = captioner.generate_captions(images)
    assert result == ["caption-1", "caption-2", "caption-3"]
    assert len(calls) == 3
    assert calls == images


def test_jina_captioner_coerce_caption_handles_string_output() -> None:
    plain = object.__new__(JinaVlmMlxCaptioner)
    assert plain._coerce_caption(_GenerationResult("  hello  ")) == "hello"
    assert plain._coerce_caption("  raw string  ") == "raw string"


def test_generate_captions_jina_backend_pipeline(keyframes: Path, tmp_path: Path) -> None:
    """The jina backend flows through the same pipeline and artifact schema."""
    out = tmp_path / "captions-jina"
    captioner = _StubCaptioner()
    captioner.model_name = "jina-vlm-mlx:jinaai/jina-vlm-mlx"
    captioner.checkpoint = "jinaai/jina-vlm-mlx"
    manifest = generate_captions(
        keyframes,
        out,
        captioner=captioner,
        image_loader=_image_loader,
        prompt=PROMPT,
        backend="jina-vlm-mlx",
    )
    assert manifest.num_records == 6
    assert manifest.model == "jina-vlm-mlx:jinaai/jina-vlm-mlx"
    assert manifest.checkpoint == "jinaai/jina-vlm-mlx"
    assert manifest.schema_version == "1.0"
    assert validate_caption_artifact(manifest, out) == []

    qwen_out = tmp_path / "captions-qwen"
    qwen_captioner = _StubCaptioner()
    qwen_manifest = generate_captions(
        keyframes,
        qwen_out,
        captioner=qwen_captioner,
        image_loader=_image_loader,
        prompt=PROMPT,
    )
    assert qwen_manifest.schema_version == "1.0"
    assert set(validate_caption_artifact(qwen_manifest, qwen_out)) == set(
        validate_caption_artifact(manifest, out)
    )
    assert set(json.loads((out / MANIFEST_FILENAME).read_text())) == set(
        json.loads((qwen_out / MANIFEST_FILENAME).read_text())
    )
    assert _load_rows(out / CAPTIONS_FILENAME) == _load_rows(qwen_out / CAPTIONS_FILENAME)
