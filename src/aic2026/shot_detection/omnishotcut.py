"""OmniShotCut-based shot detector.

Wraps the OmniShotCut shot-cut model (UVA-Computer-Vision-Lab/OmniShotCut) in
a device-aware detector. The upstream ``omnishotcut.engine`` hardcodes CUDA in
both model loading and inference, so this wrapper replicates the windowed
inference loop on the resolved device (``cuda``, ``mps``, or ``cpu``) using the
library's building blocks (backbone, transformer, split/merge helpers). The
model is imported and loaded lazily so importing this module never requires
torch or the OmniShotCut package.

OmniShotCut's ``inference`` (``_run_on_numpy``) returns half-open
``[start, end)`` ranges; this wrapper converts them to the package's inclusive
:class:`Shot` convention (``end_frame = end - 1``). The ``clean_shot`` mode
keeps only general-cut shots (the mode used by the adaptive sampler).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aic2026.image_embedder import resolve_device
from aic2026.shot_detection import Shot, ShotDetectionError

logger = logging.getLogger(__name__)


class OmniShotCutShotDetector:
    """Detect shots in a video with the OmniShotCut model.

    Expected input:
        - checkpoint: local checkpoint path or HuggingFace repo id
          (default ``uva-cv-lab/OmniShotCut``).
        - filename: checkpoint filename inside a repo id.
        - mode: ``clean_shot`` (general cuts only) or ``default`` (all shots).
        - overlap: number of overlap frames between inference windows.
        - device: ``auto``, ``cpu``, ``mps``, or ``cuda``.

    Returns:
        - A shot detector instance.

    Raises:
        - ValueError: on invalid parameters.
    """

    name: str = "omnishotcut"

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        filename: str = "OmniShotCut_ckpt.pth",
        mode: str = "clean_shot",
        overlap: int = 20,
        device: str = "auto",
    ) -> None:
        if mode not in ("clean_shot", "default"):
            raise ValueError(f"unsupported OmniShotCut mode {mode!r}; expected clean_shot or default")
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}")
        if not isinstance(filename, str) or filename == "":
            raise ValueError("filename must be a non-empty string")
        if checkpoint is not None and not isinstance(checkpoint, str):
            raise ValueError("checkpoint must be a string path or repo id")
        self._checkpoint = checkpoint
        self._filename = filename
        self._mode = mode
        self._overlap = overlap
        self._device = device
        self._model = None  # (model, model_args) cached after first load
        self._model_load_count = 0
        self._last_fps: float | None = None
        self._last_num_frames: int | None = None

    @property
    def last_fps(self) -> float | None:
        """Return the frame rate of the most recently detected video."""
        return self._last_fps

    @property
    def last_num_frames(self) -> int | None:
        """Return the frame count of the most recently detected video."""
        return self._last_num_frames

    @property
    def model_load_count(self) -> int:
        """Return how many times the OmniShotCut model has been loaded.

        A value of ``1`` across many detected videos proves the model is
        loaded once per detector and reused, not reloaded per video.
        """
        return self._model_load_count

    def load(self) -> str:
        """Load the model eagerly and return the resolved device.

        Expected input:
            - None.

        Returns:
            - The concrete device the model was loaded on (``cuda``, ``mps``,
              or ``cpu``).

        Raises:
            - ShotDetectionError: if the checkpoint is invalid or the model
              cannot be loaded.
        """
        self._ensure_model()
        return resolve_device(self._device)

    def detect(self, video_path: Path | str) -> list[Shot]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {path}")
        started = time.monotonic()
        model, model_args = self._ensure_model()
        import numpy as np

        video_np = self._decode_video(path, model_args)
        num_frames = int(len(video_np))
        fps = self._probe_fps(path)
        self._last_fps = fps
        self._last_num_frames = num_frames

        ranges, intra_labels, _inter_labels = self._infer_ranges(video_np, model, model_args)
        if self._mode == "clean_shot":
            ranges = self._filter_clean_shot(ranges, intra_labels)
        if not ranges:
            logger.warning("OmniShotCut found no shots in %s", path)
            return []
        shots = [Shot(start_frame=int(start), end_frame=int(end) - 1) for start, end in ranges]
        logger.info(
            "OmniShotCut detected %d shots in %s (%d frames, %.1fs)",
            len(shots),
            path.name,
            num_frames,
            time.monotonic() - started,
        )
        return shots

    # ------------------------------------------------------------------ helpers

    def _resolve_checkpoint_path(self) -> str:
        """Return the local checkpoint path, downloading from HF when needed."""
        import os

        repo_or_path = self._checkpoint if self._checkpoint is not None else "uva-cv-lab/OmniShotCut"
        if os.path.exists(repo_or_path):
            return repo_or_path
        from huggingface_hub import hf_hub_download

        logger.info("downloading OmniShotCut checkpoint %s/%s", repo_or_path, self._filename)
        return hf_hub_download(repo_id=repo_or_path, filename=self._filename)

    def _ensure_model(self) -> tuple[object, object]:
        """Load and cache the OmniShotCut model on the resolved device."""
        if self._model is not None:
            return self._model
        import torch

        device = resolve_device(self._device)
        checkpoint_path = self._resolve_checkpoint_path()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "args" not in state_dict or "model" not in state_dict:
            raise ShotDetectionError(
                f"OmniShotCut checkpoint {checkpoint_path} must contain 'args' and 'model'"
            )
        model_args = state_dict["args"]

        from omnishotcut.architecture.backbone import build_backbone
        from omnishotcut.architecture.model import OmniShotCut
        from omnishotcut.architecture.transformer import build_transformer

        backbone = build_backbone(model_args)
        transformer = build_transformer(model_args)
        model = OmniShotCut(
            backbone,
            transformer,
            num_intra_relation_classes=model_args.num_intra_relation_classes,
            num_inter_relation_classes=model_args.num_inter_relation_classes,
            num_frames=model_args.max_process_window_length,
            num_queries=model_args.num_queries,
            aux_loss=model_args.aux_loss,
        )
        model.load_state_dict(state_dict["model"], strict=True)
        model.to(device)
        model.eval()
        self._model = (model, model_args)
        self._model_load_count += 1
        logger.info("OmniShotCut model loaded on device=%s (load #%d)", device, self._model_load_count)
        return self._model

    def _decode_video(self, path: Path, model_args: object) -> object:
        """Decode the video to a ``(T, H, W, 3)`` uint8 numpy array."""
        from omnishotcut.datasets.utils import _decode_video

        try:
            return _decode_video(
                str(path),
                model_args.process_width,
                model_args.process_height,
            )
        except (RuntimeError, ValueError) as exc:
            raise ShotDetectionError(f"failed to decode video {path}: {exc}") from exc

    def _probe_fps(self, path: Path) -> float:
        """Probe the video frame rate via ffprobe."""
        from omnishotcut.datasets.utils import _video_fps

        try:
            return float(_video_fps(str(path)))
        except Exception as exc:
            raise ShotDetectionError(f"failed to probe fps for {path}: {exc}") from exc

    def _infer_ranges(
        self,
        video_np: object,
        model: object,
        model_args: object,
    ) -> tuple[list[list[int]], list[int], list[int]]:
        """Run windowed inference, returning half-open ``[start, end)`` ranges.

        Replicates ``omnishotcut.engine._run_on_numpy`` on the resolved device
        (the upstream function hardcodes ``cuda``).
        """
        import torch

        from omnishotcut.engine import merge_predictions, split_videos
        from omnishotcut.datasets.transforms import Video_Augmentation_Transform

        device = resolve_device(self._device)
        max_window = model_args.max_process_window_length
        transform = Video_Augmentation_Transform(set_type="val")

        pred_boundary_full: list[dict[str, int]] = []
        for (
            video_chunk,
            _num_pad_frames,
            window_start_idx,
            valid_start_idx,
            valid_end_idx,
            valid_len,
        ) in split_videos(video_np, max_window, self._overlap):
            video_tensor = transform(video_chunk).unsqueeze(0).to(device)
            with torch.inference_mode():
                outputs = model(video_tensor)

            probas_intra = outputs["intra_clip_logits"].softmax(-1)[0, :, :-1]
            probas_inter = outputs["inter_clip_logits"].softmax(-1)[0, :, :-1]
            range_probas = outputs["pred_shot_logits"].softmax(-1)[0, :, :-1]
            query_intra_idx = probas_intra.argmax(dim=-1)
            query_inter_idx = probas_inter.argmax(dim=-1)
            query_range_idx = range_probas.argmax(dim=-1)

            pred_boundary: list[dict[str, int]] = []
            start_frame_idx_local = 0
            for keep_idx in range(len(query_intra_idx)):
                pred_intra_label = int(query_intra_idx[keep_idx].detach().cpu())
                pred_inter_label = int(query_inter_idx[keep_idx].detach().cpu())
                end_frame_idx_local = int(query_range_idx[keep_idx].detach().cpu())
                end_frame_idx_local = min(end_frame_idx_local, valid_len)
                if start_frame_idx_local >= end_frame_idx_local:
                    continue
                end_frame_idx_global = window_start_idx + end_frame_idx_local
                if valid_start_idx < end_frame_idx_global <= valid_end_idx:
                    pred_boundary.append(
                        {
                            "end_frame_idx": int(end_frame_idx_global),
                            "intra_label": int(pred_intra_label),
                            "inter_label": int(pred_inter_label),
                        }
                    )
                start_frame_idx_local = end_frame_idx_local
                if end_frame_idx_local >= valid_len:
                    break

            pred_boundary_full = merge_predictions(pred_boundary_full, pred_boundary)

        ranges: list[list[int]] = []
        intra_labels: list[int] = []
        inter_labels: list[int] = []
        start_frame_idx = 0
        for item in pred_boundary_full:
            end_frame_idx = min(int(item["end_frame_idx"]), len(video_np))
            if end_frame_idx <= start_frame_idx:
                continue
            ranges.append([int(start_frame_idx), int(end_frame_idx)])
            intra_labels.append(int(item["intra_label"]))
            inter_labels.append(int(item["inter_label"]))
            start_frame_idx = end_frame_idx
        return ranges, intra_labels, inter_labels

    def _filter_clean_shot(self, ranges: list[list[int]], intra_labels: list[int]) -> list[list[int]]:
        """Keep only ranges whose intra label is ``general`` (clean shot)."""
        from omnishotcut.label_correspondence import unique_intra_label_mapping

        general_idx = unique_intra_label_mapping["general"]
        return [r for r, label in zip(ranges, intra_labels) if label == general_idx]
