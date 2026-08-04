"""CLIP text encoding for retrieval against organizer-provided image features."""

from __future__ import annotations

import numpy as np
import open_clip
import torch


class TextEncoder:
    """Encode text queries in the CLIP embedding space used by the image index."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        """Load a CLIP text encoder in deterministic inference mode.

        Expected input: ``model_name`` and ``pretrained`` identify an OpenCLIP model;
        they must match the image features used to build the index (ViT-B/32), or
        similarities are meaningless. ``device`` is a supported torch device string.
        Returns: None.
        Raises: RuntimeError if the model cannot be loaded or moved to ``device``.
        """
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode text queries as float32 CLIP embeddings.

        Expected input: ``texts`` is a list of query strings; ``batch_size`` is a
        positive number of queries per inference batch; ``normalize`` controls
        row-wise L2 normalization.
        Returns: A float32 array of shape ``(len(texts), 512)``.
        Raises: ValueError if ``batch_size`` is not positive; RuntimeError if encoding fails.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not texts:
            return np.empty((0, 512), dtype=np.float32)

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[start : start + batch_size]).to(self.device)
                features = self.model.encode_text(tokens)
                batches.append(features.float().cpu().numpy())

        embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        if normalize:
            # Match index normalization so dot products equal cosine similarity.
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings
