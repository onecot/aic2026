"""Reusable Jina VLM (MLX) wrapper: image(s) + prompt -> generated text.

This is the small VLM seam shared by QA answer generation and standalone
caption generation. It wraps the local-first MLX port of ``jinaai/jina-vlm``
(``mlx-vlm``, Apple Silicon Metal) behind one concrete class:

    images + text prompt -> generated text

``mlx`` and ``mlx-vlm`` are imported only when the wrapper is constructed,
so importing this module never requires the optional runtime (install with
``pip install -e '.[caption-mlx]'``).
"""

from __future__ import annotations

from importlib import import_module
from platform import machine, system
from typing import Any, Sequence

#: Default Jina VLM MLX checkpoint id (Apple Silicon, 4-bit).
DEFAULT_JINA_VLM_MODEL = "jinaai/jina-vlm-mlx"

#: Runtime packages the Jina VLM MLX wrapper needs when constructed.
REQUIRED_MLX_RUNTIME_PACKAGES = (
    "mlx",
    "mlx_vlm",
    "transformers",
    "PIL",
)

#: Generic fallback instruction used when a caller supplies no prompt.
DEFAULT_VLM_PROMPT = "Describe this image."


class JinaVlmRuntimeError(RuntimeError):
    """Raised when the optional Jina VLM MLX runtime cannot be loaded or used."""


class JinaVlmMlx:
    """Concrete image+prompt -> text generator on Apple Silicon via mlx-vlm.

    Expected input:
        - model: HuggingFace checkpoint id (default
          ``jinaai/jina-vlm-mlx``).
        - device: ``auto`` or ``mps`` (MLX runs on the Metal device; ``cpu``
          and ``cuda`` are rejected with a clear error).
        - max_new_tokens: maximum generated tokens per call.

    Raises:
        - ValueError: for unsupported devices or invalid token limits.
        - JinaVlmRuntimeError: if the host is not Apple Silicon or the
          optional runtime packages are missing.
    """

    def __init__(
        self,
        model: str = DEFAULT_JINA_VLM_MODEL,
        *,
        device: str = "mps",
        max_new_tokens: int = 256,
    ) -> None:
        self._model = model
        self._checkpoint = model
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens!r}")
        self._max_new_tokens = max_new_tokens

        if device not in ("auto", "mps"):
            raise ValueError(
                f"unsupported device {device!r} for jina-vlm-mlx; "
                "MLX runs on Apple Silicon Metal and supports only 'auto' or 'mps'"
            )

        mlx_vlm = self._load_dependencies()
        self._device = "mps"
        self._model_obj, self._processor = mlx_vlm.load(self._checkpoint)
        self._config = mlx_vlm.utils.load_config(self._checkpoint)

    @staticmethod
    def _load_dependencies() -> Any:
        """Import and pre-flight the optional Jina VLM MLX runtime.

        Returns:
            - The imported ``mlx_vlm`` module.

        Raises:
            - JinaVlmRuntimeError: if a required optional dependency is
              missing or the host is not Apple Silicon.
        """
        if system() != "Darwin" or machine() != "arm64":
            raise JinaVlmRuntimeError(
                "Jina VLM MLX runs on Apple Silicon (macOS arm64) only; "
                "no alternative backend is configured"
            )
        missing = []
        for name in REQUIRED_MLX_RUNTIME_PACKAGES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)
        if missing:
            raise JinaVlmRuntimeError(
                "Jina VLM MLX requires the optional 'caption-mlx' dependency "
                f"group; missing packages: {', '.join(missing)}. "
                "Install with pip install -e '.[caption-mlx]'"
            )
        return import_module("mlx_vlm")

    @property
    def device(self) -> str:
        """The concrete runtime device (``mps``)."""
        return self._device

    @property
    def model_name(self) -> str:
        """Stable model identity recorded in manifests."""
        return f"jina-vlm-mlx:{self._model}"

    @property
    def checkpoint(self) -> str:
        """The HuggingFace checkpoint id."""
        return self._checkpoint

    @property
    def max_new_tokens(self) -> int:
        """Maximum generated tokens per call."""
        return self._max_new_tokens

    @property
    def processor(self) -> Any:
        """The loaded tokenizer/processor pair used for prompt formatting."""
        return self._processor

    @property
    def model_config(self) -> Any:
        """The checkpoint configuration loaded by mlx-vlm."""
        return self._config

    @property
    def model_obj(self) -> Any:
        """The loaded MLX model object used for generation."""
        return self._model_obj

    def generate(self, prompt: str, images: Sequence[object] | None = None) -> str:
        """Generate text for one prompt with zero or more input images.

        This is the generic seam: captioning passes one image and a caption
        prompt; QA answering passes evidence frames and an answer prompt.

        Expected input:
            - prompt: fully formatted instruction text.
            - images: backend-native images (e.g. RGB PIL images); may be
              empty/None for text-only generation.

        Returns:
            - The stripped generated text.

        Raises:
            - RuntimeError: if generation fails.
        """
        mlx_vlm = import_module("mlx_vlm")
        formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
            self._processor,
            self._config,
            prompt,
            num_images=len(images) if images else 0,
        )
        output = mlx_vlm.generate(
            self._model_obj,
            self._processor,
            formatted_prompt,
            list(images) if images else None,
            max_tokens=self._max_new_tokens,
        )
        return self.coerce_text(output)

    @staticmethod
    def coerce_text(output: object) -> str:
        """Extract text from an mlx-vlm generation result.

        ``mlx_vlm.generate`` returns an object exposing ``.text`` (or a
        plain string in some versions); both forms are handled defensively.

        Expected input:
            - output: the mlx-vlm generation result.

        Returns:
            - The stripped text.
        """
        text = getattr(output, "text", None)
        if text is None:
            text = str(output)
        return str(text).strip()
