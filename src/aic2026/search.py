"""Concrete text-to-candidates search facade over one index artifact.

``SearchEngine`` bundles the pieces every task needs to turn query text into
ranked frame candidates:

    TextQuery / str
        -> encoder (EmbeddingEncoder contract)
        -> embedding-space compatibility gate (once, at construction)
        -> IndexRetriever.search

It is the shared retrieval boundary used by the KIS / QA / TRAKE runners and
the single-query task functions; the future API layer wraps exactly this
object. It performs no I/O beyond loading the index and writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aic2026.config import Config
from aic2026.embeddings import Embedding, EmbeddingEncoder
from aic2026.index_artifact import IndexManifest
from aic2026.retrieval import IndexRetriever, RetrievalCandidate
from aic2026.task_support import require_compatible_spaces, search_candidates


class SearchEngine:
    """One encoder bound to one index artifact for candidate retrieval.

    Expected input:
        - encoder: text encoder implementing the embedding contract.
        - retriever: an :class:`aic2026.retrieval.IndexRetriever` over one
          validated index artifact.
        - top_k: default retrieval cutoff for searches on this engine.
        - allow_embedding_space_mismatch: explicit opt-in override for
          legacy indexes only; disabled by default.

    Raises:
        - ValueError: if the encoder and index embedding spaces are
          incompatible and no override was requested, or if ``top_k`` is not
          a positive integer.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        retriever: IndexRetriever,
        *,
        top_k: int = 100,
        allow_embedding_space_mismatch: bool = False,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self._encoder = encoder
        self._retriever = retriever
        self._top_k = top_k
        self._allow_mismatch = bool(allow_embedding_space_mismatch)

        spaces_compatible = encoder.embedding_space.is_compatible_with(
            retriever.manifest.embedding_space
        )
        override_used = False
        if not spaces_compatible:
            if not allow_embedding_space_mismatch:
                require_compatible_spaces(encoder.embedding_space, retriever.manifest)
            override_used = True
        self._spaces_compatible = spaces_compatible
        self._override_used = override_used

    @classmethod
    def from_parts(
        cls,
        encoder: EmbeddingEncoder,
        index_artifact_dir: Path | str,
        *,
        top_k: int = 100,
        allow_embedding_space_mismatch: bool = False,
    ) -> "SearchEngine":
        """Build an engine from an encoder and an index artifact directory.

        Expected input:
            - encoder: text encoder implementing the embedding contract.
            - index_artifact_dir: path to one validated index artifact.
            - top_k / allow_embedding_space_mismatch: see
              :meth:`__init__`.

        Returns:
            - A ready-to-use :class:`SearchEngine`.

        Raises:
            - FileNotFoundError: if the index artifact is absent.
            - ValueError: for an invalid index artifact or incompatible
              embedding spaces under strict mode.
        """
        return cls(
            encoder,
            IndexRetriever(Path(index_artifact_dir)),
            top_k=top_k,
            allow_embedding_space_mismatch=allow_embedding_space_mismatch,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        index_artifact_dir: Path | str,
        *,
        top_k: int = 100,
        allow_embedding_space_mismatch: bool = False,
    ) -> "SearchEngine":
        """Build an engine using the active model section of a config.

        The active model section is ``models.jina`` when present, otherwise
        the config model section (production default: jina-clip-v2).

        Expected input:
            - config: effective configuration (see
              :func:`aic2026.config.load_config`).
            - index_artifact_dir / top_k / allow_embedding_space_mismatch:
              see :meth:`from_parts`.

        Returns:
            - A ready-to-use :class:`SearchEngine`.

        Raises:
            - RuntimeError: if the selected backend's optional dependencies
              or model checkpoint cannot be loaded.
        """
        from aic2026.image_embedder import create_text_encoder

        return cls.from_parts(
            create_text_encoder(config),
            index_artifact_dir,
            top_k=top_k,
            allow_embedding_space_mismatch=allow_embedding_space_mismatch,
        )

    @property
    def encoder(self) -> EmbeddingEncoder:
        """The bound text encoder."""
        return self._encoder

    @property
    def retriever(self) -> IndexRetriever:
        """The loaded index retriever."""
        return self._retriever

    @property
    def manifest(self) -> IndexManifest:
        """The index artifact manifest."""
        return self._retriever.manifest

    @property
    def top_k(self) -> int:
        """The default retrieval cutoff."""
        return self._top_k

    @property
    def embedding_space(self) -> Any:
        """The encoder's declared embedding space."""
        return self._encoder.embedding_space

    @property
    def spaces_compatible(self) -> bool:
        """Whether encoder and index spaces were compatible at construction."""
        return self._spaces_compatible

    @property
    def override_used(self) -> bool:
        """Whether the explicit mismatch override was applied."""
        return self._override_used

    def encode_query(self, text: str) -> Embedding:
        """Encode one query text into the engine's embedding space.

        Expected input:
            - text: non-empty query text.

        Returns:
            - The text embedding with its declared space metadata.

        Raises:
            - ValueError: per the encoder contract for invalid text.
        """
        return self._encoder.encode_text(text)

    def search_embedding(
        self,
        query_embedding: Embedding,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        """Rank frame candidates for one already-encoded query.

        Expected input:
            - query_embedding: embedding produced by this engine's encoder.
            - top_k: optional per-call cutoff; defaults to the engine's.

        Returns:
            - Ranked retrieval candidates in descending score order.

        Raises:
            - ValueError: for dimension/space mismatches (or non-legacy
              indexes when an override was configured).
        """
        return tuple(
            search_candidates(
                self._retriever,
                query_embedding,
                top_k=top_k if top_k is not None else self._top_k,
                allow_mismatch=self._allow_mismatch,
            )
        )

    def search_text(self, text: str, *, top_k: int | None = None) -> tuple[RetrievalCandidate, ...]:
        """Encode one query text and rank frame candidates for it.

        Expected input:
            - text: non-empty query text.
            - top_k: optional per-call cutoff; defaults to the engine's.

        Returns:
            - Ranked retrieval candidates in descending score order.

        Raises:
            - ValueError: for invalid text or incompatible spaces under
              strict mode.
        """
        return self.search_embedding(self.encode_query(text), top_k=top_k)

    def search_similar(
        self, video_id: str, frame_index: int, *, top_k: int | None = None
    ) -> tuple[RetrievalCandidate, ...]:
        """Find visually similar keyframes using the embedding of an indexed frame."""
        vec = self._retriever.get_frame_embedding(video_id, frame_index)
        if vec is None:
            raise ValueError(f"Frame {video_id}:{frame_index} not found in index")
        emb = Embedding(
            kind="image",
            vector=np.asarray(vec, dtype=np.float32),
            space=self._retriever.manifest.embedding_space,
        )
        return self.search_embedding(emb, top_k=top_k)
