"""Deterministic lexical retrieval over per-video media-info text.

This module builds a lightweight BM25 text index over the ``title``,
``keywords``, and ``description`` fields of the BTC ``media-info`` files and
ranks videos for a text query. It is an independent retrieval *branch*: it
does not modify or reuse CLIP vector retrieval, the scorer, evaluation, or
the prediction-artifact schema.

Design:

- one document per video, built from ``media-info/<VIDEO_ID>.json``;
- BM25 (Okapi) scoring with field weighting (title / keywords / description);
- fully deterministic: fixed tokenizer, documents sorted by video id, ties
  resolved by video id, so identical inputs always produce identical output;
- no external libraries beyond the standard library (the raw dataset is read
  directly; no model or API is involved).

A serialized index is a self-describing JSON artifact containing the schema
version, build config, and the source documents, so it can be reloaded and
reproduces identical rankings without re-reading the raw dataset.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from aic2026.contracts import video_id_from_collection_name
from aic2026.entities import parse_media_info

#: Schema version of the serialized media-info index artifact.
MEDIA_INFO_INDEX_SCHEMA_VERSION = 1

#: Whitespace and punctuation separator used by the tokenizer.
_TOKEN_SEPARATOR = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class MediaInfoDocument:
    """Text fields of one video's media-info record.

    Expected input:
        - video_id: video id (``L*_V*``) of the document.
        - title: video title text.
        - keywords: tuple of keyword tags, in file order.
        - description: video description text.

    Returns:
        - A validated media-info document.

    Raises:
        - ValueError: if ``video_id`` is empty or not a string.
    """

    video_id: str
    title: str
    keywords: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ValueError("video_id must be a non-empty string")


@dataclass(frozen=True)
class MediaInfoHit:
    """One ranked video hit from a media-info query.

    Expected input:
        - rank: 1-based rank.
        - video_id: ranked video id.
        - score: BM25 score for the query.

    Returns:
        - A single ranked hit.

    Raises:
        - ValueError: if ``rank`` is not a positive integer.
    """

    rank: int
    video_id: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True)
class MediaInfoRetrievalConfig:
    """Configuration for the media-info BM25 index.

    Expected input:
        - k1: BM25 term-frequency saturation; must be > 0.
        - b: BM25 document-length normalization; must be in ``[0, 1]``.
        - title_weight: multiplier for ``title`` term counts; >= 0.
        - keywords_weight: multiplier for ``keywords`` term counts; >= 0.
        - description_weight: multiplier for ``description`` term counts; >= 0.
          At least one field weight must be > 0.

    Returns:
        - A validated BM25 configuration.

    Raises:
        - ValueError: if any value violates its constraints.
    """

    k1: float = 1.5
    b: float = 0.75
    title_weight: float = 3.0
    keywords_weight: float = 2.0
    description_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.k1, (int, float)) or isinstance(self.k1, bool) or self.k1 <= 0:
            raise ValueError("k1 must be a number > 0")
        if not isinstance(self.b, (int, float)) or isinstance(self.b, bool) or not (0 <= self.b <= 1):
            raise ValueError("b must be a number in [0, 1]")
        weights = (
            self.title_weight,
            self.keywords_weight,
            self.description_weight,
        )
        for name, value in zip(
            ("title_weight", "keywords_weight", "description_weight"), weights
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a number >= 0")
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one field weight must be > 0")


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into deterministic lower-cased tokens.

    Expected input:
        - text: free text (Vietnamese or otherwise).

    Returns:
        - Tuple of lower-cased word tokens in order of appearance; empty
          string or punctuation-only text yields an empty tuple.

    Raises:
        - No exceptions.
    """
    return tuple(token for token in _TOKEN_SEPARATOR.split(text.lower()) if token) if text else ()


def _weighted_counts(
    tokens: Sequence[str], weight: float
) -> dict[str, float]:
    """Return ``{token: weight * count}`` over a token sequence."""
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + weight
    return counts


def _field_token_lists(
    document: MediaInfoDocument,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ``(title_tokens, keyword_tokens, description_tokens)``."""
    title_tokens = tokenize(document.title)
    keyword_tokens = tokenize(" ".join(document.keywords))
    description_tokens = tokenize(document.description)
    return title_tokens, keyword_tokens, description_tokens


class MediaInfoIndex:
    """BM25 text index over media-info documents.

    Build from parsed documents or directly from a raw ``media-info``
    directory, save to / load from a JSON artifact, and query by text.
    """

    def __init__(
        self,
        documents: Sequence[MediaInfoDocument],
        config: MediaInfoRetrievalConfig | None = None,
    ) -> None:
        self._config = config or MediaInfoRetrievalConfig()
        self._documents = _ordered_documents(documents)
        self._video_to_index = {
            document.video_id: index for index, document in enumerate(self._documents)
        }
        self._stats, self._lengths = self._build_stats()
        self._avgdl = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._df = self._compute_document_frequencies()

    @classmethod
    def build(
        cls,
        documents: Sequence[MediaInfoDocument],
        config: MediaInfoRetrievalConfig | None = None,
    ) -> "MediaInfoIndex":
        """Build an index from parsed media-info documents.

        Expected input:
            - documents: media-info documents; duplicate video ids raise.
            - config: optional BM25 configuration (defaults used otherwise).

        Returns:
            - A new :class:`MediaInfoIndex`.

        Raises:
            - ValueError: if a document is not a :class:`MediaInfoDocument` or
              two documents share a video id.
        """
        return cls(documents, config=config)

    @classmethod
    def build_from_dir(
        cls,
        media_info_dir: Path | str,
        config: MediaInfoRetrievalConfig | None = None,
        strict: bool = True,
    ) -> "MediaInfoIndex":
        """Build an index from a raw ``media-info`` directory.

        Expected input:
            - media_info_dir: directory of ``<VIDEO_ID>.json`` files.
            - config: optional BM25 configuration (defaults used otherwise).
            - strict: when ``True``, raise if any media-info file is
              unreadable or malformed; when ``False``, skip problem files and
              record them via :attr:`problems`.

        Returns:
            - A new :class:`MediaInfoIndex` over the successfully parsed
              files.

        Raises:
            - FileNotFoundError: if ``media_info_dir`` does not exist.
            - ValueError: if ``strict`` and at least one file could not be
              loaded.
        """
        documents, problems = _load_media_info_documents(media_info_dir)
        if strict and problems:
            detail = "; ".join(f"{name}: {message}" for name, message in problems)
            raise ValueError(
                f"media-info dir has unreadable/malformed files: {detail}"
            )
        index = cls(documents, config=config)
        index._problems = tuple(problems)
        return index

    @classmethod
    def load(cls, path: Path | str) -> "MediaInfoIndex":
        """Load an index from a serialized JSON artifact.

        Expected input:
            - path: path to a media-info index artifact produced by
              :meth:`save`.

        Returns:
            - A :class:`MediaInfoIndex` whose documents and configuration
              reproduce the saved rankings exactly.

        Raises:
            - FileNotFoundError: if ``path`` does not exist.
            - json.JSONDecodeError: if the file is not valid JSON.
            - ValueError: if the artifact uses an unsupported schema version
              or fails validation.
        """
        payload = _read_json(Path(path))
        schema_version = payload.get("schema_version")
        if schema_version != MEDIA_INFO_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported media-info index schema version: {schema_version!r}"
            )
        config_payload = payload["config"]
        documents = tuple(
            MediaInfoDocument(
                video_id=entry["video_id"],
                title=entry["title"],
                keywords=tuple(entry["keywords"]),
                description=entry["description"],
            )
            for entry in payload["documents"]
        )
        return cls.build(documents, config=MediaInfoRetrievalConfig(**config_payload))

    @property
    def config(self) -> MediaInfoRetrievalConfig:
        """Return the BM25 configuration used by this index."""
        return self._config

    @property
    def documents(self) -> tuple[MediaInfoDocument, ...]:
        """Return the indexed documents, sorted by video id."""
        return self._documents

    @property
    def video_ids(self) -> tuple[str, ...]:
        """Return the indexed video ids, sorted."""
        return tuple(document.video_id for document in self._documents)

    @property
    def num_documents(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    @property
    def problems(self) -> tuple[tuple[str, str], ...]:
        """Return ``(filename, message)`` problems from a lenient build."""
        return self._problems

    def query(self, text: str, top_k: int = 10) -> list[MediaInfoHit]:
        """Rank videos by BM25 relevance of the query text.

        Expected input:
            - text: query text (tokenized internally).
            - top_k: positive integer maximum number of hits to return.

        Returns:
            - Ranked hits with strictly increasing 1-based ranks and
              non-increasing scores, in deterministic order (ties resolved by
              video id). Only documents with a positive score are returned, so
              the result may contain fewer than ``top_k`` hits.

        Raises:
            - ValueError: if ``top_k`` is not a positive integer.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        query_terms = tuple(dict.fromkeys(tokenize(text)))
        if not query_terms or not self._documents:
            return []

        scores = self._score(query_terms)
        ranked = [
            index
            for index in sorted(
                range(len(self._documents)),
                key=lambda index: (-scores[index], self._documents[index].video_id),
            )
            if scores[index] > 0.0
        ][:top_k]
        return [
            MediaInfoHit(
                rank=rank,
                video_id=self._documents[index].video_id,
                score=scores[index],
            )
            for rank, index in enumerate(ranked, start=1)
        ]

    def save(self, path: Path | str) -> None:
        """Write this index to a JSON artifact atomically.

        Expected input:
            - path: target path for the serialized index artifact.

        Returns:
            - None.

        Raises:
            - OSError: if the artifact cannot be written.
        """
        payload = {
            "schema_version": MEDIA_INFO_INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config": {
                "k1": self._config.k1,
                "b": self._config.b,
                "title_weight": self._config.title_weight,
                "keywords_weight": self._config.keywords_weight,
                "description_weight": self._config.description_weight,
            },
            "num_documents": self.num_documents,
            "documents": [
                {
                    "video_id": document.video_id,
                    "title": document.title,
                    "keywords": list(document.keywords),
                    "description": document.description,
                }
                for document in self._documents
            ],
        }
        _write_json_atomic(Path(path), payload)

    def _build_stats(self) -> tuple[list[dict[str, float]], list[float]]:
        """Compute weighted per-document term counts and document lengths."""
        stats: list[dict[str, float]] = []
        lengths: list[float] = []
        for document in self._documents:
            title_tokens, keyword_tokens, description_tokens = _field_token_lists(document)
            counts: dict[str, float] = {}
            for tokens, weight in (
                (title_tokens, self._config.title_weight),
                (keyword_tokens, self._config.keywords_weight),
                (description_tokens, self._config.description_weight),
            ):
                for token, weighted_count in _weighted_counts(tokens, weight).items():
                    counts[token] = counts.get(token, 0.0) + weighted_count
            stats.append(counts)
            lengths.append(
                self._config.title_weight * len(title_tokens)
                + self._config.keywords_weight * len(keyword_tokens)
                + self._config.description_weight * len(description_tokens)
            )
        return stats, lengths

    def _compute_document_frequencies(self) -> dict[str, int]:
        """Return ``{term: number of documents containing the term}``."""
        frequencies: dict[str, int] = {}
        for stats in self._stats:
            for term in stats:
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies

    def _score(self, query_terms: tuple[str, ...]) -> list[float]:
        """Return the BM25 score of every document for the query terms."""
        num_documents = len(self._documents)
        scores = [0.0] * num_documents
        for term in query_terms:
            df = self._df.get(term)
            if df is None:
                continue
            idf = math.log(
                1.0
                + (num_documents - df + 0.5) / (df + 0.5)
            )
            for index in range(num_documents):
                tf = self._stats[index].get(term, 0.0)
                if tf == 0.0:
                    continue
                denominator = tf + self._config.k1 * (
                    1.0
                    - self._config.b
                    + self._config.b * self._lengths[index] / self._avgdl
                )
                scores[index] += idf * tf * (self._config.k1 + 1.0) / denominator
        return scores


def _ordered_documents(
    documents: Sequence[MediaInfoDocument],
) -> tuple[MediaInfoDocument, ...]:
    """Validate and order documents deterministically by video id."""
    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, MediaInfoDocument):
            raise ValueError("documents must contain only MediaInfoDocument values")
        if document.video_id in seen:
            raise ValueError(f"duplicate video id in documents: {document.video_id}")
        seen.add(document.video_id)
    return tuple(sorted(documents, key=lambda document: document.video_id))


def _load_media_info_documents(
    media_info_dir: Path | str,
) -> tuple[list[MediaInfoDocument], list[tuple[str, str]]]:
    """Parse every media-info JSON under a directory into documents.

    Expected input:
        - media_info_dir: directory of ``<VIDEO_ID>.json`` files.

    Returns:
        - ``(documents, problems)``: documents in deterministic file order;
          ``problems`` is a list of ``(filename, message)`` tuples for files
          that could not be loaded. Files that do not follow the
          ``<VIDEO_ID>.json`` naming convention are reported as problems.

    Raises:
        - FileNotFoundError: if ``media_info_dir`` does not exist.
    """
    directory = Path(media_info_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"media-info directory not found: {directory}")
    documents: list[MediaInfoDocument] = []
    problems: list[tuple[str, str]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".json":
            continue
        video_id = video_id_from_collection_name(path.name)
        if video_id is None:
            problems.append((path.name, "does not follow <VIDEO_ID>.json"))
            continue
        try:
            payload = parse_media_info(path)
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            problems.append((path.name, str(exc)))
            continue
        documents.append(
            MediaInfoDocument(
                video_id=video_id,
                title=str(payload["title"]),
                keywords=tuple(str(keyword) for keyword in payload["keywords"]),
                description=str(payload["description"]),
            )
        )
    return documents, problems


def _read_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write a JSON artifact atomically via a temporary file."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp_path.replace(path)