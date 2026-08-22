"""Lexical OCR evidence between queries and cleaned chunks (Phase 5D).

Computes conservative, LOCAL lexical evidence between one query and ONE
cleaned OCR chunk at a time (representation-v2 CLEANED_CONCAT text). Tokens
are never combined across chunks. Dense cosine similarity plays no role
here; the decision mechanism is purely lexical.

Evidence tiers (frozen before evaluation):

- 3 ``FULL_PHRASE``: every query content token appears contiguously in the
  chunk (>= 2 tokens).
- 2 ``PARTIAL_PHRASE`` (satisfies gate G1): some contiguous run of >= 2
  query content tokens appears in the chunk.
- 1 ``RARE_OVERLAP`` (satisfies only gate G2): >= 2 distinct overlapping
  content tokens including at least one rare token
  (document frequency <= ``AnalysisConfig.rare_max_df``).
- No evidence otherwise; common-token-only overlap never qualifies.

Normalization reuses the shared Phase 3B lexical machinery
(:mod:`aic2026.ocr_failure_analysis`: :func:`normalize_tokens`,
:func:`content_tokens`, :func:`find_phrases`, :func:`build_document_frequencies`)
plus one local punctuation-stripping step so query punctuation cannot break
token equality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from aic2026.ocr_failure_analysis import (
    AnalysisConfig,
    build_document_frequencies,
    content_tokens,
    find_phrases,
    normalize_tokens,
)

#: Evidence tier labels (higher is stronger).
EVIDENCE_TIERS = ("RARE_OVERLAP", "PARTIAL_PHRASE", "FULL_PHRASE")

#: Gate names and the minimum tier each accepts.
GATES = ("G1", "G2")
GATE_MIN_TIER: dict[str, int] = {"G1": 2, "G2": 1}

_PUNCTUATION = frozenset(",.;:!?\"'`()[]{}<>«»…–—-/\\|+=*#@%&~^$_")


def strip_punctuation(text: str) -> str:
    """Return ``text`` lowercase with punctuation replaced by spaces.

    Expected input:
        - text: arbitrary query or OCR text.

    Returns:
        - Lowercased text where every punctuation character became a space
          and whitespace runs collapsed to one space. Accents preserved.

    Raises:
        - ValueError: if ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {text!r}")
    chars = []
    for char in text.lower():
        chars.append(" " if char in _PUNCTUATION else char)
    return " ".join("".join(chars).split())


def query_content_tokens(query_text: str) -> tuple[str, ...]:
    """Return punctuation-stripped, stopword-free query tokens."""
    return tuple(content_tokens(strip_punctuation(query_text)))


@dataclass(frozen=True)
class LexicalEvidence:
    """Best lexical evidence found in ONE cleaned chunk for one query.

    Fields:
        - chunk_id / video_id / frame_start / frame_end: chunk identity.
        - tier: 3 (FULL_PHRASE), 2 (PARTIAL_PHRASE), or 1 (RARE_OVERLAP).
        - tier_name: human-readable label of the tier.
        - matched_phrase: longest contiguous matched phrase (None for
          RARE_OVERLAP).
        - overlap_tokens: distinct overlapping content tokens, sorted.
        - rare_tokens: overlapping tokens with df <= rare_max_df, sorted.
        - min_rare_df: smallest document frequency among rare tokens
          (None when no rare token exists).
        - gate: strongest satisfied gate ("G1" or "G2"), or None.
        - chunk_text: the cleaned chunk text that produced the evidence.
    """

    chunk_id: str
    video_id: str
    frame_start: int
    frame_end: int
    tier: int
    tier_name: str
    matched_phrase: str | None
    overlap_tokens: tuple[str, ...]
    rare_tokens: tuple[str, ...]
    min_rare_df: int | None
    gate: str | None
    chunk_text: str


def evaluate_chunk_evidence(
    query_text: str,
    chunk: Any,
    document_frequencies: Mapping[str, int],
    config: AnalysisConfig | None = None,
) -> LexicalEvidence | None:
    """Compute the best lexical evidence of ONE chunk for one query.

    Expected input:
        - query_text: arbitrary query text.
        - chunk: object exposing ``chunk_id``, ``video_id``,
          ``frame_start``, ``frame_end``, and cleaned ``text``.
        - document_frequencies: corpus-level DF map over the cleaned chunk
          artifact (:func:`build_document_frequencies`).
        - config: thresholds; defaults to ``AnalysisConfig()``.

    Returns:
        - :class:`LexicalEvidence` or None when the chunk shares fewer than
          two meaningful tokens or only common tokens with the query.

    Raises:
        - ValueError: inherited from text handling.
    """
    config = config or AnalysisConfig()
    q_tokens = list(query_content_tokens(query_text))
    if len(q_tokens) < 2:
        return None
    c_tokens = normalize_tokens(chunk.text)
    if not c_tokens:
        return None
    c_set = set(c_tokens)

    phrases = find_phrases(q_tokens, c_tokens, min_len=2)
    overlap = sorted(set(q_tokens) & c_set)
    rare = [t for t in overlap if document_frequencies.get(t, 0) <= config.rare_max_df]
    full_query_contiguous = bool(
        len(q_tokens) >= 2 and find_phrases(q_tokens, c_tokens, min_len=len(q_tokens))
    )

    if full_query_contiguous:
        tier, name = 3, "FULL_PHRASE"
    elif phrases:
        tier, name = 2, "PARTIAL_PHRASE"
    elif len(overlap) >= 2 and rare:
        tier, name = 1, "RARE_OVERLAP"
    else:
        return None

    return LexicalEvidence(
        chunk_id=str(chunk.chunk_id),
        video_id=str(chunk.video_id),
        frame_start=int(chunk.frame_start),
        frame_end=int(chunk.frame_end),
        tier=tier,
        tier_name=name,
        matched_phrase=phrases[0] if phrases else None,
        overlap_tokens=tuple(overlap),
        rare_tokens=tuple(sorted(rare)),
        min_rare_df=min((document_frequencies.get(t, 0) for t in rare), default=None),
        gate=max((g for g, min_tier in GATE_MIN_TIER.items() if tier >= min_tier), default=None),
        chunk_text=chunk.text,
    )


def passes_gate(evidence: LexicalEvidence | None, gate: str) -> bool:
    """Return whether evidence satisfies a named gate.

    Expected input:
        - evidence: possibly None.
        - gate: ``"G1"`` or ``"G2"``.

    Raises:
        - ValueError: if the gate name is unknown.
    """
    if gate not in GATE_MIN_TIER:
        raise ValueError(f"unknown gate: {gate!r}")
    if evidence is None:
        return False
    return evidence.tier >= GATE_MIN_TIER[gate]


def best_video_evidence(
    query_text: str,
    chunks_of_video: Sequence[Any],
    document_frequencies: Mapping[str, int],
    config: AnalysisConfig | None = None,
) -> LexicalEvidence | None:
    """Return the strongest evidence across ONE video's chunks.

    Chunks are examined independently (locality guarantee); ties resolve
    deterministically by higher tier, longer phrase, rarer minimum DF, then
    earlier chunk_id.

    Expected input:
        - chunks_of_video: cleaned chunks belonging to ONE candidate video.
        - document_frequencies / config: as in :func:`evaluate_chunk_evidence`.

    Returns:
        - The best :class:`LexicalEvidence` or None.

    Raises:
        - ValueError: inherited from text handling.
    """
    best: LexicalEvidence | None = None
    best_key: tuple | None = None
    for chunk in chunks_of_video:
        evidence = evaluate_chunk_evidence(query_text, chunk, document_frequencies, config)
        if evidence is None:
            continue
        key = (
            -evidence.tier,
            -(len(evidence.matched_phrase.split()) if evidence.matched_phrase else 0),
            evidence.min_rare_df if evidence.min_rare_df is not None else 10**9,
            evidence.chunk_id,
        )
        if best_key is None or key < best_key:
            best, best_key = evidence, key
    return best


def document_frequencies_from_texts(texts: Iterable[str]) -> dict[str, int]:
    """Build corpus DF over cleaned chunk texts via the shared machinery.

    Expected input:
        - texts: cleaned text of EVERY chunk of the representation artifact.

    Returns:
        - Deterministic term -> chunk-count map.

    Raises:
        - ValueError: inherited from normalization.
    """
    from types import SimpleNamespace

    return build_document_frequencies([SimpleNamespace(text=t) for t in texts])


# ------------------------------------------------------------------ reranking


@dataclass(frozen=True)
class RoutedRerankConfig:
    """Explicit configuration for one routed rerank run.

    Fields:
        - pool_size: number of top visual frames forming the pool.
        - gate: ``"G1"``, ``"G2"``, or None. None disables OCR entirely and
          makes the output identical to the visual baseline ordering.
        - analysis: lexical thresholds (defaults apply when None).

    Raises:
        - ValueError: if any field violates its contract.
    """

    pool_size: int = 100
    gate: str | None = None
    analysis: AnalysisConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pool_size, int) or isinstance(self.pool_size, bool) or self.pool_size < 1:
            raise ValueError(f"pool_size must be a positive integer, got {self.pool_size!r}")
        if self.gate is not None and self.gate not in GATE_MIN_TIER:
            raise ValueError(f"gate must be one of {GATES} or None, got {self.gate!r}")


@dataclass(frozen=True)
class RerankedFrameCandidate:
    """One visual frame after routed OCR reranking, with diagnostics.

    Fields:
        - rank: 1-based position after reranking.
        - video_id / keyframe_ordinal / frame_index / pts_time / score:
          identity copied from the visual candidate.
        - visual_rank: 1-based rank in the incoming visual pool.
        - ocr_supported: whether the video passed the configured gate.
        - evidence_tier: best evidence tier of the video (0 = unsupported).
        - tier_name / matched_chunk_id / matched_phrase / min_rare_df /
          evidence_frame_start / evidence_frame_end: diagnostics of the best
          evidence (None when unsupported).
    """

    rank: int
    video_id: str
    keyframe_ordinal: int
    frame_index: int
    pts_time: float
    score: float
    visual_rank: int
    ocr_supported: bool
    evidence_tier: int
    tier_name: str | None = None
    matched_chunk_id: str | None = None
    matched_phrase: str | None = None
    min_rare_df: int | None = None
    evidence_frame_start: int | None = None
    evidence_frame_end: int | None = None


def routed_candidate_rerank(
    candidates: Sequence[Any],
    *,
    config: RoutedRerankConfig,
    evidence_by_video: Mapping[str, LexicalEvidence],
) -> list[RerankedFrameCandidate]:
    """Rerank visual candidates using gated lexical OCR evidence ONLY.

    Ordering key per frame is ``(-video_tier, visual_rank)`` where the tier
    is 0 for videos without gate-passing evidence. Consequences (frozen):

    - unsupported videos keep their exact relative visual order;
    - supported videos are grouped by evidence strength above unsupported
      ones, visual rank deciding inside each group;
    - frames of one video always keep their original visual order;
    - no candidate outside the pool can appear;
    - dense OCR cosine is never consulted.

    Expected input:
        - candidates: visual-ranked candidates with unique ascending ranks.
        - config: :class:`RoutedRerankConfig`.
        - evidence_by_video: precomputed best evidence per candidate video
          (:func:`best_video_evidence`); entries for non-pool videos are
          ignored, so OCR cannot expand membership.

    Returns:
        - Reranked candidates carrying full diagnostics. With
          ``config.gate is None`` the output order equals the input order.

    Raises:
        - ValueError: on malformed inputs.
    """
    if not candidates:
        return []
    ranks = [int(c.rank) for c in candidates]
    if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("candidates must be ranked with unique ascending ranks")
    for video_id, evidence in evidence_by_video.items():
        if evidence.video_id != video_id:
            raise ValueError(f"evidence keyed {video_id!r} carries video_id {evidence.video_id!r}")

    drafts: list[tuple[int, int, dict[str, Any]]] = []
    for candidate in candidates:
        evidence = evidence_by_video.get(candidate.video_id)
        supported = (
            config.gate is not None
            and evidence is not None
            and passes_gate(evidence, config.gate)
        )
        tier = evidence.tier if supported else 0
        fields = {
            "video_id": candidate.video_id,
            "keyframe_ordinal": int(candidate.keyframe_ordinal),
            "frame_index": int(candidate.frame_index),
            "pts_time": float(candidate.pts_time),
            "score": float(candidate.score),
            "visual_rank": int(candidate.rank),
            "ocr_supported": bool(supported),
            "evidence_tier": int(tier),
            "tier_name": evidence.tier_name if supported else None,
            "matched_chunk_id": evidence.chunk_id if supported else None,
            "matched_phrase": evidence.matched_phrase if supported else None,
            "min_rare_df": evidence.min_rare_df if supported else None,
            "evidence_frame_start": evidence.frame_start if supported else None,
            "evidence_frame_end": evidence.frame_end if supported else None,
        }
        drafts.append((-tier, int(candidate.rank), fields))
    drafts.sort(key=lambda item: (item[0], item[1]))
    return [
        RerankedFrameCandidate(rank=position, **fields)
        for position, (_, _, fields) in enumerate(drafts, start=1)
    ]
