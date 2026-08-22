"""Lexical failure analysis: are visual retrieval misses recoverable from OCR text?

Diagnostic tooling for Phase 3B planning. For every visual-retrieval failure
(a query whose ground-truth video/frame region is not satisfied within the
visual top-k, using the existing frame semantics of
:func:`aic2026.kis_scoring.is_kis_correct`), this module inspects the OCR
chunks of the ground-truth video and computes transparent lexical
diagnostics:

- normalized token overlap between query and ground-truth OCR (case
  insensitive, Unicode/accents preserved via the shared
  :func:`aic2026.ocr_generation.normalize_ocr_text`);
- phrase/subphrase containment (contiguous multi-token matches);
- corpus-level document frequency per term to separate common words from
  potentially discriminative rare terms;
- temporal alignment of the evidence relative to the ground-truth frame
  range.

Every failure receives one deterministic classification:

- ``NO_GT_OCR_TEXT``: the GT video has no OCR chunk with content tokens.
- ``NO_LEXICAL_OVERLAP``: GT OCR exists but shares no query terms.
- ``PHRASE_OR_STRONG_MATCH``: a contiguous phrase of at least
  ``min_phrase_tokens`` query tokens appears in GT OCR.
- ``RARE_TERM_MATCH``: at least one overlapping query term has document
  frequency <= ``rare_max_df`` chunks.
- ``COMMON_TERM_ONLY``: overlap exists but only common terms.

``TEMPORAL_MISMATCH`` is reported as a boolean flag (evidence exists in the
GT video but not within ``gt_near_frames`` of the GT range), not as a
primary category, so rarity information is never lost.

This module performs no retrieval, no ranking, and no scoring; callers
supply the visual candidates. It never claims expected BM25 gains — it only
measures whether lexical evidence exists in the failures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from aic2026.evaluation import EvaluationQuery, EvaluationSet, EvaluationTask
from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_generation import normalize_ocr_text

#: Failure classification categories in precedence order.
CATEGORIES = (
    "NO_GT_OCR_TEXT",
    "NO_LEXICAL_OVERLAP",
    "PHRASE_OR_STRONG_MATCH",
    "RARE_TERM_MATCH",
    "COMMON_TERM_ONLY",
)

#: Categories that indicate lexically recoverable evidence.
PROMISING_CATEGORIES = ("PHRASE_OR_STRONG_MATCH", "RARE_TERM_MATCH")

#: Centralized stopword policy (lowercase). Kept small and explicit; these
#: tokens are excluded from overlap and document-frequency diagnostics.
STOPWORDS: frozenset[str] = frozenset(
    {
        # Vietnamese function words
        "một", "hai", "các", "những", "của", "và", "hoặc", "trên", "trong",
        "dưới", "với", "được", "có", "là", "gì", "nào", "đang", "đã", "sẽ",
        "này", "kia", "đó", "ở", "từ", "theo", "cho", "bị", "mỗi", "nhiều",
        "ít", "rất", "vào", "ra", "lên", "xuống",
        # English function words
        "a", "an", "the", "of", "on", "in", "with", "and", "or", "is",
        "are", "was", "what", "which", "that", "this", "to", "for", "at",
        "by", "it",
    }
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Explicit thresholds for one analysis run.

    Fields:
        - rare_max_df: a matched term is "rare" when it appears in at most
          this many OCR chunks corpus-wide.
        - min_phrase_tokens: minimum contiguous query tokens forming a
          strong phrase match.
        - gt_near_frames: chunk-to-GT-range gap tolerated when deciding
          whether evidence is temporally aligned.
        - max_excerpts: bounded number of OCR excerpts per failure record.
        - excerpt_chars: maximum characters per excerpt.
    """

    rare_max_df: int = 10
    min_phrase_tokens: int = 2
    gt_near_frames: int = 250
    max_excerpts: int = 3
    excerpt_chars: int = 200

    def __post_init__(self) -> None:
        for name in ("rare_max_df", "min_phrase_tokens", "gt_near_frames", "max_excerpts", "excerpt_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.min_phrase_tokens < 1:
            raise ValueError("min_phrase_tokens must be >= 1")


@dataclass(frozen=True)
class TermMatch:
    """One overlapping query term with its corpus document frequency.

    Fields:
        - term: the shared normalized token.
        - df: number of OCR chunks containing the term.
        - rare: ``df <= AnalysisConfig.rare_max_df``.
    """

    term: str
    df: int
    rare: bool


@dataclass(frozen=True)
class FailureAnalysis:
    """Full diagnostic record for one visual retrieval failure.

    Fields:
        - query_id / task / query_text_vi: evaluation query identity.
        - gt_video_id / gt_frame_start / gt_frame_end: existing ground-truth
          semantics (video + inclusive original-frame range).
        - visual_top_k: cutoff used for the failure definition.
        - gt_video_in_visual_top_k: whether any visual candidate came from
          the GT video.
        - gt_best_visual_rank: best (smallest) rank of a GT-video candidate,
          or None when absent.
        - gt_chunks_total / gt_chunks_with_text: GT-video OCR availability.
        - temporally_near_overlap_chunks: chunks with lexical overlap whose
          distance to the GT range is <= ``gt_near_frames``.
        - matched_terms: overlapping terms with DF/rarity, vocabulary order.
        - matched_phrases: contiguous multi-token query phrases found in GT
          OCR (longest first).
        - excerpts: bounded OCR text excerpts from overlapping chunks.
        - classification: one of :data:`CATEGORIES`.
        - temporal_mismatch: lexical evidence exists but none temporally
          near the GT range.
        - promising: classification is in :data:`PROMISING_CATEGORIES`.
    """

    query_id: str
    task: str
    query_text_vi: str
    gt_video_id: str
    gt_frame_start: int
    gt_frame_end: int
    visual_top_k: int
    gt_video_in_visual_top_k: bool
    gt_best_visual_rank: int | None
    gt_chunks_total: int
    gt_chunks_with_text: int
    temporally_near_overlap_chunks: int
    matched_terms: tuple[TermMatch, ...]
    matched_phrases: tuple[str, ...]
    excerpts: tuple[dict[str, Any], ...]
    classification: str
    temporal_mismatch: bool
    promising: bool


def normalize_tokens(text: str) -> list[str]:
    """Return whitespace tokens of the shared OCR normalization of ``text``.

    Expected input:
        - text: arbitrary query or OCR text.

    Returns:
        - Lowercased, whitespace-normalized tokens with accents preserved.

    Raises:
        - No exceptions (non-string input yields an empty list).
    """
    if not isinstance(text, str):
        return []
    return normalize_ocr_text(text).split()


def content_tokens(text: str) -> list[str]:
    """Return normalized tokens minus the centralized stopwords."""
    return [token for token in normalize_tokens(text) if token not in STOPWORDS]


def find_phrases(
    query_tokens: Sequence[str],
    text_tokens: Sequence[str],
    *,
    min_len: int,
) -> list[str]:
    """Return contiguous query-token phrases present in the text tokens.

    Expected input:
        - query_tokens: normalized query tokens (stopwords removed).
        - text_tokens: normalized tokens of one OCR chunk.
        - min_len: minimum phrase length to report (>= 1).

    Returns:
        - Matched phrases, longest first, deduplicated.

    Raises:
        - No exceptions.
    """
    if min_len < 1:
        raise ValueError(f"min_len must be >= 1, got {min_len!r}")
    n = len(query_tokens)
    m = len(text_tokens)
    found: list[str] = []
    seen: set[str] = set()
    for length in range(min(n, m), min_len - 1, -1):
        for start in range(0, n - length + 1):
            phrase_tokens = query_tokens[start : start + length]
            for offset in range(0, m - length + 1):
                if list(text_tokens[offset : offset + length]) == list(phrase_tokens):
                    phrase = " ".join(phrase_tokens)
                    if phrase not in seen:
                        seen.add(phrase)
                        found.append(phrase)
                    break
        if found:
            break
    return found


def build_document_frequencies(chunks: Sequence[OcrChunk]) -> dict[str, int]:
    """Count in how many OCR chunks each content token occurs.

    Expected input:
        - chunks: OCR chunks of the complete artifact (corpus-level DF).

    Returns:
        - Mapping from normalized content token to chunk count, sorted by
          term for deterministic serialization.

    Raises:
        - No exceptions.
    """
    counter: Counter[str] = Counter()
    for chunk in chunks:
        counter.update(set(content_tokens(chunk.text)))
    return dict(sorted(counter.items()))


def chunk_distance_to_range(chunk: OcrChunk, gt_start: int, gt_end: int) -> int:
    """Return the frame gap between a chunk range and an inclusive GT range.

    Expected input:
        - chunk: OCR chunk with ``frame_start``/``frame_end``.
        - gt_start / gt_end: inclusive ground-truth frame range.

    Returns:
        - 0 when the ranges intersect; otherwise the smallest gap between
          the two ranges in frames.

    Raises:
        - No exceptions.
    """
    if chunk.frame_start > gt_end:
        return chunk.frame_start - gt_end
    if chunk.frame_end < gt_start:
        return gt_start - chunk.frame_end
    return 0


def frame_in_gt(query: EvaluationQuery, video_id: str, frame_index: int) -> bool:
    """Return whether a frame prediction satisfies the query's GT region.

    Mirrors the frame semantics of
    :func:`aic2026.kis_scoring.is_kis_correct` (video match + inclusive
    original-frame range) so KIS and QA retrieval share one rule; no new
    ground truth is introduced.
    """
    if video_id != query.ground_truth_video_id.value:
        return False
    gt = query.ground_truth_frame_range
    return gt.start <= frame_index <= gt.end


def analyze_failure(
    query: EvaluationQuery,
    visual_candidates: Sequence[Any],
    gt_chunks: Sequence[OcrChunk],
    document_frequencies: dict[str, int],
    config: AnalysisConfig,
) -> FailureAnalysis:
    """Compute lexical diagnostics for one already-identified failure.

    Expected input:
        - query: canonical KIS or QA evaluation query.
        - visual_candidates: ranked objects exposing ``rank``,
          ``video_id``, ``frame_index`` (the visual top-k list).
        - gt_chunks: OCR chunks belonging to the GT video.
        - document_frequencies: corpus-level DF from
          :func:`build_document_frequencies`.
        - config: analysis thresholds.

    Returns:
        - The full :class:`FailureAnalysis` record.

    Raises:
        - ValueError: if the config is invalid.
    """
    gt_video = query.ground_truth_video_id.value
    gt_range = query.ground_truth_frame_range
    gt_ranks = [
        int(c.rank) for c in visual_candidates if c.video_id == gt_video
    ]
    query_terms = content_tokens(query.query_text_vi)
    query_set = set(query_terms)

    chunks_with_text = [c for c in gt_chunks if content_tokens(c.text)]
    matched_terms: dict[str, TermMatch] = {}
    matched_phrases: list[str] = []
    near_overlap = 0
    overlapping: list[tuple[int, str, OcrChunk]] = []  # (-overlap_count, chunk_id, chunk)

    for chunk in chunks_with_text:
        chunk_token_list = normalize_tokens(chunk.text)
        chunk_set = set(chunk_token_list)
        overlap = query_set & chunk_set
        if not overlap:
            continue
        for term in sorted(overlap):
            if term not in matched_terms:
                df = document_frequencies.get(term, 0)
                matched_terms[term] = TermMatch(term=term, df=df, rare=df <= config.rare_max_df)
        phrases = find_phrases(query_terms, chunk_token_list, min_len=config.min_phrase_tokens)
        for phrase in phrases:
            if phrase not in matched_phrases:
                matched_phrases.append(phrase)
        distance = chunk_distance_to_range(chunk, gt_range.start, gt_range.end)
        if distance <= config.gt_near_frames:
            near_overlap += 1
        overlapping.append((-len(overlap), chunk.chunk_id, chunk))

    if matched_phrases:
        classification = "PHRASE_OR_STRONG_MATCH"
    elif any(term.rare for term in matched_terms.values()):
        classification = "RARE_TERM_MATCH"
    elif matched_terms:
        classification = "COMMON_TERM_ONLY"
    elif chunks_with_text:
        classification = "NO_LEXICAL_OVERLAP"
    else:
        classification = "NO_GT_OCR_TEXT"

    lexical_match = classification in PROMISING_CATEGORIES or classification == "COMMON_TERM_ONLY"
    temporal_mismatch = bool(lexical_match and near_overlap == 0)

    overlapping.sort(key=lambda item: (item[0], item[1]))
    excerpts = tuple(
        {
            "chunk_id": chunk.chunk_id,
            "frame_start": chunk.frame_start,
            "frame_end": chunk.frame_end,
            "distance_to_gt": chunk_distance_to_range(chunk, gt_range.start, gt_range.end),
            "text": chunk.text[: config.excerpt_chars],
        }
        for _, _, chunk in overlapping[: config.max_excerpts]
    )

    ordered_terms = tuple(matched_terms[term] for term in sorted(matched_terms))
    return FailureAnalysis(
        query_id=query.query_id,
        task=query.task.value,
        query_text_vi=query.query_text_vi,
        gt_video_id=gt_video,
        gt_frame_start=gt_range.start,
        gt_frame_end=gt_range.end,
        visual_top_k=len(visual_candidates),
        gt_video_in_visual_top_k=bool(gt_ranks),
        gt_best_visual_rank=min(gt_ranks) if gt_ranks else None,
        gt_chunks_total=len(gt_chunks),
        gt_chunks_with_text=len(chunks_with_text),
        temporally_near_overlap_chunks=near_overlap,
        matched_terms=ordered_terms,
        matched_phrases=tuple(matched_phrases),
        excerpts=excerpts,
        classification=classification,
        temporal_mismatch=temporal_mismatch,
        promising=classification in PROMISING_CATEGORIES,
    )


def run_failure_analysis(
    eval_set: EvaluationSet,
    task: EvaluationTask | str,
    visual_ranked_by_query: dict[str, Sequence[Any]],
    chunks: Sequence[OcrChunk],
    *,
    config: AnalysisConfig,
    visual_top_k: int,
) -> tuple[list[FailureAnalysis], int]:
    """Analyze every visual failure of one task in an evaluation set.

    Expected input:
        - eval_set: loaded canonical evaluation set.
        - task: task to iterate (KIS or QA).
        - visual_ranked_by_query: mapping from query id to the ranked
          visual candidates (objects with ``rank``/``video_id``/
          ``frame_index``).
        - chunks: all OCR chunks of the merged artifact.
        - config: analysis thresholds.
        - visual_top_k: cutoff defining a failure (candidates beyond this
          rank are ignored).

    Returns:
        - ``(failures, total_queries)`` where failures keep evaluation-set
          order.

    Raises:
        - ValueError: if the task has no queries or a query lacks visual
          candidates.
    """
    chunks_by_video: dict[str, list[OcrChunk]] = {}
    for chunk in chunks:
        chunks_by_video.setdefault(chunk.video_id, []).append(chunk)
    dfs = build_document_frequencies(chunks)

    failures: list[FailureAnalysis] = []
    total = 0
    for query in eval_set.iter_queries(task):
        total += 1
        candidates = list(visual_ranked_by_query.get(query.query_id, []))[:visual_top_k]
        if not candidates:
            raise ValueError(f"missing visual candidates for query {query.query_id!r}")
        hit = any(frame_in_gt(query, c.video_id, c.frame_index) for c in candidates)
        if hit:
            continue
        failures.append(
            analyze_failure(
                query,
                candidates,
                chunks_by_video.get(query.ground_truth_video_id.value, []),
                dfs,
                config,
            )
        )
    return failures, total


def summarize(failures: Sequence[FailureAnalysis], total_queries: int) -> dict[str, Any]:
    """Return aggregate counts over one task's failure analyses.

    Expected input:
        - failures: analyses in evaluation order.
        - total_queries: number of queries of the task (failure denominator).

    Returns:
        - Dict with transparent aggregate counts and percentages.

    Raises:
        - No exceptions.
    """
    counts = Counter(f.classification for f in failures)
    with_text = sum(1 for f in failures if f.gt_chunks_with_text > 0)
    with_overlap = sum(1 for f in failures if f.matched_terms or f.matched_phrases)
    rare = sum(1 for f in failures if f.classification == "RARE_TERM_MATCH")
    phrase = sum(1 for f in failures if f.classification == "PHRASE_OR_STRONG_MATCH")
    temporal = sum(1 for f in failures if f.temporal_mismatch)
    promising = sum(1 for f in failures if f.promising)

    def pct(count: int) -> float:
        return round(100.0 * count / total_queries, 2) if total_queries else 0.0

    return {
        "total_queries": total_queries,
        "visual_failures": len(failures),
        "failures_with_gt_ocr_text": with_text,
        "failures_with_lexical_overlap": with_overlap,
        "failures_rare_term_match": rare,
        "failures_phrase_match": phrase,
        "failures_common_term_only": counts.get("COMMON_TERM_ONLY", 0),
        "failures_no_lexical_overlap": counts.get("NO_LEXICAL_OVERLAP", 0),
        "failures_no_gt_ocr_text": counts.get("NO_GT_OCR_TEXT", 0),
        "temporal_mismatch_cases": temporal,
        "promising_failures": promising,
        "percentages": {
            "visual_failures": pct(len(failures)),
            "lexical_overlap": pct(with_overlap),
            "rare_term": pct(rare),
            "phrase_match": pct(phrase),
            "temporal_mismatch": pct(temporal),
            "promising": pct(promising),
        },
    }


def shortlist(failures: Sequence[FailureAnalysis], limit: int = 10) -> list[FailureAnalysis]:
    """Return the most promising failures in deterministic order.

    Expected input:
        - failures: analyses of one task.
        - limit: maximum number of records to return.

    Returns:
        - Promising failures ordered by category strength (phrase before
          rare), then by fewest matched common terms, then query id.

    Raises:
        - No exceptions.
    """
    category_rank = {"PHRASE_OR_STRONG_MATCH": 0, "RARE_TERM_MATCH": 1}
    selected = [f for f in failures if f.promising]
    selected.sort(
        key=lambda f: (
            category_rank[f.classification],
            -len(f.matched_phrases),
            min((t.df for t in f.matched_terms if t.rare), default=0),
            f.query_id,
        )
    )
    return selected[:limit]
