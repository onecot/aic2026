"""Evaluation reliability and OCR signal audit (Phase 5A).

Deterministic, dependency-light analysis answering whether the conclusion
"naive OCR fusion has no value" is supported by the current evaluation
protocol. Pure statistics over per-query paired results; no retrieval, no
fusion, no model inference happens here.

Inputs are per-query paired observations: for one query, a baseline and a
variant each produce

- ``first_frame_rank``: 1-based rank of the first candidate satisfying the
  existing frame correctness rule (None = miss within the pool);
- ``first_video_rank``: 1-based rank of the first candidate from the
  ground-truth video (None = video absent).

Reliability tooling:

- win/tie/loss accounting on first-correct ranks;
- sign-flip paired permutation test on Hit@K deltas (deterministic seed);
- percentile bootstrap CI for the mean Hit@K delta;
- leave-one-query-out sensitivity of the mean delta.

Decomposition and stratification:

- video recall vs frame hit at every K, plus baseline->variant transition
  counts (video/frame joint outcomes);
- data-driven OCR-evidence strata reusing the Phase 3B lexical machinery
  (:mod:`aic2026.ocr_failure_analysis`): NO_GT_OCR_EVIDENCE,
  GT_OCR_NO_QUERY_OVERLAP, WEAK_OCR_SIGNAL, STRONG_LEXICAL_OCR_SIGNAL.

Small samples (QA n=10) are flagged by callers; nothing here overclaims
significance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from aic2026.ocr_chunks import OcrChunk
from aic2026.ocr_failure_analysis import (
    AnalysisConfig,
    build_document_frequencies,
    content_tokens,
    find_phrases,
    normalize_tokens,
)

#: Cutoffs shared with the project's KIS reporting.
AUDIT_K: tuple[int, ...] = (1, 5, 20, 50, 100)

#: OCR evidence strata (data-driven, assigned by :func:`assign_stratum`).
STRATA = (
    "NO_GT_OCR_EVIDENCE",
    "GT_OCR_NO_QUERY_OVERLAP",
    "WEAK_OCR_SIGNAL",
    "STRONG_LEXICAL_OCR_SIGNAL",
)


def _require(condition: bool, message: str) -> None:
    """Raise ValueError unless ``condition`` holds."""
    if not condition:
        raise ValueError(message)


# ------------------------------------------------------------------ ranks/hits


def first_rank(
    ranked_pairs: Sequence[tuple[str, int]],
    *,
    gt_video_id: str,
    gt_start: int | None = None,
    gt_end: int | None = None,
) -> int | None:
    """Return the 1-based rank of the first matching entry, or None.

    With ``gt_start``/``gt_end`` given, a match requires the frame to lie in
    the inclusive range (frame rule); otherwise any GT-video entry matches
    (video rule). Mirrors the repository's existing semantics.
    """
    for position, (video_id, frame_index) in enumerate(ranked_pairs, start=1):
        if video_id != gt_video_id:
            continue
        if gt_start is None or gt_start <= frame_index <= (gt_end if gt_end is not None else gt_start):
            return position
    return None


def hits_by_k(first_rank_value: int | None, cutoffs: Sequence[int] = AUDIT_K) -> dict[int, bool]:
    """Return hit indicators at each cutoff for one first-rank value."""
    return {k: first_rank_value is not None and first_rank_value <= k for k in cutoffs}


# ------------------------------------------------------------------ reliability


def win_tie_loss(
    baseline_ranks: Sequence[int | None],
    variant_ranks: Sequence[int | None],
) -> dict[str, int]:
    """Account paired outcomes on first-correct ranks (lower is better).

    Expected input:
        - baseline_ranks / variant_ranks: aligned per-query first ranks
          (None = miss).

    Returns:
        - ``{"win": ..., "tie": ..., "loss": ..., "n": ...}`` where a win
          means the variant found a correct result at a better rank (or
          found one the baseline missed).

    Raises:
        - ValueError: if the sequences differ in length.
    """
    _require(len(baseline_ranks) == len(variant_ranks), "aligned per-query lists required")
    wins = ties = losses = 0
    for base, variant in zip(baseline_ranks, variant_ranks):
        if base == variant:
            ties += 1
        elif variant is None:
            losses += 1
        elif base is None:
            wins += 1
        elif variant < base:
            wins += 1
        else:
            losses += 1
    return {"win": wins, "tie": ties, "loss": losses, "n": len(baseline_ranks)}


def hit_deltas(
    baseline_hits: Sequence[bool],
    variant_hits: Sequence[bool],
) -> np.ndarray:
    """Return per-query Hit deltas as float array (+1/0/-1)."""
    _require(len(baseline_hits) == len(variant_hits), "aligned per-query lists required")
    return np.array(
        [float(v) - float(b) for b, v in zip(baseline_hits, variant_hits)], dtype=np.float64
    )


def permutation_test(
    deltas: Sequence[float],
    *,
    seed: int = 20260821,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Two-sided sign-flip paired permutation test on the mean delta.

    Expected input:
        - deltas: per-query paired deltas (e.g. Hit@K differences).
        - seed: deterministic RNG seed.
        - n_resamples: number of sign-flip resamples.

    Returns:
        - ``{"mean_delta", "observed_stat", "p_value", "n", "n_resamples",
          "seed"}``. With n == 0 the test is undefined and all values are
          None except counts. With n == 1 the only possible p-value is 1.0.

    Raises:
        - No exceptions for empty input.
    """
    values = np.asarray(list(deltas), dtype=np.float64)
    n = values.size
    if n == 0:
        return {
            "mean_delta": None,
            "observed_stat": None,
            "p_value": None,
            "n": 0,
            "n_resamples": n_resamples,
            "seed": seed,
        }
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_resamples):
        signs = rng.choice((-1.0, 1.0), size=n)
        stat = float((values * signs).mean())
        if abs(stat) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "mean_delta": observed,
        "observed_stat": observed,
        "p_value": (extreme + 1) / (n_resamples + 1),
        "n": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def bootstrap_ci(
    deltas: Sequence[float],
    *,
    seed: int = 20260821,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap CI for the mean per-query delta.

    Returns:
        - ``{"mean_delta", "ci_low", "ci_high", "confidence", "n",
          "n_resamples", "seed"}``; None values when n == 0.

    Raises:
        - ValueError: if ``confidence`` is not in (0, 1).
    """
    _require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence!r}")
    values = np.asarray(list(deltas), dtype=np.float64)
    n = values.size
    if n == 0:
        return {
            "mean_delta": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
            "n": 0,
            "n_resamples": n_resamples,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_delta": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence,
        "n": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def leave_one_out(deltas: Sequence[float]) -> dict[str, Any]:
    """Leave-one-query-out sensitivity of the mean delta.

    Returns:
        - ``{"mean_delta", "loo_min", "loo_max", "sign_flips",
          "influential_indices"}`` where ``sign_flips`` is True when removing
          some query changes the sign of the mean delta, and
          ``influential_indices`` lists the removals that flip the sign.

    Raises:
        - No exceptions (empty input yields None statistics).
    """
    values = list(deltas)
    n = len(values)
    if n == 0:
        return {
            "mean_delta": None,
            "loo_min": None,
            "loo_max": None,
            "sign_flips": False,
            "influential_indices": [],
        }
    total = sum(values)
    base_sign = 0.0 if total == 0 else (1.0 if total > 0 else -1.0)
    loo = [(total - v) / (n - 1) for v in values]
    influential = [
        i for i, m in enumerate(loo) if (0.0 if m == 0 else (1.0 if m > 0 else -1.0)) != base_sign
    ]
    return {
        "mean_delta": total / n,
        "loo_min": min(loo),
        "loo_max": max(loo),
        "sign_flips": bool(influential),
        "influential_indices": influential,
    }


# ------------------------------------------------------------------ decomposition


def transition_key(
    baseline_video_hit: bool,
    baseline_frame_hit: bool,
    variant_video_hit: bool,
    variant_frame_hit: bool,
) -> str:
    """Return a readable baseline->variant transition label.

    Joint outcome per system is one of ``VIDEO_MISS``, ``FRAME_ONLY``
    (video hit, frame miss), or ``BOTH`` (video+frame hit).
    """
    def joint(video_hit: bool, frame_hit: bool) -> str:
        if frame_hit:
            return "BOTH"
        return "FRAME_ONLY" if video_hit else "VIDEO_MISS"

    return f"{joint(baseline_video_hit, baseline_frame_hit)}->{joint(variant_video_hit, variant_frame_hit)}"


def transition_matrix(
    baseline_video: Sequence[bool],
    baseline_frame: Sequence[bool],
    variant_video: Sequence[bool],
    variant_frame: Sequence[bool],
) -> dict[str, int]:
    """Count baseline->variant joint-outcome transitions across queries."""
    _require(
        len(baseline_video) == len(variant_video) == len(baseline_frame) == len(variant_frame),
        "aligned per-query lists required",
    )
    counter: Counter[str] = Counter()
    for bv, bf, vv, vf in zip(baseline_video, baseline_frame, variant_video, variant_frame):
        counter[transition_key(bv, bf, vv, vf)] += 1
    return dict(sorted(counter.items()))


# ------------------------------------------------------------------ stratification


def assign_stratum(
    query_text: str,
    gt_chunks: Sequence[OcrChunk],
    document_frequencies: dict[str, int],
    config: AnalysisConfig | None = None,
) -> str:
    """Assign one data-driven OCR-evidence stratum to a query.

    Uses the Phase 3B lexical machinery (shared normalization/stopwords,
    phrase matching, corpus document frequencies):

    - ``NO_GT_OCR_EVIDENCE``: the GT video has no chunk with content tokens;
    - ``GT_OCR_NO_QUERY_OVERLAP``: GT OCR exists but shares no query terms;
    - ``STRONG_LEXICAL_OCR_SIGNAL``: a contiguous multi-token query phrase
      appears in GT OCR, or an overlapping term has df <= ``rare_max_df``;
    - ``WEAK_OCR_SIGNAL``: overlap exists but only through common terms.

    Expected input:
        - query_text: the query text.
        - gt_chunks: OCR chunks belonging to the ground-truth video.
        - document_frequencies: corpus-level DF map.
        - config: thresholds (defaults to :class:`AnalysisConfig`).

    Returns:
        - One of :data:`STRATA`.
    """
    config = config or AnalysisConfig()
    chunks_with_text = [c for c in gt_chunks if content_tokens(c.text)]
    if not chunks_with_text:
        return "NO_GT_OCR_EVIDENCE"
    query_terms = content_tokens(query_text)
    query_set = set(query_terms)
    has_overlap = False
    has_rare = False
    has_phrase = False
    for chunk in chunks_with_text:
        chunk_tokens = normalize_tokens(chunk.text)
        overlap = query_set & set(chunk_tokens)
        if not overlap:
            continue
        has_overlap = True
        if any(document_frequencies.get(term, 0) <= config.rare_max_df for term in overlap):
            has_rare = True
        if find_phrases(query_terms, chunk_tokens, min_len=config.min_phrase_tokens):
            has_phrase = True
    if not has_overlap:
        return "GT_OCR_NO_QUERY_OVERLAP"
    if has_phrase or has_rare:
        return "STRONG_LEXICAL_OCR_SIGNAL"
    return "WEAK_OCR_SIGNAL"


def stratum_deltas(
    strata: Sequence[str],
    deltas: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Summarize mean/count of per-query deltas grouped by stratum."""
    _require(len(strata) == len(deltas), "aligned per-query lists required")
    grouped: dict[str, list[float]] = {}
    for stratum, delta in zip(strata, deltas):
        grouped.setdefault(stratum, []).append(float(delta))
    return {
        stratum: {"n": len(values), "mean_delta": sum(values) / len(values)}
        for stratum, values in sorted(grouped.items())
    }


# ------------------------------------------------------------------ artifact audit


def chunk_artifact_stats(chunks: Sequence[OcrChunk]) -> dict[str, Any]:
    """Compute representation/dilution statistics over OCR chunks.

    Reports observed distributions only; no dilution conclusion is drawn
    here.
    """
    if not chunks:
        return {"num_chunks": 0}
    source_counts = [c.source_count for c in chunks]
    lengths = [len(c.text) for c in chunks]
    unique_lines: list[int] = []
    duplicate_ratios: list[float] = []
    spans = [c.frame_end - c.frame_start + 1 for c in chunks]
    dominated = 0
    for chunk in chunks:
        lines = [line.strip() for line in chunk.text.split(" ") if line.strip()]
        normalized_lines = [" ".join(normalize_tokens(line)) for line in lines]
        counts = Counter(normalized_lines)
        if counts:
            most_common_count = counts.most_common(1)[0][1]
            if len(normalized_lines) and most_common_count / len(normalized_lines) >= 0.5:
                dominated += 1
        unique_lines.append(len(set(normalized_lines)))
        duplicate_ratios.append(
            1.0 - (len(set(normalized_lines)) / len(normalized_lines)) if normalized_lines else 0.0
        )

    dfs = build_document_frequencies(chunks)
    df_values = sorted(dfs.values())

    def dist(values: Sequence[float]) -> dict[str, float]:
        arr = np.sort(np.asarray(values, dtype=np.float64))

        def q(p: float) -> float:
            return float(arr[min(len(arr) - 1, int(p * len(arr)))]) if len(arr) else 0.0

        return {
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "p50": q(0.5),
            "p90": q(0.9),
            "p99": q(0.99),
            "max": float(arr.max()) if len(arr) else 0.0,
        }

    return {
        "num_chunks": len(chunks),
        "source_records_per_chunk": dist(source_counts),
        "text_length_chars": dist(lengths),
        "unique_normalized_words": dist(unique_lines),
        "duplicate_ratio": dist(duplicate_ratios),
        "frame_span": dist(spans),
        "fraction_dominated_by_repeated_text": dominated / len(chunks),
        "vocab_size": len(dfs),
        "token_df_distribution": {
            "p50": df_values[len(df_values) // 2] if df_values else 0,
            "p90": df_values[min(len(df_values) - 1, int(0.9 * len(df_values)))] if df_values else 0,
            "max": df_values[-1] if df_values else 0,
            "terms_with_df_le_10": sum(1 for v in df_values if v <= 10),
        },
    }
