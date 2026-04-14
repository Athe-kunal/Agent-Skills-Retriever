"""Evaluation helpers for skill-name retrieval quality."""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence

from loguru import logger as log

from ast_skills.data_gen.datamodels import RetrieverDataModel


class RetrievalEvalResult(NamedTuple):
    """Aggregated retrieval metrics from seed-question evaluation."""

    total_queries: int
    misses: int
    hit_counts: dict[int, int]
    hit_rates: dict[int, float]
    mrr_at_k: dict[int, float]
    mean_reciprocal_rank: float
    mean_first_relevant_rank: float | None
    ndcg_at_k: dict[int, float] | None


class _AggregateSums(NamedTuple):
    """Intermediate sums and counts used to build final metrics."""

    misses: int
    hit_counts: dict[int, int]
    rr_sums_at_k: dict[int, float]
    rr_sum: float
    rank_sum: int
    found_count: int
    dcg_sums_at_k: dict[int, float]


def _normalize_ks(ks: Sequence[int]) -> tuple[int, ...]:
    """Returns sorted, unique positive cutoffs."""
    normalized_ks = tuple(sorted({k for k in ks if k > 0}))
    if not normalized_ks:
        raise ValueError("ks must contain at least one positive integer.")
    return normalized_ks


def _normalize_name(name: str) -> str:
    """Normalizes a skill name for robust equality checks."""
    return name.strip().casefold()


def _iter_seed_question_targets(
    models: Sequence[RetrieverDataModel],
) -> list[tuple[str, str]]:
    """Flattens models into (question, expected_skill_name) pairs."""
    pairs: list[tuple[str, str]] = []
    for model in models:
        expected_name = model.name.strip()
        if not expected_name:
            continue
        for question in model.seed_questions:
            normalized_question = question.strip()
            if normalized_question:
                pairs.append((normalized_question, expected_name))
    log.info(f"{len(pairs)=}")
    return pairs


def _first_relevant_rank(
    retrieved_names: Sequence[str],
    expected_name: str,
) -> int | None:
    """Returns 1-indexed rank of expected name; None when absent."""
    normalized_expected_name = _normalize_name(expected_name)
    for index, retrieved_name in enumerate(retrieved_names, start=1):
        if _normalize_name(retrieved_name) == normalized_expected_name:
            return index
    return None


def _init_int_map(ks: tuple[int, ...]) -> dict[int, int]:
    """Creates zero-initialized int map for each cutoff."""
    return {k: 0 for k in ks}


def _init_float_map(ks: tuple[int, ...]) -> dict[int, float]:
    """Creates zero-initialized float map for each cutoff."""
    return {k: 0.0 for k in ks}


def _safe_divide(value: float, count: int) -> float:
    """Returns value / count, or 0.0 when count is zero."""
    if count <= 0:
        return 0.0
    return value / count


def _evaluate_pairs(
    pairs: Sequence[tuple[str, str]],
    retrieve_fn: Callable[[str, int], Sequence[str]],
    ks: tuple[int, ...],
    include_ndcg: bool,
) -> _AggregateSums:
    """Collects intermediate sums for all queries."""
    misses = 0
    rr_sum = 0.0
    rank_sum = 0
    found_count = 0
    max_k = max(ks)
    hit_counts = _init_int_map(ks)
    rr_sums_at_k = _init_float_map(ks)
    dcg_sums_at_k = _init_float_map(ks)

    for question, expected_name in pairs:
        retrieved_names = retrieve_fn(question, max_k)
        rank = _first_relevant_rank(retrieved_names, expected_name)
        if rank is None:
            misses += 1
            continue

        reciprocal_rank = 1.0 / rank
        rr_sum += reciprocal_rank
        rank_sum += rank
        found_count += 1

        for k in ks:
            if rank <= k:
                hit_counts[k] += 1
                rr_sums_at_k[k] += reciprocal_rank
                if include_ndcg:
                    dcg_sums_at_k[k] += 1.0 / math.log2(rank + 1.0)

    return _AggregateSums(
        misses=misses,
        hit_counts=hit_counts,
        rr_sums_at_k=rr_sums_at_k,
        rr_sum=rr_sum,
        rank_sum=rank_sum,
        found_count=found_count,
        dcg_sums_at_k=dcg_sums_at_k,
    )


def evaluate_retriever_hits(
    models: Sequence[RetrieverDataModel],
    retrieve_fn: Callable[[str, int], Sequence[str]],
    ks: Sequence[int] = (1, 2, 3, 5, 10),
    *,
    include_ndcg: bool = False,
) -> RetrievalEvalResult:
    """Evaluates retrieval quality for seed-question queries.

    Args:
      models: Gold rows with expected skill names and seed questions.
      retrieve_fn: Callback of shape `(query, top_k) -> list[str]`.
      ks: Cutoffs used for hit@k and MRR@k.
      include_ndcg: Whether to compute nDCG@k (off by default).

    Returns:
      Aggregated metrics across all seed questions.
    """
    normalized_ks = _normalize_ks(ks)
    pairs = _iter_seed_question_targets(models)
    total_queries = len(pairs)

    if include_ndcg:
        log.info("nDCG enabled for single-relevant evaluation labels.")
    aggregates = _evaluate_pairs(pairs, retrieve_fn, normalized_ks, include_ndcg)

    hit_rates = {
        k: _safe_divide(float(aggregates.hit_counts[k]), total_queries)
        for k in normalized_ks
    }
    mrr_at_k = {
        k: _safe_divide(aggregates.rr_sums_at_k[k], total_queries) for k in normalized_ks
    }
    mean_reciprocal_rank = _safe_divide(aggregates.rr_sum, total_queries)
    mean_first_relevant_rank: float | None = None
    if aggregates.found_count > 0:
        mean_first_relevant_rank = aggregates.rank_sum / aggregates.found_count

    ndcg_at_k: dict[int, float] | None = None
    if include_ndcg:
        ndcg_at_k = {
            k: _safe_divide(aggregates.dcg_sums_at_k[k], total_queries)
            for k in normalized_ks
        }

    result = RetrievalEvalResult(
        total_queries=total_queries,
        misses=aggregates.misses,
        hit_counts=aggregates.hit_counts,
        hit_rates=hit_rates,
        mrr_at_k=mrr_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_first_relevant_rank=mean_first_relevant_rank,
        ndcg_at_k=ndcg_at_k,
    )
    log.info(f"{total_queries=}, {result.misses=}, {result.hit_rates=}")
    log.info(
        f"{result.mrr_at_k=}, {result.mean_reciprocal_rank=}, "
        f"{result.mean_first_relevant_rank=}"
    )
    return result
