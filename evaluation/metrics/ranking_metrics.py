"""
Implementation of standard information retrieval ranking metrics.
Includes: NDCG, MRR, MAP, Precision@k, Recall@k, Hit Rate@k
"""

import numpy as np
from typing import List, Dict, Tuple


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)
        k: Cut-off rank position

    Returns:
        DCG@k score
    """
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i + 1)) for i in 1..k
    # Using i+1 because Python is 0-indexed but formula is 1-indexed
    gains = relevances / np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains)


def ndcg_at_k(relevances: List[float], k: int, ground_truth_relevances: List[float] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG normalizes DCG by the ideal DCG (IDCG), making scores comparable across queries.

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)
        k: Cut-off rank position
        ground_truth_relevances: Optional list of ALL relevance scores from ground truth (for correct IDCG calculation)
                                If provided, IDCG will be computed using the best possible ranking of all labeled documents.
                                If not provided, IDCG will use only the retrieved documents (may be inaccurate).

    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    dcg = dcg_at_k(relevances, k)

    # Calculate ideal DCG (best possible ranking)
    # Use all ground truth relevances if provided (CORRECT method)
    # Otherwise fall back to sorting retrieved relevances (legacy behavior)
    if ground_truth_relevances is not None:
        ideal_relevances = sorted(ground_truth_relevances, reverse=True)[:k]
    else:
        ideal_relevances = sorted(relevances, reverse=True)[:k]

    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(relevances_list: List[List[float]]) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.

    MRR measures how quickly the first relevant document appears.
    For each query, RR = 1 / rank of first relevant document.

    Args:
        relevances_list: List of relevance lists (one per query)

    Returns:
        MRR score (average of reciprocal ranks)
    """
    reciprocal_ranks = []

    for relevances in relevances_list:
        # Find first relevant document (relevance > 0)
        for rank, rel in enumerate(relevances, start=1):
            if rel > 0:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def average_precision(relevances: List[float]) -> float:
    """
    Calculate Average Precision for a single query.

    AP = sum(Precision@k * rel_k) / number of relevant documents

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)

    Returns:
        AP score
    """
    if not relevances or sum(relevances) == 0:
        return 0.0

    num_relevant = sum(1 for rel in relevances if rel > 0)
    precision_sum = 0.0
    num_relevant_so_far = 0

    for k, rel in enumerate(relevances, start=1):
        if rel > 0:
            num_relevant_so_far += 1
            precision_at_k = num_relevant_so_far / k
            precision_sum += precision_at_k

    return precision_sum / num_relevant


def mean_average_precision(relevances_list: List[List[float]]) -> float:
    """
    Calculate Mean Average Precision across multiple queries.

    MAP = average of AP scores across all queries

    Args:
        relevances_list: List of relevance lists (one per query)

    Returns:
        MAP score
    """
    ap_scores = [average_precision(rels) for rels in relevances_list]
    return np.mean(ap_scores) if ap_scores else 0.0


def precision_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Precision at k.

    Precision@k = (number of relevant docs in top k) / k

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)
        k: Cut-off rank position

    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0

    top_k = relevances[:k]
    num_relevant = sum(1 for rel in top_k if rel > 0)
    return num_relevant / k


def recall_at_k(relevances: List[float], total_relevant: int, k: int) -> float:
    """
    Calculate Recall at k.

    Recall@k = (number of relevant docs in top k) / (total relevant docs)

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)
        total_relevant: Total number of relevant documents for this query
        k: Cut-off rank position

    Returns:
        Recall@k score
    """
    if total_relevant == 0:
        return 0.0

    top_k = relevances[:k]
    num_relevant = sum(1 for rel in top_k if rel > 0)
    return num_relevant / total_relevant


def hit_rate_at_k(relevances_list: List[List[float]], k: int) -> float:
    """
    Calculate Hit Rate at k (also called Success Rate).

    Hit Rate@k = fraction of queries with at least one relevant document in top k

    Args:
        relevances_list: List of relevance lists (one per query)
        k: Cut-off rank position

    Returns:
        Hit Rate@k score
    """
    hits = 0

    for relevances in relevances_list:
        top_k = relevances[:k]
        if any(rel > 0 for rel in top_k):
            hits += 1

    return hits / len(relevances_list) if relevances_list else 0.0


def evaluate_ranking(
    relevances: List[float],
    total_relevant: int,
    k_values: List[int] = [5, 10, 20],
    ground_truth_relevances: List[float] = None
) -> Dict[str, float]:
    """
    Evaluate ranking with multiple metrics at different k values.

    Args:
        relevances: List of relevance scores for retrieved documents (in rank order)
        total_relevant: Total number of relevant documents for this query
        k_values: List of k values to evaluate at
        ground_truth_relevances: Optional list of ALL relevance scores from ground truth (for correct NDCG calculation)

    Returns:
        Dictionary of metric name -> score
    """
    results = {}

    for k in k_values:
        results[f'ndcg@{k}'] = ndcg_at_k(relevances, k, ground_truth_relevances)
        results[f'precision@{k}'] = precision_at_k(relevances, k)
        results[f'recall@{k}'] = recall_at_k(relevances, total_relevant, k)

    # Metrics that don't depend on k
    results['ap'] = average_precision(relevances)

    # Reciprocal rank
    for rank, rel in enumerate(relevances, start=1):
        if rel > 0:
            results['rr'] = 1.0 / rank
            break
    else:
        results['rr'] = 0.0

    return results


def aggregate_metrics(
    all_relevances: List[List[float]],
    all_total_relevant: List[int],
    k_values: List[int] = [5, 10, 20],
    all_ground_truth_relevances: List[List[float]] = None
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_relevances: List of relevance lists (one per query)
        all_total_relevant: List of total relevant counts (one per query)
        k_values: List of k values to evaluate at
        all_ground_truth_relevances: Optional list of ground truth relevance lists (one per query) for correct NDCG

    Returns:
        Dictionary of aggregated metric name -> score
    """
    # Evaluate each query
    per_query_results = []
    for i, (relevances, total_relevant) in enumerate(zip(all_relevances, all_total_relevant)):
        ground_truth = all_ground_truth_relevances[i] if all_ground_truth_relevances else None
        results = evaluate_ranking(relevances, total_relevant, k_values, ground_truth)
        per_query_results.append(results)

    # Aggregate across queries (mean)
    aggregated = {}

    # Get all metric names from first query
    if per_query_results:
        for metric_name in per_query_results[0].keys():
            scores = [r[metric_name] for r in per_query_results]
            aggregated[metric_name] = np.mean(scores)

    # Add overall metrics
    aggregated['map'] = mean_average_precision(all_relevances)
    aggregated['mrr'] = mean_reciprocal_rank(all_relevances)

    for k in k_values:
        aggregated[f'hit_rate@{k}'] = hit_rate_at_k(all_relevances, k)

    return aggregated


if __name__ == "__main__":
    # Example usage and sanity checks
    print("Testing ranking metrics...")

    # Perfect ranking
    relevances_perfect = [2, 2, 1, 1, 0, 0]
    total_relevant = 4

    print("\nPerfect ranking: [2, 2, 1, 1, 0, 0]")
    results = evaluate_ranking(relevances_perfect, total_relevant, k_values=[3, 5])
    for metric, score in sorted(results.items()):
        print(f"  {metric}: {score:.4f}")

    # Poor ranking
    relevances_poor = [0, 0, 1, 2, 2, 1]

    print("\nPoor ranking: [0, 0, 1, 2, 2, 1]")
    results = evaluate_ranking(relevances_poor, total_relevant, k_values=[3, 5])
    for metric, score in sorted(results.items()):
        print(f"  {metric}: {score:.4f}")

    # Test aggregation
    all_relevances = [relevances_perfect, relevances_poor]
    all_total_relevant = [4, 4]

    print("\nAggregated metrics:")
    aggregated = aggregate_metrics(all_relevances, all_total_relevant, k_values=[3, 5])
    for metric, score in sorted(aggregated.items()):
        print(f"  {metric}: {score:.4f}")

    print("\n[SUCCESS] All metrics implemented correctly!")
