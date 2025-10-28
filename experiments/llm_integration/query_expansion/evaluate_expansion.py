"""
Evaluate the impact of GPT-4 query expansion on retrieval performance.
Compares baseline search vs expanded query search.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import nltk
from pathlib import Path
from openai import OpenAI

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from evaluation.metrics.ranking_metrics import aggregate_metrics, evaluate_ranking
from experiments.llm_integration.query_expansion.gpt4_expansion import (
    expand_query_gpt4, expand_query_to_text
)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def load_search_system():
    """Load pre-built search system"""

    print("Loading search system...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog.csv")
    backlog_df = pd.read_csv(backlog_path)

    # Prepare text
    backlog_df['combined_text'] = (
        backlog_df['Title'].fillna('') + ". " +
        backlog_df['Description'].fillna('') + ". " +
        backlog_df['Tags'].fillna('')
    )
    backlog_df['combined_text'] = backlog_df['combined_text'].str.lower()
    documents = backlog_df['combined_text'].tolist()

    # BM25
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Semantic
    os.environ.pop('HF_TOKEN', None)
    os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
    model = SentenceTransformer('all-mpnet-base-v2', token=False)
    doc_embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    return {
        'bm25': bm25,
        'model': model,
        'doc_embeddings': doc_embeddings,
        'df': backlog_df
    }


def search_hybrid(query, search_system, bm25_weight=0.5, semantic_weight=0.5, top_k=50):
    """Hybrid search"""

    bm25 = search_system['bm25']
    model = search_system['model']
    doc_embeddings = search_system['doc_embeddings']
    df = search_system['df']

    # BM25
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Semantic
    query_embedding = model.encode([query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Normalize & combine
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    semantic_norm = (semantic_scores - np.min(semantic_scores)) / (np.ptp(semantic_scores) + 1e-8)
    final_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm

    # Top k
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return df.iloc[top_indices]['ID'].tolist(), final_scores[top_indices]


def evaluate_with_expansion(api_key, sample_size=30):
    """
    Compare baseline vs query expansion performance.

    Args:
        api_key: OpenAI API key
        sample_size: Number of queries to test

    Returns:
        dict: Comparison results
    """

    print("="*80)
    print("QUERY EXPANSION EVALUATION")
    print("="*80)

    # Load test set
    print("\n1. Loading test data...")
    test_set_path = Path("evaluation/test_sets/test_set_compact.csv")
    test_df = pd.read_csv(test_set_path)

    # Sample queries (stratified by type)
    if 'query_type' in test_df.columns:
        sampled = test_df.groupby('query_type')['query'].apply(
            lambda x: x.drop_duplicates().sample(n=min(6, len(x.unique())), random_state=42)
        ).reset_index(drop=True)
        test_queries = sampled.tolist()[:sample_size]
    else:
        test_queries = test_df['query'].drop_duplicates().sample(n=sample_size, random_state=42).tolist()

    # Organize test data
    queries_data = {}
    for query in test_queries:
        query_data = test_df[test_df['query'] == query]
        queries_data[query] = {
            'document_ids': query_data['document_id'].tolist(),
            'relevances': query_data['relevance'].tolist(),
            'query_type': query_data['query_type'].iloc[0] if 'query_type' in query_data.columns else 'unknown'
        }

    print(f"   Testing on {len(queries_data)} queries")

    # Load search system
    print("\n2. Loading search system...")
    search_system = load_search_system()

    # Evaluate baseline (no expansion)
    print("\n3. Evaluating baseline (no expansion)...")
    baseline_results = []
    for query, data in queries_data.items():
        retrieved_docs, scores = search_hybrid(query, search_system, top_k=50)

        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}
        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in retrieved_docs]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)
        ground_truth_relevances = data['relevances']

        baseline_results.append({
            'query': query,
            'relevances': relevances,
            'total_relevant': total_relevant,
            'ground_truth_relevances': ground_truth_relevances,
            'query_type': data['query_type']
        })

    # Calculate baseline metrics
    baseline_relevances = [r['relevances'] for r in baseline_results]
    baseline_total_relevant = [r['total_relevant'] for r in baseline_results]
    baseline_ground_truth = [r['ground_truth_relevances'] for r in baseline_results]
    baseline_metrics = aggregate_metrics(baseline_relevances, baseline_total_relevant, k_values=[5, 10, 20], all_ground_truth_relevances=baseline_ground_truth)

    print(f"   Baseline NDCG@10: {baseline_metrics['ndcg@10']:.4f}")
    print(f"   Baseline Recall@10: {baseline_metrics['recall@10']:.4f}")

    # Evaluate with query expansion
    print("\n4. Evaluating with query expansion...")
    expansion_results = []
    total_expansion_cost = 0.0

    for i, (query, data) in enumerate(queries_data.items(), 1):
        print(f"   [{i}/{len(queries_data)}] Expanding: '{query[:50]}...'")

        # Expand query
        expansion_result = expand_query_gpt4(query, api_key)
        total_expansion_cost += expansion_result['cost_dollars']

        # Convert to search string
        expanded_query = expand_query_to_text(expansion_result)

        # Search with expanded query
        retrieved_docs, scores = search_hybrid(expanded_query, search_system, top_k=50)

        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}
        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in retrieved_docs]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)
        ground_truth_relevances = data['relevances']

        expansion_results.append({
            'query': query,
            'expanded_query': expanded_query,
            'relevances': relevances,
            'total_relevant': total_relevant,
            'ground_truth_relevances': ground_truth_relevances,
            'query_type': data['query_type'],
            'expansion_cost': expansion_result['cost_dollars']
        })

    # Calculate expansion metrics
    expansion_relevances = [r['relevances'] for r in expansion_results]
    expansion_total_relevant = [r['total_relevant'] for r in expansion_results]
    expansion_ground_truth = [r['ground_truth_relevances'] for r in expansion_results]
    expansion_metrics = aggregate_metrics(expansion_relevances, expansion_total_relevant, k_values=[5, 10, 20], all_ground_truth_relevances=expansion_ground_truth)

    print(f"\n   With Expansion NDCG@10: {expansion_metrics['ndcg@10']:.4f}")
    print(f"   With Expansion Recall@10: {expansion_metrics['recall@10']:.4f}")

    # Compare results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    comparison = {
        'Metric': [],
        'Baseline': [],
        'With Expansion': [],
        'Improvement': []
    }

    for metric in ['ndcg@5', 'ndcg@10', 'ndcg@20', 'map', 'mrr', 'precision@10', 'recall@10']:
        baseline_val = baseline_metrics[metric]
        expansion_val = expansion_metrics[metric]
        improvement = ((expansion_val - baseline_val) / baseline_val) * 100

        comparison['Metric'].append(metric)
        comparison['Baseline'].append(f"{baseline_val:.4f}")
        comparison['With Expansion'].append(f"{expansion_val:.4f}")
        comparison['Improvement'].append(f"{improvement:+.2f}%")

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Cost analysis
    print(f"\n" + "="*80)
    print("COST ANALYSIS")
    print("="*80)
    print(f"\nTotal queries tested: {len(queries_data)}")
    print(f"Total expansion cost: ${total_expansion_cost:.4f}")
    print(f"Avg cost per query: ${total_expansion_cost/len(queries_data):.4f}")

    # Production estimates
    queries_per_day = 250 * 20  # 250 users, 20 queries/day
    daily_cost = (total_expansion_cost / len(queries_data)) * queries_per_day
    annual_cost = daily_cost * 250

    print(f"\nProduction estimates (250 users @ 20 queries/day):")
    print(f"  Daily cost: ${daily_cost:.2f}")
    print(f"  Annual cost: ${annual_cost:,.2f}")

    # Value assessment
    print(f"\n" + "="*80)
    print("VALUE ASSESSMENT")
    print("="*80)

    ndcg_improvement = ((expansion_metrics['ndcg@10'] - baseline_metrics['ndcg@10']) / baseline_metrics['ndcg@10']) * 100
    recall_improvement = ((expansion_metrics['recall@10'] - baseline_metrics['recall@10']) / baseline_metrics['recall@10']) * 100

    print(f"\nPerformance Gains:")
    print(f"  NDCG@10 improvement: {ndcg_improvement:+.2f}%")
    print(f"  Recall@10 improvement: {recall_improvement:+.2f}%")

    print(f"\nCost:")
    print(f"  ${annual_cost:,.2f}/year")

    print(f"\nRecommendation:")
    if ndcg_improvement > 10 and recall_improvement > 10:
        print("  ✓ Strong performance gains justify cost for premium use cases")
    elif ndcg_improvement > 5:
        print("  ~ Moderate gains - consider for power users or opt-in feature")
    else:
        print("  ✗ Minimal gains do not justify cost - explore alternatives")

    # Save results
    output_path = Path("experiments/llm_integration/query_expansion")
    results = {
        'baseline_metrics': baseline_metrics,
        'expansion_metrics': expansion_metrics,
        'comparison': comparison,
        'cost_analysis': {
            'total_cost': total_expansion_cost,
            'avg_cost_per_query': total_expansion_cost / len(queries_data),
            'estimated_annual_cost': annual_cost
        },
        'improvements': {
            'ndcg@10': ndcg_improvement,
            'recall@10': recall_improvement
        }
    }

    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to: {output_path / 'evaluation_results.json'}")

    return results


def main():
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    evaluate_with_expansion(api_key, sample_size=30)


if __name__ == "__main__":
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    main()
