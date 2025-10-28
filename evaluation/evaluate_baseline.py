"""
Evaluate baseline search systems: BM25, Semantic, and Hybrid.
Compares performance using standard ranking metrics.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import nltk
from pathlib import Path
from collections import defaultdict
import json

# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent))
from metrics.ranking_metrics import aggregate_metrics, evaluate_ranking

# Import model components
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    """Tokenize text using NLTK"""
    return nltk.word_tokenize(text.lower())


def load_test_set(test_set_path):
    """Load and organize test set by query"""
    test_df = pd.read_csv(test_set_path)

    # Group by query
    queries_data = {}
    for query in test_df['query'].unique():
        query_data = test_df[test_df['query'] == query]
        queries_data[query] = {
            'document_ids': query_data['document_id'].tolist(),
            'relevances': query_data['relevance'].tolist(),
            'query_type': query_data['query_type'].iloc[0] if 'query_type' in query_data.columns else 'unknown'
        }

    return queries_data


def build_search_system(backlog_df):
    """Build BM25 and semantic search indices"""

    # Prepare combined text
    backlog_df['combined_text'] = (
        backlog_df['Title'].fillna('') + ". " +
        backlog_df['Description'].fillna('') + ". " +
        backlog_df['Type'].fillna('') + " " +
        backlog_df['Component'].fillna('') + " " +
        backlog_df['Severity'].fillna('')
    )
    backlog_df['combined_text'] = backlog_df['combined_text'].str.lower()

    documents = backlog_df['combined_text'].tolist()

    # Build BM25 index
    print("  Building BM25 index...")
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Build semantic embeddings
    print("  Loading sentence transformer model...")
    # Disable HF token usage to avoid authentication issues
    os.environ.pop('HF_TOKEN', None)
    os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
    model = SentenceTransformer('all-mpnet-base-v2', token=False)

    print("  Computing embeddings...")
    doc_embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    return {
        'bm25': bm25,
        'tokenized_docs': tokenized_docs,
        'model': model,
        'doc_embeddings': doc_embeddings,
        'df': backlog_df
    }


def search_bm25(query, search_system, top_k=50):
    """Perform BM25 search"""
    bm25 = search_system['bm25']
    df = search_system['df']

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top k
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_docs = df.iloc[top_indices]

    return top_docs['ID'].tolist(), scores[top_indices]


def search_semantic(query, search_system, top_k=50):
    """Perform semantic search"""
    model = search_system['model']
    doc_embeddings = search_system['doc_embeddings']
    df = search_system['df']

    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Get top k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_docs = df.iloc[top_indices]

    return top_docs['ID'].tolist(), similarities[top_indices]


def search_hybrid(query, search_system, bm25_weight=0.5, semantic_weight=0.5, top_k=50):
    """Perform hybrid search (BM25 + Semantic)"""
    bm25 = search_system['bm25']
    model = search_system['model']
    doc_embeddings = search_system['doc_embeddings']
    df = search_system['df']

    # BM25 scores
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Semantic scores
    query_embedding = model.encode([query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Normalize scores
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    semantic_norm = (semantic_scores - np.min(semantic_scores)) / (np.ptp(semantic_scores) + 1e-8)

    # Combine
    final_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm

    # Get top k
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_docs = df.iloc[top_indices]

    return top_docs['ID'].tolist(), final_scores[top_indices]


def evaluate_system(queries_data, search_function, search_system, system_name, top_k=50):
    """Evaluate a search system on all queries"""

    print(f"\nEvaluating {system_name}...")

    all_relevances = []
    all_total_relevant = []
    all_ground_truth_relevances = []
    per_query_results = []

    for query, data in queries_data.items():
        # Perform search
        retrieved_docs, scores = search_function(query, search_system, top_k=top_k)

        # Map retrieved docs to relevance scores
        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}

        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in retrieved_docs]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)

        # Store ground truth relevances for correct NDCG calculation
        ground_truth_relevances = data['relevances']

        all_relevances.append(relevances)
        all_total_relevant.append(total_relevant)
        all_ground_truth_relevances.append(ground_truth_relevances)

        # Per-query metrics
        per_query_metrics = evaluate_ranking(relevances, total_relevant, k_values=[5, 10, 20], ground_truth_relevances=ground_truth_relevances)
        per_query_metrics['query'] = query
        per_query_metrics['query_type'] = data['query_type']
        per_query_results.append(per_query_metrics)

    # Aggregate metrics
    aggregated = aggregate_metrics(all_relevances, all_total_relevant, k_values=[5, 10, 20], all_ground_truth_relevances=all_ground_truth_relevances)

    # Analyze by query type
    query_type_results = defaultdict(list)
    for result in per_query_results:
        qtype = result['query_type']
        query_type_results[qtype].append(result)

    # Calculate per-type averages
    type_averages = {}
    for qtype, results in query_type_results.items():
        if results:
            type_avg = {
                'count': len(results),
                'ndcg@10': np.mean([r['ndcg@10'] for r in results]),
                'precision@10': np.mean([r['precision@10'] for r in results]),
                'recall@10': np.mean([r['recall@10'] for r in results]),
            }
            type_averages[qtype] = type_avg

    return {
        'aggregated': aggregated,
        'per_query': per_query_results,
        'by_query_type': type_averages,
        'system_name': system_name
    }


def print_results(results):
    """Print evaluation results in a nice format"""

    system_name = results['system_name']
    metrics = results['aggregated']

    print(f"\n{'='*60}")
    print(f"{system_name} - Aggregated Metrics")
    print(f"{'='*60}")

    # Main metrics
    print(f"\nNDCG:")
    print(f"  NDCG@5:  {metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  NDCG@20: {metrics['ndcg@20']:.4f}")

    print(f"\nPrecision:")
    print(f"  P@5:  {metrics['precision@5']:.4f}")
    print(f"  P@10: {metrics['precision@10']:.4f}")
    print(f"  P@20: {metrics['precision@20']:.4f}")

    print(f"\nRecall:")
    print(f"  R@5:  {metrics['recall@5']:.4f}")
    print(f"  R@10: {metrics['recall@10']:.4f}")
    print(f"  R@20: {metrics['recall@20']:.4f}")

    print(f"\nOverall:")
    print(f"  MAP: {metrics['map']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")

    print(f"\nHit Rate:")
    print(f"  HR@5:  {metrics['hit_rate@5']:.4f}")
    print(f"  HR@10: {metrics['hit_rate@10']:.4f}")
    print(f"  HR@20: {metrics['hit_rate@20']:.4f}")

    # By query type
    print(f"\n{'='*60}")
    print(f"Performance by Query Type")
    print(f"{'='*60}")

    for qtype, type_metrics in results['by_query_type'].items():
        print(f"\n{qtype.upper()} ({type_metrics['count']} queries):")
        print(f"  NDCG@10:      {type_metrics['ndcg@10']:.4f}")
        print(f"  Precision@10: {type_metrics['precision@10']:.4f}")
        print(f"  Recall@10:    {type_metrics['recall@10']:.4f}")


def compare_systems(all_results):
    """Compare multiple systems side-by-side"""

    print(f"\n\n{'='*80}")
    print(f"COMPARISON: All Systems")
    print(f"{'='*80}\n")

    # Create comparison table
    metrics_to_compare = ['ndcg@5', 'ndcg@10', 'ndcg@20', 'map', 'mrr', 'precision@10', 'recall@10']

    print(f"{'Metric':<15}", end='')
    for result in all_results:
        print(f"{result['system_name']:<15}", end='')
    print()
    print("-" * (15 + 15 * len(all_results)))

    for metric in metrics_to_compare:
        print(f"{metric:<15}", end='')
        for result in all_results:
            value = result['aggregated'][metric]
            print(f"{value:>14.4f} ", end='')
        print()

    # Highlight best performer
    print(f"\n{'='*80}")
    print("Best Performer by Metric:")
    print(f"{'='*80}")

    for metric in metrics_to_compare:
        scores = [(result['system_name'], result['aggregated'][metric]) for result in all_results]
        best = max(scores, key=lambda x: x[1])
        print(f"  {metric:<15}: {best[0]:<15} ({best[1]:.4f})")


def save_results(all_results, output_dir):
    """Save results to JSON and CSV"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    comparison_data = []
    for result in all_results:
        row = {'system': result['system_name']}
        row.update(result['aggregated'])
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path / "baseline_comparison.csv", index=False)

    # Save detailed per-query results
    for result in all_results:
        per_query_df = pd.DataFrame(result['per_query'])
        # Clean filename - remove special characters
        clean_name = result['system_name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        filename = f"{clean_name}_per_query.csv"
        per_query_df.to_csv(output_path / filename, index=False)

    # Save JSON summary
    summary = {
        'systems': [
            {
                'name': result['system_name'],
                'aggregated_metrics': result['aggregated'],
                'by_query_type': result['by_query_type']
            }
            for result in all_results
        ]
    }

    with open(output_path / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUCCESS] Results saved to {output_path}/")


def main():
    """Run baseline evaluation"""

    print("="*80)
    print("BASELINE EVALUATION")
    print("="*80)

    # Load synthetic backlog
    print("\n1. Loading synthetic backlog...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog_from_queries.csv")

    if not backlog_path.exists():
        print("Error: synthetic_backlog_from_queries.csv not found.")
        return

    backlog_df = pd.read_csv(backlog_path)
    print(f"   Loaded {len(backlog_df)} backlog items")

    # Load test set
    print("\n2. Loading test set...")
    test_set_path = Path("evaluation/test_sets/test_set_llm_compact.csv")

    if not test_set_path.exists():
        print("Error: test_set_llm_compact.csv not found.")
        return

    queries_data = load_test_set(test_set_path)
    print(f"   Loaded {len(queries_data)} test queries")

    # Build search systems
    print("\n3. Building search systems...")
    search_system = build_search_system(backlog_df)
    print("   Search systems ready!")

    # Evaluate systems
    print("\n4. Running evaluations...")

    all_results = []

    # BM25 Only
    bm25_results = evaluate_system(
        queries_data,
        search_bm25,
        search_system,
        "BM25 Only"
    )
    all_results.append(bm25_results)
    print_results(bm25_results)

    # Semantic Only
    semantic_results = evaluate_system(
        queries_data,
        search_semantic,
        search_system,
        "Semantic Only"
    )
    all_results.append(semantic_results)
    print_results(semantic_results)

    # Hybrid (50/50)
    def hybrid_search_50_50(query, search_sys, top_k=50):
        return search_hybrid(query, search_sys, bm25_weight=0.5, semantic_weight=0.5, top_k=top_k)

    hybrid_results = evaluate_system(
        queries_data,
        hybrid_search_50_50,
        search_system,
        "Hybrid (50/50)"
    )
    all_results.append(hybrid_results)
    print_results(hybrid_results)

    # Compare systems
    compare_systems(all_results)

    # Save results
    save_results(all_results, "evaluation/results")

    print("\n" + "="*80)
    print("[SUCCESS] Baseline evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    main()
