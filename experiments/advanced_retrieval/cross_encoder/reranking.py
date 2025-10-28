"""
Two-stage retrieval with cross-encoder reranking.

Stage 1: Fast bi-encoder retrieves top-50 candidates
Stage 2: Accurate cross-encoder reranks to top-10

Cross-encoders are more accurate but slower (encode query+doc together).
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import time
import nltk
from pathlib import Path
from sentence_transformers import CrossEncoder

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from evaluation.metrics.ranking_metrics import aggregate_metrics

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def load_search_system():
    """Load bi-encoder search system"""

    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog_from_queries.csv")
    backlog_df = pd.read_csv(backlog_path)

    backlog_df['combined_text'] = (
        backlog_df['Title'].fillna('') + ". " +
        backlog_df['Description'].fillna('') + ". " +
        backlog_df['Tags'].fillna('')
    )
    backlog_df['combined_text'] = backlog_df['combined_text'].str.lower()
    documents = backlog_df['combined_text'].tolist()

    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

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


def retrieve_candidates_hybrid(query, search_system, top_k=50):
    """
    Stage 1: Fast hybrid retrieval of candidates.

    Args:
        query: Search query
        search_system: Search system dict
        top_k: Number of candidates to retrieve

    Returns:
        DataFrame: Top-k candidate documents with scores
    """

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

    # Combine
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
    semantic_norm = (semantic_scores - np.min(semantic_scores)) / (np.ptp(semantic_scores) + 1e-8)
    final_scores = 0.5 * bm25_norm + 0.5 * semantic_norm

    # Top k
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    candidates = df.iloc[top_indices].copy()
    candidates['retrieval_score'] = final_scores[top_indices]

    return candidates


def rerank_with_cross_encoder(query, candidates, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', top_k=10):
    """
    Stage 2: Accurate reranking with cross-encoder.

    Args:
        query: Search query
        candidates: DataFrame of candidate documents
        model_name: Cross-encoder model to use
        top_k: Number of results to return after reranking

    Returns:
        DataFrame: Reranked top-k documents
    """

    # Load cross-encoder
    cross_encoder = CrossEncoder(model_name)

    # Prepare pairs (query, document)
    pairs = [[query, doc] for doc in candidates['combined_text'].tolist()]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Rerank by cross-encoder scores
    candidates = candidates.copy()
    candidates['cross_encoder_score'] = scores

    reranked = candidates.sort_values('cross_encoder_score', ascending=False).head(top_k)

    return reranked


def evaluate_reranking(test_set_path, search_system, cross_encoder_model):
    """
    Evaluate two-stage retrieval vs baseline.

    Args:
        test_set_path: Path to test set
        search_system: Loaded search system
        cross_encoder_model: Cross-encoder model name

    Returns:
        dict: Comparison results
    """

    print("="*80)
    print("CROSS-ENCODER RERANKING EVALUATION")
    print("="*80)

    # Load test set
    print("\n1. Loading test set...")
    test_df = pd.read_csv(test_set_path)

    test_queries = test_df['query'].unique()
    queries_data = {}
    for query in test_queries:
        query_data = test_df[test_df['query'] == query]
        queries_data[query] = {
            'document_ids': query_data['document_id'].tolist(),
            'relevances': query_data['relevance'].tolist(),
        }

    print(f"   Loaded {len(queries_data)} queries")

    # Evaluate baseline (Stage 1 only)
    print("\n2. Evaluating baseline (hybrid retrieval only)...")
    baseline_results = []
    for query, data in queries_data.items():
        candidates = retrieve_candidates_hybrid(query, search_system, top_k=10)

        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}
        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in candidates['ID'].tolist()]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)
        ground_truth_relevances = data['relevances']

        baseline_results.append({
            'relevances': relevances,
            'total_relevant': total_relevant,
            'ground_truth_relevances': ground_truth_relevances
        })

    baseline_relevances = [r['relevances'] for r in baseline_results]
    baseline_total_relevant = [r['total_relevant'] for r in baseline_results]
    baseline_ground_truth = [r['ground_truth_relevances'] for r in baseline_results]
    baseline_metrics = aggregate_metrics(baseline_relevances, baseline_total_relevant, k_values=[5, 10], all_ground_truth_relevances=baseline_ground_truth)

    print(f"   Baseline NDCG@10: {baseline_metrics['ndcg@10']:.4f}")

    # Evaluate with cross-encoder reranking
    print(f"\n3. Evaluating with cross-encoder reranking...")
    print(f"   Model: {cross_encoder_model}")

    reranked_results = []
    total_retrieval_time = 0.0
    total_reranking_time = 0.0

    for i, (query, data) in enumerate(queries_data.items(), 1):
        if i % 10 == 0:
            print(f"   [{i}/{len(queries_data)}] Processing...")

        # Stage 1: Retrieve candidates
        start = time.time()
        candidates = retrieve_candidates_hybrid(query, search_system, top_k=50)
        retrieval_time = time.time() - start
        total_retrieval_time += retrieval_time

        # Stage 2: Rerank
        start = time.time()
        reranked = rerank_with_cross_encoder(query, candidates, model_name=cross_encoder_model, top_k=10)
        reranking_time = time.time() - start
        total_reranking_time += reranking_time

        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}
        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in reranked['ID'].tolist()]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)
        ground_truth_relevances = data['relevances']

        reranked_results.append({
            'relevances': relevances,
            'total_relevant': total_relevant,
            'ground_truth_relevances': ground_truth_relevances
        })

    reranked_relevances = [r['relevances'] for r in reranked_results]
    reranked_total_relevant = [r['total_relevant'] for r in reranked_results]
    reranked_ground_truth = [r['ground_truth_relevances'] for r in reranked_results]
    reranked_metrics = aggregate_metrics(reranked_relevances, reranked_total_relevant, k_values=[5, 10], all_ground_truth_relevances=reranked_ground_truth)

    print(f"\n   With Reranking NDCG@10: {reranked_metrics['ndcg@10']:.4f}")

    # Latency analysis
    avg_retrieval = total_retrieval_time / len(queries_data)
    avg_reranking = total_reranking_time / len(queries_data)

    print(f"\n4. Latency Analysis:")
    print(f"   Avg retrieval time (Stage 1): {avg_retrieval*1000:.1f}ms")
    print(f"   Avg reranking time (Stage 2): {avg_reranking*1000:.1f}ms")
    print(f"   Total avg latency: {(avg_retrieval + avg_reranking)*1000:.1f}ms")
    print(f"   Additional latency from reranking: +{avg_reranking*1000:.1f}ms")

    # Results comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    comparison = []
    for metric in ['ndcg@5', 'ndcg@10', 'map', 'mrr', 'precision@10', 'recall@10']:
        baseline_val = baseline_metrics[metric]
        reranked_val = reranked_metrics[metric]
        improvement = ((reranked_val - baseline_val) / baseline_val) * 100

        comparison.append({
            'Metric': metric,
            'Baseline': f"{baseline_val:.4f}",
            'With Reranking': f"{reranked_val:.4f}",
            'Improvement': f"{improvement:+.2f}%"
        })

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Save results
    output_path = Path("experiments/advanced_retrieval/cross_encoder")
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        'baseline_metrics': baseline_metrics,
        'reranked_metrics': reranked_metrics,
        'latency': {
            'avg_retrieval_ms': avg_retrieval * 1000,
            'avg_reranking_ms': avg_reranking * 1000,
            'total_avg_ms': (avg_retrieval + avg_reranking) * 1000
        },
        'improvements': {
            metric: ((reranked_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
            for metric in ['ndcg@5', 'ndcg@10', 'map', 'mrr']
        }
    }

    with open(output_path / "reranking_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Results saved to: {output_path / 'reranking_results.json'}")

    return results


def main():
    print("Loading search system...")
    search_system = load_search_system()

    print("\nRunning cross-encoder reranking evaluation...")
    test_set_path = Path("evaluation/test_sets/test_set_llm_compact.csv")

    # Test with MiniLM cross-encoder (fast, good balance)
    results = evaluate_reranking(test_set_path, search_system, 'cross-encoder/ms-marco-MiniLM-L-6-v2')


if __name__ == "__main__":
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    main()
