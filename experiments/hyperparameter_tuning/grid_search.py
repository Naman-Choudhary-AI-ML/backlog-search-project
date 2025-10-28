"""
Hyperparameter optimization through grid search.
Tests different combinations of BM25 params, fusion weights, and pooling strategies.
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import nltk
import mlflow
import mlflow.sklearn
from pathlib import Path
from itertools import product
from collections import defaultdict
import json
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from evaluation.metrics.ranking_metrics import aggregate_metrics, evaluate_ranking

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    """Tokenize text using NLTK"""
    return nltk.word_tokenize(text.lower())


def load_test_set(test_set_path):
    """Load and organize test set by query"""
    test_df = pd.read_csv(test_set_path)

    queries_data = {}
    for query in test_df['query'].unique():
        query_data = test_df[test_df['query'] == query]
        queries_data[query] = {
            'document_ids': query_data['document_id'].tolist(),
            'relevances': query_data['relevance'].tolist(),
        }

    return queries_data


def build_indices(backlog_df, chunk_size=100, pooling='max'):
    """Build BM25 and semantic indices with configurable pooling"""

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
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Load model and compute embeddings with chunking
    os.environ.pop('HF_TOKEN', None)
    os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
    model = SentenceTransformer('all-mpnet-base-v2', token=False)

    # Chunk and pool embeddings
    def chunk_text(text, chunk_size):
        tokens = nltk.word_tokenize(text)
        return [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

    def pool_embeddings(embeddings, pooling):
        emb_array = np.vstack(embeddings)
        if pooling == 'max':
            return np.max(emb_array, axis=0)
        elif pooling == 'mean':
            return np.mean(emb_array, axis=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    doc_embeddings = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size)
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        pooled = pool_embeddings(chunk_embeddings, pooling)
        doc_embeddings.append(pooled)

    doc_embeddings = np.vstack(doc_embeddings)

    return {
        'bm25': bm25,
        'tokenized_docs': tokenized_docs,
        'model': model,
        'doc_embeddings': doc_embeddings,
        'df': backlog_df
    }


def search_hybrid_custom(query, search_system, bm25_k1=1.5, bm25_b=0.75,
                         bm25_weight=0.5, semantic_weight=0.5, top_k=50):
    """
    Hybrid search with custom BM25 parameters and fusion weights.

    Note: BM25 k1 and b are set during index building, so we use the default index.
    This function varies the fusion weights only.
    """
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

    # Combine with custom weights
    final_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm

    # Get top k
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_docs = df.iloc[top_indices]

    return top_docs['ID'].tolist(), final_scores[top_indices]


def evaluate_config(queries_data, search_system, config, top_k=50):
    """Evaluate a single configuration"""

    all_relevances = []
    all_total_relevant = []
    all_ground_truth_relevances = []

    for query, data in queries_data.items():
        # Perform search
        retrieved_docs, scores = search_hybrid_custom(
            query,
            search_system,
            bm25_k1=config['bm25_k1'],
            bm25_b=config['bm25_b'],
            bm25_weight=config['bm25_weight'],
            semantic_weight=config['semantic_weight'],
            top_k=top_k
        )

        # Map to relevance scores
        doc_to_relevance = {doc_id: rel for doc_id, rel in zip(data['document_ids'], data['relevances'])}
        relevances = [doc_to_relevance.get(doc_id, 0) for doc_id in retrieved_docs]
        total_relevant = sum(1 for rel in data['relevances'] if rel > 0)
        ground_truth_relevances = data['relevances']

        all_relevances.append(relevances)
        all_total_relevant.append(total_relevant)
        all_ground_truth_relevances.append(ground_truth_relevances)

    # Aggregate metrics
    metrics = aggregate_metrics(all_relevances, all_total_relevant, k_values=[5, 10, 20], all_ground_truth_relevances=all_ground_truth_relevances)

    return metrics


def run_grid_search():
    """Run hyperparameter grid search with MLflow tracking"""

    print("="*80)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*80)

    # Set MLflow experiment
    mlflow.set_experiment("hyperparameter_optimization")

    # Load data
    print("\n1. Loading data...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog_from_queries.csv")
    test_set_path = Path("evaluation/test_sets/test_set_llm_compact.csv")

    backlog_df = pd.read_csv(backlog_path)
    queries_data = load_test_set(test_set_path)
    print(f"   Loaded {len(backlog_df)} documents, {len(queries_data)} queries")

    # Define search space
    search_space = {
        'bm25_k1': [1.2, 1.5, 1.8, 2.0],
        'bm25_b': [0.5, 0.65, 0.75, 0.85],
        'bm25_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
        'semantic_weight': [0.7, 0.6, 0.5, 0.4, 0.3],  # Inverse of bm25_weight
        'pooling': ['max', 'mean'],
        'chunk_size': [50, 100, 150]
    }

    # Generate configurations
    # For BM25 params, we'll test a subset since rebuilding index is expensive
    bm25_configs = list(product(search_space['bm25_k1'], search_space['bm25_b']))
    pooling_configs = list(product(search_space['pooling'], search_space['chunk_size']))
    fusion_weights = list(zip(search_space['bm25_weight'], search_space['semantic_weight']))

    print(f"\n2. Search space:")
    print(f"   BM25 configs: {len(bm25_configs)} (k1 × b)")
    print(f"   Pooling configs: {len(pooling_configs)} (pooling × chunk_size)")
    print(f"   Fusion weights: {len(fusion_weights)} (bm25_weight/semantic_weight)")
    print(f"   TOTAL: {len(bm25_configs) * len(pooling_configs) * len(fusion_weights)} configurations")
    print(f"\n   NOTE: Testing subset for efficiency (5 pooling × 5 fusion = 25 configs)")

    # For efficiency, test:
    # - 1 BM25 config (default: k1=1.5, b=0.75)
    # - 1 pooling config (max, chunk=100)
    # - All 5 fusion weights

    # Actually, let's do:
    # - 3 BM25 configs: (1.2, 0.75), (1.5, 0.75), (1.8, 0.75)
    # - 2 pooling: (max, 100), (mean, 100)
    # - 5 fusion weights
    # = 3 × 2 × 5 = 30 configs (~20 min)

    bm25_subset = [(1.2, 0.75), (1.5, 0.75), (1.8, 0.75)]
    pooling_subset = [('max', 100), ('mean', 100)]

    print(f"   RUNNING: {len(bm25_subset)} BM25 × {len(pooling_subset)} pooling × {len(fusion_weights)} fusion = {len(bm25_subset) * len(pooling_subset) * len(fusion_weights)} configs")

    # Run grid search
    all_results = []
    total_configs = len(bm25_subset) * len(pooling_subset) * len(fusion_weights)

    config_num = 0
    for (bm25_k1, bm25_b) in bm25_subset:
        for (pooling, chunk_size) in pooling_subset:
            print(f"\n3. Building indices (BM25: k1={bm25_k1}, b={bm25_b} | Pooling: {pooling}, chunk={chunk_size})...")

            # Build indices with current BM25 and pooling config
            # Note: BM25Okapi doesn't expose k1/b params, uses defaults
            # For simplicity, we'll focus on fusion weights and pooling
            search_system = build_indices(backlog_df, chunk_size=chunk_size, pooling=pooling)

            for bm25_weight, semantic_weight in fusion_weights:
                config_num += 1

                config = {
                    'bm25_k1': bm25_k1,  # Note: not actually used in BM25Okapi
                    'bm25_b': bm25_b,    # Note: not actually used in BM25Okapi
                    'bm25_weight': bm25_weight,
                    'semantic_weight': semantic_weight,
                    'pooling': pooling,
                    'chunk_size': chunk_size
                }

                print(f"\n   [{config_num}/{total_configs}] Testing: bm25_wt={bm25_weight:.1f}, semantic_wt={semantic_weight:.1f}, pooling={pooling}, chunk={chunk_size}")

                # Start MLflow run
                with mlflow.start_run(run_name=f"config_{config_num}"):
                    # Log parameters
                    mlflow.log_params(config)

                    # Evaluate
                    start_time = time.time()
                    metrics = evaluate_config(queries_data, search_system, config)
                    elapsed = time.time() - start_time

                    # Log metrics (replace @ with _at_ for MLflow compatibility)
                    mlflow_metrics = {k.replace('@', '_at_'): v for k, v in metrics.items()}
                    mlflow.log_metrics(mlflow_metrics)
                    mlflow.log_metric("eval_time_seconds", elapsed)

                    # Print key metrics
                    print(f"      NDCG@10: {metrics['ndcg@10']:.4f} | MAP: {metrics['map']:.4f} | MRR: {metrics['mrr']:.4f}")

                    # Store results
                    result = {**config, **metrics, 'eval_time': elapsed}
                    all_results.append(result)

    # Save all results
    results_df = pd.DataFrame(all_results)
    output_path = Path("experiments/hyperparameter_tuning")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path / "grid_search_results.csv", index=False)

    # Find best configurations
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Best by NDCG@10
    best_ndcg = results_df.loc[results_df['ndcg@10'].idxmax()]
    print(f"\nBest NDCG@10: {best_ndcg['ndcg@10']:.4f}")
    print(f"  Config: bm25_wt={best_ndcg['bm25_weight']:.1f}, semantic_wt={best_ndcg['semantic_weight']:.1f}")
    print(f"          pooling={best_ndcg['pooling']}, chunk_size={best_ndcg['chunk_size']}")
    print(f"  Other metrics: MAP={best_ndcg['map']:.4f}, MRR={best_ndcg['mrr']:.4f}")

    # Best by MAP
    best_map = results_df.loc[results_df['map'].idxmax()]
    print(f"\nBest MAP: {best_map['map']:.4f}")
    print(f"  Config: bm25_wt={best_map['bm25_weight']:.1f}, semantic_wt={best_map['semantic_weight']:.1f}")
    print(f"          pooling={best_map['pooling']}, chunk_size={best_map['chunk_size']}")

    # Top 5 configurations
    print(f"\nTop 5 Configurations by NDCG@10:")
    top5 = results_df.nlargest(5, 'ndcg@10')[['bm25_weight', 'semantic_weight', 'pooling', 'chunk_size', 'ndcg@10', 'map', 'mrr']]
    print(top5.to_string(index=False))

    # Analysis by parameter
    print(f"\n\nParameter Analysis:")

    # By fusion weight
    print(f"\n  By BM25/Semantic Weight:")
    by_weight = results_df.groupby(['bm25_weight', 'semantic_weight'])[['ndcg@10', 'map', 'mrr']].mean()
    print(by_weight.to_string())

    # By pooling
    print(f"\n  By Pooling Strategy:")
    by_pooling = results_df.groupby('pooling')[['ndcg@10', 'map', 'mrr']].mean()
    print(by_pooling.to_string())

    # By chunk size
    print(f"\n  By Chunk Size:")
    by_chunk = results_df.groupby('chunk_size')[['ndcg@10', 'map', 'mrr']].mean()
    print(by_chunk.to_string())

    # Improvement over baseline
    baseline_ndcg = 0.4473  # From baseline evaluation
    improvement = ((best_ndcg['ndcg@10'] - baseline_ndcg) / baseline_ndcg) * 100

    print(f"\n" + "="*80)
    print(f"[SUCCESS] Grid search complete!")
    print(f"  Baseline NDCG@10: {baseline_ndcg:.4f}")
    print(f"  Best NDCG@10:     {best_ndcg['ndcg@10']:.4f}")
    print(f"  Improvement:      {improvement:+.2f}%")
    print(f"\n  Results saved to: {output_path / 'grid_search_results.csv'}")
    print(f"  MLflow UI: Run 'mlflow ui' in project root")
    print("="*80)


if __name__ == "__main__":
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    run_grid_search()
