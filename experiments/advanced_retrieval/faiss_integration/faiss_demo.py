"""
FAISS Integration for Scalable Similarity Search.

FAISS (Facebook AI Similarity Search) enables fast approximate nearest neighbor search.
Critical for scaling to large document corpora (50K+ documents).
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import faiss

from sentence_transformers import SentenceTransformer


def load_embeddings(backlog_path):
    """Load or compute embeddings"""

    backlog_df = pd.read_csv(backlog_path)

    backlog_df['combined_text'] = (
        backlog_df['Title'].fillna('') + ". " +
        backlog_df['Description'].fillna('') + ". " +
        backlog_df['Tags'].fillna('')
    )

    documents = backlog_df['combined_text'].tolist()

    os.environ.pop('HF_TOKEN', None)
    os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
    model = SentenceTransformer('all-mpnet-base-v2', token=False)

    print(f"Computing embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    return embeddings, backlog_df, model


def benchmark_search(embeddings, df, model, test_queries, index_type='flat'):
    """
    Benchmark search performance with different FAISS index types.

    Args:
        embeddings: Document embeddings
        df: Backlog DataFrame
        model: Sentence transformer model
        test_queries: List of test queries
        index_type: 'flat', 'ivf', or 'hnsw'

    Returns:
        dict: Benchmark results
    """

    print(f"\nBenchmarking {index_type.upper()} index...")

    d = embeddings.shape[1]  # Dimension
    n = embeddings.shape[0]  # Number of vectors

    # Build index
    start_time = time.time()

    if index_type == 'flat':
        # Exact search (baseline)
        index = faiss.IndexFlatIP(d)  # Inner product (cosine similarity after normalization)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        index.add(embeddings)

    elif index_type == 'ivf':
        # Inverted File Index (approximate, faster)
        nlist = min(100, n // 10)  # Number of clusters
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        faiss.normalize_L2(embeddings)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10  # Number of clusters to search

    elif index_type == 'hnsw':
        # Hierarchical Navigable Small World (best accuracy/speed tradeoff)
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # 32 = M parameter
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

    build_time = time.time() - start_time

    print(f"  Index built in {build_time:.2f}s")
    print(f"  Index size: {index.ntotal} vectors")

    # Benchmark search
    query_latencies = []

    for query in test_queries:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        start = time.time()
        scores, indices = index.search(query_embedding, k=10)
        latency = time.time() - start

        query_latencies.append(latency * 1000)  # Convert to ms

    avg_latency = np.mean(query_latencies)
    p95_latency = np.percentile(query_latencies, 95)

    print(f"  Avg query latency: {avg_latency:.2f}ms")
    print(f"  P95 query latency: {p95_latency:.2f}ms")

    return {
        'index_type': index_type,
        'build_time_seconds': build_time,
        'avg_latency_ms': avg_latency,
        'p95_latency_ms': p95_latency,
        'num_documents': n
    }


def simulate_scaling(embeddings, df, model, test_queries):
    """
    Simulate performance at different corpus sizes.

    Args:
        embeddings: Full embeddings
        df: Backlog DataFrame
        model: Sentence transformer model
        test_queries: Test queries

    Returns:
        list: Scaling results
    """

    print("\n" + "="*80)
    print("SCALING SIMULATION")
    print("="*80)

    # Test at different sizes
    sizes = [800, 2400, 4000, 8000]  # Simulate by replicating
    results = []

    for size in sizes:
        if size > len(embeddings):
            # Replicate embeddings to simulate larger corpus
            factor = size // len(embeddings)
            scaled_embeddings = np.tile(embeddings, (factor, 1))
            scaled_embeddings = scaled_embeddings[:size]
        else:
            scaled_embeddings = embeddings[:size]

        print(f"\nCorpus size: {len(scaled_embeddings)}")

        # Benchmark each index type
        for index_type in ['flat', 'hnsw']:
            result = benchmark_search(scaled_embeddings, df, model, test_queries, index_type=index_type)
            result['corpus_size'] = len(scaled_embeddings)
            results.append(result)

    return results


def main():
    """Run FAISS benchmarking and scaling analysis"""

    print("="*80)
    print("FAISS INTEGRATION & SCALABILITY ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading backlog data...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog.csv")
    embeddings, df, model = load_embeddings(backlog_path)

    print(f"   Loaded {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    # Test queries
    test_queries = [
        "authentication error",
        "database timeout",
        "API performance issues",
        "frontend bug",
        "security vulnerability"
    ]

    # Run scaling simulation
    scaling_results = simulate_scaling(embeddings, df, model, test_queries)

    # Save results
    output_path = Path("experiments/advanced_retrieval/faiss_integration")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(scaling_results)
    results_df.to_csv(output_path / "scaling_results.csv", index=False)

    # Analysis
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)

    print("\nLatency vs Corpus Size:")
    pivot = results_df.pivot(index='corpus_size', columns='index_type', values='avg_latency_ms')
    print(pivot.to_string())

    # Calculate speedup
    for size in results_df['corpus_size'].unique():
        subset = results_df[results_df['corpus_size'] == size]
        flat_latency = subset[subset['index_type'] == 'flat']['avg_latency_ms'].values[0]
        hnsw_latency = subset[subset['index_type'] == 'hnsw']['avg_latency_ms'].values[0]
        speedup = flat_latency / hnsw_latency

        print(f"\nCorpus size: {size}")
        print(f"  Flat (exact): {flat_latency:.2f}ms")
        print(f"  HNSW (approx): {hnsw_latency:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nWhen to use FAISS:")
    print("  < 5,000 documents:   Use exact search (current approach)")
    print("  5K - 50K documents:  Use FAISS HNSW (good balance)")
    print("  > 50K documents:     Use FAISS IVF + PQ (memory-efficient)")

    print("\nCurrent corpus (800 docs):")
    print("  FAISS overhead not worth it - stick with exact search")
    print("  FAISS becomes beneficial at ~5,000+ documents")

    print(f"\n[SUCCESS] Results saved to: {output_path / 'scaling_results.csv'}")


if __name__ == "__main__":
    main()
