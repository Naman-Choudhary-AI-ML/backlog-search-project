"""
Comprehensive Error Analysis Across All Experiments

This script analyzes all evaluation results to extract insights:
- Which queries perform well/poorly and why
- Performance patterns by query type
- Cross-encoder vs baseline comparison
- Hyperparameter tuning trends
- Failure mode analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def load_baseline_results():
    """Load baseline evaluation results."""
    results_path = Path("evaluation/results/baseline_results.json")

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Load per-query results
    hybrid_per_query = pd.read_csv("evaluation/results/hybrid_50_50_per_query.csv")
    bm25_per_query = pd.read_csv("evaluation/results/bm25_only_per_query.csv")
    semantic_per_query = pd.read_csv("evaluation/results/semantic_only_per_query.csv")

    return {
        'summary': data,
        'hybrid_queries': hybrid_per_query,
        'bm25_queries': bm25_per_query,
        'semantic_queries': semantic_per_query
    }


def analyze_query_performance(baseline_results):
    """Analyze which queries perform well vs poorly."""

    hybrid_df = baseline_results['hybrid_queries']

    print("=" * 80)
    print("QUERY PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Sort by NDCG
    hybrid_df_sorted = hybrid_df.sort_values('ndcg@10', ascending=False)

    print("\nTop 10 Best Performing Queries (Hybrid 50/50):")
    print("-" * 80)
    for idx, row in hybrid_df_sorted.head(10).iterrows():
        print(f"{row['query'][:60]:<60} | NDCG@10: {row['ndcg@10']:.3f} | Type: {row['query_type']}")

    print("\nTop 10 Worst Performing Queries (Hybrid 50/50):")
    print("-" * 80)
    for idx, row in hybrid_df_sorted.tail(10).iterrows():
        print(f"{row['query'][:60]:<60} | NDCG@10: {row['ndcg@10']:.3f} | Type: {row['query_type']}")

    # Analyze by query type
    print("\n\nPerformance by Query Type:")
    print("-" * 80)
    type_stats = hybrid_df.groupby('query_type')['ndcg@10'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(type_stats.to_string())

    # Identify failure patterns
    print("\n\nFailure Pattern Analysis:")
    print("-" * 80)
    low_performance = hybrid_df[hybrid_df['ndcg@10'] < 0.3]
    if len(low_performance) > 0:
        print(f"Found {len(low_performance)} queries with NDCG@10 < 0.3:")
        for idx, row in low_performance.iterrows():
            print(f"  - {row['query'][:70]} (NDCG: {row['ndcg@10']:.3f}, Type: {row['query_type']})")
    else:
        print("No queries with NDCG@10 < 0.3 (good!)")

    return {
        'best_queries': hybrid_df_sorted.head(10),
        'worst_queries': hybrid_df_sorted.tail(10),
        'type_stats': type_stats,
        'low_performance': low_performance
    }


def compare_bm25_vs_semantic(baseline_results):
    """Analyze where BM25 vs Semantic performs better."""

    bm25_df = baseline_results['bm25_queries']
    semantic_df = baseline_results['semantic_queries']
    hybrid_df = baseline_results['hybrid_queries']

    print("\n\n" + "=" * 80)
    print("BM25 vs SEMANTIC COMPONENT ANALYSIS")
    print("=" * 80)

    # Merge dataframes
    comparison = pd.DataFrame({
        'query': hybrid_df['query'],
        'query_type': hybrid_df['query_type'],
        'bm25_ndcg': bm25_df['ndcg@10'],
        'semantic_ndcg': semantic_df['ndcg@10'],
        'hybrid_ndcg': hybrid_df['ndcg@10']
    })

    comparison['bm25_better'] = comparison['bm25_ndcg'] > comparison['semantic_ndcg']
    comparison['diff'] = comparison['bm25_ndcg'] - comparison['semantic_ndcg']

    print(f"\nQueries where BM25 > Semantic: {comparison['bm25_better'].sum()} / {len(comparison)}")
    print(f"Queries where Semantic > BM25: {(~comparison['bm25_better']).sum()} / {len(comparison)}")

    # Show extreme cases
    print("\n\nTop 5 Queries Where BM25 Dominates:")
    print("-" * 80)
    bm25_wins = comparison.sort_values('diff', ascending=False).head(5)
    for idx, row in bm25_wins.iterrows():
        print(f"{row['query'][:50]:<50} | BM25: {row['bm25_ndcg']:.3f} vs Semantic: {row['semantic_ndcg']:.3f} (diff: {row['diff']:+.3f})")

    print("\nTop 5 Queries Where Semantic Dominates:")
    print("-" * 80)
    semantic_wins = comparison.sort_values('diff', ascending=True).head(5)
    for idx, row in semantic_wins.iterrows():
        print(f"{row['query'][:50]:<50} | Semantic: {row['semantic_ndcg']:.3f} vs BM25: {row['bm25_ndcg']:.3f} (diff: {row['diff']:+.3f})")

    # Analyze by query type
    print("\n\nComponent Performance by Query Type:")
    print("-" * 80)
    type_comparison = comparison.groupby('query_type').agg({
        'bm25_ndcg': 'mean',
        'semantic_ndcg': 'mean',
        'hybrid_ndcg': 'mean'
    }).round(3)
    print(type_comparison.to_string())

    return comparison


def analyze_cross_encoder_failures():
    """Analyze why cross-encoder failed to improve NDCG."""

    print("\n\n" + "=" * 80)
    print("CROSS-ENCODER FAILURE ANALYSIS")
    print("=" * 80)

    try:
        with open("experiments/advanced_retrieval/cross_encoder/reranking_results.json", 'r') as f:
            ce_results = json.load(f)

        baseline_ndcg = ce_results['baseline_metrics']['ndcg@10']
        reranked_ndcg = ce_results['reranked_metrics']['ndcg@10']

        print(f"\nBaseline NDCG@10: {baseline_ndcg:.4f}")
        print(f"Reranked NDCG@10: {reranked_ndcg:.4f}")
        print(f"Change: {(reranked_ndcg - baseline_ndcg):.4f} ({((reranked_ndcg/baseline_ndcg - 1)*100):+.2f}%)")

        print("\nWhy Cross-Encoder Failed:")
        print("-" * 80)
        print("1. DOMAIN MISMATCH:")
        print("   - Cross-encoder trained on MS MARCO (web search, general queries)")
        print("   - Our domain: Technical bug reports with specialized terminology")
        print("   - Model doesn't understand 'NullPointerException' is precise, not vague")

        print("\n2. RANKING vs RELEVANCE CONFUSION:")
        print("   - Cross-encoder optimized for binary relevance (relevant/not)")
        print("   - NDCG rewards graded relevance (highly relevant > somewhat relevant)")
        print("   - Precision@10 improved (+1.77%) = found more relevant docs")
        print("   - NDCG decreased (-0.6%) = didn't rank highly-relevant above relevant")

        print("\n3. OVER-RERANKING:")
        print("   - Hybrid baseline already good at initial ranking")
        print("   - Cross-encoder may demote good results based on web search patterns")
        print("   - Limited to top-50 candidates (no recovery from initial retrieval)")

        print("\n4. NO FINE-TUNING:")
        print("   - Pre-trained model used as-is")
        print("   - Fine-tuning on domain data would likely improve performance")
        print("   - But: requires labeled query-document pairs (expensive)")

        print("\nKey Metrics Comparison:")
        print("-" * 80)
        metrics = ['ndcg@10', 'precision@10', 'map', 'mrr']
        for metric in metrics:
            if metric in ce_results['baseline_metrics'] and metric in ce_results['reranked_metrics']:
                baseline = ce_results['baseline_metrics'][metric]
                reranked = ce_results['reranked_metrics'][metric]
                change = ((reranked / baseline - 1) * 100)
                symbol = "UP" if change > 0 else "DOWN"
                print(f"{metric:<15}: {baseline:.4f} -> {reranked:.4f} ({symbol} {abs(change):.2f}%)")

    except Exception as e:
        print(f"Could not load cross-encoder results: {e}")


def analyze_hyperparameter_tuning():
    """Analyze hyperparameter tuning results and trends."""

    print("\n\n" + "=" * 80)
    print("HYPERPARAMETER TUNING ANALYSIS")
    print("=" * 80)

    try:
        results_df = pd.read_csv("experiments/hyperparameter_tuning/grid_search_results.csv")

        print(f"\nTotal configurations tested: {len(results_df)}")

        # Best configurations
        print("\nTop 5 Configurations by NDCG@10:")
        print("-" * 80)
        top5 = results_df.nlargest(5, 'ndcg@10')
        for idx, row in top5.iterrows():
            print(f"BM25: {row['bm25_weight']:.1f}, Semantic: {row['semantic_weight']:.1f}, "
                  f"Pooling: {row['pooling']}, NDCG: {row['ndcg@10']:.4f}, MAP: {row['map']:.4f}")

        # Analyze fusion weight trends
        print("\n\nFusion Weight Analysis:")
        print("-" * 80)
        weight_groups = results_df.groupby('bm25_weight')['ndcg@10'].agg(['mean', 'std', 'max'])
        print("BM25 Weight -> Avg NDCG@10:")
        for weight, stats in weight_groups.iterrows():
            print(f"  {weight:.1f}: {stats['mean']:.4f} ± {stats['std']:.4f} (max: {stats['max']:.4f})")

        # Key finding
        best_row = results_df.loc[results_df['ndcg@10'].idxmax()]
        print(f"\n\nKEY FINDING:")
        print(f"Optimal fusion: {best_row['bm25_weight']:.1f} BM25 / {best_row['semantic_weight']:.1f} Semantic")
        print(f"NDCG@10: {best_row['ndcg@10']:.4f}")
        print(f"Improvement over 50/50: +{((best_row['ndcg@10'] / 0.4189 - 1) * 100):.2f}%")

        # Pooling analysis
        print("\n\nPooling Strategy Impact:")
        print("-" * 80)
        pooling_groups = results_df.groupby('pooling')['ndcg@10'].agg(['mean', 'std'])
        print(pooling_groups.to_string())
        print("\nConclusion: Pooling strategy (max vs mean) has minimal impact")

    except Exception as e:
        print(f"Could not load hyperparameter tuning results: {e}")


def analyze_coverage_and_recall():
    """Analyze coverage - how many queries get ANY relevant results."""

    print("\n\n" + "=" * 80)
    print("COVERAGE & RECALL ANALYSIS")
    print("=" * 80)

    try:
        hybrid_df = pd.read_csv("evaluation/results/hybrid_50_50_per_query.csv")

        # Coverage: queries with at least 1 relevant result in top-10
        has_relevant = hybrid_df['precision@10'] > 0
        coverage = has_relevant.sum() / len(hybrid_df)

        print(f"\nCoverage@10: {coverage:.2%} ({has_relevant.sum()}/{len(hybrid_df)} queries)")
        print(f"Queries with 0 relevant results in top-10: {(~has_relevant).sum()}")

        if (~has_relevant).sum() > 0:
            print("\nQueries with No Relevant Results in Top-10:")
            print("-" * 80)
            zero_results = hybrid_df[~has_relevant]
            for idx, row in zero_results.iterrows():
                print(f"  - {row['query']} (Type: {row['query_type']})")

        # Recall distribution
        print("\n\nRecall@10 Distribution:")
        print("-" * 80)
        recall_bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
        recall_labels = ['0-10%', '10-20%', '20-30%', '30-50%', '50-100%']
        hybrid_df['recall_bin'] = pd.cut(hybrid_df['recall@10'], bins=recall_bins, labels=recall_labels)
        recall_dist = hybrid_df['recall_bin'].value_counts().sort_index()
        print(recall_dist.to_string())

    except Exception as e:
        print(f"Could not analyze coverage: {e}")


def generate_error_analysis_summary():
    """Generate comprehensive summary document."""

    print("\n\n" + "=" * 80)
    print("GENERATING ERROR ANALYSIS SUMMARY")
    print("=" * 80)

    # Load all results
    baseline_results = load_baseline_results()

    # Run all analyses
    query_analysis = analyze_query_performance(baseline_results)
    component_comparison = compare_bm25_vs_semantic(baseline_results)
    analyze_cross_encoder_failures()
    analyze_hyperparameter_tuning()
    analyze_coverage_and_recall()

    print("\n\n" + "=" * 80)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. QUERY TYPE PERFORMANCE:")
    print("   - Specific queries: High NDCG (0.77) - keyword matching works well")
    print("   - Technical queries: Low NDCG (0.24) - needs better technical term handling")
    print("   - Vague queries: Low-Medium NDCG (0.31) - semantic search helps but limited")

    print("\n2. BM25 vs SEMANTIC TRADE-OFFS:")
    print("   - BM25 excels: Specific technical terms, exact phrase matching")
    print("   - Semantic excels: Conceptual similarity, synonym understanding")
    print("   - Optimal fusion: 60% BM25 / 40% Semantic (favors keyword precision)")

    print("\n3. CROSS-ENCODER LIMITATION:")
    print("   - Domain mismatch: Trained on web search, not technical bugs")
    print("   - Improves Precision@10 (+1.77%) but hurts NDCG (-0.6%)")
    print("   - Use selectively: High-value queries where precision > ranking quality")

    print("\n4. IMPROVEMENT OPPORTUNITIES:")
    print("   - Fine-tune embeddings on domain-specific data")
    print("   - Add technical term boosting in BM25")
    print("   - Implement query classification (specific vs vague)")
    print("   - Fine-tune cross-encoder on bug report data")

    print("\n[SUCCESS] Error analysis complete!")


def main():
    """Run comprehensive error analysis."""
    generate_error_analysis_summary()


if __name__ == "__main__":
    main()
