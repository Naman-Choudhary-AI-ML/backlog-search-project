"""
RAG (Retrieval-Augmented Generation) Summarization Pipeline.
Retrieves relevant documents and generates intelligent summaries using GPT-4.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import nltk
from pathlib import Path
from openai import OpenAI

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


RAG_SUMMARY_PROMPT_TEMPLATE = """You are an intelligent assistant helping engineers understand their backlog search results.

User Query: "{query}"

Retrieved Backlog Items:
{items}

Task: Analyze these backlog items and provide:
1. A concise summary (2-3 sentences) of what these items are about
2. Group them by theme/category if applicable
3. Identify which items are highest priority or most critical
4. Provide a recommendation on what to focus on first

Format your response as JSON:
{{
    "summary": "Brief overview of the items...",
    "themes": {{
        "theme1": ["ITEM-001", "ITEM-002"],
        "theme2": ["ITEM-003"]
    }},
    "high_priority": ["ITEM-001"],
    "recommendation": "Focus on X because..."
}}

Return ONLY valid JSON, no markdown."""


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def load_search_system():
    """Load search system"""

    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog.csv")
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


def search_hybrid(query, search_system, top_k=10):
    """Hybrid search returning top documents"""

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
    results = df.iloc[top_indices].copy()
    results['score'] = final_scores[top_indices]

    return results


def generate_rag_summary(query, retrieved_items, api_key, model="gpt-4-turbo"):
    """
    Generate intelligent summary using RAG.

    Args:
        query: User's search query
        retrieved_items: DataFrame of retrieved backlog items
        api_key: OpenAI API key
        model: GPT model to use

    Returns:
        dict: Summary with cost/latency metrics
    """

    client = OpenAI(api_key=api_key)

    # Format items for prompt
    items_text = ""
    for idx, row in retrieved_items.iterrows():
        items_text += f"ID: {row['ID']}\n"
        items_text += f"Title: {row['Title']}\n"
        items_text += f"Type: {row['Type']}\n"
        items_text += f"Priority: {row['Priority']}\n"
        items_text += f"Description: {row['Description'][:200]}...\n\n"

    prompt = RAG_SUMMARY_PROMPT_TEMPLATE.format(query=query, items=items_text)

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a technical project manager analyzing backlog items."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )

        latency = time.time() - start_time

        # Parse response
        content = response.choices[0].message.content.strip()

        # Handle markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        summary_data = json.loads(content)

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * 10) + (output_tokens / 1_000_000 * 30)

        return {
            'query': query,
            'summary': summary_data,
            'latency_seconds': latency,
            'cost_dollars': cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model': model,
            'num_items_retrieved': len(retrieved_items)
        }

    except Exception as e:
        print(f"Error generating summary: {e}")
        return {
            'query': query,
            'summary': None,
            'latency_seconds': time.time() - start_time,
            'cost_dollars': 0.0,
            'error': str(e)
        }


def run_rag_experiment(api_key, sample_queries=None):
    """
    Run RAG summarization experiment.

    Args:
        api_key: OpenAI API key
        sample_queries: List of queries to test (if None, uses defaults)

    Returns:
        dict: Experiment results
    """

    print("="*80)
    print("RAG SUMMARIZATION EXPERIMENT")
    print("="*80)

    # Default test queries if none provided
    if sample_queries is None:
        sample_queries = [
            "authentication problems",
            "database performance issues",
            "API timeout errors",
            "frontend UI bugs",
            "export functionality",
            "security vulnerabilities"
        ]

    print(f"\n1. Testing RAG summarization on {len(sample_queries)} queries")

    # Load search system
    print("\n2. Loading search system...")
    search_system = load_search_system()

    # Run RAG pipeline for each query
    results = []
    total_cost = 0.0
    total_latency = 0.0

    for i, query in enumerate(sample_queries, 1):
        print(f"\n[{i}/{len(sample_queries)}] Query: '{query}'")

        # Step 1: Retrieve relevant items
        print(f"   Retrieving top 10 items...")
        retrieved_items = search_hybrid(query, search_system, top_k=10)

        print(f"   Found {len(retrieved_items)} items")
        print(f"      Top result: {retrieved_items.iloc[0]['Title'][:60]}...")

        # Step 2: Generate summary
        print(f"   Generating RAG summary...")
        summary_result = generate_rag_summary(query, retrieved_items, api_key)

        if 'error' not in summary_result:
            total_cost += summary_result['cost_dollars']
            total_latency += summary_result['latency_seconds']

            print(f"   ✓ Summary generated")
            print(f"      Summary: {summary_result['summary'].get('summary', 'N/A')[:100]}...")
            print(f"      Cost: ${summary_result['cost_dollars']:.4f}, Latency: {summary_result['latency_seconds']:.2f}s")
        else:
            print(f"   ✗ Error: {summary_result['error']}")

        results.append(summary_result)

    # Save results
    output_path = Path("experiments/llm_integration/rag_summarization")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "rag_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\nCost Analysis:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost per query: ${total_cost/len(successful):.4f}")
        print(f"  Avg latency per query: {total_latency/len(successful):.2f}s")

        # Production estimates
        queries_per_day = 250 * 20  # 250 users, 20 queries/day
        daily_cost = (total_cost / len(successful)) * queries_per_day
        annual_cost = daily_cost * 250

        print(f"\nProduction estimates (250 users @ 20 queries/day):")
        print(f"  Daily cost: ${daily_cost:.2f}")
        print(f"  Annual cost: ${annual_cost:,.2f}")

        # User experience impact
        print(f"\nUser Experience:")
        print(f"  Time to read 10 raw results: ~45 seconds (estimated)")
        print(f"  Time to read RAG summary: ~10 seconds")
        print(f"  Time saved per query: ~35 seconds")
        print(f"  Trade-off: +{total_latency/len(successful):.1f}s latency for summary generation")

    print(f"\n[SUCCESS] Results saved to: {output_path / 'rag_results.json'}")

    return {
        'results': results,
        'total_cost': total_cost,
        'avg_cost': total_cost / len(successful) if successful else 0,
        'avg_latency': total_latency / len(successful) if successful else 0
    }


def main():
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        print("\nSet your OpenAI API key:")
        print("  set OPENAI_API_KEY=your-key-here")
        return

    run_rag_experiment(api_key)


if __name__ == "__main__":
    # Ensure NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    main()
