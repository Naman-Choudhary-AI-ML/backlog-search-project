"""
Query Expansion using GPT-4.
Expands vague queries with synonyms, technical terms, and related concepts.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from openai import OpenAI
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


EXPANSION_PROMPT_TEMPLATE = """You are a technical search query expansion assistant for a software bug tracking system.

Given a user's search query, expand it with:
1. Synonyms and alternative phrasings
2. Related technical terms
3. Common variations

User Query: "{query}"

Return ONLY a JSON object with this exact structure (no markdown, no code blocks):
{{
    "original": "{query}",
    "synonyms": ["synonym1", "synonym2", "synonym3"],
    "technical_terms": ["technical_term1", "technical_term2"],
    "related_concepts": ["concept1", "concept2"]
}}

Keep expansions relevant to software development (bugs, features, APIs, databases, authentication, etc.).
Maximum 3-4 terms per category."""


def expand_query_gpt4(query, api_key, model="gpt-4-turbo"):
    """
    Expand a query using GPT-4.

    Args:
        query: Original search query
        api_key: OpenAI API key
        model: GPT model to use

    Returns:
        dict: Expansion results with cost/latency
    """

    client = OpenAI(api_key=api_key)

    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=query)

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a query expansion assistant. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for consistent results
            max_tokens=300
        )

        latency = time.time() - start_time

        # Extract response
        content = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks if present)
        if content.startswith("```"):
            # Remove markdown code blocks
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        expansion = json.loads(content)

        # Calculate cost (GPT-4-turbo pricing as of 2025)
        # Input: $10/1M tokens, Output: $30/1M tokens
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * 10) + (output_tokens / 1_000_000 * 30)

        return {
            'original_query': query,
            'expansion': expansion,
            'latency_seconds': latency,
            'cost_dollars': cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model': model
        }

    except Exception as e:
        print(f"Error expanding query '{query}': {e}")
        return {
            'original_query': query,
            'expansion': {'original': query, 'synonyms': [], 'technical_terms': [], 'related_concepts': []},
            'latency_seconds': time.time() - start_time,
            'cost_dollars': 0.0,
            'error': str(e)
        }


def expand_query_to_text(expansion_result):
    """
    Convert expansion result to a single search string.

    Args:
        expansion_result: Result from expand_query_gpt4

    Returns:
        str: Combined query string
    """
    exp = expansion_result['expansion']
    original = exp.get('original', '')
    synonyms = ' '.join(exp.get('synonyms', []))
    technical = ' '.join(exp.get('technical_terms', []))
    related = ' '.join(exp.get('related_concepts', []))

    # Combine (give more weight to original by repeating it)
    combined = f"{original} {original} {synonyms} {technical} {related}"

    return combined


def run_query_expansion_experiment(api_key, sample_size=20):
    """
    Run query expansion experiment on a sample of test queries.

    Args:
        api_key: OpenAI API key
        sample_size: Number of queries to test

    Returns:
        dict: Experiment results
    """

    print("="*80)
    print("GPT-4 QUERY EXPANSION EXPERIMENT")
    print("="*80)

    # Load test queries
    print("\n1. Loading test queries...")
    test_set_path = Path("evaluation/test_sets/test_set_compact.csv")
    test_df = pd.read_csv(test_set_path)

    # Sample diverse queries (stratified by query type if available)
    if 'query_type' in test_df.columns:
        # Sample from each type
        sampled = test_df.groupby('query_type')['query'].apply(
            lambda x: x.drop_duplicates().sample(n=min(4, len(x.unique())), random_state=42)
        ).reset_index(drop=True)
        queries = sampled.tolist()[:sample_size]
    else:
        queries = test_df['query'].drop_duplicates().sample(n=sample_size, random_state=42).tolist()

    print(f"   Selected {len(queries)} diverse queries for expansion")

    # Expand queries
    print(f"\n2. Expanding queries with GPT-4...")
    expansions = []
    total_cost = 0.0
    total_latency = 0.0

    for i, query in enumerate(queries, 1):
        print(f"   [{i}/{len(queries)}] Expanding: '{query[:50]}...'")

        result = expand_query_gpt4(query, api_key)
        expansions.append(result)

        total_cost += result['cost_dollars']
        total_latency += result['latency_seconds']

        # Show expansion
        if 'error' not in result:
            print(f"      → Synonyms: {result['expansion'].get('synonyms', [])}")
            print(f"      → Technical: {result['expansion'].get('technical_terms', [])}")
            print(f"      → Cost: ${result['cost_dollars']:.4f}, Latency: {result['latency_seconds']:.2f}s")

    # Save expansions
    output_path = Path("experiments/llm_integration/query_expansion")
    output_path.mkdir(parents=True, exist_ok=True)

    expansions_df = pd.DataFrame(expansions)
    expansions_df.to_json(output_path / "expansions.json", orient='records', indent=2)

    # Summary statistics
    print(f"\n3. Summary Statistics:")
    print(f"   Total queries expanded: {len(expansions)}")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Avg cost per query: ${total_cost/len(expansions):.4f}")
    print(f"   Total latency: {total_latency:.2f}s")
    print(f"   Avg latency per query: {total_latency/len(expansions):.2f}s")

    # Estimate production cost
    total_queries_in_test = test_df['query'].nunique()
    estimated_total_cost = (total_cost / len(expansions)) * total_queries_in_test

    print(f"\n4. Production Estimates:")
    print(f"   Full test set: {total_queries_in_test} queries")
    print(f"   Estimated cost: ${estimated_total_cost:.2f}")

    # Daily/annual estimates (example: 250 users, 20 queries/user/day)
    queries_per_day = 250 * 20  # 5000 queries/day
    daily_cost = (total_cost / len(expansions)) * queries_per_day
    annual_cost = daily_cost * 250  # 250 working days

    print(f"\n   For 250 users @ 20 queries/day:")
    print(f"      Daily cost: ${daily_cost:.2f}")
    print(f"      Annual cost: ${annual_cost:,.2f}")

    print(f"\n[SUCCESS] Query expansion experiment complete!")
    print(f"   Results saved to: {output_path / 'expansions.json'}")

    return {
        'expansions': expansions,
        'total_cost': total_cost,
        'avg_cost_per_query': total_cost / len(expansions),
        'avg_latency': total_latency / len(expansions),
        'estimated_annual_cost': annual_cost
    }


def main():
    """Run query expansion experiment"""

    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your OpenAI API key:")
        print("  Windows: set OPENAI_API_KEY=your-key-here")
        print("  Linux/Mac: export OPENAI_API_KEY=your-key-here")
        print("\nOr pass it as an argument to this script")
        return

    # Run experiment
    run_query_expansion_experiment(api_key, sample_size=20)


if __name__ == "__main__":
    main()
