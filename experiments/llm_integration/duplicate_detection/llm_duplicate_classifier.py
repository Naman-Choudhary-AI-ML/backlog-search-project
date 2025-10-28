"""
LLM-based duplicate detection.
Uses GPT-4 to identify semantic duplicates that cosine similarity might miss.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


DUPLICATE_DETECTION_PROMPT = """You are a duplicate detection assistant for a software bug tracking system.

Determine if these two backlog items are duplicates (same underlying issue).

Item 1:
ID: {id1}
Title: {title1}
Description: {desc1}

Item 2:
ID: {id2}
Title: {title2}
Description: {desc2}

Consider:
- Do they describe the same root cause?
- Do they have the same symptoms?
- Would fixing one fix the other?

Return ONLY a JSON object:
{{
    "is_duplicate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation..."
}}"""


def classify_duplicate_llm(item1, item2, api_key, model="gpt-4-turbo"):
    """
    Classify if two items are duplicates using LLM.

    Args:
        item1: First backlog item (dict or Series)
        item2: Second backlog item
        api_key: OpenAI API key
        model: GPT model to use

    Returns:
        dict: Classification result with cost
    """

    client = OpenAI(api_key=api_key)

    prompt = DUPLICATE_DETECTION_PROMPT.format(
        id1=item1['ID'],
        title1=item1['Title'],
        desc1=item1['Description'][:300],  # Limit length
        id2=item2['ID'],
        title2=item2['Title'],
        desc2=item2['Description'][:300]
    )

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a duplicate detection expert. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Very low for consistent classification
            max_tokens=200
        )

        latency = time.time() - start_time

        # Parse response
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * 10) + (output_tokens / 1_000_000 * 30)

        return {
            'item1_id': item1['ID'],
            'item2_id': item2['ID'],
            'is_duplicate': result.get('is_duplicate', False),
            'confidence': result.get('confidence', 0.0),
            'reasoning': result.get('reasoning', ''),
            'latency_seconds': latency,
            'cost_dollars': cost
        }

    except Exception as e:
        print(f"Error classifying {item1['ID']} vs {item2['ID']}: {e}")
        return {
            'item1_id': item1['ID'],
            'item2_id': item2['ID'],
            'is_duplicate': False,
            'confidence': 0.0,
            'latency_seconds': time.time() - start_time,
            'cost_dollars': 0.0,
            'error': str(e)
        }


def find_duplicate_candidates(backlog_df, cosine_threshold=0.75, top_k=5):
    """
    Find potential duplicate pairs using cosine similarity.

    Args:
        backlog_df: DataFrame of backlog items
        cosine_threshold: Minimum similarity to consider
        top_k: Max candidates per item

    Returns:
        list: List of (item1, item2, cosine_sim) tuples
    """

    print("Computing embeddings for duplicate detection...")

    # Prepare text
    backlog_df['combined_text'] = (
        backlog_df['Title'].fillna('') + ". " +
        backlog_df['Description'].fillna('')
    )

    os.environ.pop('HF_TOKEN', None)
    os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
    model = SentenceTransformer('all-mpnet-base-v2', token=False)

    documents = backlog_df['combined_text'].tolist()
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Find high-similarity pairs
    candidates = []

    for i in range(len(backlog_df)):
        # Get similar items (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self

        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_sims = similarities[top_indices]

        for j, sim in zip(top_indices, top_sims):
            if sim >= cosine_threshold and i < j:  # i < j to avoid duplicates
                candidates.append((
                    backlog_df.iloc[i],
                    backlog_df.iloc[j],
                    sim
                ))

    print(f"Found {len(candidates)} candidate pairs (cosine >= {cosine_threshold})")

    return candidates


def run_duplicate_detection_experiment(api_key, max_pairs=30):
    """
    Run duplicate detection experiment.

    Compares:
    1. Cosine similarity only
    2. LLM classification
    3. Ensemble (cosine + LLM)

    Args:
        api_key: OpenAI API key
        max_pairs: Maximum pairs to test

    Returns:
        dict: Experiment results
    """

    print("="*80)
    print("LLM DUPLICATE DETECTION EXPERIMENT")
    print("="*80)

    # Load backlog
    print("\n1. Loading backlog data...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog.csv")
    backlog_df = pd.read_csv(backlog_path)
    print(f"   Loaded {len(backlog_df)} items")

    # Find candidates
    print("\n2. Finding duplicate candidates...")
    candidates = find_duplicate_candidates(backlog_df, cosine_threshold=0.75, top_k=3)

    # Sample for testing
    if len(candidates) > max_pairs:
        # Stratify by similarity range
        high_sim = [c for c in candidates if c[2] >= 0.85]
        med_sim = [c for c in candidates if 0.75 <= c[2] < 0.85]

        sample_high = high_sim[:max_pairs//2] if high_sim else []
        sample_med = med_sim[:max_pairs//2] if med_sim else []
        test_candidates = sample_high + sample_med
    else:
        test_candidates = candidates

    print(f"   Testing on {len(test_candidates)} pairs")

    # Classify with LLM
    print("\n3. Classifying with LLM...")
    results = []
    total_cost = 0.0

    for i, (item1, item2, cosine_sim) in enumerate(test_candidates, 1):
        print(f"   [{i}/{len(test_candidates)}] {item1['ID']} vs {item2['ID']} (cosine: {cosine_sim:.3f})")

        llm_result = classify_duplicate_llm(item1, item2, api_key)
        total_cost += llm_result['cost_dollars']

        result = {
            **llm_result,
            'cosine_similarity': cosine_sim
        }

        if 'error' not in llm_result:
            print(f"       LLM: {'DUPLICATE' if llm_result['is_duplicate'] else 'NOT DUPLICATE'} (conf: {llm_result['confidence']:.2f})")

        results.append(result)

    # Save results
    output_path = Path("experiments/llm_integration/duplicate_detection")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "classification_results.csv", index=False)

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    successful = [r for r in results if 'error' not in r]

    print(f"\nClassifications: {len(successful)}/{len(results)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Avg cost per pair: ${total_cost/len(successful) if successful else 0:.4f}")

    # Agreement analysis
    if successful:
        # For different cosine thresholds, compare LLM vs cosine
        for threshold in [0.80, 0.85, 0.90]:
            cosine_duplicates = sum(1 for r in successful if r['cosine_similarity'] >= threshold)
            llm_duplicates = sum(1 for r in successful if r['is_duplicate'])
            both_agree = sum(1 for r in successful if r['cosine_similarity'] >= threshold and r['is_duplicate'])

            print(f"\nThreshold: {threshold}")
            print(f"  Cosine says duplicate: {cosine_duplicates}")
            print(f"  LLM says duplicate: {llm_duplicates}")
            print(f"  Both agree: {both_agree}")

        # Confidence distribution
        confidences = [r['confidence'] for r in successful if r['is_duplicate']]
        if confidences:
            print(f"\nLLM Confidence (for duplicates):")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Min: {np.min(confidences):.3f}")
            print(f"  Max: {np.max(confidences):.3f}")

    # Production estimates
    print(f"\n" + "="*80)
    print("PRODUCTION ESTIMATES")
    print("="*80)

    # Assume 1% of backlog are potential duplicates needing LLM check
    backlog_size = 10000  # Example
    duplicate_rate = 0.01
    pairs_to_check = int(backlog_size * duplicate_rate)

    daily_cost = (total_cost / len(successful)) * pairs_to_check if successful else 0
    annual_cost = daily_cost * 250

    print(f"\nScenario: 10,000 item backlog, 1% potential duplicates")
    print(f"  Pairs to check: {pairs_to_check}")
    print(f"  Cost per check: ${total_cost/len(successful) if successful else 0:.4f}")
    print(f"  Total cost: ${daily_cost:.2f}")
    print(f"  Annual (if checked quarterly): ${daily_cost * 4:.2f}")

    print(f"\n[SUCCESS] Results saved to: {output_path / 'classification_results.csv'}")

    return {
        'results': results,
        'total_cost': total_cost,
        'avg_cost': total_cost / len(successful) if successful else 0
    }


def main():
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    run_duplicate_detection_experiment(api_key, max_pairs=30)


if __name__ == "__main__":
    main()
