"""
Master script to run all experiments in sequence.
Run this after setting OPENAI_API_KEY environment variable.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(description, command):
    """Run a command and report status"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)

    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Run all experiments"""

    print("="*80)
    print("RUNNING ALL EXPERIMENTS")
    print("="*80)

    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n⚠ WARNING: OPENAI_API_KEY not set")
        print("   LLM experiments will be skipped")
        print("\n   To set API key:")
        print("      Windows: set OPENAI_API_KEY=your-key-here")
        print("      Linux/Mac: export OPENAI_API_KEY=your-key-here")
        skip_llm = True
    else:
        print(f"\n✓ OpenAI API key found")
        skip_llm = False

    results = {}

    # 1. Hyperparameter grid search
    results['grid_search'] = run_command(
        "1. Hyperparameter Grid Search",
        "conda run -n spotlight python experiments/hyperparameter_tuning/grid_search.py"
    )

    # 2. Cross-encoder reranking
    results['cross_encoder'] = run_command(
        "2. Cross-Encoder Reranking",
        "conda run -n spotlight python experiments/advanced_retrieval/cross_encoder/reranking.py"
    )

    # 3. FAISS integration
    results['faiss'] = run_command(
        "3. FAISS Scalability Demo",
        "conda run -n spotlight python experiments/advanced_retrieval/faiss_integration/faiss_demo.py"
    )

    # 4. LLM experiments (if API key available)
    if not skip_llm:
        results['query_expansion'] = run_command(
            "4. GPT-4 Query Expansion",
            "conda run -n spotlight python experiments/llm_integration/query_expansion/evaluate_expansion.py"
        )

        results['rag_summarization'] = run_command(
            "5. RAG Summarization",
            "conda run -n spotlight python experiments/llm_integration/rag_summarization/rag_pipeline.py"
        )

        results['duplicate_detection'] = run_command(
            "6. LLM Duplicate Detection",
            "conda run -n spotlight python experiments/llm_integration/duplicate_detection/llm_duplicate_classifier.py"
        )
    else:
        print("\n" + "="*80)
        print("SKIPPING LLM EXPERIMENTS (No API key)")
        print("="*80)

    # Summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    for name, success in results.items():
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"  {name:30s}: {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\n  Total: {successful}/{total} successful")

    if successful == total:
        print("\n🎉 ALL EXPERIMENTS COMPLETE!")
    else:
        print(f"\n⚠ {total - successful} experiments failed - check logs above")


if __name__ == "__main__":
    main()
