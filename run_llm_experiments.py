"""
Wrapper script to run all LLM experiments with OpenAI API key.
This bypasses environment variable issues with conda run.
"""

import os
import sys
from pathlib import Path


def run_query_expansion():
    """Run GPT-4 query expansion experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT 1/3: GPT-4 Query Expansion")
    print("="*80)

    # Import and run
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.llm_integration.query_expansion.evaluate_expansion import main

    try:
        main()
        return True
    except Exception as e:
        print(f"\nERROR in query expansion: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_rag_summarization():
    """Run RAG summarization experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT 2/3: RAG Summarization")
    print("="*80)

    from experiments.llm_integration.rag_summarization.rag_pipeline import main

    try:
        main()
        return True
    except Exception as e:
        print(f"\nERROR in RAG summarization: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_duplicate_detection():
    """Run LLM duplicate detection experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT 3/3: LLM Duplicate Detection")
    print("="*80)

    from experiments.llm_integration.duplicate_detection.llm_duplicate_classifier import main

    try:
        main()
        return True
    except Exception as e:
        print(f"\nERROR in duplicate detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all LLM experiments"""

    # Check if API key is set
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-key-here'  # Linux/Mac")
        print("  set OPENAI_API_KEY=your-key-here       # Windows")
        return

    print("="*80)
    print("RUNNING ALL LLM EXPERIMENTS")
    print("="*80)
    print(f"\nOpenAI API Key: {'✓ Set' if os.environ.get('OPENAI_API_KEY') else '✗ Not set'}")

    results = {}

    # Run experiments
    results['query_expansion'] = run_query_expansion()
    results['rag_summarization'] = run_rag_summarization()
    results['duplicate_detection'] = run_duplicate_detection()

    # Summary
    print("\n\n" + "="*80)
    print("LLM EXPERIMENTS SUMMARY")
    print("="*80)

    for name, success in results.items():
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"  {name:30s}: {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\n  Total: {successful}/{total} successful")

    if successful == total:
        print("\n🎉 ALL LLM EXPERIMENTS COMPLETE!")
    else:
        print(f"\n⚠ {total - successful} experiments failed")


if __name__ == "__main__":
    main()
