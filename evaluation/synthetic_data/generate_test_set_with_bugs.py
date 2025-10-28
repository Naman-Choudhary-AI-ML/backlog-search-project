"""
Generate complete test set with bugs for each query using GPT-4o.

This script creates a realistic test dataset by:
1. Using 50 diverse queries (from test_queries_unique_50.csv)
2. For each query, generating 45 bugs with controlled relevance:
   - 15 irrelevant (relevance=0)
   - 15 somewhat relevant (relevance=1)
   - 15 highly relevant (relevance=2)
3. Total: 50 queries × 45 bugs = 2,250 labeled pairs

Output format matches old test_set_compact.csv for compatibility.
"""

import os
import json
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI


def generate_bugs_for_query(client, query, query_type, query_index):
    """Generate exactly 45 bugs for a single query with controlled relevance distribution."""

    # Enhanced prompt with clear instructions and context
    prompt = f"""You are a senior software engineer creating a test dataset for evaluating a bug search system. Your task is to generate EXACTLY 45 realistic bug reports for the search query below.

SEARCH QUERY: "{query}"
QUERY TYPE: {query_type}

TASK: Generate 45 bug reports with the following EXACT distribution:
• First 15 bugs: IRRELEVANT (relevance=0)
  - These bugs should be about completely different issues
  - User searching for "{query}" would NOT want these results
  - Example: If query is "memory leak", irrelevant might be "button color is wrong"

• Next 15 bugs: SOMEWHAT RELEVANT (relevance=1)
  - These bugs are related but not a perfect match
  - User might find these marginally helpful
  - Example: If query is "memory leak", somewhat relevant might be "application runs slowly"

• Last 15 bugs: HIGHLY RELEVANT (relevance=2)
  - These bugs are directly about the query issue
  - User searching for "{query}" would definitely want these results
  - Example: If query is "memory leak", highly relevant is "OutOfMemoryError in service"

CRITICAL REQUIREMENTS:
1. Generate EXACTLY 45 bug reports (15 + 15 + 15 = 45)
2. Each bug must be realistic and detailed
3. Use diverse technical terminology (NullPointerException, API timeout, CSS bug, etc.)
4. Vary bug types: crash, performance, ui, security, integration, data, feature
5. Vary components: Authentication, Dashboard, API, Database, etc.
6. Vary severities: Critical, High, Medium, Low
7. Make bug IDs unique using format: TYPE-{query_index:02d}XX (e.g., CRA-{query_index:02d}01, UI-{query_index:02d}15)

OUTPUT FORMAT: Return a JSON array of exactly 45 objects with this structure:
[
  {{
    "bug_id": "CRA-{query_index:02d}01",
    "title": "Concise bug title (max 80 chars)",
    "description": "Realistic 2-3 sentence description with technical details",
    "type": "crash",
    "component": "Authentication",
    "severity": "Critical",
    "relevance": 0
  }},
  ... (44 more bugs)
]

EXAMPLE:
If the query is "NullPointerException in authentication":
- IRRELEVANT (0): "Dashboard chart colors are incorrect", "Mobile app icon missing", "Email notification delay"
- SOMEWHAT RELEVANT (1): "Login slow performance", "Session timeout too short", "User credentials validation error"
- HIGHLY RELEVANT (2): "NullPointerException when authenticating user", "NPE in AuthService.validate()", "Authentication crashes with null user object"

DOUBLE-CHECK before responding:
✓ Total bugs = 45 (or as close as possible)
✓ First 15 have relevance=0 (irrelevant)
✓ Next 15 have relevance=1 (somewhat relevant)
✓ Last 15 have relevance=2 (highly relevant)
✓ All bug_ids are unique
✓ Output is valid JSON array

Return ONLY the JSON array, no other text."""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better instruction following
                messages=[
                    {"role": "system", "content": "You are a software engineer creating high-quality test data for ML evaluation. Follow instructions precisely. Always generate exactly the requested number of items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=6000  # Increased for 45 bugs
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])  # Remove first and last line
                if content.startswith("json"):
                    content = content[4:].strip()

            bugs = json.loads(content)

            # Accept any reasonable number of bugs (30-50 range)
            if len(bugs) >= 30 and len(bugs) <= 50:
                # Verify distribution
                rel_counts = {}
                for bug in bugs:
                    rel = bug.get('relevance', -1)
                    rel_counts[rel] = rel_counts.get(rel, 0) + 1

                # Check if distribution is reasonable (allowing tolerance)
                if (rel_counts.get(0, 0) >= 10 and
                    rel_counts.get(1, 0) >= 10 and
                    rel_counts.get(2, 0) >= 10):
                    # Good enough - accept it
                    if len(bugs) != 45:
                        print(f"  Note: Generated {len(bugs)} bugs (target: 45) - ACCEPTED")
                    return bugs
                else:
                    print(f"  WARNING: Poor distribution for {query[:40]}: {rel_counts}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(1)
                        continue
                    else:
                        # Accept it anyway if retries exhausted
                        print(f"  Accepting anyway (retries exhausted)")
                        return bugs
            else:
                print(f"  WARNING: Got {len(bugs)} bugs (outside 30-50 range)")
                if attempt < max_retries - 1:
                    print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(1)
                    continue
                else:
                    # Still accept if we have something
                    if len(bugs) > 0:
                        print(f"  Accepting {len(bugs)} bugs (retries exhausted)")
                        return bugs
                    else:
                        print(f"  No bugs generated - FAILED")
                        return None

        except json.JSONDecodeError as e:
            print(f"  ERROR: JSON parse failed for query '{query[:40]}': {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                print(f"  Skipping query after {max_retries} attempts")
                return None
        except Exception as e:
            print(f"  ERROR: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(1)
                continue
            else:
                return None

    return None


def generate_complete_test_set(n_queries=50, api_key=None):
    """Generate complete test set with bugs for each query."""

    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")

    client = OpenAI(api_key=api_key)

    # Load queries (use deduplicated unique queries)
    queries_df = pd.read_csv('evaluation/test_sets/test_queries_unique_50.csv')
    queries_df = queries_df.head(n_queries)

    print("=" * 80)
    print("GENERATING COMPLETE TEST SET WITH GPT-4o")
    print("=" * 80)
    print(f"\nQueries: {len(queries_df)}")
    print(f"Target bugs per query: 45 (15 irrelevant, 15 relevant, 15 highly relevant)")
    print(f"Target total pairs: {len(queries_df) * 45}")
    print(f"\nUsing GPT-4o for high-quality instruction following")
    print("Note: Accepting variable sizes if GPT doesn't generate exactly 45\n")

    all_rows = []
    failed_queries = []

    for i, row in queries_df.iterrows():
        query = row['query']
        query_type = row['query_type']

        print(f"[{i+1}/{len(queries_df)}] Generating bugs for: \"{query[:50]}...\"")

        bugs = generate_bugs_for_query(client, query, query_type, i)

        if bugs is None:
            print(f"  FAILED - Skipping query")
            failed_queries.append(query)
            continue

        # Convert to test set format
        for bug in bugs:
            all_rows.append({
                'query': query,
                'document_id': bug['bug_id'],
                'bug_title': bug.get('title', 'Untitled'),
                'bug_description': bug.get('description', ''),
                'type': bug.get('type', 'unknown'),
                'component': bug.get('component', 'unknown'),
                'severity': bug.get('severity', 'Medium'),
                'relevance': bug.get('relevance', 1),
                'query_type': query_type
            })

        print(f"  SUCCESS: Generated {len(bugs)} bugs")

        # Rate limiting
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(queries_df)} queries completed")
            time.sleep(2)  # Avoid rate limits

    # Create DataFrame
    test_set_df = pd.DataFrame(all_rows)

    # Verify distribution
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal pairs: {len(test_set_df)}")
    print(f"Unique queries: {test_set_df['query'].nunique()}")
    print(f"Avg bugs per query: {len(test_set_df) / test_set_df['query'].nunique():.1f}")

    if failed_queries:
        print(f"\nFailed queries ({len(failed_queries)}):")
        for fq in failed_queries:
            print(f"  - {fq}")

    print("\nRelevance distribution:")
    rel_counts = test_set_df['relevance'].value_counts().sort_index()
    for rel, count in rel_counts.items():
        pct = (count / len(test_set_df)) * 100
        print(f"  Relevance {rel}: {count:4d} ({pct:5.1f}%)")

    print("\nQuery type distribution:")
    print(test_set_df['query_type'].value_counts())

    # Save test set (compact format for evaluation)
    test_set_compact = test_set_df[['query', 'document_id', 'relevance', 'query_type']]
    output_path_compact = Path("evaluation/test_sets/test_set_llm_compact.csv")
    test_set_compact.to_csv(output_path_compact, index=False)
    print(f"\n[SAVED] Compact test set: {output_path_compact}")

    # Save full test set (with all bug details)
    output_path_full = Path("evaluation/test_sets/test_set_llm_full.csv")
    test_set_df.to_csv(output_path_full, index=False)
    print(f"[SAVED] Full test set: {output_path_full}")

    # Also save bugs as a separate corpus
    bugs_df = test_set_df[['document_id', 'bug_title', 'bug_description', 'type', 'component', 'severity']].drop_duplicates(subset=['document_id'])
    output_path_bugs = Path("evaluation/synthetic_data/synthetic_backlog_from_queries.csv")
    bugs_df.columns = ['ID', 'Title', 'Description', 'Type', 'Component', 'Severity']
    bugs_df.to_csv(output_path_bugs, index=False)
    print(f"[SAVED] Bug corpus: {output_path_bugs}")

    print("\nNext steps:")
    print("  1. Review data quality in test_set_llm_full.csv")
    print("  2. Run baseline evaluation: python evaluation/evaluate_baseline.py")
    print("  3. Run grid search: python experiments/hyperparameter_tuning/grid_search.py")

    return test_set_df


if __name__ == "__main__":
    # Generate test set
    test_set_df = generate_complete_test_set(n_queries=50)
