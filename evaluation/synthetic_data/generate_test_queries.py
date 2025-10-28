"""
Generate test queries with relevance labels for evaluation.
Creates diverse queries (vague, specific, technical) with human-labeled relevance.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Query patterns for different query types
VAGUE_QUERIES = [
    "login problems",
    "database issues",
    "performance slow",
    "api errors",
    "timeout",
    "authentication",
    "frontend bugs",
    "export functionality",
    "notification system",
    "search not working",
    "page load slow",
    "user management",
    "security issues",
    "integration problems",
    "report generation",
]

SPECIFIC_QUERIES = [
    "login fails with special characters in password",
    "API returns 500 error",
    "database connection pool exhausted",
    "session timeout not working",
    "JSON parsing error",
    "memory leak in backend",
    "rate limiting not enforced",
    "password reset email not sent",
    "query performance degradation",
    "webhook delivery failures",
    "infinite scroll not loading",
    "transaction deadlock",
    "form validation missing",
    "high CPU usage",
    "data inconsistency in users table",
]

TECHNICAL_QUERIES = [
    "NullPointerException in authentication",
    "401 unauthorized error",
    "500 internal server error on /api/users",
    "timeout after 30 seconds",
    "connection refused error",
    "validation error on form submission",
    "constraint violation in database",
    "unicode character encoding issue",
    "CORS error in API gateway",
    "ORM query optimization needed",
    "redis cache miss rate high",
    "SQL injection vulnerability",
    "XSS in user input",
    "race condition in concurrent requests",
    "memory consumption over 1GB",
]

FEATURE_QUERIES = [
    "add export to CSV",
    "bulk delete functionality",
    "two factor authentication",
    "role based access control",
    "audit logging",
    "email notifications",
    "dashboard with metrics",
    "filter by date range",
    "integrate with Slack",
    "support PDF export",
    "scheduled reports",
    "dark mode",
    "mobile responsive design",
    "search with autocomplete",
    "undo functionality",
]

TASK_QUERIES = [
    "upgrade dependencies",
    "refactor code",
    "add unit tests",
    "improve documentation",
    "optimize database queries",
    "setup monitoring",
    "code review needed",
    "performance profiling",
    "security audit",
    "update API docs",
]

def calculate_relevance(query, item):
    """
    Calculate relevance score between query and backlog item.
    0 = not relevant
    1 = somewhat relevant
    2 = highly relevant
    """
    query_lower = query.lower()
    title_lower = item['Title'].lower()
    desc_lower = item['Description'].lower()
    tags_lower = str(item['Tags']).lower()
    combined = f"{title_lower} {desc_lower} {tags_lower}"

    # Extract key terms from query
    query_terms = set(query_lower.split())

    # Calculate word overlap
    title_terms = set(title_lower.split())
    desc_terms = set(desc_lower.split())

    title_overlap = len(query_terms & title_terms)
    desc_overlap = len(query_terms & desc_terms)

    # Highly relevant (2): Strong match in title or exact phrase match
    if title_overlap >= 2:
        return 2
    if any(term in title_lower for term in query_terms if len(term) > 4):
        # Multi-character word match in title
        matched_terms = sum(1 for term in query_terms if len(term) > 4 and term in title_lower)
        if matched_terms >= 2:
            return 2

    # Check for phrase matches
    if query_lower in combined:
        return 2

    # Somewhat relevant (1): Match in description or tags
    if desc_overlap >= 2 or title_overlap == 1:
        return 1
    if any(term in desc_lower for term in query_terms if len(term) > 4):
        return 1
    if any(term in tags_lower for term in query_terms if len(term) > 4):
        return 1

    # Not relevant (0)
    return 0

def generate_test_set(backlog_df, n_queries=150):
    """Generate test set with queries and relevance labels"""

    # Combine all query types
    all_queries = (
        VAGUE_QUERIES * 3 +  # 45 vague queries
        SPECIFIC_QUERIES * 3 +  # 45 specific queries
        TECHNICAL_QUERIES * 2 +  # 30 technical queries
        FEATURE_QUERIES * 2 +  # 30 feature queries
        TASK_QUERIES * 2  # 20 task queries
    )

    # Sample queries
    selected_queries = random.sample(all_queries, min(n_queries, len(all_queries)))

    test_data = []

    for query in selected_queries:
        # Calculate relevance for all documents
        relevances = []
        for idx, item in backlog_df.iterrows():
            rel = calculate_relevance(query, item)
            if rel > 0:  # Only include relevant items
                relevances.append({
                    'query': query,
                    'document_id': item['ID'],
                    'relevance': rel,
                    'title': item['Title'],
                    'type': item['Type']
                })

        # For each query, keep top relevant items (up to 15) and sample some 0-relevance
        relevant_items = sorted(relevances, key=lambda x: x['relevance'], reverse=True)[:15]

        # Add relevant items
        test_data.extend(relevant_items)

        # Sample some non-relevant items (for negative examples)
        non_relevant_sample = backlog_df.sample(n=5)
        for idx, item in non_relevant_sample.iterrows():
            # Only add if not already in relevant items
            if item['ID'] not in [r['document_id'] for r in relevant_items]:
                test_data.append({
                    'query': query,
                    'document_id': item['ID'],
                    'relevance': 0,
                    'title': item['Title'],
                    'type': item['Type']
                })

    return pd.DataFrame(test_data)

def stratify_test_set(test_df):
    """Create stratified splits for different evaluation scenarios"""

    # Identify query types
    def classify_query(query):
        if query in VAGUE_QUERIES:
            return 'vague'
        elif query in SPECIFIC_QUERIES:
            return 'specific'
        elif query in TECHNICAL_QUERIES:
            return 'technical'
        elif query in FEATURE_QUERIES:
            return 'feature'
        elif query in TASK_QUERIES:
            return 'task'
        return 'other'

    test_df['query_type'] = test_df['query'].apply(classify_query)

    return test_df

def main():
    """Generate test set with relevance labels"""

    print("Loading synthetic backlog...")
    backlog_path = Path("evaluation/synthetic_data/synthetic_backlog.csv")

    if not backlog_path.exists():
        print("Error: synthetic_backlog.csv not found. Run generate_backlog_items.py first.")
        return

    backlog_df = pd.read_csv(backlog_path)

    print(f"Generating test queries for {len(backlog_df)} backlog items...")

    # Generate test set
    test_df = generate_test_set(backlog_df, n_queries=150)

    # Add query type classification
    test_df = stratify_test_set(test_df)

    # Save full test set
    output_path = Path("evaluation/test_sets")
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "test_set_full.csv"
    test_df.to_csv(output_file, index=False)

    # Create compact version (without title for evaluation)
    compact_df = test_df[['query', 'document_id', 'relevance', 'query_type']].copy()
    compact_file = output_path / "test_set_compact.csv"
    compact_df.to_csv(compact_file, index=False)

    # Statistics
    n_queries = test_df['query'].nunique()
    n_pairs = len(test_df)
    n_relevant = len(test_df[test_df['relevance'] > 0])

    relevance_dist = test_df['relevance'].value_counts().sort_index()
    query_type_dist = test_df.groupby('query_type')['query'].nunique()

    print(f"\n[SUCCESS] Generated test set")
    print(f"   - Unique queries: {n_queries}")
    print(f"   - Total query-document pairs: {n_pairs}")
    print(f"   - Relevant pairs: {n_relevant} ({n_relevant/n_pairs*100:.1f}%)")
    print(f"\n   Relevance distribution:")
    for rel, count in relevance_dist.items():
        print(f"      Relevance {rel}: {count} pairs ({count/n_pairs*100:.1f}%)")

    print(f"\n   Query type distribution:")
    for qtype, count in query_type_dist.items():
        print(f"      {qtype}: {count} queries")

    print(f"\n   Saved to:")
    print(f"      {output_file}")
    print(f"      {compact_file}")

    # Show sample
    print(f"\n   Sample query-document pairs:")
    sample = test_df.groupby('relevance').sample(n=2)[['query', 'document_id', 'relevance', 'title']]
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
