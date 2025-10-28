"""
Generate high-quality synthetic backlog data using GPT-4.

This script creates realistic, diverse software bug reports with:
- Varied writing styles (junior vs senior engineers)
- Different bug types (crash, performance, UI, security, integration)
- Domain-specific technical terminology
- Realistic complexity and details
"""

import os
import json
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI

# Bug types and their characteristics
BUG_TYPES = {
    'crash': ['NullPointerException', 'SegFault', 'OutOfMemoryError', 'StackOverflow', 'crash', 'application freeze'],
    'performance': ['slow', 'latency', 'timeout', 'memory leak', 'CPU spike', 'inefficient query'],
    'ui': ['button not working', 'layout broken', 'display issue', 'CSS problem', 'responsive design'],
    'security': ['XSS vulnerability', 'SQL injection', 'authentication bypass', 'privilege escalation', 'CSRF'],
    'integration': ['API failure', 'webhook not firing', 'third-party integration', 'SSO issue', 'service timeout'],
    'data': ['data corruption', 'migration failed', 'database deadlock', 'inconsistent state', 'validation error'],
    'feature': ['feature not working as expected', 'incorrect behavior', 'missing functionality', 'edge case']
}

COMPONENTS = [
    'Authentication', 'User Management', 'Dashboard', 'Reporting', 'API Gateway',
    'Payment Processing', 'Notification Service', 'Search', 'Analytics',
    'File Upload', 'Database', 'Cache Layer', 'Admin Panel', 'Mobile App',
    'Integration Service', 'Logging System', 'Configuration', 'Deployment Pipeline'
]

SEVERITIES = ['Critical', 'High', 'Medium', 'Low']

WRITING_STYLES = ['senior', 'junior', 'detailed', 'brief']


def generate_bug_report(client, bug_type, component, severity, style, index):
    """Generate a single realistic bug report using GPT-4."""

    keywords = BUG_TYPES[bug_type]
    keyword_hint = ', '.join(keywords[:2])

    if style == 'senior':
        style_instruction = "Write like an experienced senior engineer: concise, technical, with specific error messages and stack traces."
    elif style == 'junior':
        style_instruction = "Write like a junior engineer: more descriptive, less technical jargon, explaining symptoms in detail."
    elif style == 'detailed':
        style_instruction = "Write a detailed bug report with multiple paragraphs, steps to reproduce, and environment details."
    else:  # brief
        style_instruction = "Write a brief, to-the-point bug report in 2-3 sentences."

    prompt = f"""Generate a realistic software bug report for a {component} component.

Bug type: {bug_type} (related to: {keyword_hint})
Severity: {severity}
Style: {style_instruction}

The bug should be realistic and specific. Include technical details appropriate to the style.
Generate ONLY the bug description/title, no other text.
Keep it under 100 words."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini to reduce costs
            messages=[
                {"role": "system", "content": "You are a software engineer writing bug reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=200
        )

        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        print(f"Error generating bug {index}: {e}")
        return f"{severity} {bug_type} issue in {component} component requiring investigation"


def generate_test_query(client, query_type, index):
    """Generate a realistic test query."""

    if query_type == 'specific':
        prompt = """Generate a specific, technical search query that a developer would use to find a particular bug.
Examples: "NullPointerException in login API", "memory leak in dashboard component", "SQL injection in user search"
Generate ONE query only, no explanation. Make it realistic and specific."""

    elif query_type == 'vague':
        prompt = """Generate a vague, general search query that a user might use.
Examples: "login not working", "page is slow", "error message appears"
Generate ONE query only, no explanation. Make it brief and non-technical."""

    elif query_type == 'feature':
        prompt = """Generate a search query about a feature request or enhancement.
Examples: "add export to CSV", "improve search filters", "support dark mode"
Generate ONE query only, no explanation."""

    elif query_type == 'task':
        prompt = """Generate a search query about a development task or improvement.
Examples: "refactor authentication module", "update dependencies", "improve documentation"
Generate ONE query only, no explanation."""

    else:  # technical
        prompt = """Generate a technical search query with specific error messages or technical terms.
Examples: "ConnectionTimeout in webhook service", "Redis cache invalidation issue", "CORS error in API"
Generate ONE query only, no explanation. Use technical terminology."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a developer searching for bugs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # High diversity
            max_tokens=50
        )

        query = response.choices[0].message.content.strip().strip('"').strip("'")
        return query

    except Exception as e:
        print(f"Error generating query {index}: {e}")
        return f"{query_type} query {index}"


def estimate_cost(n_bugs, n_queries):
    """Estimate API costs."""
    # GPT-4o-mini: $0.150 / 1M input tokens, $0.600 / 1M output tokens
    # Rough estimate: ~150 input tokens/bug, ~100 output tokens/bug
    # ~50 input tokens/query, ~20 output tokens/query

    bug_cost = n_bugs * (150 * 0.150 / 1_000_000 + 100 * 0.600 / 1_000_000)
    query_cost = n_queries * (50 * 0.150 / 1_000_000 + 20 * 0.600 / 1_000_000)

    total = bug_cost + query_cost
    return total


def generate_synthetic_backlog(n_documents=800, api_key=None):
    """Generate synthetic backlog with GPT-4."""

    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("GENERATING SYNTHETIC BACKLOG WITH GPT-4")
    print("=" * 80)

    estimated_cost = estimate_cost(n_documents, 0)
    print(f"\nEstimated cost for {n_documents} bug reports: ${estimated_cost:.2f}")
    print("Using GPT-4o-mini for cost efficiency\n")

    bugs = []

    # Distribute bugs across types, components, severities
    bug_types_list = list(BUG_TYPES.keys())

    for i in range(n_documents):
        # Distribute evenly with some randomness
        bug_type = bug_types_list[i % len(bug_types_list)]
        component = COMPONENTS[i % len(COMPONENTS)]
        severity = SEVERITIES[i % len(SEVERITIES)]
        style = WRITING_STYLES[i % len(WRITING_STYLES)]

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_documents} bug reports...")

        description = generate_bug_report(
            client, bug_type, component, severity, style, i + 1
        )

        # Create bug ID
        prefix = bug_type.upper()[:3]
        if bug_type in ['feature', 'task']:
            prefix = bug_type.upper()[:4]

        bug_id = f"{prefix}-{i:04d}"

        bugs.append({
            'ID': bug_id,
            'Title': description[:100],  # First 100 chars as title
            'Description': description,
            'Type': bug_type,
            'Component': component,
            'Severity': severity
        })

        # Rate limiting - small delay
        if (i + 1) % 10 == 0:
            time.sleep(1)  # Avoid rate limits

    # Create DataFrame
    df = pd.DataFrame(bugs)

    # Save to CSV
    output_path = Path("evaluation/synthetic_data/synthetic_backlog_llm.csv")
    df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Generated {len(bugs)} bug reports")
    print(f"Saved to: {output_path}")

    return df


def generate_test_queries(n_queries=100, api_key=None):
    """Generate diverse test queries with GPT-4."""

    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")

    client = OpenAI(api_key=api_key)

    print("\n" + "=" * 80)
    print("GENERATING TEST QUERIES WITH GPT-4")
    print("=" * 80)

    estimated_cost = estimate_cost(0, n_queries)
    print(f"\nEstimated cost for {n_queries} queries: ${estimated_cost:.2f}\n")

    # Query type distribution: specific, vague, feature, task, technical
    query_types = ['specific', 'vague', 'feature', 'task', 'technical']

    queries = []

    for i in range(n_queries):
        query_type = query_types[i % len(query_types)]

        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{n_queries} queries...")

        query_text = generate_test_query(client, query_type, i + 1)

        queries.append({
            'query': query_text,
            'query_type': query_type
        })

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    # Create DataFrame
    df = pd.DataFrame(queries)

    # Save to CSV
    output_path = Path("evaluation/test_sets/test_queries_llm.csv")
    df.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Generated {len(queries)} test queries")
    print(f"Saved to: {output_path}")

    return df


def main():
    """Generate both synthetic backlog and test queries."""

    # API key from environment variable (secure practice)
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-key-here'  # Linux/Mac")
        print("  set OPENAI_API_KEY=your-key-here       # Windows")
        return

    # Estimate total cost
    total_cost = estimate_cost(800, 100)
    print(f"\nTotal estimated cost: ${total_cost:.2f}")
    print("\nThis will generate:")
    print("  - 800 high-quality bug reports")
    print("  - 100 diverse test queries")
    print("\nUsing GPT-4o-mini for cost efficiency")

    # Generate bugs
    bugs_df = generate_synthetic_backlog(n_documents=800, api_key=api_key)

    # Generate queries
    queries_df = generate_test_queries(n_queries=100, api_key=api_key)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nFiles created:")
    print("  - evaluation/synthetic_data/synthetic_backlog_llm.csv")
    print("  - evaluation/test_sets/test_queries_llm.csv")
    print(f"\nNext steps:")
    print("  1. Review generated data quality")
    print("  2. Create relevance labels for test queries")
    print("  3. Re-run evaluations with new data")


if __name__ == "__main__":
    main()
