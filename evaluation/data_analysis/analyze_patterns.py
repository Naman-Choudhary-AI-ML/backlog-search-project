"""
Analyze patterns in existing backlog data to inform synthetic data generation.
This script extracts structural patterns WITHOUT exposing actual data.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def analyze_text_patterns(df, text_column):
    """Analyze text length and structure patterns"""
    texts = df[text_column].dropna()

    return {
        "count": len(texts),
        "avg_length": int(texts.str.len().mean()),
        "median_length": int(texts.str.len().median()),
        "min_length": int(texts.str.len().min()),
        "max_length": int(texts.str.len().max()),
        "avg_word_count": int(texts.str.split().str.len().mean()),
        "median_word_count": int(texts.str.split().str.len().median()),
    }

def analyze_dataframe(df, source_name):
    """Extract patterns from a dataframe"""
    print(f"\nAnalyzing {source_name}...")

    patterns = {
        "source": source_name,
        "total_items": len(df),
        "columns": list(df.columns),
        "non_null_counts": {col: int(df[col].notna().sum()) for col in df.columns},
    }

    # Analyze text columns if they exist
    text_columns = ['Name', 'Summary', 'Title', 'Description']
    for col in text_columns:
        if col in df.columns:
            patterns[f"{col.lower()}_patterns"] = analyze_text_patterns(df, col)

    return patterns

def main():
    """Analyze all data sources and create pattern template"""

    base_path = Path("BacklogRetrievalApp/BacklogRetrievalApp/rawData")

    if not base_path.exists():
        print("Error: rawData directory not found")
        return

    all_patterns = []

    # File configurations
    files = [
        ("CO Defect Backlog.csv", "mac_roman"),
        ("CO SRS.csv", "mac_roman"),
        ("CO PRD.csv", "mac_roman"),
        ("CO_Sus_Features.csv", "utf-8"),
    ]

    for filename, encoding in files:
        filepath = base_path / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                patterns = analyze_dataframe(df, filename)
                all_patterns.append(patterns)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Create generic patterns template for synthetic data
    # These are GENERIC patterns, no actual data
    synthetic_template = {
        "metadata": {
            "description": "Generic patterns for synthetic backlog generation",
            "domain": "software_development",
            "note": "No actual client data included"
        },
        "target_counts": {
            "total_items": 800,  # Target 800 synthetic items
            "bugs": 400,
            "features": 250,
            "tasks": 150
        },
        "text_patterns": {
            "title": {
                "avg_word_count": 6,
                "range": [3, 12]
            },
            "description": {
                "avg_word_count": 40,
                "range": [15, 100]
            }
        },
        "components": [
            "Authentication",
            "API",
            "Database",
            "Frontend",
            "Backend",
            "UI/UX",
            "Security",
            "Performance",
            "Integration",
            "Reporting"
        ],
        "priorities": ["Critical", "High", "Medium", "Low"],
        "statuses": ["Open", "In Progress", "Resolved", "Closed", "Deferred"],
        "issue_types": ["Bug", "Feature", "Task", "Improvement", "Refactor"]
    }

    # Save patterns
    output_path = Path("evaluation/data_analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save analysis summary (counts only, no data)
    summary = {
        "sources_analyzed": len(all_patterns),
        "total_items_analyzed": sum(p["total_items"] for p in all_patterns),
        "note": "Structural patterns extracted, no actual data exposed"
    }

    with open(output_path / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save synthetic template
    with open(output_path / "synthetic_template.json", "w") as f:
        json.dump(synthetic_template, f, indent=2)

    print(f"\n[SUCCESS] Analysis complete!")
    print(f"   Analyzed {summary['total_items_analyzed']} items from {summary['sources_analyzed']} sources")
    print(f"   Created synthetic template: evaluation/data_analysis/synthetic_template.json")
    print(f"   No actual data was exposed or saved.")

if __name__ == "__main__":
    main()
