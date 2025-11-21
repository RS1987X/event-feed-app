#!/usr/bin/env python3
"""
Sample non-guidance press releases for false negative detection.

Creates a review dataset of PRs that were NOT flagged as guidance changes,
stratified by category to find potential false negatives.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import gcsfs
from datetime import datetime, timedelta
import random

def sample_non_guidance(
    days_back: int = 30,
    sample_size: int = 50,
    exclude_categories: list = None,
    output_file: str = "data/labeling/non_guidance_sample_for_review.csv"
):
    """
    Sample PRs that were NOT flagged as guidance changes.
    
    Args:
        days_back: How many days back to sample from
        sample_size: Number of PRs to sample
        exclude_categories: Categories to exclude (e.g., housekeeping, newsletter)
        output_file: Where to save the sample
    """
    if exclude_categories is None:
        exclude_categories = [
            "housekeeping", 
            "newsletter", 
            "ir_general",
            "general_news",
            "spam"
        ]
    
    print(f"Loading press releases from last {days_back} days...")
    
    # Load from GCS
    fs = gcsfs.GCSFileSystem()
    gcs_path = "event-feed-app-data/silver_normalized/table=press_releases"
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    # Load data
    df = pd.read_parquet(
        f"gs://{gcs_path}",
        filesystem=fs,
        filters=[
            ("release_date", ">=", start_date),
            ("release_date", "<=", end_date)
        ]
    )
    
    print(f"Loaded {len(df)} press releases")
    
    # Filter to exclude certain categories
    if exclude_categories:
        df = df[~df['category'].isin(exclude_categories)]
        print(f"After filtering categories: {len(df)}")
    
    # Load alerts to find what WAS flagged
    try:
        alerts_df = pd.read_parquet(
            "gs://event-feed-app-data/feedback/alert_payloads/signal_type=guidance_change",
            filesystem=fs
        )
        flagged_pr_ids = set(alerts_df['press_release_id'].unique())
        print(f"Found {len(flagged_pr_ids)} PRs that were flagged as guidance")
    except:
        print("Warning: Could not load alert history, assuming no PRs flagged")
        flagged_pr_ids = set()
    
    # Get NON-flagged PRs
    non_flagged = df[~df['press_release_id'].isin(flagged_pr_ids)]
    print(f"Non-flagged PRs: {len(non_flagged)}")
    
    # Stratified sampling by category
    sampled = []
    categories = non_flagged['category'].value_counts()
    
    print(f"\nSampling {sample_size} PRs stratified by category:")
    for category, count in categories.items():
        # Sample proportionally
        n_samples = max(1, int(sample_size * count / len(non_flagged)))
        category_sample = non_flagged[non_flagged['category'] == category].sample(
            n=min(n_samples, count),
            random_state=42
        )
        sampled.append(category_sample)
        print(f"  {category}: {len(category_sample)} samples")
    
    # Combine and shuffle
    result = pd.concat(sampled).sample(frac=1, random_state=42).head(sample_size)
    
    # Select relevant columns for review
    review_cols = [
        'press_release_id',
        'company_name',
        'release_date',
        'category',
        'title',
        'full_text',
        'source_url'
    ]
    result = result[review_cols].copy()
    
    # Add review columns
    result['contains_guidance'] = ''  # To be filled by reviewer: yes/no
    result['guidance_type'] = ''      # revenue/earnings/margin/etc
    result['notes'] = ''              # Any notes
    
    # Save
    result.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(result)} PRs to {output_file}")
    print(f"\nReview instructions:")
    print(f"1. Open {output_file} in a spreadsheet")
    print(f"2. For each row, mark 'contains_guidance' as yes/no")
    print(f"3. If yes, specify guidance_type and add notes")
    print(f"4. Any 'yes' entries are false negatives that should improve the model")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sample non-guidance PRs for review")
    parser.add_argument("--days", type=int, default=30, help="Days back to sample")
    parser.add_argument("--size", type=int, default=50, help="Sample size")
    parser.add_argument("--output", type=str, default="data/labeling/non_guidance_sample_for_review.csv")
    
    args = parser.parse_args()
    
    sample_non_guidance(
        days_back=args.days,
        sample_size=args.size,
        output_file=args.output
    )
