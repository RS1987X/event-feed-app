#!/usr/bin/env python3
"""
Review earnings-category PRs for missed guidance.

Earnings announcements should often contain guidance,
so this focuses review on that high-yield category.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import gcsfs
from datetime import datetime, timedelta

def review_earnings_for_guidance(
    days_back: int = 14,
    output_file: str = "data/labeling/earnings_guidance_review.csv"
):
    """
    Extract all earnings PRs that were NOT flagged for guidance.
    These are high-priority for false negative detection.
    """
    print("Loading earnings press releases...")
    
    fs = gcsfs.GCSFileSystem()
    gcs_path = "event-feed-app-data/silver_normalized/table=press_releases"
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    # Load classified data
    df = pd.read_parquet(
        f"gs://{gcs_path}",
        filesystem=fs,
        filters=[
            ("release_date", ">=", start_date),
            ("release_date", "<=", end_date),
            ("category", "==", "earnings")  # Only earnings
        ]
    )
    
    print(f"Found {len(df)} earnings press releases")
    
    # Load guidance alerts
    try:
        from event_feed_app.alerts.store import AlertStore
        store = AlertStore()
        
        # Get alerted PR IDs from local DB
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            alerts_df = pd.read_sql(
                "SELECT DISTINCT press_release_id FROM alerts WHERE signal_type='guidance_change'",
                conn
            )
        flagged_ids = set(alerts_df['press_release_id'])
        print(f"Found {len(flagged_ids)} earnings PRs with guidance alerts")
    except:
        flagged_ids = set()
    
    # Find earnings WITHOUT guidance
    no_guidance = df[~df['press_release_id'].isin(flagged_ids)]
    print(f"\nEarnings WITHOUT guidance alerts: {len(no_guidance)}")
    print(f"  (This is {len(no_guidance)/len(df)*100:.1f}% of all earnings)")
    
    # Prepare for review
    review = no_guidance[[
        'press_release_id',
        'company_name', 
        'release_date',
        'title',
        'full_text',
        'source_url'
    ]].copy()
    
    review['has_guidance'] = ''
    review['guidance_metrics'] = ''  # revenue, earnings, margin, etc.
    review['why_missed'] = ''  # notes on why system missed it
    
    review.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")
    print(f"\nExpected guidance rate in earnings: ~60-70%")
    print(f"If many of these contain guidance, you have high false negative rate!")
    
    return review

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", default="data/labeling/earnings_guidance_review.csv")
    args = parser.parse_args()
    
    review_earnings_for_guidance(args.days, args.output)
