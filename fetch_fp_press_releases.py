#!/usr/bin/env python3
"""
Fetch press releases for false positive analysis.
"""
from event_feed_app.alerts.runner import fetch_data
import pandas as pd

# False positive press release IDs from CSV
lawsuit_prs = [
    "prnewswire__7d9d4d1a828c8a12",  # Securities fraud lawsuit
    "prnewswire__e45d311b92f17f3f",  # Securities fraud lawsuit
]

market_research_prs = [
    "prnewswire__3112a9d70933f504",  # Market research report
]

# Fetch recent data (these are from Nov 24-26, 2025)
df = fetch_data(
    start_date="2025-11-20",
    end_date="2025-11-27",
    max_rows=10000,
    source="prnewswire"
)

print(f"Loaded {len(df)} press releases")
print()

print("=" * 80)
print("SECURITIES FRAUD LAWSUIT FALSE POSITIVES")
print("=" * 80)

for pr_id in lawsuit_prs:
    print(f"\n{'='*80}")
    print(f"PR ID: {pr_id}")
    print("=" * 80)
    
    pr_df = df[df['press_release_id'] == pr_id]
    if len(pr_df) > 0:
        pr = pr_df.iloc[0]
        print(f"Company: {pr.get('company_name', 'N/A')}")
        print(f"Title: {pr.get('title', 'N/A')}")
        print(f"Date: {pr.get('release_date', 'N/A')}")
        print(f"\nFull Text (first 3000 chars):")
        print("-" * 80)
        full_text = pr.get('full_text', '')
        print(full_text[:3000])
        print("-" * 80)
        
        # Look for lawsuit indicators
        text_lower = full_text.lower()
        lawsuit_keywords = [
            'class action', 'lawsuit', 'litigation', 'complaint', 
            'securities fraud', 'plaintiff', 'defendant', 'law firm',
            'rosen law firm', 'kahn swick', 'bragar eagel', 'faruqi',
            'investor rights', 'securities litigation'
        ]
        
        found_keywords = [kw for kw in lawsuit_keywords if kw in text_lower]
        if found_keywords:
            print(f"\n🚨 Lawsuit Keywords Found: {found_keywords}")
    else:
        print("❌ Could not find press release in dataset")

print("\n\n")
print("=" * 80)
print("MARKET RESEARCH REPORT FALSE POSITIVES")
print("=" * 80)

for pr_id in market_research_prs:
    print(f"\n{'='*80}")
    print(f"PR ID: {pr_id}")
    print("=" * 80)
    
    pr_df = df[df['press_release_id'] == pr_id]
    if len(pr_df) > 0:
        pr = pr_df.iloc[0]
        print(f"Company: {pr.get('company_name', 'N/A')}")
        print(f"Title: {pr.get('title', 'N/A')}")
        print(f"Date: {pr.get('release_date', 'N/A')}")
        print(f"\nFull Text (first 3000 chars):")
        print("-" * 80)
        full_text = pr.get('full_text', '')
        print(full_text[:3000])
        print("-" * 80)
        
        # Look for market research indicators
        text_lower = full_text.lower()
        research_keywords = [
            'market research', 'market report', 'market forecast',
            'industry report', 'market analysis', 'market intelligence',
            'cagr', 'compound annual growth rate', 'market size',
            'market share', 'market trends', 'forecast period',
            'research and markets', 'technavio', 'mordor intelligence',
            'grand view research', 'fortune business insights',
            'market study', 'research report'
        ]
        
        found_keywords = [kw for kw in research_keywords if kw in text_lower]
        if found_keywords:
            print(f"\n📊 Market Research Keywords Found: {found_keywords}")
    else:
        print("❌ Could not find press release in dataset")
