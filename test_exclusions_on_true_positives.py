#!/usr/bin/env python3
"""
Test that exclusion rules don't accidentally exclude true positive cases.
Loads all true_positive feedback and ensures they still generate events.
"""
import pandas as pd
from event_feed_app.alerts.runner import fetch_data
from event_feed_app.events.guidance_change.plugin2 import GuidanceChangePlugin
import yaml
from pathlib import Path

# Load true positives from CSV
tp_df = pd.read_csv("guidance_true_positives.csv")
print(f"Loaded {len(tp_df)} true positive feedback records")

# Get unique press release IDs
pr_ids = tp_df['press_release_id'].dropna().unique()
print(f"Unique press releases: {len(pr_ids)}")

# Load YAML config
config_path = Path("src/event_feed_app/configs/significant_events.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Initialize plugin with config
plugin = GuidanceChangePlugin()
plugin.configure(cfg)

# Fetch press releases (need to get date range from data)
print("\nFetching press releases from recent dates...")
df = fetch_data(
    start_date="2025-11-10",
    end_date="2025-11-27",
    max_rows=20000,
    source=None  # All sources
)

print(f"Loaded {len(df)} total press releases")

# Test each true positive
print("\n" + "=" * 80)
print("TESTING TRUE POSITIVES AGAINST EXCLUSION RULES")
print("=" * 80)

excluded_count = 0
still_detects_count = 0
not_found_count = 0
no_events_count = 0

for idx, row in tp_df.iterrows():
    pr_id = row['press_release_id']
    if pd.isna(pr_id):
        continue
    
    # Find the press release
    pr_df = df[df['press_release_id'] == pr_id]
    if len(pr_df) == 0:
        not_found_count += 1
        continue
    
    pr = pr_df.iloc[0]
    doc = {
        'title': pr.get('title', ''),
        'body': pr.get('full_text', ''),
        'press_release_id': pr_id,
        'company_name': pr.get('company_name', ''),
    }
    
    # Check if it would be excluded
    text = f"{doc['title']}\n{doc['body']}"
    is_excluded = plugin._should_exclude_document(text)
    
    # Try to detect events
    events = list(plugin.detect(doc))
    
    if is_excluded:
        excluded_count += 1
        print(f"\n❌ EXCLUDED (TRUE POSITIVE!): {pr_id}")
        print(f"   Company: {doc.get('company_name', 'N/A')}")
        print(f"   Title: {doc['title'][:100]}...")
        print(f"   Feedback ID: {row['feedback_id']}")
        
    elif len(events) == 0:
        no_events_count += 1
        print(f"\n⚠️  NO EVENTS DETECTED (but was true positive): {pr_id}")
        print(f"   Company: {doc.get('company_name', 'N/A')}")
        print(f"   Title: {doc['title'][:100]}...")
        
    else:
        still_detects_count += 1
        if (idx + 1) % 5 == 0:
            print(f"✓ Checked {idx + 1}/{len(tp_df)} true positives...")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total true positives checked: {len(tp_df)}")
print(f"  ✅ Still detects events: {still_detects_count}")
print(f"  ⚠️  No events (may be detection issue): {no_events_count}")
print(f"  ❌ EXCLUDED by new rules: {excluded_count}")
print(f"  📭 Press release not found: {not_found_count}")

if excluded_count > 0:
    print("\n⚠️  WARNING: Exclusion rules are blocking true positives!")
    print("Review the excluded cases above and adjust rules if needed.")
else:
    print("\n✅ SUCCESS: No true positives were excluded by the new rules!")

if no_events_count > 0:
    print(f"\nℹ️  Note: {no_events_count} true positives didn't generate events.")
    print("This may be due to changes in detection logic, not exclusions.")
