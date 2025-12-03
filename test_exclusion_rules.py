#!/usr/bin/env python3
"""
Test that exclusion rules work correctly for known false positives.
"""
from event_feed_app.alerts.runner import fetch_data
from event_feed_app.events.guidance_change.plugin import GuidanceChangePlugin
import pandas as pd
import yaml

# Load YAML config
with open("src/event_feed_app/configs/significant_events.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize and configure plugin with YAML
plugin = GuidanceChangePlugin()
plugin.configure(cfg)

# False positive press release IDs that should now be excluded
test_cases = [
    ("prnewswire__7d9d4d1a828c8a12", "Securities fraud lawsuit #1"),
    ("prnewswire__e45d311b92f17f3f", "Securities fraud lawsuit #2"),
    ("prnewswire__3112a9d70933f504", "Market research report"),
]

# Fetch data
df = fetch_data(
    start_date="2025-11-20",
    end_date="2025-11-27",
    max_rows=10000,
    source="prnewswire"
)

print(f"Loaded {len(df)} press releases")
print()

print("=" * 80)
print("TESTING EXCLUSION RULES")
print("=" * 80)
print()

for pr_id, description in test_cases:
    print(f"Testing: {description}")
    print(f"PR ID: {pr_id}")
    print("-" * 80)
    
    pr_df = df[df['press_release_id'] == pr_id]
    if len(pr_df) == 0:
        print("❌ Press release not found in dataset")
        print()
        continue
    
    pr = pr_df.iloc[0]
    doc = {
        'title': pr.get('title', ''),
        'body': pr.get('full_text', ''),
        'press_release_id': pr_id,
    }
    
    # Test detection
    import os
    os.environ['GUIDANCE_DEBUG'] = '1'  # Enable debug output
    
    events = list(plugin.detect(doc))
    
    os.environ['GUIDANCE_DEBUG'] = '0'
    
    if len(events) == 0:
        print("✅ EXCLUDED - No events detected (as expected)")
    else:
        print(f"❌ FAILED - {len(events)} events still detected:")
        for evt in events[:2]:  # Show first 2
            print(f"   - {evt.get('metric', 'N/A')}: {evt.get('value_low', 'N/A')}")
    
    print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
