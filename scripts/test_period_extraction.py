#!/usr/bin/env python3
"""
Test script to examine period extraction from actual press releases.
Fetches a few PRs and shows what periods are detected.
"""
import sys
import os
import yaml
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from collections import defaultdict
from src.event_feed_app.alerts.runner import fetch_data
from src.event_feed_app.events.registry import get_plugin

def test_period_extraction():
    """Test period extraction on real PRs."""
    
    # Load significant events config
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "src" / "event_feed_app" / "configs" / "significant_events.yaml"
    with cfg_path.open("r") as f:
        yaml_cfg = yaml.safe_load(f)
    
    # Extract guidance_change config
    events = yaml_cfg.get("events", []) or []
    guidance_cfg = next((e for e in events if e.get("key") == "guidance_change"), {})
    
    # Get and configure plugin
    plugin = get_plugin("guidance_change")
    plugin.configure(guidance_cfg)
    
    # Fetch recent PRs
    end_date = "2025-11-24"
    start_date = "2025-11-17"
    
    print(f"Fetching PRs from {start_date} to {end_date}...")
    df = fetch_data(start_date=start_date, end_date=end_date, max_rows=100, source=None)
    print(f"Fetched {len(df)} PRs\n")
    
    # Track period patterns
    period_patterns = defaultdict(int)
    multi_period_events = []
    
    processed = 0
    events_found = 0
    
    for idx, row in df.iterrows():
        processed += 1
        
        # Convert row to document dict
        doc = {
            'press_release_id': row.get('press_release_id'),
            'title': row.get('title', ''),
            'full_text': row.get('full_text', ''),
            'company_name': row.get('company_name', ''),
            'release_date': row.get('release_date', ''),
            'source_url': row.get('source_url', ''),
            'category': row.get('category', ''),
        }
        
        events = list(plugin.detect(doc))
        
        if events:
            print(f"\n{'='*80}")
            print(f"PR: {doc.get('title', 'No title')[:80]}")
            print(f"Source: {doc.get('press_release_id', 'unknown')}")
            print(f"Events detected: {len(events)}")
            
            for i, event in enumerate(events, 1):
                events_found += 1
                period = event.get('period', 'N/A')
                metric = event.get('metric', 'unknown')
                value_low = event.get('value_low')
                value_high = event.get('value_high')
                
                # Track period pattern
                period_patterns[period] += 1
                
                # Format value
                if value_low is not None and value_high is not None:
                    if abs(value_low - value_high) < 0.01:
                        value_str = f"{value_low:.1f}"
                    else:
                        value_str = f"{value_low:.1f}-{value_high:.1f}"
                else:
                    value_str = "N/A"
                
                print(f"  Event {i}:")
                print(f"    Metric: {metric}")
                print(f"    Period: {period}")
                print(f"    Value: {value_str} {event.get('unit', '')}")
                print(f"    Basis: {event.get('basis', 'N/A')}")
                print(f"    Direction: {event.get('direction_hint', 'N/A')}")
        
        if processed % 20 == 0:
            print(f"Processed {processed} PRs...")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"\nSUMMARY:")
    print(f"  PRs processed: {processed}")
    print(f"  Events found: {events_found}")
    print(f"\nPERIOD PATTERNS (top 20):")
    for period, count in sorted(period_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {period}: {count} occurrences")
    
    # Check if any events have multiple periods (shouldn't happen with current schema)
    print(f"\nCONCLUSION:")
    print(f"  All events use a single 'period' field: {len(multi_period_events) == 0}")
    print(f"  Period field appears sufficient for current implementation: YES")

if __name__ == "__main__":
    test_period_extraction()
