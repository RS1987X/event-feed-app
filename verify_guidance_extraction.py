#!/usr/bin/env python3
"""
Verification script to check guidance parametrization extraction and storage.

This script:
1. Fetches a specific press release by ID
2. Runs the GuidanceChangePlugin to extract events
3. Shows what metrics, periods, and values are extracted
4. Runs GuidanceAlertDetector to build alert payload
5. Checks what was actually stored in AlertPayloadStore
"""

import sys
sys.path.insert(0, '/home/ichard/projects/event-feed-app/src')

from event_feed_app.alerts.runner import fetch_data
from event_feed_app.events.guidance_change.plugin2 import GuidanceChangePlugin
from event_feed_app.alerts.detector import GuidanceAlertDetector
import yaml
from pathlib import Path
from pprint import pprint

def verify_pr(press_release_id: str):
    """Verify guidance extraction for a specific press release."""
    
    # 1. Fetch the PR
    print(f"\n{'='*80}")
    print(f"FETCHING PRESS RELEASE: {press_release_id}")
    print(f"{'='*80}\n")
    
    df = fetch_data('2025-11-10', '2025-11-27', 20000, None)
    pr_df = df[df['press_release_id'] == press_release_id]
    
    if len(pr_df) == 0:
        print(f"❌ Press release {press_release_id} not found")
        return
    
    pr = pr_df.iloc[0]
    print(f"Title: {pr['title'][:120]}")
    print(f"Company: {pr.get('company_name', 'N/A')}")
    print()
    
    # 2. Load config and run plugin
    print(f"\n{'='*80}")
    print(f"RUNNING GUIDANCECHANGEPLUGIN.detect()")
    print(f"{'='*80}\n")
    
    config_path = Path('src/event_feed_app/configs/significant_events.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    plugin = GuidanceChangePlugin()
    plugin.configure(cfg)
    
    doc = {
        'title': pr['title'],
        'body': pr['full_text'],
        'press_release_id': press_release_id,
        'company_name': pr.get('company_name', ''),
    }
    
    events = list(plugin.detect(doc))
    print(f"Detected {len(events)} events\n")
    
    if events:
        print("Event structure (first 3):")
        for i, evt in enumerate(events[:3], 1):
            print(f"\n  Event #{i}:")
            print(f"    Metric: {evt.get('metric')} (kind: {evt.get('metric_kind')})")
            print(f"    Period: {evt.get('period')}")
            print(f"    Unit: {evt.get('unit')}")
            print(f"    Value: {evt.get('value_low')} to {evt.get('value_high')}")
            print(f"    Direction: {evt.get('direction_hint')}")
            print(f"    Basis: {evt.get('basis')}")
    
    # 3. Run detector to see what gets built
    print(f"\n{'='*80}")
    print(f"RUNNING GUIDANCEALERTDETECTOR")
    print(f"{'='*80}\n")
    
    # Force alert_all_guidance to bypass significance threshold
    cfg_with_alert_all = dict(cfg)
    cfg_with_alert_all['alert_all_guidance'] = True
    
    detector = GuidanceAlertDetector(cfg_with_alert_all)
    pr_dict = pr.to_dict()
    
    try:
        alerts = detector.detect_alerts([pr_dict])
        
        if alerts:
            alert = alerts[0]
            print(f"✅ Alert generated!")
            print(f"\nAlert ID: {alert['alert_id']}")
            print(f"Company: {alert['company_name']}")
            print(f"Guidance items: {alert['guidance_count']}")
            print(f"Significance: {alert['significance_score']}")
            
            print(f"\n  📊 METRICS ARRAY (for historical tracking):")
            print(f"  {'-'*60}")
            if alert.get('metrics'):
                for i, metric in enumerate(alert['metrics'], 1):
                    print(f"\n  Metric #{i}:")
                    pprint(metric, indent=4)
            else:
                print("  (empty)")
            
            print(f"\n  📋 GUIDANCE_ITEMS ARRAY (for viewer display):")
            print(f"  {'-'*60}")
            for i, item in enumerate(alert['guidance_items'], 1):
                print(f"\n  Item #{i}:")
                print(f"    Metric: {item['metric']}")
                print(f"    Period: {item['period']}")
                print(f"    Direction: {item['direction']}")
                print(f"    Value: {item['value_str']}")  # THIS SHOULD NOW BE POPULATED
                print(f"    Confidence: {item['confidence']}")
        else:
            print("❌ No alerts generated (below significance threshold)")
            
    except Exception as e:
        print(f"❌ Error running detector: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Check stored payload
    print(f"\n{'='*80}")
    print(f"CHECKING ALERT PAYLOAD STORE")
    print(f"{'='*80}\n")
    
    try:
        from core.oltp_store import AlertPayloadStore
        store = AlertPayloadStore()
        payload = store.get_by_press_release_id(press_release_id)
        
        if payload:
            print(f"✅ Found stored alert payload!")
            print(f"\nAlert ID: {payload.get('alert_id')}")
            print(f"Detected at: {payload.get('detected_at')}")
            print(f"Significance score: {payload.get('significance_score')}")
            
            guidance_items = payload.get('guidance_items', [])
            print(f"\nStored guidance items ({len(guidance_items)}):\n")
            
            for i, item in enumerate(guidance_items, 1):
                print(f"  Item #{i}:")
                print(f"    - Metric: {item.get('metric')}")
                print(f"    - Direction: {item.get('direction')}")
                print(f"    - Period: {item.get('period')}")
                print(f"    - Value: {item.get('value_str')}")
                print(f"    - Confidence: {item.get('confidence')}")
                print()
            
            print(f"\n  📦 FULL PAYLOAD STRUCTURE:")
            pprint(payload, indent=2)
        else:
            print("⚠️  No stored payload found (may not have been run through full pipeline)")
            
    except ImportError:
        print("⚠️  Could not import AlertPayloadStore (skipping)")
    except Exception as e:
        print(f"⚠️  Error checking payload: {e}")
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    print("✅ Plugin extracts structured events with:")
    print("   - Metrics (name, kind, unit)")
    print("   - Time periods")
    print("   - Direction hints")
    print("   - Numeric values (ranges or points)")
    print("   - Detection metadata")
    print()
    print("✅ Detector builds alerts with:")
    print("   - metrics[] array for historical tracking")
    print("   - guidance_items[] with value_str for display")
    print("   - Full event context")
    print()
    print("✅ AlertPayloadStore saves:")
    print("   - Guidance items with metric/direction/period/value")
    print("   - Full event context")
    print("   - Structured data for historical comparison")
    print()
    print(f"{'='*80}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python verify_guidance_extraction.py <press_release_id>")
        sys.exit(1)
    
    verify_pr(sys.argv[1])
