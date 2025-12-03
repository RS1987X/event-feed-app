#!/usr/bin/env python3
"""
Test directional cue logic for "target" word.
Verifies that "target" before metric = guidance, but "metric ... target" = not guidance.
"""

import yaml
from pathlib import Path
from event_feed_app.events.registry import get_plugin

def test_directional_cue():
    """Test that 'target' requires CUE→METRIC order"""
    
    # Load config and configure plugin
    config_path = Path("src/event_feed_app/configs/significant_events.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    events_cfg = {e["key"]: e for e in cfg.get("events", [])}
    guidance_cfg = events_cfg.get("guidance_change")
    
    # Get the configured plugin instance
    plugin = get_plugin("guidance_change")
    plugin.configure(guidance_cfg)
    
    # Debug: check if directional cues were loaded
    print(f"\nDebug: Directional cues loaded: {plugin.directional_cues}")
    print(f"Debug: Forward cue terms count: {len(plugin.fwd_cue_terms)}")
    print(f"Debug: Metric terms count: {len(plugin.fin_metric_terms)}")
    
    # Test case 1: VALID - "target" BEFORE "revenue" (should detect)
    doc1 = {
        "title": "Company targets revenue of $100M for 2026",
        "body": "The management announced they target revenue of USD 100 million for full year 2026.",
        "release_date": "2025-12-01",
        "source_type": "issuer_pr",
    }
    
    # Test case 2: INVALID - "revenue" BEFORE "target" (should NOT detect)
    doc2 = {
        "title": "Neola Medical granted US patent",
        "body": "Granted US patent for disposable probes, revenue model in target market.",
        "release_date": "2025-12-01",
        "source_type": "issuer_pr",
    }
    
    # Test case 3: VALID - "expects" before "revenue" (non-directional cue, either order OK)
    doc3 = {
        "title": "Company expects strong revenue",
        "body": "The company expects revenue to be EUR 200-250 million in Q4 2025.",
        "release_date": "2025-12-01",
        "source_type": "issuer_pr",
    }
    
    print("\n" + "="*70)
    print("TEST 1: 'target revenue' (CUE → METRIC) - Should DETECT")
    print("="*70)
    print(f"Text: {doc1['title']} {doc1['body'][:80]}")
    results1 = list(plugin.detect(doc1))
    print(f"Detected {len(results1)} event(s)")
    if results1:
        print(f"  ✓ PASS - Detected guidance (correct)")
        for r in results1:
            print(f"    - {r.get('metric_key')}: {r.get('value')} {r.get('unit')}")
    else:
        print(f"  ✗ FAIL - No detection (should have detected)")
    
    print("\n" + "="*70)
    print("TEST 2: 'revenue ... target market' (METRIC → CUE) - Should NOT detect")
    print("="*70)
    results2 = list(plugin.detect(doc2))
    print(f"Detected {len(results2)} event(s)")
    if results2:
        print(f"  ✗ FAIL - Detected guidance (should have been suppressed)")
        for r in results2:
            print(f"    - {r.get('metric_key')}: {r.get('value')} {r.get('unit')}")
            print(f"    - excerpt: {r.get('excerpt', '')[:100]}")
    else:
        print(f"  ✓ PASS - No detection (correct)")
    
    print("\n" + "="*70)
    print("TEST 3: 'expects revenue' (non-directional cue) - Should DETECT")
    print("="*70)
    results3 = list(plugin.detect(doc3))
    print(f"Detected {len(results3)} event(s)")
    if results3:
        print(f"  ✓ PASS - Detected guidance (correct)")
        for r in results3:
            print(f"    - {r.get('metric_key')}: {r.get('value')} {r.get('unit')}")
    else:
        print(f"  ✗ FAIL - No detection (should have detected)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    test1_pass = len(results1) > 0
    test2_pass = len(results2) == 0
    test3_pass = len(results3) > 0
    
    print(f"Test 1 (target → revenue): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (revenue → target): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    print(f"Test 3 (expects revenue):  {'PASS ✓' if test3_pass else 'FAIL ✗'}")
    
    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\nOverall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    
    return all_pass

if __name__ == "__main__":
    success = test_directional_cue()
    exit(0 if success else 1)
