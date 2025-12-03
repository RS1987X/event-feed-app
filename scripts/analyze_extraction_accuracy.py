#!/usr/bin/env python3
"""
Analyze extraction accuracy from detailed feedback data.

This script processes the extraction_feedback metadata to measure:
- Per-field accuracy (metric, direction, period, value)
- Common error patterns
- Missing detection patterns

Usage:
    python scripts/analyze_extraction_accuracy.py
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from load_env import load_env
load_env()

from event_feed_app.alerts.feedback_store import FeedbackStore


def analyze_extraction_accuracy():
    """Analyze extraction accuracy from feedback data."""
    
    print("=" * 80)
    print("EXTRACTION ACCURACY ANALYSIS")
    print("=" * 80)
    print()
    
    feedback_store = FeedbackStore()
    
    # Fetch all guidance_change feedback
    print("📥 Fetching guidance_change feedback from GCS...")
    feedback_records = feedback_store.get_feedback_for_signal("guidance_change")
    
    if not feedback_records:
        print("❌ No feedback found for guidance_change signal")
        return
    
    print(f"✅ Found {len(feedback_records)} feedback records")
    print()
    
    # Extract extraction feedback
    total_items = 0
    correct_items = 0
    total_missing = 0
    
    # Per-field accuracy
    metric_correct = 0
    direction_correct = 0
    period_correct = 0
    value_correct = 0
    field_total = 0
    
    # Error patterns
    metric_errors = Counter()
    direction_errors = Counter()
    period_errors = Counter()
    missing_metrics = Counter()
    
    for record in feedback_records:
        metadata = record.get("metadata", {})
        extraction_fb = metadata.get("extraction_feedback", {})
        
        if not extraction_fb:
            continue
        
        # Analyze detected items
        items = extraction_fb.get("items", [])
        for item in items:
            total_items += 1
            
            if item.get("is_correct", True):
                correct_items += 1
            else:
                # Analyze what was wrong
                detected = item.get("detected", {})
                corrections = item.get("corrections", {})
                
                # Determine item_type to decide which fields to evaluate
                item_type = (detected.get("item_type") or "").lower()
                is_generic = item_type == "guidance_generic"

                # Count applicable fields per item
                # For generic guidance, skip metric/value accuracy (qualitative items)
                applicable_fields = 4 if not is_generic else 2  # direction, period only
                field_total += applicable_fields

                # Direction
                if detected.get("direction") == corrections.get("direction"):
                    direction_correct += 1
                else:
                    direction_errors[(detected.get("direction"), corrections.get("direction"))] += 1

                # Period
                if detected.get("period") == corrections.get("period"):
                    period_correct += 1
                else:
                    period_errors[(detected.get("period"), corrections.get("period"))] += 1

                if not is_generic:
                    # Metric
                    if detected.get("metric") == corrections.get("metric"):
                        metric_correct += 1
                    else:
                        metric_errors[(detected.get("metric"), corrections.get("metric"))] += 1
                    # Value
                    if detected.get("value_str") == corrections.get("value_str"):
                        value_correct += 1
        
        # Analyze missing items
        missing = extraction_fb.get("missing_items", [])
        total_missing += len(missing)
        
        for miss in missing:
            missing_metrics[miss.get("metric", "unknown")] += 1
    
    # Print results
    print("=" * 80)
    print("ITEM-LEVEL ACCURACY")
    print("=" * 80)
    
    if total_items > 0:
        item_accuracy = correct_items / total_items
        print(f"Total Items Evaluated: {total_items}")
        print(f"Correct Items: {correct_items} ({item_accuracy:.1%})")
        print(f"Incorrect Items: {total_items - correct_items} ({(1-item_accuracy):.1%})")
    else:
        print("No item-level feedback available yet")
    
    print()
    print("=" * 80)
    print("FIELD-LEVEL ACCURACY (for incorrect items)")
    print("=" * 80)
    
    if field_total > 0:
        # Compute denominators by counting applicable items
        # We approximate using totals: direction & period always applicable; metric/value only on non-generic items
        # For clearer reporting, we derive counts from errors + corrects
        dir_total = direction_correct + sum(direction_errors.values())
        per_total = period_correct + sum(period_errors.values())
        met_total = metric_correct + sum(metric_errors.values())
        val_total = value_correct  # value field tracked only for correct comparisons

        print(f"\nMetric Accuracy:    {metric_correct}/{max(met_total,1)} ({(metric_correct/max(met_total,1))*100:.1f}%)")
        print(f"Direction Accuracy: {direction_correct}/{max(dir_total,1)} ({(direction_correct/max(dir_total,1))*100:.1f}%)")
        print(f"Period Accuracy:    {period_correct}/{max(per_total,1)} ({(period_correct/max(per_total,1))*100:.1f}%)")
        # Value accuracy denominator approximated by non-generic incorrect items; if zero, guard with max(...,1)
        print(f"Value Accuracy:     {value_correct}/{max(met_total,1)} ({(value_correct/max(met_total,1))*100:.1f}%)")
    else:
        print("No field-level corrections available yet")
    
    print()
    print("=" * 80)
    print("COMMON ERROR PATTERNS")
    print("=" * 80)
    
    if metric_errors:
        print("\n📊 Metric Errors (detected → correct):")
        for (detected, correct), count in metric_errors.most_common(5):
            print(f"  {detected or 'None'} → {correct}: {count} times")
    
    if direction_errors:
        print("\n📈 Direction Errors (detected → correct):")
        for (detected, correct), count in direction_errors.most_common(5):
            print(f"  {detected or 'None'} → {correct}: {count} times")
    
    if period_errors:
        print("\n📅 Period Errors (detected → correct):")
        for (detected, correct), count in period_errors.most_common(5):
            print(f"  '{detected or 'None'}' → '{correct}': {count} times")
    
    print()
    print("=" * 80)
    print("MISSING DETECTIONS")
    print("=" * 80)
    
    print(f"\nTotal Missing Items: {total_missing}")
    
    if missing_metrics:
        print("\nMissed Metrics Breakdown:")
        for metric, count in missing_metrics.most_common():
            print(f"  {metric}: {count} ({count/total_missing*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Generate targeted recommendations
    if period_correct < (field_total // 4) * 0.7:
        print("\n⚠️  PRIORITY: Period detection accuracy is low (<70%)")
        print("   → Review period extraction logic")
        print("   → Add more period pattern tests")
    
    if total_missing > total_items * 0.3:
        print("\n⚠️  PRIORITY: High miss rate (>30%)")
        print(f"   → Missing {total_missing} items from {len(feedback_records)} alerts")
        print(f"   → Most missed: {missing_metrics.most_common(1)[0] if missing_metrics else 'N/A'}")
    
    if direction_errors:
        top_direction_error = direction_errors.most_common(1)[0]
        print(f"\n⚠️  Direction confusion: {top_direction_error[0][0]} → {top_direction_error[0][1]} ({top_direction_error[1]} times)")
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_extraction_accuracy()
