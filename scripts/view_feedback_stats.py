#!/usr/bin/env python3
"""
View alert feedback statistics from GCS.

Usage:
    python scripts/view_feedback_stats.py [--signal-type guidance_change]
    
Shows aggregated metrics about alert accuracy based on user feedback.
"""
import sys
from pathlib import Path
import argparse

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_feed_app.alerts.feedback_store import FeedbackStore


def main():
    parser = argparse.ArgumentParser(description="View feedback statistics")
    parser.add_argument("--signal-type", help="Filter by signal type")
    args = parser.parse_args()
    
    store = FeedbackStore()
    
    print("\n" + "="*60)
    print("ALERT FEEDBACK STATISTICS")
    if args.signal_type:
        print(f"Signal Type: {args.signal_type}")
    print("="*60)
    
    stats = store.get_feedback_stats(signal_type=args.signal_type)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Total Feedback: {stats['total_feedback']}")
    print(f"   ✅ Correct:     {stats['correct']}")
    print(f"   ❌ Incorrect:   {stats['incorrect']}")
    
    if stats['total_feedback'] > 0:
        accuracy_pct = stats['accuracy'] * 100
        print(f"   🎯 Accuracy:    {accuracy_pct:.1f}%")
    
    if stats['by_type']:
        print(f"\n📂 By Feedback Type:")
        for fb_type, count in sorted(stats['by_type'].items()):
            print(f"   {fb_type.ljust(20)}: {count}")
    
    if not args.signal_type and stats['by_signal_type']:
        print(f"\n🔔 By Signal Type:")
        for sig_type, count in sorted(stats['by_signal_type'].items()):
            print(f"   {sig_type.ljust(25)}: {count}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
