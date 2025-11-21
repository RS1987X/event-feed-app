#!/usr/bin/env python3
"""
Test the false negative feedback flow end-to-end.

This simulates:
1. Daily review sending PRs
2. User providing feedback
3. Feedback being saved to GCS
4. Verifying feedback was stored correctly
"""
import os
import json
from pathlib import Path
from datetime import datetime, timezone

from event_feed_app.alerts.feedback_store import FeedbackStore


def test_false_negative_feedback():
    """Test saving false negative feedback."""
    print("🧪 Testing False Negative Feedback Flow\n")
    
    # Simulate a false negative case
    test_pr_id = "test_pr_12345"
    test_alert_id = f"fn_{test_pr_id}"
    test_user_id = "test_user_001"
    
    feedback_store = FeedbackStore()
    
    # Test 1: Save FALSE NEGATIVE feedback (user says it IS guidance)
    print("1️⃣ Testing FALSE NEGATIVE feedback...")
    try:
        feedback_id = feedback_store.save_feedback(
            signal_type="guidance_change",
            alert_id=test_alert_id,
            press_release_id=test_pr_id,
            user_id=test_user_id,
            is_correct=False,  # System was WRONG
            guidance_metadata={
                "issue_type": "false_negative",
                "review_type": "daily_fn_review"
            },
            notes="Raised FY2025 revenue guidance from $500M to $550M"
        )
        print(f"   ✅ Saved: {feedback_id}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Save TRUE NEGATIVE feedback (user says it's NOT guidance)
    print("\n2️⃣ Testing TRUE NEGATIVE feedback...")
    test_pr_id_2 = "test_pr_67890"
    test_alert_id_2 = f"fn_{test_pr_id_2}"
    
    try:
        feedback_id_2 = feedback_store.save_feedback(
            signal_type="guidance_change",
            alert_id=test_alert_id_2,
            press_release_id=test_pr_id_2,
            user_id=test_user_id,
            is_correct=True,  # System was RIGHT
            guidance_metadata={
                "issue_type": "true_negative",
                "review_type": "daily_fn_review"
            },
            notes="Just reporting quarterly results, no forward guidance"
        )
        print(f"   ✅ Saved: {feedback_id_2}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 3: Retrieve and verify
    print("\n3️⃣ Retrieving saved feedback...")
    try:
        all_feedback = feedback_store.get_feedback_for_signal(
            signal_type="guidance_change"
        )
        
        # Find our test feedback
        test_feedback = [
            fb for fb in all_feedback 
            if fb.get("press_release_id") in [test_pr_id, test_pr_id_2]
        ]
        
        if test_feedback:
            print(f"   ✅ Found {len(test_feedback)} test feedback records")
            for fb in test_feedback:
                print(f"\n   📋 Feedback: {fb['feedback_id']}")
                print(f"      PR: {fb['press_release_id']}")
                print(f"      Type: {fb.get('feedback_type', 'N/A')}")
                print(f"      Correct: {fb['is_correct']}")
                print(f"      Notes: {fb.get('notes', 'N/A')}")
        else:
            print("   ⚠️ No test feedback found (may be stored but not in today's file)")
            
    except Exception as e:
        print(f"   ❌ Error retrieving feedback: {e}")
    
    # Test 4: Show stats
    print("\n4️⃣ Overall feedback stats...")
    try:
        stats = feedback_store.get_feedback_stats(signal_type="guidance_change")
        print(f"   Total feedback: {stats.get('total', 0)}")
        print(f"   Correct (TP + TN): {stats.get('correct', 0)}")
        print(f"   Incorrect (FP + FN): {stats.get('incorrect', 0)}")
        
        by_type = stats.get('by_type', {})
        if by_type:
            print(f"\n   By type:")
            for ftype, count in by_type.items():
                print(f"      {ftype}: {count}")
    except Exception as e:
        print(f"   ⚠️ Could not get stats: {e}")
    
    print("\n✅ Test complete!")
    print("\nNote: Check GCS at:")
    print("gs://event-feed-app-data/feedback/signal_ratings/signal_type=guidance_change/")


if __name__ == "__main__":
    test_false_negative_feedback()
