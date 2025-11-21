#!/usr/bin/env python3
"""
Test the GCS-based alert feedback system.

Tests:
1. Save feedback to GCS
2. Retrieve feedback
3. Get statistics
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from event_feed_app.alerts.feedback_store import FeedbackStore


def main():
    print("\n🧪 Testing GCS-based Alert Feedback System")
    print("=" * 60)
    
    store = FeedbackStore()
    
    # Test 1: Save correct feedback with guidance metadata
    print("\n1️⃣ Saving 'correct' feedback with metadata...")
    
    try:
        feedback_id = store.save_feedback(
            signal_type="guidance_change",
            alert_id="alert_test123",
            press_release_id="1980c93f9f819a6d",
            user_id="user_test",
            is_correct=True,
            guidance_metadata={
                "detected_metrics": ["revenue", "eps"],
                "direction": "up",
                "period": "Q4_2025",
                "guidance_type": "raise"
            },
            notes="Test feedback - alert was accurate"
        )
        print(f"   ✅ Save successful: {feedback_id}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
    
    # Test 2: Save incorrect feedback with issue details
    print("\n2️⃣ Saving 'incorrect' feedback with correction details...")
    
    try:
        feedback_id = store.save_feedback(
            signal_type="guidance_change",
            alert_id="alert_test456",
            press_release_id="abc123def456",
            user_id="user_test",
            is_correct=False,
            guidance_metadata={
                "issue_type": "wrong_metric",
                "correct_metrics": ["margin"],
                "correct_direction": "down",
                "correct_period": "FY_2025"
            },
            notes="Test feedback - detected revenue instead of margin"
        )
        print(f"   ✅ Save successful: {feedback_id}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
    
    # Test 3: Retrieve feedback for specific signal
    print(f"\n3️⃣ Retrieving feedback for signal: alert_test123")
    try:
        feedback = store.get_feedback_for_signal(
            signal_type="guidance_change",
            alert_id="alert_test123"
        )
        print(f"   Found {len(feedback)} feedback entries:")
        for fb in feedback:
            print(f"   - {fb['feedback_type']}: {fb['is_correct']}")
            if fb.get('metadata'):
                print(f"     Metadata: {fb['metadata']}")
    except Exception as e:
        print(f"   ❌ Retrieval failed: {e}")
    
    # Test 4: Get aggregated statistics
    print("\n4️⃣ Getting feedback statistics...")
    try:
        stats = store.get_feedback_stats(signal_type="guidance_change")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Correct: {stats['correct']}")
        print(f"   Incorrect: {stats['incorrect']}")
        print(f"   Accuracy: {stats['accuracy'] * 100:.1f}%")
        print(f"   By type: {stats['by_type']}")
    except Exception as e:
        print(f"   ❌ Stats failed: {e}")
    
    print("\n✅ All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
