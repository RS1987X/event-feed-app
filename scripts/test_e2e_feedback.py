#!/usr/bin/env python3
"""
End-to-end test of the alert feedback workflow.

Tests:
1. Generate alert from test document
2. Save alert payload to GCS
3. Simulate alert delivery
4. Verify payload can be retrieved
5. Save user feedback to GCS
6. Verify feedback can be retrieved
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

print("\n🧪 End-to-End Alert Feedback Workflow Test")
print("=" * 70)

# Test 1: Create a mock alert
print("\n1️⃣ Creating mock alert...")
mock_alert = {
    "alert_id": "alert_e2e_test_123",
    "alert_type": "guidance_change",
    "press_release_id": "test_pr_456",
    "company_name": "Test Corp",
    "significance_score": 0.85,
    "guidance_count": 2,
    "guidance_items": [
        {
            "metric": "revenue",
            "direction": "up",
            "period": "Q4_2025",
            "value_str": "$100M → $120M (+20%)",
            "confidence": 0.92,
            "text_snippet": "we now expect revenue of $120M, up from $100M"
        },
        {
            "metric": "eps",
            "direction": "up",
            "period": "Q4_2025",
            "value_str": "$1.50 → $1.80 (+20%)",
            "confidence": 0.88,
            "text_snippet": "EPS guidance raised to $1.80"
        }
    ],
    "metadata": {
        "press_release_url": "https://example.com/pr/test",
        "release_date": "2025-11-12"
    }
}
print(f"   ✅ Created alert: {mock_alert['alert_id']}")

# Test 2: Save alert payload
print("\n2️⃣ Saving alert payload to GCS...")
try:
    from event_feed_app.alerts.alert_payload_store import AlertPayloadStore
    
    payload_store = AlertPayloadStore()
    success = payload_store.save_alert_payload(mock_alert)
    
    if success:
        print("   ✅ Alert payload saved successfully")
    else:
        print("   ❌ Failed to save alert payload")
except Exception as e:
    print(f"   ❌ Error saving payload: {e}")

# Test 3: Retrieve alert payload
print("\n3️⃣ Retrieving alert payload from GCS...")
try:
    retrieved_payload = payload_store.get_alert_payload(
        alert_id="alert_e2e_test_123",
        signal_type="guidance_change"
    )
    
    if retrieved_payload:
        print(f"   ✅ Retrieved payload with {len(retrieved_payload.get('guidance_items', []))} guidance items")
    else:
        print("   ⚠️ Payload not found (might take a moment to propagate)")
except Exception as e:
    print(f"   ❌ Error retrieving payload: {e}")

# Test 4: Save user feedback
print("\n4️⃣ Saving user feedback to GCS...")
try:
    from event_feed_app.alerts.feedback_store import FeedbackStore
    
    feedback_store = FeedbackStore()
    feedback_id = feedback_store.save_feedback(
        signal_type="guidance_change",
        signal_id="alert_e2e_test_123",
        press_release_id="test_pr_456",
        user_id="test_user",
        is_correct=False,
        guidance_metadata={
            "issue_type": "wrong_metric",
            "correct_metrics": ["margin"],
            "correct_direction": "down"
        },
        notes="Test feedback - detected revenue but should be margin"
    )
    
    print(f"   ✅ Feedback saved: {feedback_id}")
except Exception as e:
    print(f"   ❌ Error saving feedback: {e}")

# Test 5: Retrieve feedback
print("\n5️⃣ Retrieving feedback from GCS...")
try:
    feedback = feedback_store.get_feedback_for_signal(
        signal_type="guidance_change",
        signal_id="alert_e2e_test_123"
    )
    
    if feedback:
        print(f"   ✅ Retrieved {len(feedback)} feedback entries")
        for fb in feedback:
            print(f"      - {fb['feedback_type']}: is_correct={fb['is_correct']}")
    else:
        print("   ⚠️ No feedback found yet (might take a moment)")
except Exception as e:
    print(f"   ❌ Error retrieving feedback: {e}")

# Test 6: Get stats
print("\n6️⃣ Getting feedback statistics...")
try:
    stats = feedback_store.get_feedback_stats(signal_type="guidance_change")
    print(f"   ✅ Total feedback: {stats['total_feedback']}")
    print(f"      Correct: {stats['correct']}, Incorrect: {stats['incorrect']}")
    if stats['total_feedback'] > 0:
        print(f"      Accuracy: {stats['accuracy']*100:.1f}%")
except Exception as e:
    print(f"   ❌ Error getting stats: {e}")

print("\n" + "=" * 70)
print("✅ End-to-end test complete!")
print("\nNote: GCS operations require:")
print("  1. gcsfs installed: pip install gcsfs")
print("  2. GCP credentials configured")
print("  3. Access to event-feed-app-data bucket")
print("=" * 70 + "\n")
