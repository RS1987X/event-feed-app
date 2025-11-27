#!/usr/bin/env python3
"""
Analyze feedback for guidance_change classifier to identify improvement opportunities.

Fetches all guidance_change feedback from GCS and provides:
- Overall accuracy metrics
- False positive analysis (what was incorrectly flagged)
- False negative analysis (what was missed)
- Common patterns in errors
- Suggestions for improvement
"""
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
from event_feed_app.alerts.feedback_store import FeedbackStore
from event_feed_app.alerts.alert_payload_store import AlertPayloadStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_guidance_feedback():
    """Fetch and analyze all guidance_change feedback."""
    
    print("=" * 80)
    print("GUIDANCE CHANGE CLASSIFIER - FEEDBACK ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize stores
    feedback_store = FeedbackStore()
    payload_store = AlertPayloadStore()
    
    # Fetch all guidance feedback
    print("📥 Fetching guidance_change feedback from GCS...")
    feedback_records = feedback_store.get_feedback_for_signal("guidance_change")
    
    if not feedback_records:
        print("❌ No feedback found for guidance_change signal")
        return
    
    print(f"✅ Found {len(feedback_records)} feedback records")
    print()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(feedback_records)
    
    # === OVERALL METRICS ===
    print("=" * 80)
    print("📊 OVERALL METRICS")
    print("=" * 80)
    
    total = len(df)
    correct = df['is_correct'].sum()
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0
    
    print(f"Total Feedback: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print()
    
    # Feedback type breakdown
    print("Breakdown by Feedback Type:")
    feedback_counts = df['feedback_type'].value_counts()
    for fb_type, count in feedback_counts.items():
        print(f"  {fb_type}: {count} ({count/total*100:.1f}%)")
    print()
    
    # === FALSE POSITIVES ANALYSIS ===
    print("=" * 80)
    print("❌ FALSE POSITIVES (Incorrectly Flagged)")
    print("=" * 80)
    
    false_positives = df[df['feedback_type'] == 'false_positive']
    print(f"Count: {len(false_positives)}")
    
    if len(false_positives) > 0:
        print()
        print("Issue Type Breakdown:")
        
        # Extract issue types from metadata
        issue_types = []
        for _, row in false_positives.iterrows():
            metadata = row.get('metadata', {})
            issue = metadata.get('issue_type', 'unspecified')
            issue_types.append(issue)
        
        issue_counter = Counter(issue_types)
        for issue, count in issue_counter.most_common():
            print(f"  {issue}: {count}")
        
        print()
        print("Sample False Positives:")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(false_positives.head(10).iterrows(), 1):
            alert_id = row['alert_id']
            metadata = row.get('metadata', {})
            notes = row.get('notes', '')
            
            # Try to load the full alert payload
            try:
                payload = payload_store.load_alert_payload(alert_id)
                if payload:
                    title = payload.get('title', 'N/A')
                    company = payload.get('company', 'N/A')
                    detected_metrics = metadata.get('detected_metrics', [])
                    
                    print(f"\n{idx}. Alert ID: {alert_id}")
                    print(f"   Company: {company}")
                    print(f"   Title: {title[:100]}...")
                    print(f"   Detected Metrics: {detected_metrics}")
                    print(f"   Issue: {metadata.get('issue_type', 'unspecified')}")
                    if notes:
                        print(f"   Notes: {notes}")
                else:
                    print(f"\n{idx}. Alert ID: {alert_id}")
                    print(f"   (Could not load payload)")
                    print(f"   Issue: {metadata.get('issue_type', 'unspecified')}")
                    if notes:
                        print(f"   Notes: {notes}")
            except Exception as e:
                print(f"\n{idx}. Alert ID: {alert_id}")
                print(f"   (Error loading payload: {e})")
    
    print()
    
    # === FALSE NEGATIVES ANALYSIS ===
    print("=" * 80)
    print("❌ FALSE NEGATIVES (Should Have Been Flagged)")
    print("=" * 80)
    
    # False negatives come from the daily review process (alert_id starts with "fn_")
    false_negatives = df[df['feedback_type'] == 'false_negative']
    print(f"Count: {len(false_negatives)}")
    
    if len(false_negatives) > 0:
        print()
        print("Sample False Negatives:")
        print("-" * 80)
        
        for idx, (_, row) in enumerate(false_negatives.head(10).iterrows(), 1):
            alert_id = row['alert_id']
            metadata = row.get('metadata', {})
            notes = row.get('notes', '')
            
            # Try to load the full alert payload
            try:
                payload = payload_store.load_alert_payload(alert_id)
                if payload:
                    title = payload.get('title', 'N/A')
                    company = payload.get('company', 'N/A')
                    
                    print(f"\n{idx}. Alert ID: {alert_id}")
                    print(f"   Company: {company}")
                    print(f"   Title: {title[:100]}...")
                    print(f"   Why Missed: {metadata.get('reason', 'Not specified')}")
                    if notes:
                        print(f"   Notes: {notes}")
                else:
                    print(f"\n{idx}. Alert ID: {alert_id}")
                    print(f"   (Could not load payload)")
                    if notes:
                        print(f"   Notes: {notes}")
            except Exception as e:
                print(f"\n{idx}. Alert ID: {alert_id}")
                print(f"   (Error loading payload: {e})")
    
    print()
    
    # === TRUE POSITIVES ANALYSIS ===
    print("=" * 80)
    print("✅ TRUE POSITIVES (Correctly Flagged)")
    print("=" * 80)
    
    true_positives = df[df['feedback_type'] == 'true_positive']
    print(f"Count: {len(true_positives)}")
    
    if len(true_positives) > 0:
        print()
        print("Metric Type Distribution:")
        
        # Extract metrics from metadata
        all_metrics = []
        for _, row in true_positives.iterrows():
            metadata = row.get('metadata', {})
            detected = metadata.get('detected_metrics', [])
            all_metrics.extend(detected)
        
        metric_counter = Counter(all_metrics)
        for metric, count in metric_counter.most_common(10):
            print(f"  {metric}: {count}")
    
    print()
    
    # === IMPROVEMENT SUGGESTIONS ===
    print("=" * 80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("=" * 80)
    print()
    
    if len(false_positives) > 0:
        print("False Positive Reduction:")
        issue_counter = Counter(issue_types)
        
        for issue, count in issue_counter.most_common(3):
            print(f"  • Address '{issue}' issues ({count} occurrences)")
            
            if issue == 'not_guidance':
                print(f"    - Add patterns to exclude non-guidance metrics")
                print(f"    - Improve context detection (e.g., historical vs. forward-looking)")
            elif issue == 'wrong_metric':
                print(f"    - Refine metric extraction patterns")
                print(f"    - Add validation for metric types")
            elif issue == 'reaffirmation':
                print(f"    - Add reaffirmation detection (no change language)")
                print(f"    - Check for 'maintains', 'reiterates', 'confirms' patterns")
    
    if len(false_negatives) > 0:
        print()
        print("False Negative Reduction:")
        print(f"  • Review {len(false_negatives)} missed cases")
        print(f"  • Extract common patterns from missed guidance")
        print(f"  • Consider expanding detection rules")
    
    print()
    
    # === EXPORT DETAILED DATA ===
    print("=" * 80)
    print("💾 EXPORTING DETAILED DATA")
    print("=" * 80)
    
    # Export to CSV for further analysis
    output_file = "guidance_feedback_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Exported full feedback data to: {output_file}")
    
    # Create separate files for each category
    if len(false_positives) > 0:
        false_positives.to_csv("guidance_false_positives.csv", index=False)
        print(f"✅ Exported false positives to: guidance_false_positives.csv")
    
    if len(false_negatives) > 0:
        false_negatives.to_csv("guidance_false_negatives.csv", index=False)
        print(f"✅ Exported false negatives to: guidance_false_negatives.csv")
    
    if len(true_positives) > 0:
        true_positives.to_csv("guidance_true_positives.csv", index=False)
        print(f"✅ Exported true positives to: guidance_true_positives.csv")
    
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_guidance_feedback()
