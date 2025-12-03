#!/usr/bin/env python3
"""
Refresh the labeled guidance examples from GCS feedback.

This pulls the latest user feedback from GCS and exports it to CSV files
for use in the test suite.

Usage:
    python scripts/refresh_labeled_data.py
    
This will update:
    - guidance_true_positives.csv
    - guidance_false_positives.csv  
    - guidance_false_negatives.csv
"""
import sys
import subprocess

print("=== Refreshing Labeled Guidance Data ===\n")
print("Running analyze_guidance_feedback.py to pull latest from GCS...\n")

# Run the existing analysis script which exports CSVs
result = subprocess.run(
    ["python", "analyze_guidance_feedback.py"],
    capture_output=False,
    text=True
)

if result.returncode == 0:
    print("\n✅ Successfully refreshed labeled data from GCS feedback")
    print("\nUpdated files:")
    print("  - guidance_true_positives.csv")
    print("  - guidance_false_positives.csv")
    print("  - guidance_false_negatives.csv")
    print("\nNow run: pytest tests/guidance_change_tests/test_labeled_cases.py -v -s")
else:
    print("\n❌ Failed to refresh data")
    sys.exit(1)
