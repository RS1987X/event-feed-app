# Extraction-Level Feedback System

## Overview

The feedback system now supports **two-level validation**:

1. **Classification Level**: Is this earnings guidance? (YES/NO)
2. **Extraction Level**: Are the parsed parameters correct? (per-item validation)

## How It Works

### For Users (in Viewer)

When reviewing an alert, you can now:

1. **Validate each detected item**:
   - ✓ Mark as correct if all fields are right
   - ✗ Mark as incorrect and specify corrections:
     - Metric (revenue, EPS, EBIT, etc.)
     - Direction (raise, lower, maintain, etc.)
     - Period (Q4_2025, FY_2025, etc.)
     - Value ($100M → $120M)

2. **Add missing items**:
   - Click "+ Add Missing Item"
   - Fill in all fields for what the system missed
   - Optional: Specify where in text it appears

3. **Submit feedback**:
   - Even if classification is correct, feedback captures extraction quality
   - Enables targeted improvements to extraction logic

### Feedback Data Structure

```json
{
  "feedback_id": "fb_xxx",
  "signal_type": "guidance_change",
  "alert_id": "abc123",
  "is_correct": true,
  "metadata": {
    "extraction_feedback": {
      "items": [
        {
          "item_index": 0,
          "is_correct": false,
          "detected": {
            "metric": "revenue",
            "direction": "raise",
            "period": "FY_2025",
            "value_str": "$500M"
          },
          "corrections": {
            "metric": "revenue",
            "direction": "raise", 
            "period": "Q4_2025",
            "value_str": "$500M → $520M"
          }
        }
      ],
      "missing_items": [
        {
          "metric": "eps",
          "direction": "raise",
          "period": "FY_2025",
          "value_str": "$2.50 → $2.75",
          "location_hint": "paragraph 3"
        }
      ],
      "overall_correct": false
    }
  }
}
```

## Analysis

### View Extraction Accuracy

```bash
python scripts/analyze_extraction_accuracy.py
```

This shows:
- **Item-level accuracy**: What % of detected items are fully correct?
- **Field-level accuracy**: For incorrect items, which fields are wrong?
  - Metric accuracy
  - Direction accuracy
  - Period accuracy (typically hardest!)
  - Value accuracy
- **Common error patterns**: Most frequent mistakes
- **Missing detection patterns**: What metrics are most often missed?

### Example Output

```
================================================================================
ITEM-LEVEL ACCURACY
================================================================================
Total Items Evaluated: 45
Correct Items: 28 (62.2%)
Incorrect Items: 17 (37.8%)

================================================================================
FIELD-LEVEL ACCURACY (for incorrect items)
================================================================================

Metric Accuracy:    15/17 (88.2%)
Direction Accuracy: 14/17 (82.4%)
Period Accuracy:    8/17 (47.1%)   ⚠️ LOW!
Value Accuracy:     11/17 (64.7%)

================================================================================
COMMON ERROR PATTERNS
================================================================================

📅 Period Errors (detected → correct):
  'FY_2025' → 'Q4_2025': 4 times
  'Q4_2024' → 'FY_2025': 3 times
  'None' → 'Q1_2026': 2 times

================================================================================
MISSING DETECTIONS
================================================================================

Total Missing Items: 12

Missed Metrics Breakdown:
  eps: 6 (50.0%)
  margin: 3 (25.0%)
  ebitda: 2 (16.7%)
  capex: 1 (8.3%)

================================================================================
RECOMMENDATIONS
================================================================================

⚠️  PRIORITY: Period detection accuracy is low (<70%)
   → Review period extraction logic
   → Add more period pattern tests

⚠️  PRIORITY: High miss rate (>30%)
   → Missing 12 items from 28 alerts
   → Most missed: eps
```

## Integration with Tests

Create targeted tests for extraction accuracy:

```python
# tests/guidance_change_tests/test_extraction_accuracy.py

def test_period_extraction_accuracy(plugin, extraction_feedback_df):
    """Test that we extract the correct time period."""
    correct = 0
    total = 0
    
    for _, row in extraction_feedback_df.iterrows():
        if not row['is_correct']:
            detected_period = row['detected_period']
            correct_period = row['corrected_period']
            total += 1
            if detected_period == correct_period:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    assert accuracy >= 0.80, f"Period accuracy too low: {accuracy:.1%}"
```

## Workflow

1. **Generate alerts** with current config
2. **Review in viewer** with extraction-level feedback
3. **Run analysis** to identify weak spots
4. **Make targeted improvements** to extraction logic
5. **Test** with labeled feedback
6. **Repeat** until satisfactory accuracy

## Metrics Goals

- **Item-level accuracy**: 80%+ (all fields correct)
- **Metric accuracy**: 95%+ (usually easy to detect)
- **Direction accuracy**: 90%+ (raise vs lower)
- **Period accuracy**: 80%+ (hardest - lots of variation)
- **Value accuracy**: 85%+ (range vs single value)
- **Miss rate**: <20% (catch most guidance items)

## Benefits

1. **Precision**: Know exactly which extraction component needs work
2. **Efficiency**: Fix specific issues instead of guessing
3. **Measurable**: Track improvement over time
4. **Testable**: Build test suite from real corrections
5. **User-driven**: Improvements based on actual errors seen
