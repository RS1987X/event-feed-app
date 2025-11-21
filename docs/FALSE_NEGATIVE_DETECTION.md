# False Negative Detection Strategy

## Problem
Currently, feedback only covers **false positives** (PRs incorrectly flagged as guidance).
We need feedback on **false negatives** (guidance PRs that were missed).

## Current Metrics
- Total feedback: 17 records
- True positives: 11 (64.7%)
- False positives: 6 (35.3%)
- **False negatives: UNKNOWN** ← This is the gap!

---

## Solutions

### 1. Random Sampling (Recommended First Step)
**Script:** `scripts/ml/sample_non_guidance_for_review.py`

**How it works:**
- Samples PRs that were NOT flagged as guidance
- Stratified by category for balanced coverage
- Outputs CSV for manual review

**Usage:**
```bash
python scripts/ml/sample_non_guidance_for_review.py --days 30 --size 100
# Creates: data/labeling/non_guidance_sample_for_review.csv

# Review in spreadsheet:
# - Mark contains_guidance: yes/no
# - Specify guidance_type if yes
# - Add notes
```

**Benefits:**
- Unbiased sample across all categories
- Catches unexpected guidance in non-earnings PRs
- Provides baseline false negative rate

---

### 2. Earnings-Focused Review (High Yield)
**Script:** `scripts/ml/review_earnings_for_guidance.py`

**How it works:**
- Filters to earnings category only
- Excludes PRs that were flagged
- ~60-70% of earnings should have guidance

**Usage:**
```bash
python scripts/ml/review_earnings_for_guidance.py --days 14
# Creates: data/labeling/earnings_guidance_review.csv
```

**Benefits:**
- High probability of finding false negatives
- Earnings is the primary use case
- Quick validation of recall rate

---

### 3. User-Reported False Negatives (Future)

**Add to Telegram Bot:**
Allow users to forward PRs with a command like:
```
/report_missed <press_release_id> <why>
```

**Implementation:**
1. Add Telegram command handler
2. Store in `gs://event-feed-app-data/feedback/false_negatives/`
3. Include in training data

---

### 4. Periodic Automatic Review

**Cron job to detect potential false negatives:**

```python
# Compare guidance alert rate vs. historical baseline
# If alert rate drops suddenly, flag for review

def detect_anomalies():
    # Last 7 days
    recent_rate = guidance_alerts / total_earnings
    
    # Historical baseline
    historical_rate = 0.65  # 65% of earnings have guidance
    
    if recent_rate < historical_rate * 0.8:  # 20% drop
        alert("Potential false negative spike!")
        generate_review_sample()
```

---

## Recommended Workflow

### Week 1-2: Initial Assessment
1. Run `review_earnings_for_guidance.py` for last 30 days
2. Manually review ~50 earnings PRs
3. Calculate false negative rate
4. Document patterns in missed guidance

### Week 3-4: Broader Coverage  
1. Run `sample_non_guidance_for_review.py` for 100 random PRs
2. Review across all categories
3. Find unexpected guidance sources
4. Update taxonomy rules

### Ongoing: Continuous Improvement
1. Weekly earnings review (automated)
2. Monthly random sampling
3. User feedback via Telegram
4. Track metrics over time

---

## Expected Metrics After Implementation

**Current (estimated):**
- Precision: ~65% (11 TP / 17 total flagged)
- Recall: **Unknown**
- F1: **Cannot calculate**

**After false negative detection:**
- Precision: ~65%
- Recall: ~70-80% (target)
- F1: ~67-72%

---

## Files Created

1. `scripts/ml/sample_non_guidance_for_review.py` - Random sampling
2. `scripts/ml/review_earnings_for_guidance.py` - Earnings-focused
3. Output: `data/labeling/non_guidance_sample_for_review.csv`
4. Output: `data/labeling/earnings_guidance_review.csv`

---

## Next Steps

1. **Run earnings review first** (highest yield):
   ```bash
   python scripts/ml/review_earnings_for_guidance.py --days 30
   ```

2. **Review the CSV** and mark which ones contain guidance

3. **Analyze patterns** - Why did the system miss them?
   - Wrong keywords?
   - Unusual phrasing?
   - Missing metric types?
   - Period format issues?

4. **Update the guidance plugin** based on findings

5. **Re-run and measure** improvement in recall
