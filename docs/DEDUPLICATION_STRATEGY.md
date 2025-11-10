# Deduplication Strategy - Clarification

## Two Levels of Deduplication

### 1. Metric-Level Deduplication (✅ Already Built-In)

**Handled by**: `GuidanceChangePlugin.prior_key()` and `compare()`

**How it works:**
- Tracks state per (company_id, period, metric, basis)
- Compares new values against stored prior state
- Only triggers alert if values actually changed beyond threshold
- Example: If ACME updates revenue guidance twice in the same document or different documents, only the first or changed values trigger alerts

**State Storage:**
- The plugin's `prior_key()` generates: `(company_id, period, metric, basis)`
- Prior state is stored and loaded via the `AlertStore.guidance_prior_state` table
- The `compare()` method calculates deltas and determines if change is significant

**Example:**
```python
# First document: ACME raises revenue guidance 10-15%
prior_key = ("ACME_123", "FY2025", "revenue", "organic")
# No prior state → triggers alert

# Second document (same day): ACME confirms same 10-15% revenue guidance
prior_key = ("ACME_123", "FY2025", "revenue", "organic")
# Prior state exists, values unchanged → NO alert (compare() returns delta=0)

# Third document (week later): ACME raises revenue guidance to 15-20%
prior_key = ("ACME_123", "FY2025", "revenue", "organic")
# Prior state exists, values changed → triggers alert (delta > threshold)
```

### 2. Document-Level Deduplication (What AlertDeduplicator Does)

**Handled by**: `AlertDeduplicator`

**How it works:**
- Prevents the SAME press release from being processed multiple times
- Uses document ID (press_release_id) as dedup key
- Time-windowed to handle pipeline retries

**Use cases:**
- Pipeline retry processes same document twice
- Duplicate documents in RSS/Gmail feed
- Testing scenarios

**Example:**
```python
# Document PR_001 processed first time
event_id = "PR_001"
# No prior processing → alerts generated

# Same document PR_001 processed again (pipeline retry)
event_id = "PR_001"
# Already processed within window → skipped

# Different document PR_002 with same guidance
event_id = "PR_002"
# Different document ID → processed (metric-level dedup handles the rest)
```

## Why Both Are Needed

### Scenario 1: Same Document, Multiple Processes
```
Time 0: RSS feed ingests PR_001 (ACME raises guidance)
Time 1: Orchestrator processes PR_001 → Alert sent ✓
Time 2: Pipeline retry processes PR_001 again
        → AlertDeduplicator: SKIP (same document)
```

### Scenario 2: Different Documents, Same Guidance
```
Day 1: PR_001 - ACME raises revenue guidance to 10-15%
       → GuidanceChangePlugin: No prior → Alert sent ✓
       → Prior state saved: (ACME, FY2025, revenue) = 10-15%

Day 2: PR_002 - ACME reconfirms 10-15% revenue guidance
       → GuidanceChangePlugin: compare() finds delta=0 → NO alert
       (Plugin handles this, AlertDeduplicator not involved)

Day 3: PR_003 - ACME raises guidance to 15-20%
       → GuidanceChangePlugin: compare() finds delta=5pp → Alert sent ✓
       → Prior state updated: (ACME, FY2025, revenue) = 15-20%
```

### Scenario 3: Multiple Metrics in Same Document
```
PR_001: ACME updates revenue AND margin guidance
        → GuidanceChangePlugin.detect() yields 2 candidates:
          1. (ACME, FY2025, revenue, organic)
          2. (ACME, FY2025, operating_margin, reported)
        → Both get separate prior_key comparisons
        → Both generate alerts if changed
        → AlertDeduplicator sees both alerts from PR_001:
          - First alert: event_id=PR_001 → processed
          - Second alert: event_id=PR_001 → SKIP? NO!
          
ISSUE: AlertDeduplicator shouldn't skip the second metric!
```

## Solution: Update Dedup Key

The `AlertDeduplicator` should use `(event_id, metric, period)` to allow multiple metrics from the same document:

```python
def _generate_dedup_key(self, alert: Dict[str, Any]) -> str:
    """Allow multiple alerts from same document if different metrics."""
    event_id = alert.get("event_id", "")
    
    # Include first metric to allow multiple metrics per document
    metrics = alert.get("metrics", [])
    metric_key = metrics[0].get("metric") if metrics else "unknown"
    
    key_parts = [event_id, metric_key]
    content = "|".join(str(p) for p in key_parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
```

## Recommendation: Simplify for Now

Given that:
1. **GuidanceChangePlugin already handles guidance-level deduplication perfectly**
2. **Your pipeline likely has its own document deduplication** (via `dedup_df()` in orchestrator)
3. **Document retries are rare** in your current setup

**You could make AlertDeduplicator optional or even remove it** for the initial implementation:

```python
class AlertOrchestrator:
    def __init__(self, ..., enable_document_dedup: bool = False):
        self.enable_document_dedup = enable_document_dedup
        if enable_document_dedup:
            self.deduplicator = AlertDeduplicator(...)
        else:
            self.deduplicator = None
    
    def process_documents(self, docs):
        alerts = self.detector.detect_alerts(docs)
        
        if self.enable_document_dedup and self.deduplicator:
            # Optional document-level dedup
            alerts = [a for a in alerts if not self.deduplicator.is_duplicate(a)]
        
        # Deliver alerts...
```

This keeps the code simpler and relies on the plugin's authoritative deduplication.

## Summary

✅ **Metric-level deduplication**: Handled by `GuidanceChangePlugin` (trust it!)  
⚠️ **Document-level deduplication**: Handled by `AlertDeduplicator` (optional, may need refinement)  
💡 **Recommendation**: Make document-level dedup optional for initial deployment
