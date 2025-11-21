# Alert System Integration Guide

## How It Works: End-to-End Data Flow

### 1. Data Ingestion (Existing)

```
RSS/Gmail → Bronze Layer (GCS) → Silver Layer (GCS parquet)
```

Your existing jobs:
- `gcr_job_rss.py` - Fetches from RSS feeds
- `gcr_job_main.py` - Fetches from Gmail
- Both write to GCS silver bucket: `event-feed-app-data/silver_normalized/table=press_releases/`

### 2. Classification Pipeline (Existing)

```python
# src/event_feed_app/pipeline/orchestrator.py
from event_feed_app.taxonomy.rules_engine import classify_batch

# Documents are classified by rules (earnings, M&A, personnel, etc.)
classified_df = classify_batch(docs, keywords=keywords)
```

### 3. Event Detection (Existing - GuidanceChangePlugin)

```python
# src/event_feed_app/events/guidance_change/plugin.py
class GuidanceChangePlugin(EventPlugin):
    
    def detect(self, doc):
        """Extract guidance metrics from document"""
        # Returns candidates like:
        # {company, period, metric, value, basis, ...}
    
    def prior_key(self, cand):
        """Create unique key for deduplication"""
        return (company, period, metric, basis)
    
    def compare(self, cand, prior):
        """Compare new guidance vs prior"""
        # Returns: {delta, pct_change, direction, ...}
    
    def decide_and_score(self, cand, comparison):
        """Determine if guidance change is significant"""
        # Returns: score 0.0-1.0
```

**This is the core!** The plugin already:
- ✅ Extracts guidance metrics (revenue, EBITDA, etc.)
- ✅ Tracks prior state per (company, period, metric, basis)
- ✅ Calculates deltas and significance scores
- ✅ Handles deduplication via prior_key/compare

### 4. Alert System (NEW - Your Addition)

```python
# src/event_feed_app/alerts/detector.py
class GuidanceAlertDetector:
    def __init__(self):
        # Uses the EXISTING GuidanceChangePlugin
        self.guidance_plugin = get_plugin("guidance_change")
    
    def detect_alerts(self, docs):
        """
        For each document:
        1. Let plugin.gate() filter out non-guidance docs
        2. Let plugin.detect() extract candidates
        3. Let plugin.prior_key() + compare() check for changes
        4. Let plugin.decide_and_score() determine significance
        5. Build alert dict if score >= threshold
        """
        alerts = []
        for doc in docs:
            if self.guidance_plugin.gate(doc):  # Plugin's gating logic
                candidates = self.guidance_plugin.detect(doc)
                for cand in candidates:
                    # Get prior state
                    key = self.guidance_plugin.prior_key(cand)
                    prior = self.store.get_prior_state(*key)
                    
                    # Compare
                    comparison = self.guidance_plugin.compare(cand, prior)
                    
                    # Score
                    result = self.guidance_plugin.decide_and_score(cand, comparison)
                    
                    if result['score'] >= self.min_significance:
                        alert = self._build_alert(doc, cand, comparison, result)
                        alerts.append(alert)
                        
                        # Update prior state
                        self.store.save_prior_state(*key, cand)
        
        return alerts
```

### 5. Alert Delivery (NEW)

```python
# src/event_feed_app/alerts/delivery.py
class AlertDelivery:
    def deliver(self, alert):
        """Send alert via configured channels"""
        users = self.store.get_users_for_alert(alert)
        
        for user in users:
            for channel in user['channels']:
                if channel == 'telegram':
                    self._deliver_telegram(alert, user)
                elif channel == 'email':
                    self._deliver_email(alert, user)
                elif channel == 'webhook':
                    self._deliver_webhook(alert, user)
```

## Integration Points

### Option A: Batch Processing (Recommended for Testing)

Process historical data from GCS:

```bash
# Test with last 7 days of data
python3 scripts/test_alerts_with_gcp_data.py --days 7 --dry-run

# Real run (will deliver alerts!)
python3 scripts/test_alerts_with_gcp_data.py --days 3
```

The script:
1. Fetches press releases from GCS silver bucket
2. Applies housekeeping/newsletter filters
3. Classifies with rules_engine
4. Filters for earnings documents
5. Processes through GuidanceChangePlugin → alerts
6. Delivers via Telegram/email

### Option B: Real-Time Integration (Production)

Add to your existing pipeline after classification:

```python
# In src/event_feed_app/pipeline/orchestrator.py
# Add after line ~200 (after classification)

from event_feed_app.alerts.orchestration import AlertOrchestrator
import os

def run_with_alerts(cfg: Settings):
    """Enhanced orchestrator with alert support."""
    
    # ... existing code ...
    # (load data, apply gating, classify, etc.)
    
    # After classification:
    classified_df = classify_batch(docs, keywords=keywords)
    
    # NEW: Process earnings docs for alerts
    if os.getenv("ENABLE_ALERTS", "false").lower() == "true":
        logger.info("Processing guidance alerts...")
        
        # Filter for earnings
        earnings_mask = classified_df["rule_pred_cid"] == "earnings_report"
        earnings_docs = classified_df[earnings_mask].to_dict("records")
        
        if earnings_docs:
            # Initialize alert system
            alert_config = {
                "min_significance": float(os.getenv("ALERT_MIN_SIGNIFICANCE", "0.7")),
                "smtp": {
                    "host": os.getenv("SMTP_HOST"),
                    "port": int(os.getenv("SMTP_PORT", "587")),
                    "username": os.getenv("SMTP_USERNAME"),
                    "password": os.getenv("SMTP_PASSWORD"),
                    "from_address": os.getenv("SMTP_FROM_ADDRESS"),
                },
                "telegram": {
                    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
                }
            }
            
            alert_orch = AlertOrchestrator(
                config=alert_config,
                enable_document_dedup=False  # Trust plugin dedup
            )
            
            # Process alerts
            stats = alert_orch.process_documents(earnings_docs)
            logger.info(f"Alerts: {stats}")
    
    # ... rest of existing pipeline ...
```

### Option C: RSS/Gmail Job Integration (Real-Time Streaming)

Add to `gcr_job_rss.py` or `gcr_job_main.py` after writing to GCS:

```python
# In gcr_job_rss.py or gcr_job_main.py
from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.taxonomy.rules_engine import classify_batch
from event_feed_app.taxonomy.keywords.keyword import keywords

def process_with_alerts(doc_dict):
    """Process a single document for alerts."""
    
    # Quick classification
    classified = classify_batch([doc_dict], keywords=keywords)
    
    # Check if earnings
    if classified[0].get("rule_pred_cid") == "earnings_report":
        # Process for alert
        alert_orch = AlertOrchestrator()  # Uses env vars
        alert_orch.process_single_document(classified[0])

# In main loop after writing to GCS:
for item in rss_items:
    doc_dict = {
        "press_release_id": pr_id,
        "title_clean": title,
        "full_text_clean": body,
        ...
    }
    
    # Write to GCS (existing)
    write_to_gcs(doc_dict)
    
    # NEW: Check for alerts
    if os.getenv("ENABLE_REALTIME_ALERTS") == "true":
        process_with_alerts(doc_dict)
```

## Testing Strategy

### Phase 1: Dry Run with Historical Data

```bash
# See what would be detected (no delivery)
python3 scripts/test_alerts_with_gcp_data.py --days 30 --dry-run
```

Expected output:
```
Alert Processing Results:
  Documents processed: 247
  Alerts detected: 12
  Alerts delivered: 0 (dry run)
```

### Phase 2: Test Delivery to Yourself

1. Set up your user preferences:
```bash
python3 scripts/alert_cli.py user <your-email> \
  --channels telegram \
  --telegram-chat-id <your-chat-id> \
  --min-significance 0.5
```

2. Run with a small date range:
```bash
python3 scripts/test_alerts_with_gcp_data.py --days 3
```

3. Check Telegram for alerts!

### Phase 3: Production Users

1. Add all users:
```python
from event_feed_app.alerts.store import AlertStore

store = AlertStore()

# Portfolio managers
store.save_user_preferences(
    user_id="pm1@fund.com",
    preferences={
        "channels": ["telegram", "email"],
        "telegram_chat_id": "111111111",
        "email": "pm1@fund.com",
        "min_significance": 0.7,
        "companies": ["AAPL", "MSFT"],  # Watchlist
    }
)

# Analysts (all companies)
store.save_user_preferences(
    user_id="analyst@fund.com",
    preferences={
        "channels": ["telegram"],
        "telegram_chat_id": "222222222",
        "min_significance": 0.6,
        "companies": [],  # All companies
    }
)
```

2. Enable in production pipeline:
```bash
export ENABLE_ALERTS=true
export ALERT_MIN_SIGNIFICANCE=0.7
```

## How Deduplication Works

**Two-Level System:**

### Level 1: Metric-Level (Plugin - Automatic)
```python
# GuidanceChangePlugin.prior_key()
key = (company="ACME", period="Q4-2024", metric="revenue", basis="guidance")

# Only alerts if:
# - New guidance found for this (company, period, metric, basis)
# - Value changed from prior state
# - Change is significant (delta > threshold)
```

**This is the authoritative dedup.** The plugin won't detect the same guidance twice.

### Level 2: Document-Level (Optional - Disabled by Default)
```python
# AlertDeduplicator.is_duplicate()
# Checks: "Did we already process this press_release_id?"
# Only useful if pipeline retries same document
```

**Not needed in production** because GCS writes are idempotent and plugin handles metric-level.

## Configuration

All settings via `.env`:

```bash
# Telegram (required for Telegram alerts)
TELEGRAM_BOT_TOKEN=123456789:ABC-DEF...

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_ADDRESS=alerts@yourcompany.com

# Alert thresholds
ALERT_MIN_SIGNIFICANCE=0.7

# Enable in production
ENABLE_ALERTS=true
ENABLE_REALTIME_ALERTS=true
```

## Monitoring & Debugging

### View all alerts:
```bash
python3 scripts/alert_cli.py list
```

### View user preferences:
```bash
python3 scripts/alert_cli.py users
python3 scripts/alert_cli.py user analyst@fund.com
```

### Check alert database:
```bash
sqlite3 data/state/alerts.db

SELECT 
    alert_id,
    company_name,
    summary,
    significance_score,
    created_at
FROM alerts
ORDER BY created_at DESC
LIMIT 10;
```

### Send test alert:
```bash
python3 scripts/send_test_telegram.py <chat-id>
```

## Troubleshooting

**No alerts detected?**
- Check significance threshold (try --min-significance 0.3)
- Verify documents contain guidance keywords
- Check plugin gating isn't filtering them out
- Run with --dry-run to see detection without delivery

**Alerts not delivered?**
- Verify .env has correct credentials
- Check user preferences are set up
- Look at delivery_failures in stats
- Test with scripts/send_test_telegram.py

**Duplicate alerts?**
- Plugin dedup should handle this automatically
- Check that prior_state is being saved correctly
- Verify press_release_id is unique per document

**Too many/few alerts?**
- Adjust ALERT_MIN_SIGNIFICANCE (0.0-1.0)
- Use company watchlists in user preferences
- Fine-tune plugin's decide_and_score() thresholds

## Next Steps

1. **Test with dry run:**
   ```bash
   python3 scripts/test_alerts_with_gcp_data.py --days 7 --dry-run
   ```

2. **Deliver to yourself:**
   ```bash
   # Add your preferences
   python3 scripts/alert_cli.py user <your-email> --channels telegram --telegram-chat-id <id>
   
   # Run for real
   python3 scripts/test_alerts_with_gcp_data.py --days 3
   ```

3. **Integrate into production:**
   - Add alert processing to `orchestrator.py` (Option B)
   - Or add to RSS/Gmail jobs (Option C)
   - Set `ENABLE_ALERTS=true` in production .env
   - Monitor with `alert_cli.py list`

4. **Scale to team:**
   - Add all users with `alert_cli.py user`
   - Set up different thresholds per role
   - Use company watchlists for focused alerts
   - Monitor delivery stats

Ready to test? Run:
```bash
python3 scripts/test_alerts_with_gcp_data.py --days 7 --dry-run
```
