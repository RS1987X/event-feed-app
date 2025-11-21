# Alert System

A comprehensive alert system for detecting and delivering notifications about significant corporate events, with a focus on earnings guidance changes.

## Overview

This module provides:
- **Alert Detection**: Uses the `GuidanceChangePlugin` to identify significant guidance changes
- **Document-level Deduplication**: Prevents processing the same press release twice
- **Multi-channel Delivery**: Email, webhooks, and GUI notifications
- **User Management**: Per-user preferences, watchlists, and significance thresholds
- **Audit Trail**: Complete history of alerts and deliveries

**Important Note on Deduplication:**

The `GuidanceChangePlugin` already handles **metric-level deduplication** via its `prior_key()` and `compare()` methods. It tracks state per (company, period, metric, basis) and only triggers alerts when values actually change.

The `AlertDeduplicator` in this module only handles **document-level deduplication** - preventing the same press release from being processed multiple times due to pipeline retries or duplicate documents in the feed.

## Architecture

```
alerts/
├── __init__.py           # Module exports
├── detector.py           # GuidanceAlertDetector - event detection
├── deduplicator.py       # AlertDeduplicator - spam prevention
├── delivery.py           # AlertDelivery - multi-channel notifications
├── store.py             # AlertStore - persistence and state
├── orchestration.py     # AlertOrchestrator - workflow coordination
└── templates/
    └── email_guidance.html  # HTML email template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pyyaml requests  # Already in your requirements
```

### 2. Set Up User Preferences

```bash
# Using the CLI tool
python scripts/alert_cli.py user investor@example.com \
    --email investor@example.com \
    --watchlist "ACME Corporation,TechCo Inc." \
    --min-significance 0.75 \
    --channels email

# Verify setup
python scripts/alert_cli.py users
```

### 3. Configure SMTP (Optional for Email Delivery)

Set environment variables or update config:

```bash
export ALERT_SMTP_HOST="smtp.gmail.com"
export ALERT_SMTP_PORT="587"
export ALERT_SMTP_USERNAME="your-email@gmail.com"
export ALERT_SMTP_PASSWORD="your-app-password"
export ALERT_FROM_ADDRESS="alerts@event-feed-app.com"
```

### 4. Process Documents

```python
from event_feed_app.alerts.orchestration import AlertOrchestrator

# Initialize
config = {
    "min_significance": 0.7,
    "smtp": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "from_address": "alerts@event-feed-app.com",
    }
}
orchestrator = AlertOrchestrator(config=config)

# Process documents
docs = [...]  # Your classified documents
stats = orchestrator.process_documents(docs)
print(f"Alerts delivered: {stats['alerts_delivered']}")
```

## Integration Points

### With Orchestrator Pipeline

Add to `src/event_feed_app/pipeline/orchestrator.py`:

```python
from event_feed_app.alerts.orchestration import AlertOrchestrator

def run(cfg: Settings):
    # ... existing pipeline code ...
    
    # After classification and consensus
    alert_config = {
        "min_significance": cfg.alert_min_significance,
        "smtp": {
            "host": cfg.alert_smtp_host,
            "port": cfg.alert_smtp_port,
            "username": cfg.alert_smtp_username,
            "password": cfg.alert_smtp_password,
            "from_address": cfg.alert_from_address,
        }
    }
    alert_orchestrator = AlertOrchestrator(config=alert_config)
    
    # Filter for earnings documents
    earnings_mask = base_df["category_final"] == "earnings_report"
    earnings_docs = base_df[earnings_mask].to_dict("records")
    
    # Generate and deliver alerts
    if earnings_docs:
        alert_stats = alert_orchestrator.process_documents(earnings_docs)
        logger.info(f"Alert processing: {alert_stats}")
```

### With RSS/Gmail Ingestion

For real-time alerts, add to ingestion jobs:

```python
# In jobs/ingestion/rss/gcr_job_rss.py or gcr_job_main.py

from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.taxonomy.rules_engine import classify_batch

# After extracting and normalizing a press release
doc = {
    "press_release_id": pr_id,
    "title_clean": title,
    "full_text_clean": body,
    "company_name": company,
    "release_date": date,
    "source_url": url,
}

# Quick classification
classified = classify_batch([doc], keywords=keywords)

# If earnings-related, check for alerts
if classified[0].get("rule_pred_cid") == "earnings_report":
    orchestrator = AlertOrchestrator()
    orchestrator.process_single_document(classified[0])
```

## CLI Usage

The `scripts/alert_cli.py` tool provides command-line management:

```bash
# List recent alerts
python scripts/alert_cli.py list --limit 20

# Show user preferences
python scripts/alert_cli.py user investor@example.com --show

# Update user preferences
python scripts/alert_cli.py user investor@example.com \
    --email investor@example.com \
    --watchlist "Company A,Company B" \
    --min-significance 0.8 \
    --channels email,webhook \
    --webhook https://api.example.com/alerts

# List all users
python scripts/alert_cli.py users

# Send test alert
python scripts/alert_cli.py test investor@example.com
```

## Configuration

### Alert Configuration

```python
config = {
    # Detection
    "min_significance": 0.7,  # Minimum score to trigger alert (0-1)
    
    # Deduplication
    "dedup_window_days": 7,   # Days to consider for duplicates
    "dedup_backend": "memory", # "memory" or "firestore"
    
    # SMTP (for email delivery)
    "smtp": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "from_address": "alerts@event-feed-app.com",
    },
    
    # GuidanceChangePlugin configuration
    "guidance_plugin_config": {
        # ... plugin-specific config from YAML
    }
}
```

### User Preferences Schema

```python
{
    "user_id": "investor@example.com",
    "email_address": "investor@example.com",
    "watchlist": ["Company A", "Company B"],  # Empty = all companies
    "alert_types": ["guidance_change", "earnings_report"],
    "min_significance": 0.75,  # User-specific threshold
    "delivery_channels": ["email", "webhook", "gui"],
    "webhook_url": "https://api.example.com/alerts",
    "active": true
}
```

## Database Schema

The alert system uses SQLite with the following tables:

- **alerts**: Stores all generated alerts
- **alert_deliveries**: Tracks delivery status per user/channel
- **guidance_prior_state**: Stores prior guidance for comparison
- **user_preferences**: User configuration and watchlists

Database location: `data/state/alerts.db`

## Examples

See `examples/alert_integration_example.py` for comprehensive examples:

```bash
python examples/alert_integration_example.py
```

## Testing

```python
# Test alert detection
from event_feed_app.alerts.detector import GuidanceAlertDetector

detector = GuidanceAlertDetector(min_significance=0.7)
alerts = detector.detect_alerts(test_docs)
print(f"Detected {len(alerts)} alerts")

# Test deduplication
from event_feed_app.alerts.deduplicator import AlertDeduplicator

dedup = AlertDeduplicator(window_days=7)
is_dup = dedup.is_duplicate(alert)
print(f"Duplicate: {is_dup}")

# Test delivery (without actually sending)
from event_feed_app.alerts.delivery import AlertDelivery

delivery = AlertDelivery()  # No SMTP config = dry run
delivery.deliver(alert, [user_prefs])
```

## Monitoring

Check alert deliveries:

```python
from event_feed_app.alerts.store import AlertStore

store = AlertStore()

# Get recent alerts
recent = store.get_recent_alerts(limit=10)
for alert in recent:
    print(f"{alert['detected_at']}: {alert['company_name']} - {alert['summary']}")
```

## Production Deployment

For production use:

1. **Use Firestore for Deduplication**
   ```python
   config["dedup_backend"] = "firestore"
   ```

2. **Secure SMTP Credentials**
   - Use environment variables or Secret Manager
   - Never commit credentials to git

3. **Set Up Cloud Functions** (for real-time)
   - GCS object creation triggers Cloud Function
   - Function runs classification + alert generation
   - Publishes to Pub/Sub for delivery workers

4. **Enable Monitoring**
   - Log all alerts to Cloud Logging
   - Set up alerts for delivery failures
   - Track metrics (detection rate, delivery success rate)

5. **Configure Rate Limiting**
   - Limit alerts per user per day
   - Implement backoff for failed deliveries

## Troubleshooting

**No alerts detected:**
- Check `min_significance` threshold
- Verify GuidanceChangePlugin is configured
- Ensure documents have required fields (title_clean, full_text_clean)

**Alerts not delivered:**
- Verify user preferences are active
- Check SMTP configuration
- Review logs for delivery errors
- Confirm user watchlist includes the company

**Duplicate alerts:**
- Check deduplication window settings
- Verify deduplication backend is working
- Review dedup key generation logic

## Future Enhancements

- [ ] Slack/Teams integration
- [ ] SMS delivery via Twilio
- [ ] Alert priority levels
- [ ] User feedback loop (useful/not useful)
- [ ] ML-based significance scoring
- [ ] Multi-language support
- [ ] Alert scheduling (daily digest vs. real-time)

## Support

For issues or questions:
1. Check logs in `data/state/alerts.db`
2. Review `EARNINGS_GUIDANCE_ALERT_STRATEGY.md`
3. Run examples to verify setup
4. Check user preferences with `alert_cli.py`
