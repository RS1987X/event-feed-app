# Quick Start Guide: Earnings Guidance Alerts

Get earnings guidance alerts running in 5 minutes.

## Step 1: Set Up User (30 seconds)

```bash
cd /home/ichard/projects/event-feed-app

# Create your first user
python scripts/alert_cli.py user your-email@example.com \
    --email your-email@example.com \
    --min-significance 0.7 \
    --channels email

# Verify it worked
python scripts/alert_cli.py users
```

## Step 2: Configure SMTP (1 minute)

### Option A: Gmail (Recommended for Testing)

1. Get an App Password: https://myaccount.google.com/apppasswords
2. Update your config:

```python
# Create config/alerts_config.py or set environment variables
export ALERT_SMTP_HOST="smtp.gmail.com"
export ALERT_SMTP_PORT="587"
export ALERT_SMTP_USERNAME="your-email@gmail.com"
export ALERT_SMTP_PASSWORD="your-app-password"
export ALERT_FROM_ADDRESS="alerts@event-feed-app.com"
```

### Option B: Skip Email (Test Without Delivery)

```python
# Alerts will be detected and stored but not emailed
# Check them with: python scripts/alert_cli.py list
```

## Step 3: Test the System (2 minutes)

```bash
# Send a test alert to verify email delivery works
python scripts/alert_cli.py test your-email@example.com

# Check your email inbox!
```

## Step 4: Process Real Documents (1 minute)

```python
# Create test_alerts.py
from event_feed_app.alerts.orchestration import AlertOrchestrator

# Your SMTP config (or leave empty to skip delivery)
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

# Sample document with guidance change
test_doc = {
    "press_release_id": "test_001",
    "title_clean": "ACME Corp Raises Revenue Guidance for FY2025",
    "full_text_clean": """
        ACME Corporation today announced it is raising its revenue 
        guidance for fiscal year 2025 to $500-550 million, up from 
        the previous guidance of $450-480 million. Operating margin 
        guidance is reaffirmed at approximately 20%.
    """,
    "release_date": "2025-11-10",
    "company_name": "ACME Corporation",
    "source_url": "https://example.com/acme-guidance",
}

# Process and deliver alert
result = orchestrator.process_single_document(test_doc)
if result:
    print(f"✓ Alert generated: {result['summary']}")
else:
    print("✗ No alert generated (below threshold or gated out)")
```

Run it:
```bash
python test_alerts.py
```

## Step 5: Check Results

```bash
# List all alerts
python scripts/alert_cli.py list

# Check your email for the alert notification!
```

## What You Should See

### In Terminal:
```
✓ Alert generated and delivered:
  Alert ID: abc123...
  Company: ACME Corporation
  Summary: Revenue guidance raised for FY2025
  Score: 0.850
```

### In Your Email:
```
Subject: 🔔 Guidance Alert: ACME Corporation

ACME Corporation has updated their guidance:

  ↑ Revenue: Raised to 10-15% growth
  → Operating Margin: Confirmed at ~20%

Period: FY2025
Significance Score: 85/100

[View Full Press Release]
```

### In Database:
```bash
sqlite3 data/oltp/alerts.db "SELECT * FROM alerts ORDER BY detected_at DESC LIMIT 1;"
```

## Troubleshooting

### "No alert generated"
- **Reason**: Document didn't pass the gate check or significance threshold
- **Solution**: Lower `min_significance` to 0.5 or check the GuidanceChangePlugin config

### "SMTP authentication failed"
- **Reason**: Wrong password or 2FA not configured
- **Solution**: Use Gmail App Password, not your regular password

### "User not found"
- **Reason**: User preferences not saved
- **Solution**: Run `python scripts/alert_cli.py user ...` first

### "No users to deliver alert to"
- **Reason**: No active users in database
- **Solution**: Create at least one user with Step 1

## Next Steps

### Integrate with Your Pipeline

Add to `src/event_feed_app/pipeline/orchestrator.py`:

```python
from event_feed_app.alerts.orchestration import AlertOrchestrator

def run(cfg: Settings):
    # ... existing code ...
    
    # After classification
    alert_config = {
        "min_significance": 0.7,
        "smtp": {...}  # Load from cfg
    }
    alert_orch = AlertOrchestrator(config=alert_config)
    
    # Filter earnings documents
    earnings_docs = base_df[base_df["category_final"] == "earnings_report"]
    
    # Generate alerts
    if len(earnings_docs) > 0:
        stats = alert_orch.process_documents(earnings_docs.to_dict("records"))
        logger.info(f"Alerts: {stats}")
```

### Add Real-Time Alerts

Add to `jobs/ingestion/rss/gcr_job_rss.py`:

```python
from event_feed_app.alerts.orchestration import AlertOrchestrator

# After parsing each press release
orchestrator = AlertOrchestrator()
orchestrator.process_single_document(normalized_doc)
```

### Monitor Alerts

```bash
# See recent alerts
watch -n 60 'python scripts/alert_cli.py list --limit 5'

# Check delivery status
sqlite3 data/oltp/alerts.db "
  SELECT alert_id, user_id, status, delivered_at 
  FROM alert_deliveries 
  ORDER BY delivered_at DESC 
  LIMIT 10;
"
```

## Advanced Usage

### Company Watchlist

Only get alerts for specific companies:

```bash
python scripts/alert_cli.py user your-email@example.com \
    --watchlist "Microsoft,Apple,Tesla"
```

### Webhook Delivery

Receive alerts as HTTP POST:

```bash
python scripts/alert_cli.py user your-email@example.com \
    --webhook https://your-api.com/webhooks/alerts \
    --channels webhook
```

### Multiple Users

Set up different users with different preferences:

```bash
# Executive: High-priority only, all companies
python scripts/alert_cli.py user ceo@company.com \
    --min-significance 0.9 \
    --channels email

# Analyst: All guidance changes, specific watchlist
python scripts/alert_cli.py user analyst@company.com \
    --min-significance 0.6 \
    --watchlist "Competitor1,Competitor2,Competitor3" \
    --channels email,webhook
```

## You're Done! 🎉

Your earnings guidance alert system is now:
- ✅ Detecting guidance changes automatically
- ✅ Filtering by significance score
- ✅ Delivering to your preferred channels
- ✅ Tracking all alerts and deliveries

For more details:
- **Full Documentation**: `src/event_feed_app/alerts/README.md`
- **Strategy Guide**: `EARNINGS_GUIDANCE_ALERT_STRATEGY.md`
- **Examples**: `examples/alert_integration_example.py`
- **Implementation**: `ALERT_IMPLEMENTATION_SUMMARY.md`
