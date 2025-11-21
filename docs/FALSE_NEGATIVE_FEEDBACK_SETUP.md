# False Negative Feedback System

Complete setup for collecting feedback on missed guidance alerts via Telegram.

## Architecture

```
Daily Review (Cron)          Telegram Bot              Storage
     │                            │                       │
     ├─> Samples 10 PRs           │                       │
     ├─> Sends to Telegram ──────>│                       │
     ├─> Saves tracking.json      │                       │
     │                            │                       │
     │   User replies with 👍/👎   │                       │
     │                       <────┤                       │
     │                            │                       │
     │                            ├─> FeedbackStore ─────>│
     │                            │                       │
     │                            │                  GCS JSONL
     │                            │         (same as positive feedback)
```

## Components

### 1. Daily Review Script
**Module:** `event_feed_app.alerts.review.daily_review`
**Entry point:** `event-alerts-review`

**What it does:**
- Samples N earnings PRs that were NOT flagged
- Sends each PR as individual Telegram message
- Saves message_id → press_release_id mapping to `data/labeling/daily_review_YYYYMMDD.json`

**Cron schedule:**
```bash
# Runs daily at 10 AM (7 days/week)
0 10 * * * /home/ichard/projects/event-feed-app/deployment/run_false_negative_review.sh
```

**Manual usage:**
```bash
event-alerts-review --size 10 --days 7 --category earnings
# or
python -m event_feed_app.alerts.review.daily_review --size 10 --days 7 --category earnings
```

### 2. Telegram Feedback Handler
**Module:** `event_feed_app.alerts.review.telegram_handler`
**Entry point:** `event-alerts-telegram`

**What it does:**
- Listens for user replies to daily review messages
- Parses 👍/👎 or `/correct`/`/missed` commands
- Extracts optional notes from reply text
- Saves to GCS via `FeedbackStore` (same as positive feedback)

**Run modes:**
```bash
# Polling mode (for testing)
event-alerts-telegram --poll
# or
python -m event_feed_app.alerts.review.telegram_handler --poll

# Webhook mode (for production)
event-alerts-telegram
```

### 3. Feedback Storage
**Module:** `event_feed_app.alerts.feedback_store.FeedbackStore`

**GCS location:**
```
gs://event-feed-app-data/feedback/signal_ratings/signal_type=guidance_change/YYYY-MM-DD.jsonl
```

**Record format:**
```json
{
  "feedback_id": "fb_abc123",
  "signal_type": "guidance_change",
  "alert_id": "fn_press_release_id",
  "press_release_id": "actual_pr_id",
  "user_id": "telegram_user_id",
  "is_correct": false,
  "feedback_type": "false_negative",
  "submitted_at": "2025-11-21T10:15:00Z",
  "metadata": {
    "issue_type": "false_negative",
    "review_type": "daily_fn_review"
  },
  "notes": "Raised Q4 revenue outlook to $550M"
}
```

## User Workflow

1. **Receive daily review** (10 AM daily)
   - Get 10 earnings PRs that weren't flagged
   - Each PR sent as separate message with company, title, date, link

2. **Reply to any PR** with:
   - `👍` or `/correct` = Not guidance (system correct)
   - `👎` or `/missed` = IS guidance (false negative)

3. **Add notes** (optional):
   ```
   /missed Raised Q4 revenue outlook to $550M
   ```

4. **Get confirmation**:
   ```
   ✅ Feedback saved: fb_abc123
   📌 Marked as FALSE NEGATIVE (should have been flagged)
   Company: Acme Corp
   Note: Raised Q4 revenue outlook to $550M
   ```

## Feedback Types

| User Action | is_correct | feedback_type | Meaning |
|------------|-----------|---------------|---------|
| 👎 /missed | False | false_negative | System MISSED guidance (bad) |
| 👍 /correct | True | true_negative | System CORRECTLY ignored non-guidance (good) |

## Setup Instructions

### 1. Verify Cron Job

```bash
crontab -l | grep false-negative-review
```

Should show:
```
0 10 * * * /home/ichard/projects/event-feed-app/deployment/run_false_negative_review.sh >> /tmp/false-negative-review.log 2>&1
```

### 2. Start Telegram Bot Handler

**Option A: Polling (testing)**
```bash
source venv/bin/activate
event-alerts-telegram --poll
```

**Option B: Webhook (production)**

Set up webhook with Telegram:
```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=https://your-domain.com/webhook"
```

Then run:
```bash
event-alerts-telegram
```

### 3. Test the Flow

```bash
# Test feedback storage
python -m event_feed_app.alerts.review.test_feedback

# Manually trigger daily review
event-alerts-review --size 5 --days 2

# Check saved tracking
cat data/labeling/daily_review_$(date +%Y%m%d).json
```

### 4. Verify Feedback in GCS

```bash
# List feedback files
gsutil ls gs://event-feed-app-data/feedback/signal_ratings/signal_type=guidance_change/

# View recent feedback
gsutil cat gs://event-feed-app-data/feedback/signal_ratings/signal_type=guidance_change/$(date +%Y-%m-%d).jsonl
```

## Monitoring

### Check Daily Review Logs
```bash
tail -f /tmp/false-negative-review.log
```

### Check Bot Handler Logs
```bash
# If running as systemd service
journalctl -u telegram-feedback-handler -f

# If running in terminal
# Logs printed to stdout
```

### Feedback Stats
```python
from event_feed_app.alerts.feedback_store import FeedbackStore

store = FeedbackStore()
stats = store.get_feedback_stats(signal_type="guidance_change")

print(f"Total: {stats['total']}")
print(f"False negatives: {stats['by_type'].get('false_negative', 0)}")
print(f"True negatives: {stats['by_type'].get('true_negative', 0)}")
```

## Next Steps

1. **Run bot handler** - Start in polling mode for testing
2. **Test manually** - Send `/missed` reply to a test message
3. **Verify GCS** - Check feedback saved correctly
4. **Deploy webhook** - Set up production webhook endpoint
5. **Monitor metrics** - Track false negative rate over time
6. **Improve model** - Use feedback to retrain guidance detection

## Troubleshooting

**Bot not responding?**
- Check `TELEGRAM_BOT_TOKEN` in `.env`
- Verify bot handler is running: `ps aux | grep telegram_feedback_handler`
- Check logs for errors

**Feedback not saving?**
- Check GCS permissions: `gsutil ls gs://event-feed-app-data/feedback/`
- Verify tracking file exists: `ls data/labeling/daily_review_*.json`
- Test FeedbackStore directly: `python scripts/alerts/test_feedback_flow.py`

**Can't find PR in tracking data?**
- Make sure replying to message from today's review
- Check tracking file date matches: `ls data/labeling/`
- Review may be too old (tracking files rotate daily)
