# Alert Feedback Implementation Summary

## What We Built

A complete feedback system that allows you to rate alert correctness from your iPhone and capture detailed information about what the system detected vs what was actually in the press release.

## Key Components

### 1. Alert Payload Storage (`alert_payload_store.py`)
- **Purpose**: Save what the system detected when alerts are generated
- **Storage**: `gs://event-feed-app-data/feedback/alert_payloads/signal_type=<type>/YYYY-MM-DD.jsonl`
- **Contains**: Detected metrics, direction, period, confidence, text snippets

### 2. User Feedback Storage (`feedback_store.py`)
- **Purpose**: Save user ratings on alert correctness
- **Storage**: `gs://event-feed-app-data/feedback/signal_ratings/signal_type=<type>/YYYY-MM-DD.jsonl`
- **Contains**: Correctness flag, issue type, correct values, notes

### 3. Web Viewer (`view_pr_web.py`)
- **Purpose**: Mobile-friendly interface for reading PRs and submitting feedback
- **Shows**: 
  - Full press release content
  - What the system detected (from alert payload)
  - Feedback form with signal-specific fields
- **Deployment**: Cloud Run (~$0.05/month)

### 4. Alert Delivery Enhancement (`delivery.py`)
- **Purpose**: Save alert payloads when delivering alerts
- **Enhancement**: Include `alert_id` and `signal_type` in viewer URLs

### 5. Alert Detector Enhancement (`detector.py`)
- **Purpose**: Structure guidance items for easy display
- **Enhancement**: Added `guidance_items` array with formatted detection details

## User Workflow

```
1. System detects guidance change
   └─> Saves alert payload to GCS

2. User receives Telegram alert on iPhone
   └─> Message includes link to cloud viewer

3. User clicks link
   └─> Opens Streamlit app in browser

4. Viewer displays:
   📄 Press Release Content
   🔍 What System Detected:
      ✓ Metric: revenue
      ✓ Direction: up (📈)
      ✓ Period: Q4_2025
      ✓ Value: $100M → $120M (+20%)
      ✓ Text: "we now expect revenue of $120M..."

5. User compares detection with actual content

6. User submits feedback:
   Option A: ✅ Correct (quick button)
   Option B: ❌ Incorrect → Detailed form:
      - What was wrong? [dropdown]
      - Correct metrics: [multi-select]
      - Correct direction: [dropdown]
      - Correct period: [text input]
      - Notes: [text area]

7. Feedback saved to GCS

8. Later: Analyze feedback for model improvement
```

## Files Created/Modified

### New Files:
- `src/event_feed_app/alerts/feedback_store.py` - GCS-based feedback storage
- `src/event_feed_app/alerts/alert_payload_store.py` - Alert detection storage
- `scripts/view_feedback_stats.py` - View aggregated stats
- `scripts/test_feedback_system.py` - Test GCS feedback
- `docs/FEEDBACK_SYSTEM.md` - Complete documentation
- `Dockerfile.viewer` - Docker image for Cloud Run
- `deploy_viewer.sh` - Deployment script

### Modified Files:
- `src/event_feed_app/alerts/delivery.py` - Save payloads, add signal_type to URLs
- `src/event_feed_app/alerts/detector.py` - Structure guidance_items for display
- `src/event_feed_app/alerts/store.py` - Added SQLite feedback table (legacy, not used)
- `scripts/view_pr_web.py` - Enhanced with payload display and detailed feedback form
- `.env.example` - Added VIEWER_BASE_URL documentation

## Data Schema

### Alert Payload:
```json
{
  "alert_id": "alert_abc",
  "alert_type": "guidance_change",
  "press_release_id": "1980c93f",
  "guidance_items": [
    {
      "metric": "revenue",
      "direction": "up",
      "period": "Q4_2025",
      "value_str": "$100M → $120M (+20%)",
      "confidence": 0.92,
      "text_snippet": "..."
    }
  ]
}
```

### User Feedback:
```json
{
  "feedback_id": "fb_xyz",
  "signal_type": "guidance_change",
  "signal_id": "alert_abc",
  "is_correct": false,
  "metadata": {
    "issue_type": "wrong_metric",
    "correct_metrics": ["margin"],
    "correct_direction": "down",
    "correct_period": "Q4_2025"
  },
  "notes": "Detected revenue but was margin"
}
```

## Cost Analysis

- **GCS Storage**: ~$0.01/month (< 1GB for years of data)
- **Cloud Run**: ~$0.05/month (70 views/month, minimal compute)
- **BigQuery** (optional): $5/TB scanned (rarely needed)
- **Total**: < $0.10/month

## Next Steps

1. **Test locally**: `python3 scripts/test_feedback_system.py`
2. **Deploy viewer**: `bash deploy_viewer.sh`
3. **Set env var**: `VIEWER_BASE_URL=https://pr-viewer-xyz.run.app`
4. **Test end-to-end**: Generate alert → receive Telegram → click link → submit feedback
5. **Analyze feedback**: `python scripts/view_feedback_stats.py --signal-type guidance_change`

## Future Enhancements

- Add user authentication (Telegram user_id integration)
- Build analytics dashboard (Streamlit multipage app)
- Export feedback to BigQuery for SQL analysis
- Add feedback trends over time
- Implement A/B testing for model improvements
