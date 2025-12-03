# Data Architecture Overview

## TL;DR - Where's My Data?

### SQLite Database (`data/state/alerts.db`)
**Purpose**: Operational data - alerts, deliveries, user config  
**Why**: Fast local access, transactional guarantees, simple deployment

### GCS Bucket (`gs://event-feed-app-data/`)
**Purpose**: Durable storage - press releases, feedback, alert payloads  
**Why**: Scalable, shareable across instances, permanent audit trail

---

## Complete Data Map

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA STORAGE LAYERS                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 1. GCS SILVER BUCKET - Source Data (Read-Only)                 │
├─────────────────────────────────────────────────────────────────┤
│ gs://event-feed-app-data/silver_normalized/                    │
│   table=press_releases/                                         │
│     source=gmail/                                               │
│     source=globenewswire/                                       │
│     source=prnewswire/                                          │
│     ...                                                         │
│                                                                 │
│ Content: Raw press releases (Parquet, partitioned by date)     │
│ Used by: Alert detector, viewer, analysis scripts              │
│ Retention: Permanent                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. SQLITE DATABASE - Operational Data (Local)                  │
├─────────────────────────────────────────────────────────────────┤
│ data/state/alerts.db                                            │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ TABLE: alerts                                               ││
│ │ Purpose: Store generated alerts                             ││
│ │ Size: ~100 bytes/row                                        ││
│ │ Columns:                                                    ││
│ │   - alert_id (PK)                                           ││
│ │   - alert_type, company_name, press_release_id             ││
│ │   - detected_at, significance_score                         ││
│ │   - summary, metrics_json, metadata_json                    ││
│ │                                                             ││
│ │ Why SQLite: Fast queries, transactional, easy to backup    ││
│ │ Why NOT GCS: Need fast "did we already alert?" checks      ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ TABLE: alert_deliveries                                     ││
│ │ Purpose: Track who received which alerts via which channel  ││
│ │ Columns:                                                    ││
│ │   - alert_id, user_id, channel (email/telegram/webhook)    ││
│ │   - delivered_at, status, error_message                     ││
│ │                                                             ││
│ │ Why SQLite: Fast deduplication ("already sent to user?")   ││
│ │ Why NOT GCS: Need immediate write + read in same pipeline  ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ TABLE: user_preferences                                     ││
│ │ Purpose: User settings, watchlists, notification config     ││
│ │ Columns:                                                    ││
│ │   - user_id (PK)                                            ││
│ │   - email_address, telegram_chat_id, watchlist_json        ││
│ │   - alert_types_json, delivery_channels_json               ││
│ │   - min_significance, webhook_url, active                   ││
│ │                                                             ││
│ │ Why SQLite: Simple CRUD, no need for distributed access    ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ TABLE: guidance_prior_state                                 ││
│ │ Purpose: Store previous guidance values for change detection││
│ │ Columns:                                                    ││
│ │   - state_key (company+period)                              ││
│ │   - metrics_json (previous values)                          ││
│ │   - updated_at                                              ││
│ │                                                             ││
│ │ Why SQLite: Need fast lookup/update in detection pipeline  ││
│ │ Why NOT GCS: Too slow for hot path                         ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ TABLE: alert_feedback (DEPRECATED - DO NOT USE)             ││
│ │ Status: Legacy, kept for schema compatibility              ││
│ │ Use instead: GCS feedback_store (see below)                ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ Backup: Entire file (~1MB), can be committed to git or cloud   │
│ Migration: Easy to move to PostgreSQL/Cloud SQL if needed      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. GCS FEEDBACK BUCKET - User Feedback (Append-Only)           │
├─────────────────────────────────────────────────────────────────┤
│ gs://event-feed-app-data/feedback/                              │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ signal_ratings/signal_type=guidance_change/YYYY-MM-DD.jsonl ││
│ │                                                             ││
│ │ Purpose: User feedback on alert correctness                 ││
│ │ Format: Newline-delimited JSON                              ││
│ │ Content:                                                    ││
│ │   {                                                         ││
│ │     "feedback_id": "fb_xxx",                                ││
│ │     "alert_id": "abc123",                                   ││
│ │     "press_release_id": "19a76a...",                        ││
│ │     "is_correct": true/false,                               ││
│ │     "feedback_type": "true_positive|false_positive|...",    ││
│ │     "metadata": {                                           ││
│ │       "extraction_feedback": {                              ││
│ │         "items": [...],  // Per-item validation            ││
│ │         "missing_items": [...],  // What we missed         ││
│ │         "overall_correct": false                            ││
│ │       }                                                     ││
│ │     }                                                       ││
│ │   }                                                         ││
│ │                                                             ││
│ │ Why GCS: Shared across instances, audit trail, analytics   ││
│ │ Why NOT SQLite: Need web access, share with team           ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ alert_payloads/signal_type=guidance_change/YYYY-MM-DD.jsonl ││
│ │                                                             ││
│ │ Purpose: What the system detected when alerts generated     ││
│ │ Format: Newline-delimited JSON                              ││
│ │ Content:                                                    ││
│ │   {                                                         ││
│ │     "alert_id": "abc123",                                   ││
│ │     "press_release_id": "19a76a...",                        ││
│ │     "title": "...",                                         ││
│ │     "full_text": "...",                                     ││
│ │     "guidance_items": [                                     ││
│ │       {                                                     ││
│ │         "metric": "revenue",                                ││
│ │         "direction": "raise",                               ││
│ │         "period": "Q4_2025",                                ││
│ │         "value_str": "$500M → $520M",                       ││
│ │         "confidence": 0.85,                                 ││
│ │         "text_snippet": "..."                               ││
│ │       }                                                     ││
│ │     ]                                                       ││
│ │   }                                                         ││
│ │                                                             ││
│ │ Why GCS: Show users what we detected, analyze patterns     ││
│ │ Why NOT SQLite: Too large (full text), need web access     ││
│ └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│ Retention: Permanent (needed for feedback analysis)            │
│ Access: Public viewer UI, analysis scripts                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 4. GCS SIGNALS BUCKET - Detection Results (Optional)           │
├─────────────────────────────────────────────────────────────────┤
│ gs://event-feed-app-data/signals/guidance_change/               │
│   aggregated/date=YYYY-MM-DD/ (Parquet)                        │
│   events/date=YYYY-MM-DD/ (Parquet)                            │
│                                                                 │
│ Purpose: Long-term analytics, ML training data                  │
│ Used by: Offline analysis, model training                       │
│ Note: Not currently used by alert system                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture?

### SQLite for Hot Path
- **Alert deduplication**: "Did we already generate this alert?" (sub-millisecond)
- **Delivery tracking**: "Did user X already receive alert Y?" (prevents duplicates)
- **Prior state**: "What was last quarter's guidance?" (needed for comparison)
- **User config**: "What's user X's watchlist?" (fast lookup during detection)

**Alternative considered**: All in GCS
**Why rejected**: Too slow for hot path queries (100-500ms vs <1ms)

### GCS for Feedback
- **Shared across instances**: Cloud Run viewer + local scripts both access same data
- **No database setup**: Just append to JSONL files
- **Audit trail**: Immutable append-only log
- **Team access**: Anyone with bucket access can analyze
- **Cost**: ~$0.10/month for thousands of feedback records

**Alternative considered**: SQLite feedback table
**Why rejected**: Not accessible from web viewer, harder to share

### GCS for Alert Payloads
- **Full context**: Store complete press release text + detection details
- **Web viewer**: Show users what was detected
- **Analysis**: "Why did we detect X as Y?"
- **Debugging**: Reproduce exact detection scenario

**Alternative considered**: Store in SQLite metadata field
**Why rejected**: Too large (full text), need web access

---

## Data Flow

```
Press Release (GCS Silver)
  ↓
Alert Detector
  ↓
┌─────────────────────────────────────────────┐
│ 1. Check prior state (SQLite)              │ ← Fast lookup
│ 2. Detect changes                           │
│ 3. Save alert (SQLite)                      │ ← Deduplication
│ 4. Save payload (GCS)                       │ ← Full context
│ 5. Check delivery log (SQLite)             │ ← Already sent?
│ 6. Deliver to users                         │
│ 7. Record delivery (SQLite)                 │ ← Audit trail
└─────────────────────────────────────────────┘
  ↓
User receives alert email/Telegram
  ↓
User clicks feedback link → Web Viewer
  ↓
┌─────────────────────────────────────────────┐
│ 1. Load payload (GCS)                       │ ← Show detection
│ 2. User provides feedback                   │
│ 3. Save feedback (GCS)                      │ ← Append to JSONL
└─────────────────────────────────────────────┘
  ↓
Analysis Scripts
  ↓
┌─────────────────────────────────────────────┐
│ 1. Read feedback (GCS)                      │
│ 2. Read payloads (GCS)                      │
│ 3. Generate reports                         │
│ 4. Identify improvements                    │
└─────────────────────────────────────────────┘
```

---

## Migration Path

If SQLite becomes a bottleneck (unlikely for single-user):

```python
# Current: SQLite
alert_store = AlertStore(db_path="data/state/alerts.db")

# Future: PostgreSQL (same interface!)
alert_store = AlertStore(
    db_url="postgresql://user:pass@host/db"
)
```

All code stays the same - just swap the storage backend.

---

## Summary Table

| Data Type | Storage | Size | Access Pattern | Why Not Alternative? |
|-----------|---------|------|----------------|---------------------|
| **Press Releases** | GCS Silver | GB | Batch read | ✓ Archive, need Parquet |
| **Alerts** | SQLite | KB | Hot path write+read | GCS too slow (<1ms needed) |
| **Deliveries** | SQLite | KB | Hot path dedup | GCS too slow |
| **User Prefs** | SQLite | KB | Config lookup | Simple CRUD, no sharing needed |
| **Prior State** | SQLite | KB | Hot path compare | GCS too slow |
| **Feedback** | GCS | KB | Append-only, shared | Need web access, team sharing |
| **Payloads** | GCS | MB | Occasional read | Full text too large for SQLite |

---

## Cost Analysis

**SQLite**: $0/month (local disk)  
**GCS Feedback**: ~$0.10/month (1000 records @ $0.02/GB/month)  
**GCS Payloads**: ~$0.50/month (50MB @ $0.02/GB/month + egress)  
**Total**: **~$0.60/month**

Compare to: Cloud SQL ($25/month minimum)

---

## Backup Strategy

**SQLite**:
```bash
# Backup entire database
cp data/state/alerts.db backups/alerts_$(date +%Y%m%d).db

# Or commit to git (small enough)
git add data/state/alerts.db
```

**GCS**:
- Automatic versioning enabled
- No manual backup needed
- Point-in-time recovery available

---

## Questions?

**Q: Why not just use GCS for everything?**  
A: Too slow for hot path queries. SQLite gives <1ms lookups, GCS is 100-500ms.

**Q: Why not just use SQLite for everything?**  
A: Can't access from web viewer, harder to share with team, no cloud deployment.

**Q: Can I query both together?**  
A: Yes! Analysis scripts load SQLite + GCS data for comprehensive reports.

**Q: What if I have multiple machines?**  
A: SQLite is single-machine. For multi-instance: move to PostgreSQL or sync via GCS.

**Q: Is feedback in both SQLite and GCS?**  
A: SQLite has empty `alert_feedback` table (legacy). **Only use GCS FeedbackStore**.
