# Earnings Guidance Commentary Alert System - Robust Strategy

## Executive Summary

This document outlines a comprehensive strategy to alert users about new earnings guidance commentary by leveraging your existing infrastructure:
- **GuidanceChangePlugin** for sophisticated guidance detection
- **Taxonomy rules** (earnings, ratings) for event classification
- **RSS/Gmail ingestion jobs** for data capture
- **OLTP store** for event persistence
- **Orchestrator pipeline** for processing

---

## Current Infrastructure Analysis

### ✅ What You Already Have

1. **Detection & Classification**
   - `GuidanceChangePlugin`: Full-featured plugin that detects guidance changes with:
     - Metric extraction (revenue, EBITDA, margins, etc.)
     - Direction detection (upgrades/downgrades)
     - Numeric range parsing (percentages, currency amounts)
     - Prior state comparison
     - Significance scoring
   - `EarningsHiSigRule`: Detects earnings reports with high confidence
   - `CreditRatingsRule`: Differentiates between credit ratings and guidance (using `GUIDANCE_WORDS`)
   - `GUIDANCE_WORDS`: ["guidance", "prognos", "utsikter", "outlook for"]

2. **Ingestion Pipeline**
   - RSS feed ingestion (`jobs/ingestion/rss/gcr_job_rss.py`)
   - Gmail ingestion (`jobs/ingestion/gmail/gcr_job_main.py`)
   - Incremental sync with Firestore state management
   - Bronze (raw) and Silver (normalized) data layers on GCS

3. **Processing Pipeline**
   - `orchestrator.py`: Main processing pipeline with:
     - Gating (housekeeping, newsletter filtering)
     - Deduplication
     - Rules-based classification
     - ML-based categorization
   - `classify_batch()`: Applies taxonomy rules to documents

4. **Storage**
   - OLTP SQLite database (`core/oltp_store.py`) with:
     - Events table
     - Companies table
     - Event-company junction table
   - Firestore for ingestion state management
   - GCS for data lake (bronze/silver layers)

5. **Event Infrastructure**
   - `EventPlugin` protocol for structured event detection
   - `GuidanceChangePlugin` already registered and tested
   - Event registry system

### ❌ What's Missing

1. **Alert Notification System**
   - No module to send alerts to users (email, webhook, dashboard)
   - No user preference management
   - No alert deduplication mechanism
   - No alert history/audit trail

2. **Real-time Processing Integration**
   - Ingestion jobs write to GCS but don't trigger immediate classification
   - No connection between GuidanceChangePlugin and notification delivery
   - No real-time alert generation from streaming data

3. **User Management**
   - No user profile/preference storage
   - No company watchlist management
   - No alert subscription configuration

---

## Robust Alert Strategy

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                              │
│  ┌──────────────┐              ┌──────────────┐                     │
│  │ Gmail Job    │              │ RSS Job      │                     │
│  │ (gcr_job)    │              │ (gcr_job)    │                     │
│  └──────┬───────┘              └──────┬───────┘                     │
│         │                             │                             │
│         └─────────────┬───────────────┘                             │
│                       ▼                                             │
│         ┌─────────────────────────────┐                             │
│         │  Normalize & Store (GCS)    │                             │
│         │  Bronze → Silver (Parquet)  │                             │
│         └─────────────┬───────────────┘                             │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CLASSIFICATION LAYER                            │
│         ┌───────────────────────────────┐                           │
│         │  Taxonomy Rules Engine        │                           │
│         │  - EarningsHiSigRule          │                           │
│         │  - GuidanceChangePlugin       │                           │
│         │  - Other taxonomy rules       │                           │
│         └─────────────┬─────────────────┘                           │
│                       │                                             │
│                       ▼                                             │
│         ┌───────────────────────────────┐                           │
│         │  Event Detection & Scoring    │                           │
│         │  - Detect guidance changes    │                           │
│         │  - Extract metrics & direction│                           │
│         │  - Calculate significance     │                           │
│         └─────────────┬─────────────────┘                           │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ALERTING LAYER (NEW)                           │
│         ┌───────────────────────────────┐                           │
│         │  Alert Generator              │                           │
│         │  - Match against watchlists   │                           │
│         │  - Apply significance filters │                           │
│         │  - Deduplicate alerts         │                           │
│         └─────────────┬─────────────────┘                           │
│                       │                                             │
│                       ▼                                             │
│         ┌───────────────────────────────┐                           │
│         │  Alert Delivery Manager       │                           │
│         │  - Email notifications        │                           │
│         │  - Webhook/API callbacks      │                           │
│         │  - GUI/Dashboard updates      │                           │
│         └─────────────┬─────────────────┘                           │
│                       │                                             │
│                       ▼                                             │
│         ┌───────────────────────────────┐                           │
│         │  Alert Storage & Audit        │                           │
│         │  - Alert history (SQLite)     │                           │
│         │  - Delivery status tracking   │                           │
│         │  - User feedback/actions      │                           │
│         └───────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

#### Phase 1: Alert Detection & Generation (Batch Mode)

**Objective**: Process existing data and identify guidance alerts without real-time delivery.

1. **Extend Orchestrator Pipeline**
   - After `classify_batch()`, run `GuidanceChangePlugin` on earnings-classified documents
   - Filter for significant guidance changes (score > threshold)
   - Store alert candidates in a new `alerts` table

2. **Create Alert Schema**
   ```python
   # alerts table
   {
     "alert_id": "unique_hash",
     "event_id": "press_release_id",
     "alert_type": "guidance_change",
     "company_name": "ACME Corp",
     "company_id": 123,
     "detected_at": "2025-11-10T14:30:00Z",
     "significance_score": 0.85,
     "summary": "Revenue guidance raised 10-15%",
     "metrics": [{"metric": "revenue", "direction": "up", "range": "10-15%"}],
     "metadata": {...}
   }
   ```

3. **Alert Deduplication**
   - Hash on (company_id, alert_type, detected_period, key_metrics)
   - Store hash in Firestore with 7-day TTL
   - Skip if duplicate detected within window

#### Phase 2: User Preferences & Watchlists

**Objective**: Allow users to configure what alerts they want to receive.

1. **User Preferences Schema**
   ```python
   # user_preferences table
   {
     "user_id": "user@example.com",
     "watchlist_companies": [123, 456, 789],  # company IDs
     "alert_types": ["guidance_change", "earnings_report"],
     "min_significance": 0.7,
     "delivery_channels": ["email", "webhook"],
     "email_address": "user@example.com",
     "webhook_url": "https://api.example.com/webhook",
     "active": true
   }
   ```

2. **Watchlist Management**
   - Simple CRUD API for managing company watchlists
   - Support for "all companies" vs. specific companies
   - Industry/sector-based filtering (future enhancement)

#### Phase 3: Alert Delivery System

**Objective**: Send alerts to users via their preferred channels.

1. **Email Delivery**
   ```python
   # Use Python's smtplib or SendGrid API
   Subject: 🔔 Guidance Alert: ACME Corp raises revenue outlook
   
   ACME Corp has updated their guidance:
   
   • Revenue: Raised to 10-15% growth (from 5-10%)
   • Operating Margin: Confirmed at ~20%
   • Period: FY2025
   
   Significance Score: 85/100
   
   View full press release: [link]
   
   ---
   Manage your alerts: [link]
   ```

2. **Webhook Delivery**
   ```python
   POST {webhook_url}
   {
     "alert_id": "...",
     "type": "guidance_change",
     "company": {...},
     "summary": "...",
     "metrics": [...],
     "press_release_url": "..."
   }
   ```

3. **GUI Updates** (if using PyQt6 GUI)
   - Real-time alert pop-ups
   - Dedicated alerts panel
   - Mark as read/dismiss functionality

#### Phase 4: Real-time Pipeline Integration

**Objective**: Generate and deliver alerts immediately as new data arrives.

1. **Streaming Processor**
   - Add post-ingestion hook in RSS/Gmail jobs
   - After writing to GCS, immediately classify new items
   - Run GuidanceChangePlugin detection
   - Generate alerts synchronously

2. **Cloud Function Trigger** (for production)
   - GCS object creation triggers Cloud Function
   - Function runs classification + alert generation
   - Publishes to Pub/Sub topic
   - Alert delivery worker consumes from Pub/Sub

---

## Implementation Plan

### Module Structure

```
src/event_feed_app/
├── alerts/
│   ├── __init__.py
│   ├── detector.py          # Alert detection logic
│   ├── deduplicator.py      # Deduplication engine
│   ├── delivery.py          # Multi-channel delivery
│   ├── preferences.py       # User preference management
│   ├── store.py            # Alert storage (SQLite/Firestore)
│   └── templates/
│       ├── email_guidance.html
│       └── email_guidance.txt
```

### Key Components

#### 1. Alert Detector (`alerts/detector.py`)

```python
from event_feed_app.events.registry import get_plugin
from typing import List, Dict, Any

class GuidanceAlertDetector:
    def __init__(self, min_significance: float = 0.7):
        self.guidance_plugin = get_plugin("guidance_change")
        self.min_significance = min_significance
    
    def detect_alerts(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process documents and return alert-worthy guidance changes.
        
        Args:
            docs: List of normalized documents (from Silver layer)
        
        Returns:
            List of alert objects ready for delivery
        """
        alerts = []
        
        for doc in docs:
            # Gate: only process earnings-related documents
            if not self.guidance_plugin.gate(doc):
                continue
            
            # Detect candidates
            candidates = list(self.guidance_plugin.detect(doc))
            
            for cand in candidates:
                # Normalize
                norm_cand = self.guidance_plugin.normalize(cand)
                
                # Compare with prior state (if any)
                prior_key = self.guidance_plugin.prior_key(norm_cand)
                prior = self._load_prior_state(prior_key)
                comparison = self.guidance_plugin.compare(norm_cand, prior)
                
                # Score significance
                is_sig, score, features = self.guidance_plugin.decide_and_score(
                    norm_cand, comparison, self.guidance_plugin.cfg
                )
                
                if is_sig and score >= self.min_significance:
                    # Build alert object
                    alert = self._build_alert(doc, norm_cand, comparison, score)
                    alerts.append(alert)
                    
                    # Update prior state
                    self._save_prior_state(prior_key, norm_cand)
        
        return alerts
    
    def _build_alert(self, doc, candidate, comparison, score):
        """Build structured alert object."""
        return {
            "alert_id": self._generate_alert_id(doc, candidate),
            "event_id": doc.get("press_release_id"),
            "alert_type": "guidance_change",
            "company_name": doc.get("company_name"),
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "significance_score": score,
            "summary": self._generate_summary(candidate, comparison),
            "metrics": candidate.get("metrics", []),
            "metadata": {
                "press_release_url": doc.get("source_url"),
                "release_date": doc.get("release_date"),
                "direction": comparison.get("direction"),
                "change_magnitude": comparison.get("magnitude")
            }
        }
```

#### 2. Alert Deduplicator (`alerts/deduplicator.py`)

```python
import hashlib
from google.cloud import firestore
from datetime import datetime, timedelta

class AlertDeduplicator:
    def __init__(self, window_days: int = 7):
        self.db = firestore.Client()
        self.window_days = window_days
        self.collection = "alert_dedup"
    
    def is_duplicate(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is duplicate within window."""
        dedup_key = self._generate_dedup_key(alert)
        doc_ref = self.db.collection(self.collection).document(dedup_key)
        doc = doc_ref.get()
        
        if doc.exists:
            # Check if within window
            created_at = doc.to_dict().get("created_at")
            if created_at and (datetime.now() - created_at).days < self.window_days:
                return True
        
        # Not duplicate - store this alert
        doc_ref.set({
            "created_at": datetime.now(),
            "alert_id": alert["alert_id"],
            "company_name": alert["company_name"]
        })
        
        return False
    
    def _generate_dedup_key(self, alert: Dict[str, Any]) -> str:
        """Generate dedup hash from alert characteristics."""
        key_parts = [
            alert.get("company_name", ""),
            alert.get("alert_type", ""),
            str(alert.get("metadata", {}).get("detected_period", "")),
            # Include metric directions to avoid suppressing legitimate changes
            "_".join(sorted([
                f"{m.get('metric')}_{m.get('direction')}" 
                for m in alert.get("metrics", [])
            ]))
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()
```

#### 3. Alert Delivery (`alerts/delivery.py`)

```python
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AlertDelivery:
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
    
    def deliver(self, alert: Dict[str, Any], users: List[Dict[str, Any]]):
        """
        Deliver alert to all matching users via their preferred channels.
        
        Args:
            alert: Alert object to deliver
            users: List of user preference objects
        """
        for user in users:
            channels = user.get("delivery_channels", ["email"])
            
            if "email" in channels:
                self._send_email(alert, user)
            
            if "webhook" in channels and user.get("webhook_url"):
                self._send_webhook(alert, user)
    
    def _send_email(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """Send email notification."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🔔 Guidance Alert: {alert['company_name']}"
            msg["From"] = self.smtp_config["from_address"]
            msg["To"] = user["email_address"]
            
            # Plain text version
            text = self._render_text_email(alert)
            msg.attach(MIMEText(text, "plain"))
            
            # HTML version (optional)
            # html = self._render_html_email(alert)
            # msg.attach(MIMEText(html, "html"))
            
            with smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"]) as server:
                server.starttls()
                server.login(self.smtp_config["username"], self.smtp_config["password"])
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {user['email_address']} for {alert['alert_id']}")
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook(self, alert: Dict[str, Any], user: Dict[str, Any]):
        """Send webhook notification."""
        try:
            response = requests.post(
                user["webhook_url"],
                json=alert,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"Webhook alert sent to {user['webhook_url']} for {alert['alert_id']}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _render_text_email(self, alert: Dict[str, Any]) -> str:
        """Render plain text email body."""
        lines = [
            f"{alert['company_name']} has updated their guidance:",
            "",
        ]
        
        for metric in alert.get("metrics", []):
            direction = "↑" if metric.get("direction") == "up" else "↓" if metric.get("direction") == "down" else "→"
            lines.append(
                f"  {direction} {metric.get('metric', 'Unknown')}: {metric.get('summary', 'Changed')}"
            )
        
        lines.extend([
            "",
            f"Significance Score: {int(alert['significance_score'] * 100)}/100",
            "",
            f"View full press release: {alert['metadata'].get('press_release_url', 'N/A')}",
            "",
            "---",
            "Manage your alerts: [TBD: user preferences URL]"
        ])
        
        return "\n".join(lines)
```

---

## Deployment Roadmap

### Step 1: Extend OLTP Database (Immediate)
- Add `alerts` table
- Add `user_preferences` table
- Add `alert_deliveries` table (audit trail)

### Step 2: Build Alert Detector (Week 1)
- Implement `GuidanceAlertDetector`
- Integrate with orchestrator in batch mode
- Test on historical data

### Step 3: Build Deduplicator (Week 1)
- Implement Firestore-based deduplication
- Test dedup logic with realistic scenarios

### Step 4: Build Delivery System (Week 2)
- Implement email delivery
- Implement webhook delivery
- Create email templates
- Test end-to-end delivery

### Step 5: User Preferences Management (Week 2)
- Create preferences schema
- Build simple API/CLI for managing preferences
- Integrate with alert matching logic

### Step 6: Real-time Integration (Week 3-4)
- Add post-ingestion hooks to RSS/Gmail jobs
- Implement streaming alert generation
- Deploy Cloud Function triggers (production)
- Monitor and optimize performance

---

## Configuration Example

```yaml
# config/alerts.yaml
alerts:
  guidance_change:
    enabled: true
    min_significance: 0.7
    dedup_window_days: 7
    
delivery:
  email:
    smtp_host: smtp.gmail.com
    smtp_port: 587
    from_address: alerts@event-feed-app.com
  
  webhook:
    timeout_seconds: 10
    retry_attempts: 3

user_defaults:
  delivery_channels:
    - email
  min_significance: 0.7
  alert_types:
    - guidance_change
    - earnings_report
```

---

## Known Limitations and Edge Cases

### Qualitative Market Commentary (Swedish)

**Pattern:** CEO commentary in Swedish quarterly reports about general market conditions using vague language about distant future.

**Example:** BMST Delårsrapport (Q3 2025)
- CEO states: "marknaden bedöms därför ta fart först senare delen av nästa år" (market assessed to take off only in latter part of next year)
- Uses "bedöms" (assessed) - not currently in Swedish forward-cue terms
- No formal guidance keywords ("prognos", "utsikter")
- **No quantified metrics or financial measures** - purely qualitative market view
- Qualitative macro commentary vs. company-specific financial guidance

**Technical Note:** 
- "bedöms" (assessed/judged) could be added to Swedish `fwd_cue.terms.sv` for broader coverage
- However, this PR correctly not flagged due to **lack of any quantified metric** (revenue, EBITDA, margin, growth rate, etc.)
- Guidance detection requires both forward-looking language AND measurable financial metrics
- Adding "bedöms" alone won't cause false positives since metric extraction would still fail

**Decision:** Intentionally NOT detected
- Vague language about general market conditions without quantified guidance
- Distant future timeframe (12+ months)
- Macro market outlook vs. specific company financial targets
- Missing numeric metrics required for actionable guidance

**Alternative Detection:** These types of statements may be better suited for a future "Future Outlook Statements" signal using LLM extraction (see SIGNAL_ROADMAP.md), which would capture qualitative management sentiment without requiring numeric metrics.

### Operational Policy Changes

**Pattern:** Business policy updates that don't involve financial guidance.

**Example:** Illinois Mutual DI Non-Medical Limits
- Announcement of increased non-medical underwriting limits
- Zero guidance keywords present
- Operational/product change, not financial outlook

**Decision:** Correctly NOT detected - these are not guidance changes.

---

## Success Metrics

1. **Detection Accuracy**
   - Precision: % of alerts that are truly significant guidance changes
   - Recall: % of actual guidance changes that trigger alerts
   - Target: >90% precision, >85% recall
   - **Current Performance:** 91% detection rate on validated true positives

2. **Delivery Performance**
   - Latency: Time from ingestion to alert delivery
   - Target: <5 minutes in real-time mode
   - Delivery success rate: >99%

3. **User Engagement**
   - Alert open rate (for emails)
   - False positive feedback rate
   - Target: <5% user-reported false positives

---

## Next Steps

1. ✅ Review this strategy document
2. ⏳ Implement Phase 1 (Alert Detection in Batch Mode)
3. ⏳ Build unit tests for alert detection
4. ⏳ Test on historical earnings/guidance press releases
5. ⏳ Implement Phase 2 (User Preferences)
6. ⏳ Implement Phase 3 (Delivery System)
7. ⏳ Deploy Phase 4 (Real-time Integration)
