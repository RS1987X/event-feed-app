# Gmail → GCP Ingestion Pipeline

This project ingests emails (e.g. press releases or event notifications) from Gmail into a structured dataset on Google Cloud.  
It is designed as a **production-ready ingestion pipeline** with incremental sync, historical backfill, idempotency, and layered storage for downstream analytics.

---

## ✨ Features

- **Automated Gmail ingestion**
  - OAuth2 credentials managed via **Google Secret Manager**.
  - Incremental updates via Gmail `historyId`.
  - Historical **backfill with configurable lookback windows**.
  - Retry logic with exponential backoff and support for Gmail `Retry-After` headers.

- **State management with Firestore**
  - Tracks ingestion watermark (last processed message ID + timestamp).
  - Persists backfill cursors and Gmail history IDs.
  - Manages cooldowns to respect Gmail API quota and retry-after windows.
  - Guarantees **strict ordering and idempotency**.

- **Data lake layering on GCS**
  - **Bronze layer**: Raw `.eml` messages + JSON metadata, partitioned by date.
  - **Silver layer**: Normalized Parquet records with extracted fields for analytics.
  - Unique pathing ensures no duplicate ingestion.

- **Robust parsing**
  - Handles both plain text and HTML emails.
  - Extracts subject, sender, and message body into normalized schema.

- **Configurable backfill**
  - Adjustable lookback horizon (`BACKFILL_MAX_AGE_DAYS`).
  - Windowed backfill slices with per-run download caps (`BACKFILL_MAX_RAW_PER_RUN`).
  - Supports resuming from last backfill cursor.

---

## 🏗️ Architecture


- **Bronze (raw layer)**: Raw emails stored as `.eml` with JSON metadata.  
- **Silver (normalized layer)**: Cleaned & structured records in Parquet format.  
- **Firestore**: Stores watermarks, history IDs, backfill cursors, and cooldown state.  
- **Secret Manager**: Manages OAuth client and refresh tokens securely.  

---

## 📂 Output Schema

### Bronze Layer
- `message.eml` (raw MIME message)
- `meta.json`:
  ```json
  {
    "id": "<gmail_id>",
    "from": "sender@example.com",
    "subject": "Example subject",
    "internal_epoch_ms": 1724137200000
  }

### Silver layer

| Column             | Type     | Description                   |
| ------------------ | -------- | ----------------------------- |
| press\_release\_id | string   | Gmail message ID (unique key) |
| release\_date      | date     | UTC date of the email         |
| ingested\_at       | datetime | UTC ingestion timestamp       |
| title              | string   | Subject line                  |
| full\_text         | string   | Extracted plain-text body     |
| from               | string   | Sender email address          |
| source             | string   | Fixed `"gmail"`               |
| source\_url        | string   | (reserved, empty for now)     |
| parser\_version    | int      | Parser version tag            |
| schema\_version    | int      | Schema version tag            |

### Configuration 

| Variable                   | Default               | Description                                    |
| -------------------------- | --------------------- | ---------------------------------------------- |
| `PROJECT_ID`               | event-feed-app-463206 | GCP project ID                                 |
| `GCS_BUCKET`               | event-feed-app-data   | Target bucket for bronze/silver data           |
| `GMAIL_LABEL`              | INBOX                 | Gmail label to ingest from                     |
| `LOOKBACK_SECS`            | 120                   | Lookback overlap window for incremental ingest |
| `BACKFILL_START_ISO`       | None                  | ISO date to anchor backfill start              |
| `BACKFILL_WINDOW_DAYS`     | 1                     | Number of days per backfill window             |
| `BACKFILL_MAX_RAW_PER_RUN` | 300                   | Max raw downloads per run                      |
| `BACKFILL_LABEL`           | INBOX                 | Label to use during backfill                   |
| `BACKFILL_MAX_AGE_DAYS`    | 90                    | Max backfill horizon (days)                    |
| `MAILBOX_TZ`               | Europe/Stockholm      | Local timezone for date queries                |


