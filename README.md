# Event Feed App

A production-ready event intelligence platform that ingests, classifies, and delivers real-time financial event alerts from press releases and corporate communications.

---

## 🎯 Overview

Event Feed App is a comprehensive pipeline for:
- **Ingesting** press releases from Gmail and RSS feeds
- **Classifying** events into 15+ categories using ML + rules
- **Detecting** high-value signals (earnings guidance changes, M&A, etc.)
- **Delivering** actionable alerts via Telegram and email with feedback loop

---

## 🏗️ Architecture

```
Gmail/RSS → Bronze (raw) → Silver (normalized) → Classification → Alert Detection → Delivery
                ↓              ↓                      ↓                ↓              ↓
              GCS/.eml      GCS/Parquet         Taxonomy v4      Significance     Telegram
                                                ML + Rules         Scoring          Email
```

**Data Flow:**
1. **Ingestion** (`jobs/ingestion/`): Gmail/RSS → Bronze → Silver layers on GCS
2. **Classification** (`src/event_feed_app/pipeline/`): Taxonomy v4 rules + ML embeddings
3. **Alert Detection** (`src/event_feed_app/alerts/`): Signal extraction & scoring
4. **Delivery** (`src/event_feed_app/delivery/`): Multi-channel distribution with feedback

**Storage:**
- **GCS**: Bronze (raw .eml) + Silver (Parquet) data lakes
- **Firestore**: Watermarks, state management, alert tracking

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/RS1987X/event-feed-app.git
cd event-feed-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Running the Pipeline

```bash
# Run classification pipeline
event-feed-run

# Run alert detection and delivery
event-alerts-run
```

See [`docs/QUICKSTART_ALERTS.md`](docs/QUICKSTART_ALERTS.md) for detailed setup instructions.

---

## 📂 Project Structure

```
event-feed-app/
├── src/event_feed_app/          # Main package
│   ├── pipeline/                # Classification orchestration
│   ├── alerts/                  # Alert detection & delivery
│   ├── taxonomy/                # Event taxonomy v4 rules
│   ├── events/                  # Signal extractors (guidance_change, etc.)
│   ├── models/                  # ML models (embeddings, TF-IDF)
│   ├── delivery/                # Telegram, email delivery
│   └── config.py                # Configuration management
├── jobs/ingestion/              # Data ingestion jobs
│   ├── gmail/                   # Gmail OAuth + API ingestion
│   └── rss/                     # RSS feed polling
├── scripts/                     # Utility scripts
│   ├── analysis/                # Data analysis tools
│   ├── debug/                   # Debugging utilities
│   ├── data/                    # Data loading/export
│   ├── ml/                      # ML experiments
│   └── gui/                     # Review/labeling apps
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── ALERT_INTEGRATION_GUIDE.md
│   ├── CREDENTIALS_SETUP.md
│   ├── GUIDANCE_CHANGE_SPEC.md
│   └── refactoring/            # Refactoring documentation
├── deployment/                  # Deployment configs
│   ├── Dockerfile.viewer
│   ├── deploy_viewer.sh
│   └── run_alerts.sh
├── data/                        # Data storage
│   ├── companies.csv            # Company reference data
│   ├── labeling/                # ML training data
│   └── oltp/                    # Local database
├── models/                      # ML model artifacts
│   └── lid.176.ftz             # Language detection model
└── archive/                     # Archived legacy code
    └── pre-refactor/           # Code before Nov 2024 refactoring
```

---

## 🎓 Key Features

### 1. Multi-Source Ingestion
- **Gmail**: OAuth2 with Secret Manager, incremental sync via `historyId`, backfill support
- **RSS**: Polling from GlobalNewsWire, Business Wire, PR Newswire
- **Layered storage**: Bronze (raw) + Silver (normalized Parquet)

### 2. Advanced Classification
- **Taxonomy v4**: 15+ categories (M&A, Earnings, Leadership, Product, etc.)
- **Hybrid approach**: Rules engine + ML embeddings (sentence-transformers)
- **TF-IDF gating**: Efficient pre-filtering before expensive embeddings

### 3. Intelligent Alert Detection
- **Guidance Change Plugin**: Deep NLP extraction of earnings guidance changes
- **Significance scoring**: Magnitude, clarity, market impact assessment
- **Deduplication**: Near-duplicate detection to avoid alert spam

### 4. Multi-Channel Delivery
- **Telegram**: Rich formatting with action buttons
- **Email**: HTML templates with company context
- **Feedback loop**: Track user reactions (👍/👎) to improve relevance

---

## 📊 Data Schema

### Silver Layer (Parquet)
| Column             | Type     | Description                   |
| ------------------ | -------- | ----------------------------- |
| press_release_id   | string   | Unique identifier             |
| release_date       | date     | Publication date (UTC)        |
| ingested_at        | datetime | Ingestion timestamp           |
| title              | string   | Press release headline        |
| full_text          | string   | Full body text                |
| from               | string   | Source email/feed             |
| source             | string   | `gmail` or `rss`              |
| source_url         | string   | Original URL (if RSS)         |

### Classification Output
Adds taxonomy category, confidence scores, matched keywords, and ML embedding similarity.

---

## ⚙️ Configuration

Key environment variables (see [`docs/CREDENTIALS_SETUP.md`](docs/CREDENTIALS_SETUP.md)):

| Variable              | Description                              |
| --------------------- | ---------------------------------------- |
| `PROJECT_ID`          | GCP project ID                           |
| `GCS_BUCKET`          | GCS bucket for data lake                 |
| `TELEGRAM_BOT_TOKEN`  | Telegram bot API token                   |
| `TELEGRAM_CHAT_ID`    | Target chat for alerts                   |
| `SMTP_HOST`           | Email server for delivery                |
| `ALERT_MIN_SCORE`     | Minimum significance score (0.0-1.0)     |

---

## 📚 Documentation

- **[Alert Integration Guide](docs/ALERT_INTEGRATION_GUIDE.md)**: End-to-end alert system
- **[Credentials Setup](docs/CREDENTIALS_SETUP.md)**: OAuth, GCP, Telegram configuration
- **[Guidance Change Spec](docs/GUIDANCE_CHANGE_SPEC.md)**: Deep dive into NLP extraction
- **[Quickstart](docs/QUICKSTART_ALERTS.md)**: Get started in 5 minutes

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_guidance_plugin.py

# With coverage
pytest --cov=src/event_feed_app
```

---

## 🔄 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run type checking
pyright

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

---

## 📝 License

This project is proprietary software. All rights reserved.

---

## 🙏 Acknowledgments

Built with:
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- [scikit-learn](https://scikit-learn.org/) for TF-IDF and ML pipeline
- [Google Cloud Platform](https://cloud.google.com/) for infrastructure
- [FastText](https://fasttext.cc/) for language detection
