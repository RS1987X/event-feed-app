# Pre-Refactoring Archive

**Archived Date**: November 21, 2025  
**Reason**: Code refactoring - consolidating into src/event_feed_app/

This directory contains code that was replaced or deemed obsolete during the refactoring.
Kept for historical reference and potential recovery.

## Structure

- `legacy-modules/` - Old core/, auth/, sources/, utils/ modules (replaced by src/event_feed_app/)
- `legacy-scripts/` - Root-level scripts replaced by organized scripts/ or jobs/
- `old-taxonomy/` - event_taxonomies/ (replaced by src/event_feed_app/taxonomy/)

## Why Archived

### Legacy Modules
- `core/` - Replaced by src/event_feed_app/data/ and src/event_feed_app/events/
- `auth/` - Integrated into jobs/ingestion or obsolete
- `sources/` - Replaced by jobs/ingestion/gmail/ and jobs/ingestion/rss/
- `utils/` - Merged into src/event_feed_app/utils/

### Legacy Scripts
- `main.py` - Old PyQt6 GUI, replaced by streamlit apps in scripts/
- `main_cloud.py` - Old cloud deployment, replaced by jobs/ingestion/
- `gcr_job_main.py` - Empty duplicate of jobs/ingestion/gmail/gcr_job_main.py

### Old Taxonomy
- `event_taxonomies/` - Taxonomy v3, replaced by v4 in src/event_feed_app/taxonomy/

## Recovery

If you need to recover any of this code:
```bash
# Copy back from archive
cp archive/pre-refactor/legacy-modules/core/company_loader.py .

# Or view in git history
git show pre-refactoring-baseline:core/company_loader.py
```
