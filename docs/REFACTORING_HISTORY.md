# Refactoring Plan

**Status**: In Progress  
**Started**: 2025-11-05  
**Target Completion**: TBD

## Overview
This project has organic growth that created:
- Dual code hierarchies (`core/` + `src/event_feed_app/`)
- ~20 root-level scripts (~3,700 LOC)
- Mixed module boundaries

## Phase 1: Stabilize (Current)
- [ ] Document current import patterns
- [ ] Create scripts inventory
- [ ] Add temporary symlinks for backward compat
- [ ] Improve .gitignore

## Phase 2: Consolidate
- [ ] Move root scripts to scripts/
- [ ] Merge core/ into src/event_feed_app/
- [ ] Update all imports
- [ ] Fix tests

## Phase 3: Optimize
- [ ] Refactor rule registration
- [ ] Centralize configuration
- [ ] Clean up data directories

---

## Current Module Map

### Active Locations
- `src/event_feed_app/` - **PRIMARY** package (pipeline, taxonomy, models, events)
- `core/` - **LEGACY** (company matching, event types, OLTP store)
- `auth/` - Gmail auth utilities
- `sources/` - Data fetchers (gmail, RSS)
- `gui/` - PyQt6 GUI app
- `event_taxonomies/` - **DUPLICATE** of src/event_feed_app/taxonomy/?

### Root Scripts (to relocate)
```
backfill_company_name.py          → scripts/data/
build_labeling_pool.py            → scripts/ml/
cant_be_mapped.py                 → scripts/analysis/
convert_token_to_plain_json.py   → scripts/auth/
debug_missing_company_names.py   → scripts/debug/
evaluate_categories.py            → scripts/eval/
export_press_releases_for_labeling.py → scripts/data/
fetch_silver_from_GCS.py         → scripts/data/
gcr_job_main.py                  → DELETE (duplicate of jobs/ingestion/gmail/)
globalnewswire_rss_fetch_test.py → scripts/debug/
inspect_db.py                     → scripts/debug/
inspect_silver_company_names.py  → scripts/analysis/
main.py                           → scripts/gui/ or tools/
main_cloud.py                     → scripts/cloud/
mapping_workflow.py               → scripts/data/
optimal_index_replication_weights.py → scripts/analysis/
review_app.py                     → scripts/gui/
test_semantics.py                 → tests/manual/
unsupervised_cluster_*.py         → scripts/ml/
```

---

## Import Patterns to Fix

### Current (messy)
```python
from core.event_types import Event           # root-level core/
from sources.gmail_fetcher import fetch      # root-level sources/
from event_feed_app.pipeline import run      # src/ package
```

### Target (clean)
```python
from event_feed_app.core.event_types import Event
from event_feed_app.sources.gmail_fetcher import fetch
from event_feed_app.pipeline import run
```

---

## Testing Strategy
1. Run existing tests before each major move
2. Keep old imports working via `__init__.py` aliases during transition
3. Add deprecation warnings to old paths
4. Remove aliases after 2 weeks

---

## Rollback Plan
- Git branch for each phase
- Tag releases: `pre-refactor`, `phase1-complete`, etc.
- Keep `core/` symlinked to new location for 1 sprint
