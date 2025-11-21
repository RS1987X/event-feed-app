# Refactoring Strategy Validation Report

**Date**: November 21, 2025  
**Branch**: `refactor/consolidate-structure`  
**Status**: ✅ Strategy validated with adjustments needed

---

## ✅ System Architecture Confirmed

### **Main Pipelines Identified:**

1. **Classification Pipeline** (`src/event_feed_app/pipeline/orchestrator.py`)
   - Purpose: Categorize all incoming press releases
   - Input: Press releases from GCS Silver layer (Parquet)
   - Processing: 
     - Gating filters (housekeeping, newsletter spam)
     - Taxonomy rules engine (15+ event categories)
     - ML-based classification (embeddings, TF-IDF)
   - Output: Categorized press releases with metadata
   - Categories: M&A, earnings, personnel, regulatory, debt, etc.

2. **Alert System** (`src/event_feed_app/alerts/`)
   - Purpose: Detect and deliver high-signal event alerts
   - Input: Classified press releases
   - Processing:
     - Event plugins (currently `guidance_change`)
     - Significance scoring
     - Deduplication
   - Output: Alerts delivered via Telegram/Email
   - Feedback loop for improvement

### **Data Flow:**
```
Gmail/RSS → Bronze (raw .eml) → Silver (parquet) → Classification → Alert Detection → Delivery
                                      ↓                    ↓
                                   GCS Storage         Alert Store
```

---

## 🔍 Legacy Module Analysis

### **Active Legacy Imports Found: 21 occurrences**

**Files still using old import patterns:**
1. `main.py` - 5 imports from core/sources
2. `main_cloud.py` - 4 imports from core/sources
3. `sources/gmail_fetcher.py` - 4 imports from core/auth
4. `sources/rss_sources.py` - 3 imports from core
5. `core/company_matcher.py` - 1 import from core
6. `core/oltp_store.py` - 1 import from core
7. `tests/test_company_matching.py` - 2 imports from core
8. `tests/test_oldb_store.py` - 1 import from core

### **Critical Questions That Need Answers:**

#### **Q1: Entry Points - Are these still used?**
- [ ] **`main.py`** - What does this do? GUI? Viewer? CLI tool?
  - Uses: `sources.gmail_fetcher`, `sources.rss_sources`, `core.event_types`, `core.oltp_store`
  - **Decision needed**: Migrate to scripts/ or delete?

- [ ] **`main_cloud.py`** - Cloud deployment script?
  - Uses: `sources.gmail_fetcher`, `sources.rss_sources`, `core.event_types`, `core.oltp_store`
  - **Decision needed**: Still used or replaced by jobs/?

#### **Q2: Ingestion - Replaced by jobs/ingestion/?**
- [ ] **`sources/gmail_fetcher.py`** - Is this old ingestion code?
  - Current ingestion: `jobs/ingestion/gmail/` exists
  - **Decision needed**: Delete or migrate?

- [ ] **`sources/rss_sources.py`** - RSS ingestion still needed?
  - Used by main.py and main_cloud.py
  - **Decision needed**: Active or obsolete?

#### **Q3: Database Layer - Still using SQLite?**
- [ ] **`core/oltp_store.py`** - SQLite database operations
  - Used by main.py, main_cloud.py
  - Current system uses GCS + Firestore
  - **Decision needed**: Legacy code to delete?

---

## 📊 Duplicate Detection Results

### **`event_taxonomies/` - ✅ CONFIRMED OBSOLETE**
- No active imports found
- Contains old v3 taxonomy files
- Current system uses `src/event_feed_app/taxonomy/` with v4
- **Action**: Safe to delete

### **Root-level Scripts Count**
- Total Python files at root: **82 files**
- Already organized in `scripts/`: **18 files**
- Need to triage: **~64 files**

---

## 🎯 Revised Refactoring Priorities

### **Phase 0: PREREQUISITE - Triage Legacy Code (Do First!)**

**Before migrating legacy modules, determine:**

1. **Identify Active vs Obsolete Entry Points**
   ```bash
   # Check what main.py does
   python main.py --help  # Does it work?
   
   # Check main_cloud.py
   python main_cloud.py --help  # Still deployed?
   ```

2. **Verify Ingestion Architecture**
   - Is `sources/gmail_fetcher.py` used or replaced by `jobs/ingestion/`?
   - Is `sources/rss_sources.py` still active for RSS feeds?
   - Document current ingestion workflow

3. **Database Usage**
   - Is SQLite (`core/oltp_store.py`) still used?
   - Or fully migrated to GCS/Firestore?

4. **Create Archive Directory**
   ```bash
   mkdir -p archive/pre-refactor/
   # Move obsolete code here instead of deleting immediately
   ```

### **Phase 1: Safe Deletions (After Triage)**

**Confirmed Safe to Delete:**
- `event_taxonomies/` - Obsolete taxonomy v3
- Any entry points confirmed unused
- Test files for deleted modules

**Archive (Don't Delete Yet):**
- Questionable scripts - move to `archive/`
- Old entry points - keep for reference
- Legacy modules - archive after migration

### **Phase 2: Migrate Active Legacy Modules**

**Only migrate modules that are actively used:**

**High Priority (Definitely Active):**
1. `core/event_types.py` → `src/event_feed_app/events/types.py`
2. `core/company_loader.py` → `src/event_feed_app/data/company_loader.py`
3. `core/company_matcher.py` → `src/event_feed_app/data/company_matcher.py`

**Conditional (Depends on Triage):**
4. `core/oltp_store.py` → Migrate or delete?
5. `auth/gmail_auth.py` → Migrate or delete?
6. `sources/gmail_fetcher.py` → Migrate or delete?
7. `sources/rss_sources.py` → Migrate or delete?

### **Phase 3: Update Import Statements**

**Files Needing Import Updates:**
- `main.py` (if keeping)
- `main_cloud.py` (if keeping)
- `sources/*.py` (if keeping)
- `tests/test_company_matching.py`
- `tests/test_oldb_store.py`

**Migration Pattern:**
```python
# OLD
from core.event_types import Event
from sources.gmail_fetcher import fetch_recent_emails

# NEW
from event_feed_app.events.types import Event
from event_feed_app.sources.gmail import fetch_recent_emails  # if migrated
# OR delete if obsolete
```

---

## 🚨 Risks & Mitigation

### **Risk 1: Breaking Active Entry Points**
**Risk**: Migrating code that's actively used in production
**Mitigation**: 
- Triage first (Phase 0)
- Keep deprecation shims for 2 weeks
- Test all entry points after migration

### **Risk 2: Hidden Dependencies**
**Risk**: Legacy modules imported dynamically or in unexpected places
**Mitigation**:
- Run full grep search before each module migration
- Check `jobs/` directory for usage
- Run full test suite after each change

### **Risk 3: Data Loss**
**Risk**: Deleting files that contain important logic/data
**Mitigation**:
- Archive instead of delete
- Git history preserves everything
- Create pre-refactor tag

---

## ✅ Pre-Refactoring Checklist

Before starting Phase 2 migration:

- [ ] **Triage main.py** - Determine if active or obsolete
- [ ] **Triage main_cloud.py** - Check if still deployed
- [ ] **Verify ingestion** - Document current workflow
- [ ] **Check database usage** - SQLite vs GCS/Firestore
- [ ] **Run full test suite** - Capture baseline
- [ ] **Create git tag** - `pre-refactoring-baseline`
- [ ] **Create archive/** - For obsolete code
- [ ] **Document findings** - Update this file with decisions

---

## 📋 Questions for Developer

**Please answer these to finalize strategy:**

1. **What is `main.py` used for?** (GUI viewer? CLI tool? Obsolete?)
   - [ ] Still actively used
   - [ ] Can be deleted/archived
   - [ ] Should be migrated to scripts/

2. **What is `main_cloud.py`?** (Production deployment? Test script?)
   - [ ] Active production code
   - [ ] Obsolete, replaced by jobs/
   - [ ] Keep for reference

3. **Is `sources/gmail_fetcher.py` still used?**
   - [ ] Yes, active ingestion
   - [ ] No, replaced by jobs/ingestion/gmail/
   - [ ] Partially - some functions still needed

4. **Is `sources/rss_sources.py` still needed?**
   - [ ] Yes, for RSS ingestion
   - [ ] No, can delete
   - [ ] Migrate to proper location

5. **Is SQLite (`core/oltp_store.py`) still used?**
   - [ ] Yes, active database
   - [ ] No, fully migrated to GCS/Firestore
   - [ ] Legacy code, can delete

6. **Root scripts priority** - Which root .py files are most important?
   - List top 5-10 that MUST be kept and organized
   - Rest can be archived

---

## 🎯 Next Actions

**Immediate (this session):**
1. Answer the questions above
2. Run exploratory commands to check entry points
3. Document findings
4. Update strategy with firm decisions

**After triage:**
1. Create archive/ directory
2. Move confirmed-obsolete code to archive
3. Delete event_taxonomies/
4. Begin Phase 2 migration of active modules

---

**Status**: Awaiting developer input on questions above before proceeding with migration.
