# Event Feed App - Comprehensive Refactoring Strategy

**Date**: November 21, 2025  
**Current State**: Main branch (alert system merged) on `refactor/consolidate-structure`  
**Objective**: Transform organically grown codebase into clear, maintainable, production-ready structure

**System Overview:**
- **Classification Pipeline**: Categorizes press releases using taxonomy rules engine (M&A, earnings, personnel, etc.)
- **Alert System**: Detects high-signal events (earnings guidance changes) and delivers alerts via Telegram/Email
- **Data Flow**: Gmail/RSS → Bronze (raw) → Silver (parquet) → Classification → Alert Detection → Delivery

---

## 🔍 Current State Analysis

### Key Issues Identified

1. **Scattered Code Structure**
   - **82 Python files** scattered at root level (outside src/tests/jobs)
   - 18 scripts already in `scripts/` directory
   - Dual hierarchies: `core/`, `auth/`, `sources/`, `utils/` at root + `src/event_feed_app/`
   - Duplicate module: `event_taxonomies/` vs `src/event_feed_app/taxonomy/`

2. **Import Inconsistency**
   - Old-style root imports: `from core.event_types import Event`
   - New-style package imports: `from event_feed_app.taxonomy.adapters import Taxonomy`
   - Mixed import patterns throughout codebase

3. **Module Boundary Confusion**
   - Legacy `core/` (4 modules): company_loader, company_matcher, event_types, oltp_store
   - Legacy `auth/`: gmail_auth
   - Legacy `sources/`: gmail_fetcher, rss_sources
   - Legacy `utils/`: time_utils
   - All duplicated or overlapping with `src/event_feed_app/`

4. **Package Configuration Issues**
   - Two setup files: `setup.py` (minimal) and `pyproject.toml` (proper)
   - Package only finds `src/event_feed_app/*` but code depends on root modules

5. **Script Organization**
   - Root level scripts include: backfill, debug, evaluation, ML, GUI, cloud deployment
   - Some already migrated to `scripts/` but inconsistently categorized

### Current Directory Structure

```
event-feed-app/
├── src/event_feed_app/          # ✅ PRIMARY package (well-organized)
│   ├── alerts/                  # Alert system
│   ├── config.py               # Settings management
│   ├── data/                   # Data utilities
│   ├── evaluation/             # Model evaluation
│   ├── events/                 # Event processing
│   ├── gating/                 # Filter gates
│   ├── models/                 # ML models (NER, TF-IDF)
│   ├── pipeline/               # Orchestration
│   ├── preprocessing/          # Data preprocessing
│   ├── representation/         # Feature engineering
│   ├── taxonomy/               # Event taxonomy & rules engine
│   ├── tools/                  # Calibration & tuning
│   └── utils/                  # Utilities (io, lang, gcs_io, text)
│
├── core/                        # ❌ LEGACY - should merge into src
│   ├── company_loader.py
│   ├── company_matcher.py
│   ├── event_types.py
│   └── oltp_store.py
│
├── auth/                        # ❌ LEGACY - should merge into src
│   └── gmail_auth.py
│
├── sources/                     # ❌ LEGACY - should merge into src
│   ├── gmail_fetcher.py
│   └── rss_sources.py
│
├── utils/                       # ❌ LEGACY - minimal, merge into src
│   └── time_utils.py
│
├── event_taxonomies/            # ❌ DUPLICATE - appears to be old version
│   └── [various taxonomy files]
│
├── scripts/                     # ⚠️ PARTIALLY organized
│   ├── alert_cli.py
│   ├── test_*.py
│   ├── view_*.py
│   └── [subdirs: analysis/, auth/, data/, debug/, eval/, gui/, ml/]
│
├── tests/                       # ✅ Test suite
├── jobs/                        # ✅ Production jobs
├── gui/                         # ❌ Should move to scripts or src
│
└── [82 root-level .py files]   # ❌ CRITICAL - must organize
    ├── backfill_company_name.py
    ├── build_labeling_pool.py
    ├── debug_missing_company_names.py
    ├── evaluate_categories.py
    ├── main.py
    ├── main_cloud.py
    ├── review_app.py
    └── [many more...]
```

---

## 🎯 Refactoring Goals

1. **Single Source of Truth**: All production code in `src/event_feed_app/`
2. **Clear Boundaries**: Scripts vs package code vs tests vs jobs
3. **Consistent Imports**: All imports use `event_feed_app.*` package path
4. **Logical Organization**: Group by function (data, ml, debugging, deployment)
5. **Maintainability**: Easy to navigate, understand, and extend
6. **Backward Compatibility**: Gradual migration with deprecation warnings

---

## 📋 Phased Refactoring Plan

### **Phase 1: Foundation & Safety (Week 1)**

**Goal**: Prepare for migration with safety nets

#### 1.1 Pre-Migration Audit
- [x] Document current state (this file)
- [ ] Run full test suite and capture baseline
- [ ] Create git tag: `pre-refactoring-baseline`
- [ ] Create branch: `refactor/consolidate-structure`
- [ ] Inventory all imports using automated tools
- [ ] Map dependencies between root modules and src

#### 1.2 Test Coverage Enhancement
- [ ] Add integration tests for critical paths
- [ ] Ensure core functionality has unit tests
- [ ] Document any missing test coverage
- [ ] Create smoke tests for main entry points

#### 1.3 Documentation
- [ ] Document public APIs in `src/event_feed_app/`
- [ ] Create module dependency diagram
- [ ] List all entry points (CLIs, jobs, apps)

---

### **Phase 2: Merge Legacy Modules into src/ (Week 2)**

**Goal**: Consolidate `core/`, `auth/`, `sources/`, `utils/` into `src/event_feed_app/`

#### 2.1 Plan Module Mapping

**VERIFIED ACTIVE STATUS - Migration Priority:**

| Legacy Module | New Location | Status | Notes |
|---------------|--------------|--------|-------|
| `core/event_types.py` | `src/event_feed_app/events/types.py` | 🔴 **ACTIVE** | Used by main.py, tests, sources |
| `core/company_loader.py` | `src/event_feed_app/data/company_loader.py` | 🔴 **ACTIVE** | Used by sources, tests |
| `core/company_matcher.py` | `src/event_feed_app/data/company_matcher.py` | 🔴 **ACTIVE** | Used by sources, tests |
| `core/oltp_store.py` | `src/event_feed_app/data/oltp_store.py` | 🔴 **ACTIVE** | Used by main.py, main_cloud.py |
| `auth/gmail_auth.py` | `src/event_feed_app/sources/auth/gmail.py` | 🔴 **ACTIVE** | Used by gmail_fetcher |
| `sources/gmail_fetcher.py` | **⚠️ VERIFY** | ⚠️ **CHECK** | Possibly replaced by jobs/ingestion? |
| `sources/rss_sources.py` | **⚠️ VERIFY** | ⚠️ **CHECK** | Used by main.py, main_cloud.py |
| `utils/time_utils.py` | `src/event_feed_app/utils/time.py` | 🟡 **MINIMAL** | Merge with existing utils |

**CRITICAL FILES TO TRIAGE FIRST:**
- `main.py` - 21 imports from legacy modules - What is this? Still used?
- `main_cloud.py` - 4 imports from legacy - Cloud deployment? Replaced?
- `sources/gmail_fetcher.py` - Is ingestion now in jobs/ingestion/?
- `sources/rss_sources.py` - Still needed for RSS ingestion?

**ACTION REQUIRED BEFORE MIGRATION:**
1. Determine if `main.py` and `main_cloud.py` are still entry points
2. Check if `sources/` is replaced by `jobs/ingestion/`
3. If obsolete, archive rather than migrate

#### 2.2 Migration Steps (Per Module)
1. Copy module to new location
2. Update imports within the module
3. Create backward-compatibility alias in old location:
   ```python
   # core/company_loader.py (deprecated)
   import warnings
   warnings.warn(
       "Importing from 'core' is deprecated. Use 'from event_feed_app.data import company_loader'",
       DeprecationWarning,
       stacklevel=2
   )
   from event_feed_app.data.company_loader import *  # noqa
   ```
4. Update all internal `src/` imports to use new paths
5. Run tests after each module migration
6. Commit atomically: "refactor: migrate core.company_loader to event_feed_app.data"

#### 2.3 Update Root-Level Code
- [ ] Update `main.py` imports
- [ ] Update `main_cloud.py` imports
- [ ] Update all scripts that import legacy modules
- [ ] Update test imports
- [ ] Update `jobs/` imports

#### 2.4 Validation
- [ ] All tests pass
- [ ] No runtime import errors
- [ ] Deprecation warnings visible but not failing
- [ ] Create git tag: `phase2-legacy-merged`

---

### **Phase 3: Organize Root-Level Scripts (Week 3)**

**Goal**: Move 82 root-level Python files into organized `scripts/` subdirectories

#### 3.1 Script Categorization & Mapping

**Data Management** → `scripts/data/`
```
backfill_company_name.py
export_press_releases_for_labeling.py
fetch_silver_from_GCS.py
mapping_workflow.py
```

**Machine Learning** → `scripts/ml/`
```
build_labeling_pool.py
unsupervised_cluster_press_releases_mu.py
unsupervised_cluster_press_releases_mu_embeddings.py
```

**Evaluation & Analysis** → `scripts/analysis/`
```
cant_be_mapped.py
evaluate_categories.py
inspect_silver_company_names.py
optimal_index_replication_weights.py
```

**Debugging & Diagnostics** → `scripts/debug/`
```
debug_missing_company_names.py
globalnewswire_rss_fetch_test.py
inspect_db.py
test_semantics.py
```

**Authentication & Setup** → `scripts/setup/`
```
convert_token_to_plain_json.py
```

**GUI Applications** → `scripts/gui/` or `tools/`
```
main.py (event viewer?)
review_app.py
gui/* (entire directory)
```

**Cloud Deployment** → `scripts/deploy/`
```
main_cloud.py
gcr_job_main.py (DELETE - duplicate of jobs/ingestion)
deploy_viewer.sh
```

**Alert Testing** → `scripts/alerts/` (already exists)
```
alert_cli.py (already in scripts/)
test_alerts_*.py (already in scripts/)
```

#### 3.2 Migration Process
1. Create subdirectory structure in `scripts/`
2. Move files in batches by category
3. Update any hardcoded paths within scripts
4. Update imports if scripts import from each other
5. Create README.md in each `scripts/` subdir explaining purpose
6. Update main README.md with script inventory
7. Delete or archive redundant scripts

#### 3.3 Entry Point Management
- [ ] Update `pyproject.toml` console scripts if needed
- [ ] Create convenience shell scripts if needed
- [ ] Document how to run each major workflow

#### 3.4 Validation
- [ ] Run key scripts to ensure they work from new location
- [ ] Update CI/CD if scripts are run automatically
- [ ] Create git tag: `phase3-scripts-organized`

---

### **Phase 4: Clean Up Duplicates & Dead Code (Week 4)**

**Goal**: Remove duplicates, deprecate old paths, clean up artifacts

#### 4.1 Handle `event_taxonomies/` Duplicate
- [ ] Compare with `src/event_feed_app/taxonomy/`
- [ ] Verify no active usage
- [ ] Archive or delete `event_taxonomies/`
- [ ] Remove from imports

#### 4.2 Remove Legacy Directories (After Deprecation Period)
- [ ] After 2 weeks of deprecation warnings, remove:
  - `core/`
  - `auth/`
  - `sources/`
  - `utils/` (root level)

#### 4.3 Clean Up Root Artifacts
- [ ] Move CSV/JSON data files to `data/` or `outputs/`
- [ ] Organize markdown docs in `docs/`
- [ ] Clean up old config files
- [ ] Update .gitignore

#### 4.4 Consolidate Documentation
```
docs/
├── architecture/
│   ├── system_overview.md
│   └── module_structure.md
├── guides/
│   ├── QUICKSTART_ALERTS.md
│   ├── CREDENTIALS_SETUP.md
│   └── TELEGRAM_SETUP.md
├── specs/
│   ├── GUIDANCE_CHANGE_SPEC.md
│   └── DEDUPLICATION_STRATEGY.md
└── operations/
    ├── ALERT_INTEGRATION_GUIDE.md
    └── GCS_LOADER_MIGRATION.md
```

#### 4.5 Package Configuration
- [ ] Remove old `setup.py` (keep `pyproject.toml` only)
- [ ] Verify all dependencies listed in `pyproject.toml`
- [ ] Consolidate `requirements.txt` and `requirements-viewer.txt`
- [ ] Update package metadata

---

### **Phase 5: Improve Internal Structure (Week 5)**

**Goal**: Refine organization within `src/event_feed_app/`

#### 5.1 Review Current src/ Organization
Current structure is mostly good, but consider:
- [ ] `src/event_feed_app/sources/` (create if not exists)
  - Move gmail_fetcher, rss_sources here
  - Add auth/ subdirectory
- [ ] `src/event_feed_app/events/` (already exists)
  - Ensure types.py is here
- [ ] `src/event_feed_app/data/` (expand)
  - Add company_loader, company_matcher, oltp_store

#### 5.2 Standardize Module Patterns
- [ ] Each module has `__init__.py` with public API
- [ ] Private helpers prefixed with `_`
- [ ] Type hints on public functions
- [ ] Docstrings on public classes/functions

#### 5.3 Configuration Management
- [ ] Centralize all config in `src/event_feed_app/config.py`
- [ ] Environment variable validation
- [ ] Config schema documentation

#### 5.4 Error Handling
- [ ] Define custom exceptions in `src/event_feed_app/exceptions.py`
- [ ] Consistent error handling patterns
- [ ] Logging standards

---

### **Phase 6: Testing & Documentation (Week 6)**

**Goal**: Ensure quality and maintainability

#### 6.1 Test Organization
```
tests/
├── unit/
│   ├── test_taxonomy/
│   ├── test_alerts/
│   ├── test_data/
│   └── ...
├── integration/
│   ├── test_pipeline_e2e.py
│   ├── test_alert_workflow.py
│   └── ...
└── fixtures/
    ├── sample_press_releases.json
    └── ...
```

#### 6.2 Documentation
- [ ] API reference (auto-generated from docstrings)
- [ ] User guides for main workflows
- [ ] Developer guide for contributors
- [ ] Architecture decision records (ADRs)

#### 6.3 Quality Checks
- [ ] Set up linting (ruff/black configured in pyproject.toml)
- [ ] Set up type checking (mypy)
- [ ] Pre-commit hooks
- [ ] CI/CD pipeline updates

---

## 🚀 Quick Wins (Can Do Immediately)

These changes are low-risk and high-impact:

1. **Move Documentation**
   - Consolidate all .md files into `docs/` subdirectories
   - Keep only README.md, REFACTORING.md at root

2. **Organize Data Files**
   - Move .csv files to `data/outputs/` or `data/inputs/`
   - Move .json config files to `data/configs/`

3. **Clean .gitignore**
   - Add common Python patterns
   - Exclude `__pycache__`, `*.pyc`, `.pytest_cache`
   - Exclude `outputs/`, `eval_outputs/`, data artifacts

4. **Scripts README**
   - Create `scripts/README.md` documenting each script category
   - Add usage examples

5. **Delete Obvious Duplicates**
   - `gcr_job_main.py` (duplicate of `jobs/ingestion/gmail/`)
   - Old taxonomy versions if confirmed unused

---

## 📊 Success Metrics

- [ ] Zero root-level .py files (except setup/config)
- [ ] All imports use `event_feed_app.*` pattern
- [ ] 100% test pass rate maintained throughout
- [ ] All deprecation warnings resolved
- [ ] Documentation covers all major features
- [ ] New developers can onboard in < 1 day
- [ ] Build/install from `pyproject.toml` works cleanly

---

## 🔄 Rollback Plan

Each phase is in a separate git branch:
- `refactor/phase1-foundation`
- `refactor/phase2-merge-legacy`
- `refactor/phase3-organize-scripts`
- `refactor/phase4-cleanup`
- `refactor/phase5-internal`
- `refactor/phase6-quality`

Git tags mark completion:
- `pre-refactoring-baseline`
- `phase1-complete`
- `phase2-complete`
- ...

If issues arise:
1. Identify broken functionality
2. Git revert to last stable tag
3. Fix issue in isolation
4. Re-apply migration

---

## 🛠️ Tools & Automation

### Import Scanner
```bash
# Find all imports from legacy modules
grep -r "^from \(core\|auth\|sources\|utils\)\." --include="*.py"
grep -r "^import \(core\|auth\|sources\|utils\)" --include="*.py"
```

### Migration Script Template
```bash
#!/bin/bash
# migrate_module.sh <old_path> <new_path>

OLD_MODULE=$1
NEW_MODULE=$2

# Move file
mkdir -p $(dirname "src/event_feed_app/$NEW_MODULE")
git mv "$OLD_MODULE.py" "src/event_feed_app/$NEW_MODULE.py"

# Create deprecation shim
cat > "$OLD_MODULE.py" << EOF
import warnings
warnings.warn(
    "Importing from '$OLD_MODULE' is deprecated. Use 'event_feed_app.$NEW_MODULE'",
    DeprecationWarning,
    stacklevel=2
)
from event_feed_app.$NEW_MODULE import *
EOF

# Update imports (requires manual review)
echo "TODO: Update imports in codebase"
echo "OLD: from $OLD_MODULE import X"
echo "NEW: from event_feed_app.$NEW_MODULE import X"
```

---

## 📝 Next Actions

**This Week** (Phase 1):
1. ✅ Create this strategy document
2. Run full test suite and capture results
3. Create baseline git tag
4. Create refactor branch
5. Build automated import inventory

**Next Week** (Phase 2):
1. Start merging legacy modules
2. Begin with least-coupled module (`utils/time_utils.py`)
3. Test thoroughly after each module

**Timeline**: 6 weeks for full refactoring (can be condensed if working full-time)

---

## 🎓 Lessons for Future Development

1. **Start with proper structure** - Avoid root-level code files
2. **Use package from day 1** - Always develop within `src/package_name/`
3. **Scripts vs Library** - Clear separation from the start
4. **Import hygiene** - Enforce absolute imports from package
5. **Incremental commits** - Small, atomic changes
6. **Tests first** - Write tests before refactoring
7. **Documentation as code** - Keep docs in sync with code

---

## 🤝 Getting Help

If stuck during refactoring:
1. Check this strategy document
2. Review git history for similar migrations
3. Run tests frequently to catch issues early
4. Ask for code review before major merges
5. Don't hesitate to roll back if something breaks

---

**Last Updated**: November 21, 2025  
**Status**: Ready to begin Phase 1  
**Owner**: Development Team  
**Priority**: High - Technical debt reduction
