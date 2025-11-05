# Scripts Directory

This directory contains operational and utility scripts for the event-feed-app project.

## Directory Structure

```
scripts/
├── data/           # Data processing, backfills, exports
├── ml/             # ML experiments, clustering, labeling
├── analysis/       # Ad-hoc analysis, reporting
├── debug/          # Debugging utilities, test scripts
├── gui/            # GUI launchers (PyQt, Streamlit)
├── auth/           # Authentication utilities
├── migrate_scripts.sh          # Migration helper (Phase 2A)
└── consolidate_modules.sh      # Module consolidation (Phase 2B)
```

## Migration Status

These scripts were moved from the project root on 2025-11-05 to improve organization.

### Original Locations (before refactor)
All these scripts were at project root. See `REFACTORING.md` for details.

## Usage

Most scripts are standalone and can be run directly:

```bash
# Example: Export data for labeling
python scripts/data/export_press_releases_for_labeling.py

# Example: Run GUI
python scripts/gui/main.py

# Example: Debug semantics
python scripts/debug/test_semantics.py
```

### Adding scripts/ to PYTHONPATH

If scripts import from `event_feed_app`, ensure it's installed or add to PYTHONPATH:

```bash
export PYTHONPATH="/home/ichard/projects/event-feed-app/src:$PYTHONPATH"
```

Or install the package in editable mode:

```bash
pip install -e .
```

## Contributing

When adding new scripts:
1. Place in appropriate subdirectory
2. Add docstring explaining purpose
3. Update this README if it's a key utility
4. Consider if it should be a CLI command in `pyproject.toml` instead
