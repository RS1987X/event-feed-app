#!/usr/bin/env bash
# consolidate_modules.sh - Move root modules into src/event_feed_app/
# Phase 2B: After scripts are moved

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE"
fi

move_module() {
    local src="$1"
    local dest="$2"
    
    if [[ ! -d "$src" ]]; then
        echo "⚠️  Skip: $src (not found)"
        return
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "📋 Would move: $src → $dest"
    else
        mkdir -p "$(dirname "$dest")"
        git mv "$src" "$dest" 2>/dev/null || mv "$src" "$dest"
        echo "✅ Moved: $src → $dest"
    fi
}

echo "🔧 Consolidating modules into src/event_feed_app/..."
echo ""

# Move root modules
move_module "auth" "src/event_feed_app/auth"
move_module "sources" "src/event_feed_app/sources"
move_module "gui" "src/event_feed_app/gui"
make_stub "auth" "event_feed_app/auth"
make_stub "sources" "event_feed_app/sources"
make_stub "gui" "event_feed_app/gui"

# Create backward-compat stubs after move (only in real run)
make_stub() {
    local pkg="$1"   # e.g., core
    local modpath="$2" # e.g., event_feed_app/core
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "📋 Would create stub package $pkg/ → import from $modpath"
        return
    fi
    mkdir -p "$pkg"
    cat > "$pkg/__init__.py" <<PY
import warnings as _warnings
_warnings.warn(
        "Deprecated import path '$pkg'. Use 'event_feed_app.$pkg' instead.",
        DeprecationWarning,
        stacklevel=2,
)
from event_feed_app.$pkg import *  # type: ignore
PY
    echo "🧩 Stub created: $pkg/__init__.py (re-exports event_feed_app.$pkg)"
}

# Handle core/ - need to check for conflicts first
if [[ -d "core" ]]; then
    echo "📦 Handling core/ (checking for conflicts with existing src/event_feed_app/core/)"
    
    if [[ -d "src/event_feed_app/core" ]]; then
        echo "⚠️  WARNING: src/event_feed_app/core/ already exists!"
        echo "   Manual merge required. Contents:"
        echo "   Root core/:"
        ls -1 core/
        echo ""
        echo "   Package core/:"
        ls -1 src/event_feed_app/core/ 2>/dev/null || echo "   (empty)"
    else
        move_module "core" "src/event_feed_app/core"
        make_stub "core" "event_feed_app/core"
    fi
fi

# Check event_taxonomies/
if [[ -d "event_taxonomies" ]]; then
    echo ""
    echo "📚 Checking event_taxonomies/ vs src/event_feed_app/taxonomy/"
    
    if [[ -d "src/event_feed_app/taxonomy" ]]; then
        echo "⚠️  Both exist - manual comparison needed:"
        echo "   Files in event_taxonomies/:"
        find event_taxonomies -name "*.py" | head -5
        echo ""
        echo "   Files in src/event_feed_app/taxonomy/:"
        find src/event_feed_app/taxonomy -name "*.py" | head -5
        echo ""
        echo "   Recommendation: Compare files, keep src/event_feed_app/taxonomy/, delete event_taxonomies/"
    else
        move_module "event_taxonomies" "src/event_feed_app/taxonomy"
        # No stub for event_taxonomies; import path differs. Prefer updating imports.
    fi
fi

echo ""
echo "✨ Module consolidation complete!"

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "📝 CRITICAL: Update imports throughout codebase:"
    echo "   OLD: from core.event_types import Event"
    echo "   NEW: from event_feed_app.core.event_types import Event"
    echo ""
    echo "   OLD: from sources.gmail_fetcher import fetch"
    echo "   NEW: from event_feed_app.sources.gmail_fetcher import fetch"
fi
