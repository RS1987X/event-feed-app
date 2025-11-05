#!/usr/bin/env bash
# migrate_scripts.sh - Move root scripts to proper locations
# Run with: bash scripts/migrate_scripts.sh --dry-run (to preview)
#           bash scripts/migrate_scripts.sh (to execute)

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be moved"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

move_file() {
    local src="$1"
    local dest="$2"
    
    if [[ ! -f "$src" ]]; then
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

echo "📦 Starting script migration..."
echo ""

# Data processing
move_file "backfill_company_name.py" "scripts/data/backfill_company_name.py"
move_file "export_press_releases_for_labeling.py" "scripts/data/export_press_releases_for_labeling.py"
move_file "fetch_silver_from_GCS.py" "scripts/data/fetch_silver_from_GCS.py"
move_file "mapping_workflow.py" "scripts/data/mapping_workflow.py"

# ML / Clustering
move_file "build_labeling_pool.py" "scripts/ml/build_labeling_pool.py"
move_file "unsupervised_cluster_press_releases_mu.py" "scripts/ml/unsupervised_cluster_press_releases_mu.py"
move_file "unsupervised_cluster_press_releases_mu_embeddings.py" "scripts/ml/unsupervised_cluster_press_releases_mu_embeddings.py"

# Analysis
move_file "cant_be_mapped.py" "scripts/analysis/cant_be_mapped.py"
move_file "inspect_silver_company_names.py" "scripts/analysis/inspect_silver_company_names.py"
move_file "optimal_index_replication_weights.py" "scripts/analysis/optimal_index_replication_weights.py"
move_file "evaluate_categories.py" "scripts/analysis/evaluate_categories.py"

# Debug / Testing
move_file "debug_missing_company_names.py" "scripts/debug/debug_missing_company_names.py"
move_file "globalnewswire_rss_fetch_test.py" "scripts/debug/globalnewswire_rss_fetch_test.py"
move_file "inspect_db.py" "scripts/debug/inspect_db.py"
move_file "test_semantics.py" "scripts/debug/test_semantics.py"

# GUI
move_file "main.py" "scripts/gui/main.py"
move_file "review_app.py" "scripts/gui/review_app.py"
move_file "main_cloud.py" "scripts/gui/main_cloud.py"

# Auth
move_file "convert_token_to_plain_json.py" "scripts/auth/convert_token_to_plain_json.py"

echo ""
echo "✨ Migration complete!"

if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo "📝 Next steps:"
    echo "  1. Update any hardcoded paths in moved scripts"
    echo "  2. Add scripts/ to PYTHONPATH if needed"
    echo "  3. Run tests: pytest tests/"
    echo "  4. Commit: git add scripts/ && git commit -m 'refactor: organize root scripts'"
fi
