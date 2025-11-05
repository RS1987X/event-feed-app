#!/usr/bin/env bash
# pre_migration_check.sh - Quick import/smoke test runner
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure package path is visible when running without install
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

echo "🔎 Running smoke tests..."
if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PY_CMD="$ROOT_DIR/venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY_CMD="$ROOT_DIR/.venv/bin/python"
else
  PY_CMD="python"
fi

$PY_CMD -m pytest -q tests/smoke || {
  echo "❌ Smoke tests failed. Aborting.";
  exit 1;
}

echo "✅ Smoke tests passed. Safe to proceed with migration."
