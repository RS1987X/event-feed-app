# SPDX-License-Identifier: MIT
# src/event_feed_app/events/registry.py
from __future__ import annotations
from typing import Dict, List
from .base import EventPlugin, validate_plugin

# ---- Import and register plugins explicitly (simple & reliable) ----
# Switch to v2/v3 guidance change plugin implementation (plugin2).
# Keep old import comment for quick rollback if needed.
from .guidance_change.plugin2 import plugin as guidance_change  # was: plugin (v0)

PLUGINS: Dict[str, EventPlugin] = {
    guidance_change.event_key: guidance_change,
}

# Validate at import time to fail fast in CI/runtime
for _p in PLUGINS.values():
    validate_plugin(_p)

def all_plugins() -> List[EventPlugin]:
    return list(PLUGINS.values())

def get_plugin(event_key: str) -> EventPlugin:
    try:
        return PLUGINS[event_key]
    except KeyError as e:
        raise KeyError(f"Unknown event_key '{event_key}'. Registered: {list(PLUGINS)}") from e
