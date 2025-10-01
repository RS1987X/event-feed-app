# SPDX-License-Identifier: MIT
# src/event_feed_app/events/base.py
from __future__ import annotations
from typing import Protocol, Mapping, Any, Iterable, Tuple, Optional, Dict, TypedDict

# ---- Common type aliases ----
Doc = Mapping[str, Any]          # normalized input doc you already pass around
Candidate = Dict[str, Any]       # event-specific candidate payload
Comparison = Dict[str, Any]      # deltas/prior-compare result
Cfg = Dict[str, Any]             # loaded YAML for this event (thresholds, scoring, patterns)

class BuiltRows(TypedDict, total=False):
    versions_row: Dict[str, Any]   # for per-event history table (e.g., guidance_versions)
    event_row: Dict[str, Any]      # for global events stream

class EventPlugin(Protocol):
    """
    Minimal interface every event plugin must implement.
    Keep implementations stateless; carry doc/cfg/state via arguments.
    """
    # Static identifiers
    event_key: str          # e.g., "guidance_change"
    taxonomy_class: str     # e.g., "earnings_report"

    # 1) Cheap prefilter: decide if this plugin should even look at the doc
    def gate(self, doc: Doc) -> bool: ...

    # 2) Find 0..N raw candidates inside the doc (event-specific)
    def detect(self, doc: Doc) -> Iterable[Candidate]: ...

    # 3) Normalize a raw candidate (enums, numbers, ranges). Return a new dict.
    def normalize(self, cand: Candidate) -> Candidate: ...

    # 4) Compute the key used to look up prior state (or return a stable tuple).
    #    If your event has no "current state", you can return an empty tuple.
    def prior_key(self, cand: Candidate) -> tuple: ...

    # 5) Compare candidate vs prior state (if any) and return a comparison dict.
    def compare(self, cand: Candidate, prior: Optional[Dict[str, Any]]) -> Comparison: ...

    # 6) Decide significance and produce a score in [0,1]. Also return debug features.
    def decide_and_score(self, cand: Candidate, comp: Comparison, cfg: Cfg) -> Tuple[bool, float, Dict[str, Any]]: ...

    # 7) Build dict rows for storage (no I/O here). Return (versions_row?, event_row?)
    def build_rows(
        self, doc: Doc, cand: Candidate, comp: Comparison, is_sig: bool, score: float
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: ...

def validate_plugin(plugin: EventPlugin) -> None:
    """
    Sanity checks at startup to fail fast if a plugin is mis-declared.
    """
    missing = []
    for attr in ("event_key", "taxonomy_class", "gate", "detect", "normalize",
                 "prior_key", "compare", "decide_and_score", "build_rows"):
        if not hasattr(plugin, attr):
            missing.append(attr)
    if missing:
        raise TypeError(f"Plugin {plugin!r} is missing: {', '.join(missing)}")
