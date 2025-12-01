# SPDX-License-Identifier: MIT
# src/event_feed_app/signals/__init__.py
"""
Signal storage and retrieval for structured event data.

This module provides canonical storage for detected signals (guidance changes,
significant orders, etc.) in a queryable Parquet format, separate from alert
delivery payloads.
"""

from .store import SignalStore
from .schema import GuidanceEventSchema, AggregatedAlertSchema

__all__ = ["SignalStore", "GuidanceEventSchema", "AggregatedAlertSchema"]
