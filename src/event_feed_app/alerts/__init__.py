# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/__init__.py
"""
Alert system for detecting and delivering notifications about significant corporate events.

This module provides:
- Alert detection from event plugins (e.g., GuidanceChangePlugin)
- Deduplication to prevent alert spam
- Multi-channel delivery (email, webhook, GUI)
- User preference management
"""

from .detector import GuidanceAlertDetector
from .deduplicator import AlertDeduplicator
from .delivery import AlertDelivery
from .store import AlertStore

__all__ = [
    "GuidanceAlertDetector",
    "AlertDeduplicator",
    "AlertDelivery",
    "AlertStore",
]
