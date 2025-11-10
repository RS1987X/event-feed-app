# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/deduplicator.py
"""
Document-level alert deduplication to prevent processing the same press release twice.

NOTE: The GuidanceChangePlugin already handles metric-level deduplication via its
prior_key() and compare() methods. This deduplicator only prevents duplicate alerts
from the SAME document being processed multiple times (e.g., due to pipeline retries).

For guidance-specific deduplication (e.g., same company updating same metric in different
documents), the plugin's state management is authoritative.
"""
from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AlertDeduplicator:
    """
    Deduplicates alerts at the DOCUMENT level to prevent processing the same 
    press release multiple times.
    
    IMPORTANT: This does NOT handle metric-level deduplication (e.g., same company 
    updating same metric in different documents). That's handled by GuidanceChangePlugin's
    prior_key() and compare() methods, which track state per (company, period, metric, basis).
    
    Use cases for this deduplicator:
    - Pipeline retries (same document processed twice)
    - Duplicate documents in the feed
    - Testing scenarios where same doc is submitted multiple times
    
    For guidance change detection, the plugin is the authoritative source of truth.
    """
    
    def __init__(
        self,
        window_days: int = 7,
        backend: str = "memory",
        firestore_client=None
    ):
        """
        Initialize the deduplicator.
        
        Args:
            window_days: Number of days to consider for duplicate detection
            backend: Storage backend ("memory" or "firestore")
            firestore_client: Firestore client instance (required if backend="firestore")
        """
        self.window_days = window_days
        self.backend = backend
        
        if backend == "firestore":
            if firestore_client is None:
                try:
                    from google.cloud import firestore
                    self.db = firestore.Client()
                except ImportError:
                    logger.warning("Firestore not available, falling back to memory backend")
                    self.backend = "memory"
                    self.db = None
            else:
                self.db = firestore_client
            self.collection = "alert_dedup"
        else:
            self.db = None
            # In-memory storage: {dedup_key: {"created_at": datetime, "alert_id": str}}
            self._memory_store: Dict[str, Dict[str, Any]] = {}
    
    def is_duplicate(self, alert: Dict[str, Any]) -> bool:
        """
        Check if alert is duplicate within the configured time window.
        
        Args:
            alert: Alert object to check
        
        Returns:
            True if duplicate, False otherwise
        """
        dedup_key = self._generate_dedup_key(alert)
        
        if self.backend == "firestore" and self.db:
            return self._is_duplicate_firestore(dedup_key, alert)
        else:
            return self._is_duplicate_memory(dedup_key, alert)
    
    def _is_duplicate_firestore(self, dedup_key: str, alert: Dict[str, Any]) -> bool:
        """Check for duplicate using Firestore backend."""
        if not self.db:
            return False
            
        try:
            doc_ref = self.db.collection(self.collection).document(dedup_key)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                if not data:
                    return False
                created_at = data.get("created_at")
                
                # Check if within window
                if created_at:
                    age_days = (datetime.now(timezone.utc) - created_at).days
                    if age_days < self.window_days:
                        logger.info(
                            f"Duplicate alert detected: {alert['alert_id']} "
                            f"(matches {data.get('alert_id')}, age={age_days}d)"
                        )
                        return True
            
            # Not duplicate - store this alert
            doc_ref.set({
                "created_at": datetime.now(timezone.utc),
                "alert_id": alert["alert_id"],
                "company_name": alert.get("company_name", ""),
                "summary": alert.get("summary", ""),
            })
            
            return False
        
        except Exception as e:
            logger.error(f"Firestore dedup check failed: {e}", exc_info=True)
            # Fail open: allow alert through if dedup check fails
            return False
    
    def _is_duplicate_memory(self, dedup_key: str, alert: Dict[str, Any]) -> bool:
        """Check for duplicate using in-memory backend."""
        # Clean up expired entries periodically
        self._cleanup_expired_memory()
        
        if dedup_key in self._memory_store:
            existing = self._memory_store[dedup_key]
            age_days = (datetime.now(timezone.utc) - existing["created_at"]).days
            
            if age_days < self.window_days:
                logger.info(
                    f"Duplicate alert detected: {alert['alert_id']} "
                    f"(matches {existing['alert_id']}, age={age_days}d)"
                )
                return True
        
        # Not duplicate - store this alert
        self._memory_store[dedup_key] = {
            "created_at": datetime.now(timezone.utc),
            "alert_id": alert["alert_id"],
            "company_name": alert.get("company_name", ""),
        }
        
        return False
    
    def _cleanup_expired_memory(self):
        """Remove expired entries from memory store."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.window_days)
        expired_keys = [
            key for key, data in self._memory_store.items()
            if data["created_at"] < cutoff
        ]
        for key in expired_keys:
            del self._memory_store[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired dedup entries")
    
    def _generate_dedup_key(self, alert: Dict[str, Any]) -> str:
        """
        Generate deduplication hash at DOCUMENT level.
        
        We use event_id (press_release_id) as the primary dedup key since the
        GuidanceChangePlugin already handles metric-level deduplication.
        
        This prevents the same document from generating duplicate alerts if
        processed multiple times (e.g., pipeline retry, duplicate in feed).
        """
        # Simple: just hash the event_id (press_release_id)
        # If the same document is processed twice, we skip the second time
        event_id = alert.get("event_id", "")
        
        if not event_id:
            # Fallback: use alert_id if no event_id
            logger.warning("No event_id in alert, using alert_id for dedup")
            return alert.get("alert_id", "unknown")
        
        return hashlib.sha256(event_id.encode("utf-8")).hexdigest()
    
    def clear(self):
        """Clear all dedup state (useful for testing)."""
        if self.backend == "memory":
            self._memory_store.clear()
            logger.info("Cleared in-memory dedup state")
        elif self.backend == "firestore" and self.db:
            # Note: Only clear in test environments!
            logger.warning("Firestore dedup clear not implemented (safety)")
