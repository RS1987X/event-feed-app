# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/orchestration.py
"""
Alert orchestration - integrates detection, deduplication, and delivery.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

from .detector import GuidanceAlertDetector
from .deduplicator import AlertDeduplicator
from .delivery import AlertDelivery
from .store import AlertStore

logger = logging.getLogger(__name__)


class AlertOrchestrator:
    """
    Orchestrates the complete alert workflow:
    1. Detect significant events from documents
    2. Match against user preferences
    3. Deliver via appropriate channels
    
    NOTE: Document-level deduplication is optional and disabled by default.
    The GuidanceChangePlugin already handles metric-level deduplication via
    its prior_key() and compare() methods.
    """
    
    def __init__(
        self,
        detector: Optional[GuidanceAlertDetector] = None,
        deduplicator: Optional[AlertDeduplicator] = None,
        delivery: Optional[AlertDelivery] = None,
        store: Optional[AlertStore] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_document_dedup: bool = False  # Disabled by default
    ):
        """
        Initialize the alert orchestrator.
        
        Args:
            detector: Alert detector instance
            deduplicator: Deduplicator instance (optional, usually not needed)
            delivery: Delivery manager instance
            store: Alert store instance
            config: Configuration dict
            enable_document_dedup: Enable document-level deduplication (default: False)
        """
        self.config = config or {}
        self.store = store or AlertStore()
        self.enable_document_dedup = enable_document_dedup
        
        self.detector = detector or GuidanceAlertDetector(
            min_significance=self.config.get("min_significance", 0.7),
            store=self.store,
            config=self.config
        )
        
        # Document dedup is optional - the plugin handles guidance-level dedup
        if enable_document_dedup:
            self.deduplicator = deduplicator or AlertDeduplicator(
                window_days=self.config.get("dedup_window_days", 7),
                backend=self.config.get("dedup_backend", "memory")
            )
        else:
            self.deduplicator = None
        
        self.delivery = delivery or AlertDelivery(
            smtp_config=self.config.get("smtp", {}),
            telegram_config=self.config.get("telegram", {}),
            store=self.store
        )
    
    def process_documents(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process documents and generate/deliver alerts.
        
        Args:
            docs: List of normalized documents to process
        
        Returns:
            Summary statistics dict
        """
        stats = {
            "docs_processed": len(docs),
            "alerts_detected": 0,
            "alerts_deduplicated": 0,  # Kept for API compatibility, but plugin handles this
            "alerts_delivered": 0,
            "delivery_failures": 0,
        }
        
        # Step 1: Detect alerts
        # Note: GuidanceChangePlugin handles metric-level deduplication via prior_key/compare
        logger.info(f"Processing {len(docs)} documents for alerts...")
        raw_alerts = self.detector.detect_alerts(docs)
        stats["alerts_detected"] = len(raw_alerts)
        
        if not raw_alerts:
            logger.info("No alerts detected")
            return stats
        
        # Step 2: Optional document-level deduplication (usually not needed)
        unique_alerts = raw_alerts
        if self.enable_document_dedup and self.deduplicator is not None:
            unique_alerts = []
            for alert in raw_alerts:
                if self.deduplicator.is_duplicate(alert):
                    stats["alerts_deduplicated"] += 1
                    logger.debug(f"Duplicate document skipped: {alert['alert_id']}")
                else:
                    unique_alerts.append(alert)
            
            logger.info(
                f"After document deduplication: {len(unique_alerts)} unique alerts "
                f"({stats['alerts_deduplicated']} duplicates filtered)"
            )
        else:
            logger.info(
                f"Document deduplication disabled, processing {len(unique_alerts)} alerts "
                f"(plugin handles metric-level dedup)"
            )
        
        if not unique_alerts:
            return stats
        
        # Step 3: Get active users
        users = self.store.get_all_active_users()
        logger.info(f"Found {len(users)} active users")
        
        if not users:
            logger.warning("No active users configured - alerts will be stored but not delivered")
            # Still save alerts for later review
            for alert in unique_alerts:
                self.store.save_alert(alert)
            return stats
        
        # Step 4: Deliver alerts
        for alert in unique_alerts:
            # Save alert first
            self.store.save_alert(alert)
            
            # Deliver to matching users
            try:
                self.delivery.deliver(alert, users)
                stats["alerts_delivered"] += 1
            except Exception as e:
                logger.error(f"Failed to deliver alert {alert['alert_id']}: {e}")
                stats["delivery_failures"] += 1
        
        logger.info(
            f"Alert processing complete: {stats['alerts_detected']} detected, "
            f"{stats['alerts_delivered']} delivered"
        )
        
        return stats
    
    def process_single_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single document (for real-time use).
        
        Args:
            doc: Normalized document
        
        Returns:
            Alert dict if one was generated and delivered, None otherwise
        """
        stats = self.process_documents([doc])
        
        if stats["alerts_delivered"] > 0:
            # Return the most recent alert for this doc
            recent = self.store.get_recent_alerts(limit=1)
            return recent[0] if recent else None
        
        return None


def create_orchestrator_from_config(config_path: str) -> AlertOrchestrator:
    """
    Create an AlertOrchestrator from a YAML/JSON config file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configured AlertOrchestrator instance
    """
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    
    if config_file.suffix in (".yaml", ".yml"):
        with open(config_file) as f:
            config = yaml.safe_load(f)
    elif config_file.suffix == ".json":
        import json
        with open(config_file) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_file.suffix}")
    
    return AlertOrchestrator(config=config.get("alerts", {}))
