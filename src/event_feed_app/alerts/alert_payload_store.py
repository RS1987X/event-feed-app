# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/alert_payload_store.py
"""
GCS-based storage for alert payloads (what the system detected).

Stores complete alert detection details to:
gs://event-feed-app-data/feedback/alert_payloads/signal_type=<type>/YYYY-MM-DD.jsonl

This enables users to see what the system detected when giving feedback.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AlertPayloadStore:
    """Store and retrieve alert detection payloads in GCS."""
    
    def __init__(self, gcs_bucket: str = "event-feed-app-data"):
        self.gcs_bucket = gcs_bucket
        self.payload_root = f"gs://{gcs_bucket}/feedback/alert_payloads"
        
    def save_alert_payload(self, alert: Dict[str, Any]) -> bool:
        """
        Save alert payload to GCS for later reference.
        
        Args:
            alert: Complete alert dictionary from detector
            
        Returns:
            True if saved successfully
            
        Example alert structure:
        {
            "alert_id": "alert_abc123",
            "alert_type": "guidance_change",
            "press_release_id": "1980c93f",
            "company_name": "Acme Corp",
            "detected_at": "2025-11-12T10:00:00Z",
            "significance_score": 0.85,
            "guidance_items": [
                {
                    "metric": "revenue",
                    "direction": "up",
                    "period": "Q4_2025",
                    "value_str": "$100M → $120M",
                    "confidence": 0.85,
                    "text_snippet": "..."
                }
            ],
            "metadata": {...}
        }
        """
        try:
            import gcsfs
            
            alert_id = alert.get("alert_id")
            if not alert_id:
                logger.warning("Alert missing alert_id, cannot save payload")
                return False
            
            signal_type = alert.get("alert_type", "unknown")
            
            # Add detected_at timestamp if not present
            if "detected_at" not in alert:
                alert["detected_at"] = datetime.now(timezone.utc).isoformat()
            
            # Determine file path: signal_type=<type>/YYYY-MM-DD.jsonl
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            file_path = f"{self.payload_root}/signal_type={signal_type}/{date_str}.jsonl"
            
            # Write as newline-delimited JSON
            # Note: gcsfs doesn't support true append mode, so we need to read-modify-write
            fs = gcsfs.GCSFileSystem()
            line = json.dumps(alert) + "\n"
            
            # Read existing content if file exists
            existing_content = ""
            if fs.exists(file_path):
                with fs.open(file_path, "r") as f:
                    content = f.read()
                    existing_content = content if isinstance(content, str) else content.decode("utf-8")
            
            # Write existing + new content
            with fs.open(file_path, "w") as f:
                f.write(existing_content)
                f.write(line)
            
            logger.info(f"Saved alert payload {alert_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save alert payload to GCS: {e}", exc_info=True)
            return False
    
    def get_alert_payload(self, alert_id: str, signal_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve alert payload by ID.
        
        Args:
            alert_id: Alert identifier
            signal_type: Signal type to search in
            
        Returns:
            Alert payload dict or None if not found
        """
        try:
            import gcsfs
            
            fs = gcsfs.GCSFileSystem()
            pattern = f"{self.payload_root}/signal_type={signal_type}/*.jsonl"
            
            files = fs.glob(pattern)
            if not files:
                logger.info(f"No payload files found for {signal_type}")
                return None
            
            # Search through files (newest first)
            for file_path in sorted(files, reverse=True):
                with fs.open(f"gs://{file_path}", "r") as f:
                    for line in f:
                        if line.strip():
                            payload = json.loads(line)
                            if payload.get("alert_id") == alert_id:
                                return payload
            
            logger.info(f"Alert payload not found: {alert_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve alert payload from GCS: {e}", exc_info=True)
            return None
    
    def get_by_press_release_id(
        self, 
        press_release_id: str, 
        signal_type: Optional[str] = None,
        max_days: int = 90
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve alert payload by press release ID.
        
        Useful for feedback analysis when you have the press_release_id
        but not the alert_id (e.g., when investigating false negatives).
        
        Args:
            press_release_id: Press release identifier
            signal_type: Optional signal type to narrow search (e.g., "guidance_change")
                        If None, searches all signal types
            max_days: Maximum number of days of files to search (default: 90)
            
        Returns:
            Alert payload dict or None if not found
        """
        try:
            import gcsfs
            from datetime import datetime, timedelta, timezone
            
            fs = gcsfs.GCSFileSystem()
            
            # Determine search patterns
            if signal_type:
                patterns = [f"{self.payload_root}/signal_type={signal_type}/*.jsonl"]
            else:
                # Search all signal types
                try:
                    signal_dirs = fs.ls(self.payload_root)
                    patterns = [f"{dir_path}/*.jsonl" for dir_path in signal_dirs]
                except:
                    # Fallback: common signal types
                    patterns = [
                        f"{self.payload_root}/signal_type=guidance_change/*.jsonl",
                        f"{self.payload_root}/signal_type=earnings_report/*.jsonl",
                    ]
            
            # Collect all matching files
            all_files = []
            for pattern in patterns:
                files = fs.glob(pattern)
                all_files.extend(files)
            
            if not all_files:
                logger.info(f"No payload files found for search")
                return None
            
            # Search through files (newest first)
            for file_path in sorted(all_files, reverse=True):
                try:
                    with fs.open(f"gs://{file_path}", "r") as f:
                        for line in f:
                            if line.strip():
                                payload = json.loads(line)
                                if payload.get("press_release_id") == press_release_id:
                                    logger.info(f"Found alert payload for PR {press_release_id} in {file_path}")
                                    return payload
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue
            
            logger.info(f"No alert payload found for press_release_id: {press_release_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve alert by press_release_id from GCS: {e}", exc_info=True)
            return None
