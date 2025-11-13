# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/feedback_store.py
"""
GCS-based feedback storage for signal ratings.

Stores user feedback on alert correctness to:
gs://event-feed-app-data/feedback/signal_ratings/signal_type=<type>/YYYY-MM-DD.jsonl
"""
from __future__ import annotations
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Store and retrieve signal feedback in GCS."""
    
    def __init__(self, gcs_bucket: str = "event-feed-app-data"):
        self.gcs_bucket = gcs_bucket
        self.feedback_root = f"gs://{gcs_bucket}/feedback/signal_ratings"
        
    def save_feedback(
        self,
        signal_type: str,
        alert_id: str,
        press_release_id: str,
        user_id: str,
        is_correct: bool,
        guidance_metadata: Dict[str, Any] | None = None,
        notes: str | None = None
    ) -> str:
        """
        Save user feedback to GCS.
        
        Args:
            signal_type: Type of signal (e.g., 'guidance_change')
            alert_id: Alert identifier (renamed from signal_id for consistency)
            press_release_id: Source press release ID
            user_id: User submitting feedback
            is_correct: True if signal was correct, False if incorrect
            guidance_metadata: Signal-specific metadata (metrics, direction, period, etc.)
            notes: Optional free-text notes
            
        Returns:
            feedback_id if saved successfully
            
        Raises:
            Exception if save fails
        """
        try:
            import gcsfs
            
            # Generate feedback record
            feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc)
            
            # Determine feedback classification
            if is_correct:
                feedback_type = "true_positive"
            else:
                # Check metadata for more specific classification
                if guidance_metadata and guidance_metadata.get("issue_type"):
                    feedback_type = guidance_metadata["issue_type"]
                else:
                    feedback_type = "false_positive"
            
            record = {
                "feedback_id": feedback_id,
                "signal_type": signal_type,
                "alert_id": alert_id,
                "press_release_id": press_release_id,
                "user_id": user_id,
                "is_correct": is_correct,
                "feedback_type": feedback_type,
                "submitted_at": now.isoformat(),
                "metadata": guidance_metadata or {},
                "notes": notes or ""
            }
            
            # Determine file path: signal_type=<type>/YYYY-MM-DD.jsonl
            date_str = now.strftime("%Y-%m-%d")
            file_path = f"{self.feedback_root}/signal_type={signal_type}/{date_str}.jsonl"
            
            # Append to file (create if doesn't exist)
            fs = gcsfs.GCSFileSystem()
            
            # Write as newline-delimited JSON
            line = json.dumps(record) + "\n"
            
            # Read-modify-write pattern (gcsfs append mode doesn't work - overwrites instead)
            existing_content = ""
            if fs.exists(file_path):
                with fs.open(file_path, "r") as f:
                    content = f.read()
                    # Handle both bytes and string
                    if isinstance(content, bytes):
                        existing_content = content.decode('utf-8')
                    else:
                        existing_content = content
            
            # Write existing + new content
            with fs.open(file_path, "w") as f:
                f.write(existing_content)
                f.write(line)
            
            logger.info(f"Saved feedback {feedback_id} to {file_path}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to save feedback to GCS: {e}", exc_info=True)
            raise
    
    def get_feedback_for_signal(
        self,
        signal_type: str,
        alert_id: str | None = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve feedback for a signal type or specific alert.
        
        Args:
            signal_type: Signal type to filter by
            alert_id: Optional specific alert ID to filter (renamed from signal_id)
            
        Returns:
            List of feedback records
        """
        try:
            import gcsfs
            import pandas as pd
            
            fs = gcsfs.GCSFileSystem()
            pattern = f"{self.feedback_root}/signal_type={signal_type}/*.jsonl"
            
            files = fs.glob(pattern)
            if not files:
                logger.info(f"No feedback files found for {signal_type}")
                return []
            
            # Read all JSONL files
            records = []
            for file_path in files:
                with fs.open(f"gs://{file_path}", "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            
            # Filter by alert_id if provided
            if alert_id:
                records = [r for r in records if r.get("alert_id") == alert_id]
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to retrieve feedback from GCS: {e}", exc_info=True)
            return []
    
    def get_feedback_stats(self, signal_type: str | None = None) -> Dict[str, Any]:
        """
        Get aggregated feedback statistics.
        
        Args:
            signal_type: Optional filter by signal type
            
        Returns:
            Dictionary with stats (total, correct, incorrect, accuracy, by_type)
        """
        try:
            import gcsfs
            
            fs = gcsfs.GCSFileSystem()
            
            if signal_type:
                pattern = f"{self.feedback_root}/signal_type={signal_type}/*.jsonl"
            else:
                pattern = f"{self.feedback_root}/signal_type=*/*.jsonl"
            
            files = fs.glob(pattern)
            if not files:
                return {
                    "total_feedback": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "accuracy": 0.0,
                    "by_type": {},
                    "by_signal_type": {}
                }
            
            # Read all records
            records = []
            for file_path in files:
                with fs.open(f"gs://{file_path}", "r") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            
            # Calculate stats
            total = len(records)
            correct = sum(1 for r in records if r.get("is_correct"))
            incorrect = total - correct
            
            # By feedback_type
            by_type: Dict[str, int] = {}
            for r in records:
                fb_type = r.get("feedback_type", "unknown")
                by_type[fb_type] = by_type.get(fb_type, 0) + 1
            
            # By signal_type
            by_signal_type: Dict[str, int] = {}
            for r in records:
                sig_type = r.get("signal_type", "unknown")
                by_signal_type[sig_type] = by_signal_type.get(sig_type, 0) + 1
            
            return {
                "total_feedback": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / total if total > 0 else 0.0,
                "by_type": by_type,
                "by_signal_type": by_signal_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
            return {
                "total_feedback": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "by_type": {},
                "by_signal_type": {}
            }
