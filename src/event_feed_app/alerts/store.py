# SPDX-License-Identifier: MIT
# src/event_feed_app/alerts/store.py
"""
Alert storage and state management.
"""
from __future__ import annotations
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AlertStore:
    """
    Manages alert persistence and prior state for comparison.
    
    Uses SQLite for local storage with the following tables:
    - alerts: Stores all generated alerts
    - alert_deliveries: Tracks delivery status per user
    - guidance_prior_state: Stores prior guidance values for comparison
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the alert store.
        
        Args:
            db_path: Path to SQLite database file (default: data/state/alerts.db)
        """
        if db_path is None:
            db_path = Path("data/state/alerts.db")
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id        TEXT    PRIMARY KEY,
                    event_id        TEXT    NOT NULL,
                    alert_type      TEXT    NOT NULL,
                    press_release_id TEXT,
                    company_name    TEXT,
                    detected_at     TEXT    NOT NULL,
                    significance_score REAL NOT NULL,
                    summary         TEXT,
                    metrics_json    TEXT,
                    metadata_json   TEXT,
                    created_at      TEXT    NOT NULL
                )
            """)
            
            # Alert deliveries table (tracks which users received which alerts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_deliveries (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id        TEXT    NOT NULL REFERENCES alerts(alert_id),
                    user_id         TEXT    NOT NULL,
                    channel         TEXT    NOT NULL,
                    delivered_at    TEXT    NOT NULL,
                    status          TEXT    NOT NULL,
                    error_message   TEXT,
                    UNIQUE(alert_id, user_id, channel)
                )
            """)
            
            # Prior state table (for guidance comparison)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS guidance_prior_state (
                    state_key       TEXT    PRIMARY KEY,
                    company_name    TEXT,
                    period          TEXT,
                    metrics_json    TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                )
            """)
            
            # User preferences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id         TEXT    PRIMARY KEY,
                    email_address   TEXT,
                    telegram_chat_id TEXT,
                    watchlist_json  TEXT,
                    alert_types_json TEXT,
                    min_significance REAL,
                    delivery_channels_json TEXT,
                    webhook_url     TEXT,
                    active          INTEGER DEFAULT 1,
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL
                )
            """)
            
            # New table: alert_feedback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_feedback (
                    feedback_id     TEXT    PRIMARY KEY,
                    alert_id        TEXT    NOT NULL,
                    press_release_id TEXT,
                    user_id         TEXT    NOT NULL,
                    is_correct      INTEGER NOT NULL,  -- 1=correct, 0=incorrect
                    feedback_type   TEXT,              -- 'true_positive', 'false_positive', etc.
                    notes           TEXT,
                    submitted_at    TEXT    NOT NULL,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_detected_at ON alerts(detected_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_company ON alerts(company_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_alert ON alert_deliveries(alert_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_alert ON alert_feedback(alert_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON alert_feedback(user_id)")

            # ---- Lightweight migrations (add columns if missing) ----
            try:
                cur = conn.execute("PRAGMA table_info(user_preferences)")
                cols = {row[1] for row in cur.fetchall()}
                if "telegram_chat_id" not in cols:
                    conn.execute("ALTER TABLE user_preferences ADD COLUMN telegram_chat_id TEXT")
            except Exception:
                # Best-effort migration; safe to ignore
                pass
            
            try:
                cur = conn.execute("PRAGMA table_info(alerts)")
                cols = {row[1] for row in cur.fetchall()}
                if "press_release_id" not in cols:
                    conn.execute("ALTER TABLE alerts ADD COLUMN press_release_id TEXT")
            except Exception:
                # Best-effort migration; safe to ignore
                pass
            
            try:
                cur = conn.execute("PRAGMA table_info(alert_feedback)")
                cols = {row[1] for row in cur.fetchall()}
                if "press_release_id" not in cols:
                    conn.execute("ALTER TABLE alert_feedback ADD COLUMN press_release_id TEXT")
            except Exception:
                # Best-effort migration; safe to ignore
                pass
    
    def save_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Save an alert to the database.
        
        Args:
            alert: Alert object to save
        
        Returns:
            True if saved, False if duplicate
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts (
                        alert_id, event_id, alert_type, press_release_id, company_name,
                        detected_at, significance_score, summary,
                        metrics_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert["alert_id"],
                    alert.get("event_id", ""),
                    alert["alert_type"],
                    alert.get("press_release_id", ""),
                    alert.get("company_name", ""),
                    alert["detected_at"],
                    alert["significance_score"],
                    alert.get("summary", ""),
                    json.dumps(alert.get("metrics", [])),
                    json.dumps(alert.get("metadata", {})),
                    datetime.now(timezone.utc).isoformat(),
                ))
            
            logger.info(f"Saved alert {alert['alert_id']} to database")
            return True
        
        except sqlite3.IntegrityError:
            logger.debug(f"Alert {alert['alert_id']} already exists in database")
            return False
        except Exception as e:
            logger.error(f"Failed to save alert: {e}", exc_info=True)
            return False
    
    def record_delivery(
        self,
        alert_id: str,
        user_id: str,
        channel: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """
        Record an alert delivery attempt.
        
        Args:
            alert_id: Alert identifier
            user_id: User identifier (email or user ID)
            channel: Delivery channel ("email", "webhook", "gui")
            status: Delivery status ("success", "failed", "pending")
            error_message: Error message if delivery failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alert_deliveries (
                        alert_id, user_id, channel, delivered_at, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert_id,
                    user_id,
                    channel,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    error_message,
                ))
            
            logger.info(f"Recorded delivery: alert={alert_id}, user={user_id}, status={status}")
        
        except Exception as e:
            logger.error(f"Failed to record delivery: {e}", exc_info=True)
    
    def get_prior_state(self, prior_key: Tuple) -> Optional[Dict[str, Any]]:
        """
        Load prior guidance state for comparison.
        
        Args:
            prior_key: Tuple key (typically company_name, period)
        
        Returns:
            Prior state dict or None if not found
        """
        if not prior_key:
            return None
        
        state_key = "|".join(str(p) for p in prior_key)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT metrics_json FROM guidance_prior_state WHERE state_key = ?",
                    (state_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
        
        except Exception as e:
            logger.error(f"Failed to load prior state: {e}", exc_info=True)
        
        return None
    
    def save_prior_state(self, prior_key: Tuple, candidate: Dict[str, Any]):
        """
        Save candidate as new prior state for future comparisons.
        
        Args:
            prior_key: Tuple key (typically company_name, period)
            candidate: Candidate dict containing metrics
        """
        if not prior_key:
            return
        
        state_key = "|".join(str(p) for p in prior_key)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO guidance_prior_state (
                        state_key, company_name, period, metrics_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    state_key,
                    candidate.get("company_name", ""),
                    candidate.get("period", ""),
                    json.dumps(candidate.get("metrics", [])),
                    datetime.now(timezone.utc).isoformat(),
                ))
            
            logger.debug(f"Saved prior state for key {state_key}")
        
        except Exception as e:
            logger.error(f"Failed to save prior state: {e}", exc_info=True)
    
    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
        
        Returns:
            List of alert dicts
        """
        alerts = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM alerts
                    ORDER BY detected_at DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor:
                    alert = dict(row)
                    alert["metrics"] = json.loads(alert.pop("metrics_json", "[]"))
                    alert["metadata"] = json.loads(alert.pop("metadata_json", "{}"))
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Failed to fetch recent alerts: {e}", exc_info=True)
        
        return alerts
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    prefs = dict(row)
                    prefs["watchlist"] = json.loads(prefs.pop("watchlist_json", "[]"))
                    prefs["alert_types"] = json.loads(prefs.pop("alert_types_json", "[]"))
                    prefs["delivery_channels"] = json.loads(prefs.pop("delivery_channels_json", "[]"))
                    return prefs
        
        except Exception as e:
            logger.error(f"Failed to fetch user preferences: {e}", exc_info=True)
        
        return None
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Save or update user preferences."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now(timezone.utc).isoformat()
                
                conn.execute("""
                    INSERT OR REPLACE INTO user_preferences (
                        user_id, email_address, telegram_chat_id, watchlist_json, alert_types_json,
                        min_significance, delivery_channels_json, webhook_url,
                        active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM user_preferences WHERE user_id = ?), ?),
                        ?)
                """, (
                    user_id,
                    preferences.get("email_address", ""),
                    preferences.get("telegram_chat_id", ""),
                    json.dumps(preferences.get("watchlist", [])),
                    json.dumps(preferences.get("alert_types", ["guidance_change"])),
                    preferences.get("min_significance", 0.7),
                    json.dumps(preferences.get("delivery_channels", ["email"])),
                    preferences.get("webhook_url", ""),
                    int(preferences.get("active", True)),
                    user_id,  # for COALESCE
                    now,
                    now,
                ))
            
            logger.info(f"Saved preferences for user {user_id}")
        
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}", exc_info=True)
    
    def save_feedback(self, alert_id: str, user_id: str, is_correct: bool, 
                     feedback_type: str | None = None, notes: str | None = None,
                     press_release_id: str | None = None) -> bool:
        """
        Save user feedback for an alert.
        
        Args:
            alert_id: Alert ID being rated
            user_id: User submitting feedback
            is_correct: True if alert was correct, False if incorrect
            feedback_type: Optional classification (e.g., 'true_positive', 'false_positive')
            notes: Optional free-text notes
            press_release_id: Optional press release ID for reference
            
        Returns:
            True if saved successfully
        """
        try:
            import uuid
            from datetime import datetime, timezone
            
            feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc).isoformat()
            
            # Auto-detect feedback_type if not provided
            if feedback_type is None:
                feedback_type = "true_positive" if is_correct else "false_positive"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alert_feedback (
                        feedback_id, alert_id, press_release_id, user_id, is_correct, 
                        feedback_type, notes, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id, alert_id, press_release_id, user_id, 
                    int(is_correct), feedback_type, notes, now
                ))
            
            logger.info(f"Saved feedback {feedback_id} for alert {alert_id}: {feedback_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feedback for alert {alert_id}: {e}", exc_info=True)
            return False
    
    def get_feedback_for_alert(self, alert_id: str) -> List[Dict[str, Any]]:
        """Get all feedback entries for a specific alert."""
        feedback = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM alert_feedback 
                    WHERE alert_id = ? 
                    ORDER BY submitted_at DESC
                """, (alert_id,))
                
                for row in cur.fetchall():
                    feedback.append(dict(row))
                    
        except Exception as e:
            logger.error(f"Failed to get feedback for alert {alert_id}: {e}", exc_info=True)
        
        return feedback
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get aggregated feedback statistics."""
        stats = {
            "total_feedback": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "by_type": {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM alert_feedback")
                stats["total_feedback"] = cur.fetchone()[0]
                
                cur = conn.execute("SELECT COUNT(*) FROM alert_feedback WHERE is_correct = 1")
                stats["correct"] = cur.fetchone()[0]
                
                cur = conn.execute("SELECT COUNT(*) FROM alert_feedback WHERE is_correct = 0")
                stats["incorrect"] = cur.fetchone()[0]
                
                if stats["total_feedback"] > 0:
                    stats["accuracy"] = stats["correct"] / stats["total_feedback"]
                
                # By type
                cur = conn.execute("""
                    SELECT feedback_type, COUNT(*) as count 
                    FROM alert_feedback 
                    GROUP BY feedback_type
                """)
                stats["by_type"] = {row[0]: row[1] for row in cur.fetchall()}
                
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}", exc_info=True)
        
        return stats
    
    def get_all_active_users(self) -> List[Dict[str, Any]]:
        """Get all active users with their preferences."""
        users = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM user_preferences WHERE active = 1"
                )
                
                for row in cursor:
                    user = dict(row)
                    user["watchlist"] = json.loads(user.pop("watchlist_json", "[]"))
                    user["alert_types"] = json.loads(user.pop("alert_types_json", "[]"))
                    user["delivery_channels"] = json.loads(user.pop("delivery_channels_json", "[]"))
                    users.append(user)
        
        except Exception as e:
            logger.error(f"Failed to fetch active users: {e}", exc_info=True)
        
        return users
