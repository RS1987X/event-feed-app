import sqlite3
import hashlib
from pathlib import Path
from core.event_types import Event

DB_PATH = Path("data/oltp/events.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                company TEXT,
                ticker TEXT
            )
        """)

def _make_event_id(event: Event) -> str:
    # Deterministic hash from content or URL
    base = event.title + event.content
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def insert_if_new(event: Event) -> bool:
    event_id = _make_event_id(event)
    #url = event.metadata.get("url")
    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute("""
                INSERT INTO events (
                    id, title, content, source, timestamp,
                    fetched_at, company, ticker
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event.title,
                event.content,
                event.source,
                event.timestamp.isoformat(),
                event.fetched_at.isoformat(),
                event.metadata.get("company", ""),
                event.metadata.get("ticker", "")
            ))
            return True
        except sqlite3.IntegrityError:
            return False  # Duplicate (same id or url)
