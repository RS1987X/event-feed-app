import sqlite3
import hashlib
from pathlib import Path
from core.event_types import Event

DB_PATH = Path("data/oltp/events.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# def init_db():
#     with sqlite3.connect(DB_PATH) as conn:
#         conn.execute("""
#             CREATE TABLE IF NOT EXISTS events (
#                 id TEXT PRIMARY KEY,
#                 title TEXT NOT NULL,
#                 content TEXT NOT NULL,
#                 source TEXT,
#                 timestamp TEXT NOT NULL,
#                 fetched_at TEXT NOT NULL,
#                 company TEXT,
#                 ticker TEXT
#             )
#         """)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # 1) New events table (no company/ticker columns)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id          TEXT    PRIMARY KEY,
            source      TEXT    NOT NULL,
            title       TEXT    NOT NULL,
            summary     TEXT,
            body        TEXT,
            url         TEXT,
            event_type  TEXT,
            timestamp   TEXT    NOT NULL,
            fetched_at  TEXT    NOT NULL
        )
        """)

        # 2) Companies lookup table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name   TEXT    UNIQUE NOT NULL,
            ticker TEXT
        )
        """)

        # 3) Junction table for event↔company links
        conn.execute("""
        CREATE TABLE IF NOT EXISTS event_company (
            event_id       TEXT    NOT NULL
                                 REFERENCES events(id)
                                 ON DELETE CASCADE,
            company_id     INTEGER NOT NULL
                                 REFERENCES companies(id)
                                 ON DELETE CASCADE,
            matched_phrase TEXT,
            PRIMARY KEY(event_id, company_id)
        )
        """)


def _make_event_id(event: Event) -> str:
    # Deterministic hash from content or URL
    base = event.title + event.content
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

# def insert_if_new(event: Event) -> bool:
#     event_id = _make_event_id(event)
#     #url = event.metadata.get("url")
#     with sqlite3.connect(DB_PATH) as conn:
#         try:
#             conn.execute("""
#                 INSERT INTO events (
#                     id, title, content, source, timestamp,
#                     fetched_at, company, ticker
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 event_id,
#                 event.title,
#                 event.content,
#                 event.source,
#                 event.timestamp.isoformat(),
#                 event.fetched_at.isoformat(),
#                 event.metadata.get("company", ""),
#                 event.metadata.get("ticker", "")
#             ))
#             return True
#         except sqlite3.IntegrityError:
#             return False  # Duplicate (same id or url)



def insert_if_new(event: Event) -> bool:
    eid = _make_event_id(event)
    with sqlite3.connect(DB_PATH) as conn:
        try:
            # 1) insert event row
            conn.execute("""
              INSERT INTO events (
                id, source, title, summary, body, url, event_type,
                timestamp, fetched_at
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
              eid,
              event.source,
              event.title,
              event.metadata.get("summary", ""),
              event.content,
              event.metadata.get("link", ""),
              event.metadata.get("event_type", ""),
              event.timestamp.isoformat(),
              event.fetched_at.isoformat(),
            ))

            # 2) for each matched company, upsert into companies & link
            for comp_name, matched_phrase in event.metadata.get("matches", []):
                # upsert into companies
                ticker = event.metadata.get("tickers", {}).get(comp_name, "")
                conn.execute(
                  "INSERT OR IGNORE INTO companies(name, ticker) VALUES (?, ?)",
                  (comp_name, ticker)
                )
                # fetch its ID
                company_id = conn.execute(
                  "SELECT id FROM companies WHERE name = ?",
                  (comp_name,)
                ).fetchone()[0]

                # insert into junction
                conn.execute(
                  "INSERT OR IGNORE INTO event_company(event_id, company_id, matched_phrase)"
                  " VALUES (?, ?, ?)",
                  (eid, company_id, matched_phrase)
                )

            return True

        except sqlite3.IntegrityError:
            # event id already exists
            return False