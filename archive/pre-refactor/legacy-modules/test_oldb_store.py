import sqlite3
from pathlib import Path
from datetime import datetime

import pytest

import core.oltp_store as store

from core.event_types import Event


def make_event(title="Test", content="Content", **kwargs):
    """Helper to build an Event with sane defaults."""
    return Event(
        title=title,
        content=content,
        source=kwargs.get("source", "unit-test"),
        timestamp=kwargs.get("timestamp", datetime.now()),
        fetched_at=kwargs.get("fetched_at", datetime.now()),
        metadata=kwargs.get("metadata", {})
    )


def test_init_db_creates_table(tmp_path, monkeypatch):
    # Redirect the DB into our temp directory
    db_file = tmp_path / "events.db"
    monkeypatch.setattr(store, "DB_PATH", db_file)
    # Ensure parent exists
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Run init_db
    store.init_db()

    # Connect directly and check sqlite_master for our table
    conn = sqlite3.connect(db_file)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    )
    assert cur.fetchone() is not None, "events table should exist after init_db()"


def test_insert_if_new_and_duplicates(tmp_path, monkeypatch):
    # Point to a fresh DB
    db_file = tmp_path / "events.db"
    monkeypatch.setattr(store, "DB_PATH", db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    store.init_db()

    # First insert of same Event → True
    e1 = make_event()
    assert store.insert_if_new(e1) is True

    # Second insert of same Event → False
    assert store.insert_if_new(e1) is False

    # A different event (different content) → True
    e2 = make_event(content="Other content")
    assert store.insert_if_new(e2) is True

    # Verify rows actually in DB
    conn = sqlite3.connect(db_file)
    rows = list(conn.execute("SELECT id, title, content FROM events"))
    assert len(rows) == 2
    ids = {row[0] for row in rows}
    # The two events' IDs should be distinct
    assert len(ids) == 2


def test_make_event_id_is_deterministic_and_sensitive():
    e1 = make_event(title="A", content="1")
    e1b = make_event(title="A", content="1")
    e2 = make_event(title="A", content="2")

    id1 = store._make_event_id(e1)
    id1b = store._make_event_id(e1b)
    id2 = store._make_event_id(e2)

    # Same title+content → same ID
    assert id1 == id1b

    # Different content → different ID
    assert id1 != id2

    # IDs should look like 64-hex characters
    assert len(id1) == 64
    int(id1, 16)  # should parse as hex without error



if __name__ == "__main__":
    # allows running this file directly: python tests/test_oltp_store.py
    import sys
    sys.exit(pytest.main([__file__]))