# tests/test_pollers.py
import pytest
from datetime import datetime

# ---- adjust this to your actual module path: -------------------------
import main as m
# -----------------------------------------------------------------------

class FakeEvent:
    """A minimal stand-in for core.event_types.Event."""
    def __init__(self, title="T", content="", source="X"):
        self.title      = title
        self.content    = content
        self.source     = source
        self.timestamp  = datetime.now()
        self.fetched_at = datetime.now()
        self.metadata   = {}

@pytest.fixture(autouse=True)
def stub_sleep(monkeypatch):
    """
    Break out of the infinite loops by having time.sleep() raise StopIteration.
    Since sleep() is *after* the try/except in each poller, StopIteration will
    bubble up and abort the loop.
    """
    monkeypatch.setattr(m.time, "sleep", lambda secs: (_ for _ in ()).throw(StopIteration))


def test_safe_add_event_invokes_QMetaObject(monkeypatch):
    from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
    from PyQt6.QtCore import QGenericArgument
    window = object()
    ev = FakeEvent()
    calls = []
    def fake_invoke(w, method, conn_type, arg):
        calls.append((w, method, conn_type, arg))
    monkeypatch.setattr(QMetaObject, "invokeMethod", fake_invoke)

    m.safe_add_event(window, ev)

    assert len(calls) == 1
    w, method, conn, arg = calls[0]
    assert w is window
    assert method == "add_event"
    ## We know the code passed a Q_ARG(object, ev), so arg should wrap our event:
    #assert hasattr(arg, 'data') or hasattr(arg, 'value'), "expected a Q_ARG wrapper"
    # We know the code passed a Q_ARG(object, ev), so arg should be a QGenericArgument
    assert isinstance(arg, QGenericArgument)


# ------- poll_gmail tests -------

def test_poll_gmail_happy(monkeypatch):
    window = object()
    evs = [FakeEvent("A"), FakeEvent("B")]
    monkeypatch.setattr(m, "fetch_recent_emails", lambda max_results: evs)
    monkeypatch.setattr(m, "insert_if_new", lambda e: True)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append((w,e)))

    # freeze time so no heartbeat
    monkeypatch.setattr(m.time, "time", lambda: 1_000)

    with pytest.raises(StopIteration):
        m.poll_gmail(window)

    # one call per event
    assert called == [(window, evs[0]), (window, evs[1])]


def test_poll_gmail_heartbeat(monkeypatch):
    window = object()
    # no emails ever
    monkeypatch.setattr(m, "fetch_recent_emails", lambda max_results: [])
    monkeypatch.setattr(m, "insert_if_new", lambda e: False)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append(e))

    # simulate:
    #   initial last_event_time = 0
    #   then now = 901 (> HEARTBEAT_INTERVAL=900)
    times = [0, 901, 901]
    monkeypatch.setattr(m.time, "time", lambda: times.pop(0))

    with pytest.raises(StopIteration):
        m.poll_gmail(window)

    # exactly one heartbeat event
    assert len(called) == 1
    hb = called[0]
    assert hb.source == "System"
    assert "No new Gmail events" in hb.title


def test_poll_gmail_error(monkeypatch):
    window = object()
    # fetch throws
    def boom(max_results): raise RuntimeError("oops")
    monkeypatch.setattr(m, "fetch_recent_emails", boom)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append(e))

    with pytest.raises(StopIteration):
        m.poll_gmail(window)

    # we should get one "Fetch error" event
    assert len(called) == 1
    err = called[0]
    assert err.source == "Gmail"
    assert "Fetch error" in err.title
    assert "oops" in err.content


# ------- poll_rss tests -------

def test_poll_rss_happy(monkeypatch):
    window = object()
    evs = [FakeEvent("R1"), FakeEvent("R2")]
    monkeypatch.setattr(m, "fetch_latest_di_headlines", lambda limit: evs)
    monkeypatch.setattr(m, "insert_if_new", lambda e: True)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append((w,e)))
    monkeypatch.setattr(m.time, "time", lambda: 42)

    with pytest.raises(StopIteration):
        m.poll_rss(window)

    assert called == [(window, evs[0]), (window, evs[1])]


def test_poll_rss_error(monkeypatch):
    window = object()
    def boom(limit): raise ValueError("uh-oh")
    monkeypatch.setattr(m, "fetch_latest_di_headlines", boom)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append(e))

    with pytest.raises(StopIteration):
        m.poll_rss(window)

    assert len(called) == 1
    e = called[0]
    assert e.source == "DI.se RSS"
    assert "RSS fetch error" in e.title
    assert "uh-oh" in e.content


# ------- poll_thomson_rss tests -------

def test_poll_thomson_happy(monkeypatch):
    window = object()
    evs = [FakeEvent("T1"), FakeEvent("T2")]
    monkeypatch.setattr(m, "fetch_thomson_rss", lambda: evs)
    monkeypatch.setattr(m, "insert_if_new", lambda e: True)

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append((w,e)))
    monkeypatch.setattr(m.time, "time", lambda: 123)

    with pytest.raises(StopIteration):
        m.poll_thomson_rss(window)

    assert called == [(window, evs[0]), (window, evs[1])]


def test_poll_thomson_error(monkeypatch):
    window = object()
    monkeypatch.setattr(m, "fetch_thomson_rss", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    called = []
    monkeypatch.setattr(m, "safe_add_event", lambda w,e: called.append(e))

    with pytest.raises(StopIteration):
        m.poll_thomson_rss(window)

    assert len(called) == 1
    e = called[0]
    assert e.source == "Thomson Reuters RSS"
    assert "RSS fetch error" in e.title
    assert "fail" in e.content


import sys
import pytest

if __name__ == "__main__":
    # argv[1:] lets you optionally pass extra pytest args, e.g. `python run_tests.py -q`
    sys.exit(pytest.main(["tests"] + sys.argv[1:]))