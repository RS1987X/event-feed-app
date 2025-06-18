from gui.app import run_gui
from core.event_types import Event
from sources.gmail_fetcher import fetch_recent_emails, test_fetch_any_email
from datetime import datetime
from threading import Thread
import time
import logging
from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
from core.oltp_store import insert_if_new, init_db

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def safe_add_event(window, event):
    QMetaObject.invokeMethod(
        window,
        "add_event",
        Qt.ConnectionType.QueuedConnection,
        Q_ARG(object, event)
    )

def poll_gmail(window):
    last_event_time = time.time()
    POLL_INTERVAL = 60  # seconds
    HEARTBEAT_INTERVAL = 900  # seconds without new events triggers a "still alive" message

    while True:
        try:
            emails = fetch_recent_emails(max_results=5)

            if emails:
                for e in emails:
                    if insert_if_new(e):
                        safe_add_event(window, e)  # Use safe_add_event to ensure thread safety
                    #window.add_event(e)
                last_event_time = time.time()
            else:
                if time.time() - last_event_time > HEARTBEAT_INTERVAL:
                    heartbeat = Event(
                            source="System",
                            title="No new Gmail events",
                            timestamp=datetime.now(),
                            fetched_at=datetime.now(),
                            content="…",
                            metadata={}
                        )
                    safe_add_event(window, heartbeat)  # Use safe_add_event to ensure thread safety
                    #window.add_event(heartbeat)
                    last_event_time = time.time()  # reset to avoid repeated heartbeat

        except Exception as e:
            error_event = Event(
                source="Gmail",
                title="Fetch error",
                timestamp=datetime.now(),
                fetched_at=datetime.now(),
                content=str(e),
                metadata={}
            )
            safe_add_event(window, error_event)  # Use safe_add_event to ensure thread safety
            #window.add_event(error_event)

        time.sleep(POLL_INTERVAL)


from sources.rss_sources import fetch_latest_di_headlines

def poll_rss(window):
    POLL_INTERVAL = 120  # check every 2 minutes
    HEARTBEAT_INTERVAL = 900  # 5 minutes
    seen_links = set()
    last_event_time = time.time()

    while True:
        try:
            headlines = fetch_latest_di_headlines(limit=10)
            new_events = 0

            for event in headlines:
                if insert_if_new(event):
                    safe_add_event(window, event)
                    new_events += 1
                # link = event.metadata.get("link")
                # if link not in seen_links:
                #     safe_add_event(window, event)  # Use safe_add_event to ensure thread safety
                #     #window.add_event(event)
                #     seen_links.add(link)
                #     new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            elif time.time() - last_event_time > HEARTBEAT_INTERVAL:
                
                heartbeat = Event(
                    source="DI.se RSS",
                    title="No new RSS items",
                    timestamp=datetime.now(),
                    fetched_at=datetime.now(),  # ← add this line
                    content="Still polling DI.se — no new articles detected.",
                    metadata={}
                )
                safe_add_event(window, heartbeat)  # Use safe_add_event to ensure thread safety
                    # window.add_event(heartbeat)
                last_event_time = time.time()

        except Exception as e:
            error_event = Event(
                source="DI.se RSS",
                title="RSS fetch error",
                timestamp=datetime.now(),
                fetched_at=datetime.now(),
                content=str(e),
                metadata={}
            )
            safe_add_event(window, error_event)  # Use safe_add_event to ensure thread safety
            #window.add_event(error_event)

        time.sleep(POLL_INTERVAL)


from sources.rss_sources import fetch_thomson_rss

def poll_thomson_rss(window):
    POLL_INTERVAL = 120             # every 2 minutes
    HEARTBEAT_INTERVAL = 900        # 5 minutes
    seen_links = set()
    last_event_time = time.time()

    while True:
        try:
            headlines = fetch_thomson_rss()
            new_events = 0

            for event in headlines:
                if insert_if_new(event):
                    safe_add_event(window, event)
                    new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            elif time.time() - last_event_time >= HEARTBEAT_INTERVAL:
                heartbeat = Event(
                    source="Thomson Reuters RSS",
                    title="No new RSS items",
                    timestamp=datetime.now(),
                    fetched_at=datetime.now(),  # ← add this line
                    content="Still polling Thomson Reuters — no new press releases.",
                    metadata={}
                )
                safe_add_event(window, heartbeat)  # Use safe_add_event to ensure thread safety
                #window.add_event(heartbeat)
                last_event_time = time.time()

        except Exception as e:
            error_event = Event(
                source="Thomson Reuters RSS",
                title="RSS fetch error",
                timestamp=datetime.now(),
                fetched_at=datetime.now(),
                content=str(e),
                metadata={}
            )
            safe_add_event(window, error_event)  # Use safe_add_event to ensure thread safety
            #window.add_event(error_event)
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    # ensure our events table exists
    init_db()
    #test_fetch_any_email(max_results=3)
     
    app, window = run_gui()
    print("GUI created")  # Debug print
    # Example fake event
    fake_event = Event(
        source="System",
        title="Startup Complete",
        timestamp=datetime.now(),
        fetched_at=datetime.now(),  # ← add this line
        content="GUI initialized and ready to receive events.",
        metadata={}
    )
    safe_add_event(window, fake_event)

    # Start background Gmail polling
    gmail_thread = Thread(target=poll_gmail, args=(window,), daemon=True)

    # Start background RSS polling
    di_rss_thread = Thread(target=poll_rss, args=(window,), daemon=True)

     # Start background RSS polling
    thomson_rss_thread = Thread(target=poll_thomson_rss, args=(window,), daemon=True)

    gmail_thread.start()
    di_rss_thread.start()
    thomson_rss_thread.start()

    print("Starting event loop")
    app.exec()
