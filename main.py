from gui.app import run_gui
from core.event_types import Event
from sources.gmail_fetcher import fetch_unread_emails, test_fetch_any_email
from datetime import datetime
from threading import Thread
import time

POLL_INTERVAL = 60  # seconds
def poll_gmail(window):
    last_event_time = time.time()
    POLL_INTERVAL = 60  # seconds
    HEARTBEAT_INTERVAL = 300  # seconds without new events triggers a "still alive" message

    while True:
        try:
            emails = fetch_unread_emails(max_results=5)

            if emails:
                for e in emails:
                    window.add_event(e)
                last_event_time = time.time()
            else:
                if time.time() - last_event_time > HEARTBEAT_INTERVAL:
                    heartbeat = Event(
                        source="System",
                        title="No new Gmail events",
                        timestamp=datetime.now(),
                        content="Polling continues... no new messages found.",
                        metadata={}
                    )
                    window.add_event(heartbeat)
                    last_event_time = time.time()  # reset to avoid repeated heartbeat

        except Exception as e:
            error_event = Event(
                source="Gmail",
                title="Fetch error",
                timestamp=datetime.now(),
                content=str(e),
                metadata={}
            )
            window.add_event(error_event)

        time.sleep(POLL_INTERVAL)


from sources.rss_sources import fetch_latest_di_headlines

def poll_rss(window):
    POLL_INTERVAL = 120  # check every 2 minutes
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    seen_links = set()
    last_event_time = time.time()

    while True:
        try:
            headlines = fetch_latest_di_headlines(limit=5)
            new_events = 0

            for event in headlines:
                link = event.metadata.get("link")
                if link not in seen_links:
                    window.add_event(event)
                    seen_links.add(link)
                    new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            else:
                if time.time() - last_event_time > HEARTBEAT_INTERVAL:
                    heartbeat = Event(
                        source="DI.se RSS",
                        title="No new RSS items",
                        timestamp=datetime.now(),
                        content="Still polling DI.se — no new articles detected.",
                        metadata={}
                    )
                    window.add_event(heartbeat)
                    last_event_time = time.time()

        except Exception as e:
            error_event = Event(
                source="DI.se RSS",
                title="RSS fetch error",
                timestamp=datetime.now(),
                content=str(e),
                metadata={}
            )
            window.add_event(error_event)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    
    #test_fetch_any_email(max_results=3)
     
    app, window = run_gui()

    # Example fake event
    fake_event = Event(
        source="System",
        title="Startup Complete",
        timestamp=datetime.now(),
        content="GUI initialized and ready to receive events.",
        metadata={}
    )
    window.add_event(fake_event)

    # Start background Gmail polling
    gmail_thread = Thread(target=poll_gmail, args=(window,), daemon=True)
    
    # Start background RSS polling
    di_rss_thread = Thread(target=poll_rss, args=(window,), daemon=True)
    
    gmail_thread.start()
    di_rss_thread.start()


    app.exec()
