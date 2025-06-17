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
    thread = Thread(target=poll_gmail, args=(window,), daemon=True)
    thread.start()

    app.exec()
