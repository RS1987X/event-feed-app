import os
import time
import logging
import json
from datetime import datetime
#from threading import Thread
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


from core.event_types import Event
from sources.gmail_fetcher import fetch_recent_emails
from sources.rss_sources import fetch_latest_di_headlines, fetch_thomson_rss
from core.oltp_store import insert_if_new, init_db

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

# Polling intervals (in seconds)
POLL_INTERVAL_GMAIL = 60
POLL_INTERVAL_RSS = 120
HEARTBEAT_INTERVAL = 900


def poll_gmail():
    """
    Periodically fetch recent Gmail messages and store new events.
    """
    last_event_time = time.time()
    while True:
        try:
            emails = fetch_recent_emails(max_results=100)
            new_events = 0
            for e in emails:
                if insert_if_new(e):
                    logging.info(f"[Gmail] New event: {e.title} at {e.fetched_at}. Sent from {e.metadata['company']}.")
                    new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            elif time.time() - last_event_time > HEARTBEAT_INTERVAL:
                logging.info("[Gmail] No new events in the last interval.")
                last_event_time = time.time()
        except Exception:
            logging.exception("[Gmail] Error fetching emails:")

        time.sleep(POLL_INTERVAL_GMAIL)


def poll_rss():
    """
    Periodically fetch DI.se RSS headlines and store new events.
    """
    last_event_time = time.time()
    while True:
        try:
            headlines = fetch_latest_di_headlines(limit=100)
            new_events = 0
            for ev in headlines:
                if insert_if_new(ev):
                    logging.info(f"[DI.se RSS] New headline: {ev.title} at {ev.fetched_at}. Company matches: {ev.metadata['matches']}." )
                    new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            elif time.time() - last_event_time > HEARTBEAT_INTERVAL:
                logging.info("[DI.se RSS] No new items in the last interval.")
                last_event_time = time.time()
        except Exception:
            logging.exception("[DI.se RSS] Error fetching headlines:")

        time.sleep(POLL_INTERVAL_RSS)


def poll_thomson_rss():
    """
    Periodically fetch Thomson Reuters RSS and store new events.
    """
    last_event_time = time.time()
    while True:
        try:
            items = fetch_thomson_rss(limit=100)
            new_events = 0
            for ev in items:
                if insert_if_new(ev):
                    logging.info(f"[Thomson RSS] New item: {ev.title} at {ev.fetched_at}. Company matches: {ev.metadata['matches']}.")
                    new_events += 1

            if new_events > 0:
                last_event_time = time.time()
            elif time.time() - last_event_time > HEARTBEAT_INTERVAL:
                logging.info("[Thomson RSS] No new items in the last interval.")
                last_event_time = time.time()
        except Exception:
            logging.exception("[Thomson RSS] Error fetching Thomson Reuters feed:")

        time.sleep(POLL_INTERVAL_RSS)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # silence default logging
        return


def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("", port), HealthHandler)
    logging.info(f"Starting health endpoint on port {port}")
    server.serve_forever()

import subprocess
#import time
#import logging

def pull_with_retries(remote="mygdrive", retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"DVC pull attempt {attempt}/{retries}")
            subprocess.check_call(["dvc", "pull", "--force", "-r", remote])
            logging.info("DVC pull succeeded")
            return
        except subprocess.CalledProcessError as e:
            logging.warning(f"DVC pull failed (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                logging.error("All DVC pull attempts failed, aborting startup")
                raise

# ── DVC PUSH WITH RETRIES ──────────────────────────────────────────────────────
def push_db_with_retries(db_path="events.db", remote="mygdrive", retries=3, delay=5):
    for attempt in range(1, retries + 1):
        start = time.time()
        try:
            logging.info(f"DB-push attempt {attempt}/{retries}")
            subprocess.check_call(["dvc", "add", db_path])
            subprocess.check_call(["dvc", "push", "-r", remote])
            logging.info(f"DB-push succeeded in {time.time() - start:.1f}s")
            return True
        except subprocess.CalledProcessError as e:
            logging.warning(f"DB-push failed (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                logging.error("All DB-push attempts failed")
                return False

def periodic_push(interval_minutes=30, **kwargs):
    """
    Runs push_db_with_retries every interval_minutes in a daemon thread.
    """
    interval = interval_minutes * 60
    logging.info(f"Starting periodic DB-push every {interval_minutes} minutes.")
    while True:
        success = push_db_with_retries(**kwargs)
        # even on success, wait full interval before next
        time.sleep(interval)

def main():
    # 1) Pull latest data
    #os.system("dvc pull --force")
    pull_with_retries()
    # 2) Initialize DB & start pollers
    init_db()
    logging.info("Initialized DB, starting pollers.")
    for target in (poll_gmail, poll_rss, poll_thomson_rss):
        t = threading.Thread(target=target, daemon=True)
        t.start()

    
    # 3) Start periodic DB-push every 30 minutes
    threading.Thread(
        target=periodic_push,
        kwargs={
            "interval_minutes": 30,
            "db_path": "events.db",
            "remote": "mygdrive",
            "retries": 3,
            "delay": 5,
        },
        daemon=True,
    ).start()


    # 3) Start HTTP server (blocks forever but leaves threads alive)
    run_http_server()


# def main():
#     # Initialize or migrate the SQLite database schema
#     init_db()
#     logging.info("Initialized database, starting headless event polling service.")

#     # Start polling threads
#     threads = [
#         Thread(target=poll_gmail, daemon=True),
#         Thread(target=poll_rss, daemon=True),
#         Thread(target=poll_thomson_rss, daemon=True),
#     ]
#     for t in threads:
#         t.start()

#     # Keep the main thread alive
#     try:
#         while True:
#             time.sleep(3600)
#     except KeyboardInterrupt:
#         logging.info("Shutting down polling service.")


if __name__ == "__main__":
    main()
