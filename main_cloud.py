import os
import time
import logging
import json
from datetime import datetime
#from threading import Thread
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from httplib2 import ServerNotFoundError
import socket 
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
RETRIES            = 3
BACKOFF            = 5          # seconds between DVC or DNS retries

# def poll_gmail():
#     """
#     Periodically fetch recent Gmail messages and store new events.
#     """
#     last_event_time = time.time()
#     while True:
#         try:
#             emails = fetch_recent_emails(max_results=100)
#             new_events = 0
#             for e in emails:
#                 if insert_if_new(e):
#                     logging.info(f"[Gmail] New event: {e.title} at {e.fetched_at}. Sent from {e.metadata['company']}.")
#                     new_events += 1

#             if new_events > 0:
#                 last_event_time = time.time()
#             elif time.time() - last_event_time > HEARTBEAT_INTERVAL:
#                 logging.info("[Gmail] No new events in the last interval.")
#                 last_event_time = time.time()
#         except Exception:
#             logging.exception("[Gmail] Error fetching emails:")

#         time.sleep(POLL_INTERVAL_GMAIL)


def poll_gmail_forever():
    last_event_time = time.time()

    while True:
        # --- Retry-aware fetch step ---
        for attempt in range(1, RETRIES+1):
            try:
                logging.info(f"[Gmail] Fetch attempt {attempt}/{RETRIES}")
                emails = fetch_recent_emails(max_results=100)
                break
            except (ServerNotFoundError, socket.gaierror) as e:
                logging.warning(f"[Gmail] DNS error: {e}")
                if attempt < RETRIES:
                    time.sleep(BACKOFF)
                else:
                    logging.error("[Gmail] All DNS retries failed; skipping this cycle")
                    emails = []
            except Exception:
                logging.exception("[Gmail] Unexpected error; skipping this cycle")
                emails = []
                break

        # --- Process results & heartbeat ---
        new_events = 0
        for e in emails:
            if insert_if_new(e):
                logging.info(f"[Gmail] New event: {e.title}")
                new_events += 1

        now = time.time()
        if new_events:
            last_event_time = now
        elif now - last_event_time > HEARTBEAT_INTERVAL:
            logging.info("[Gmail] No new events in the last hour.")
            last_event_time = now

        # --- Wait for next cycle ---
        time.sleep(POLL_INTERVAL_GMAIL)


def poll_rss_forever(limit=100):
    last_event_time = time.time()
    while True:
        # --- Retry logic around fetch ---
        for attempt in range(1, RETRIES+1):
            try:
                logging.info(f"[DI.se RSS] Fetch attempt {attempt}/{RETRIES}")
                headlines = fetch_latest_di_headlines(limit=limit)
                break
            except Exception as e:
                logging.warning(f"[DI.se RSS] Fetch failed (attempt {attempt}): {e}")
                if attempt < RETRIES:
                    time.sleep(BACKOFF)
                else:
                    logging.error("[DI.se RSS] All fetch attempts failed; skipping this cycle")
                    headlines = []
        # --- Process results & heartbeat ---
        new_events = 0
        for ev in headlines:
            if insert_if_new(ev):
                logging.info(
                    f"[DI.se RSS] New headline: {ev.title} at {ev.fetched_at}. "
                    f"Company matches: {ev.metadata['matches']}."
                )
                new_events += 1

        now = time.time()
        if new_events > 0:
            last_event_time = now
        elif now - last_event_time > HEARTBEAT_INTERVAL:
            logging.info("[DI.se RSS] No new items in the last interval.")
            last_event_time = now

        # wait for next poll
        time.sleep(POLL_INTERVAL_RSS)


def poll_thomson_rss_forever(limit=100):
    last_event_time = time.time()
    while True:
        # --- Retry logic around fetch ---
        for attempt in range(1, RETRIES+1):
            try:
                logging.info(f"[Thomson RSS] Fetch attempt {attempt}/{RETRIES}")
                items = fetch_thomson_rss(limit=limit)
                break
            except Exception as e:
                logging.warning(f"[Thomson RSS] Fetch failed (attempt {attempt}): {e}")
                if attempt < RETRIES:
                    time.sleep(BACKOFF)
                else:
                    logging.error("[Thomson RSS] All fetch attempts failed; skipping this cycle")
                    items = []

        # --- Process results & heartbeat ---
        new_events = 0
        for ev in items:
            if insert_if_new(ev):
                logging.info(
                    f"[Thomson RSS] New item: {ev.title} at {ev.fetched_at}. "
                    f"Company matches: {ev.metadata['matches']}."
                )
                new_events += 1

        now = time.time()
        if new_events > 0:
            last_event_time = now
        elif now - last_event_time > HEARTBEAT_INTERVAL:
            logging.info("[Thomson RSS] No new items in the last interval.")
            last_event_time = now

        # wait for next poll
        time.sleep(POLL_INTERVAL_RSS)


def poll_gmail(max_results=100, retries=3, backoff=5):
    """
    Poll Gmail up to max_results, retrying on DNS failures.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"[Gmail] Poll attempt {attempt}/{retries}")
            emails = fetch_recent_emails(max_results=max_results)
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
            
            return
        
        except ServerNotFoundError as e:
            logging.warning(f"[Gmail] DNS error on attempt {attempt}: {e}")
        except socket.gaierror as e:
            logging.warning(f"[Gmail] socket.gaierror on attempt {attempt}: {e}")
        except Exception as e:
            logging.error(f"[Gmail] Unexpected error: {e}", exc_info=True)
            return
        if attempt < retries:
            time.sleep(backoff)
    logging.error("[Gmail] All retry attempts failed; skipping this cycle")


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

def push_db_with_retries(
    db_path="events.db",
    remote="mygdrive",
    retries=3,
    delay=5
):
    """
    Attempt to dvc add/commit + push events.db up to `retries` times,
    with `delay` seconds between attempts. Returns True on success.
    """
    dvc_meta = f"{db_path}.dvc"
    for attempt in range(1, retries + 1):
        start = time.time()
        try:
            logging.info(f"DB-push attempt {attempt}/{retries}")

            # 1) Track or commit the file in DVC
            if not os.path.exists(dvc_meta):
                logging.info("Tracking new events.db with `dvc add`")
                subprocess.check_call(["dvc", "add", db_path])
            else:
                logging.info("Updating existing DVC entry with `dvc commit`")
                subprocess.check_call(["dvc", "commit", db_path])

            # 2) Push to remote
            subprocess.check_call(["dvc", "push", "-r", remote])

            elapsed = time.time() - start
            logging.info(f"DB-push succeeded in {elapsed:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            logging.warning(
                f"DB-push step failed on attempt {attempt} "
                f"after {elapsed:.1f}s: {e}"
            )
            if attempt < retries:
                time.sleep(delay)
            else:
                logging.error("All DB-push attempts failed; giving up.")
                return False

        except Exception:
            logging.exception("Unexpected error during DB-push; aborting retries")
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
    for target in (poll_gmail_forever, poll_rss_forever, poll_thomson_rss_forever):
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


if __name__ == "__main__":
    main()
