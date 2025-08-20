import os, json, base64, logging, io
import time, random
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Optional
import re
from datetime import timedelta

from bs4 import BeautifulSoup
import polars as pl

from email import policy
from email.parser import BytesParser

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import secretmanager, firestore, storage


# ────────────────────────────────────────────────────────────────────────────────
# Config (from env)

PROJECT_ID = os.environ.get("PROJECT_ID", "event-feed-app-463206")
BUCKET = os.environ.get("GCS_BUCKET", "event-feed-app-data")
GMAIL_LABEL = os.environ.get("GMAIL_LABEL", "INBOX")
SCHEMA_VER = 1
PARSER_VER = 1
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
LOOKBACK_SECS = int(os.environ.get("LOOKBACK_SECS", "120"))  # 2 minutes


# Backfill knobs (env-configurable)
BACKFILL_START_ISO = os.environ.get("BACKFILL_START_ISO")  # e.g. "2025-01-01"
BACKFILL_WINDOW_DAYS = int(os.environ.get("BACKFILL_WINDOW_DAYS", "1"))  # 1-day slices
BACKFILL_MAX_RAW_PER_RUN = int(os.environ.get("BACKFILL_MAX_RAW_PER_RUN", "300"))  # cap raw downloads per run
BACKFILL_LABEL = os.environ.get("BACKFILL_LABEL", GMAIL_LABEL)  # typically "INBOX"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ────────────────────────────────────────────────────────────────────────────────
# Secrets / Auth

def _secret_text(name: str) -> str:
    sm = secretmanager.SecretManagerServiceClient()
    rn = f"projects/{PROJECT_ID}/secrets/{name}/versions/latest"
    return sm.access_secret_version(request={"name": rn}).payload.data.decode()

def gmail_creds_from_existing_secrets() -> Credentials:
    client_json = json.loads(_secret_text("GMAIL_OAUTH_JSON"))
    client_blob = client_json.get("installed") or client_json.get("web") or {}
    token_json = json.loads(_secret_text("GMAIL_OAUTH_CREDENTIALS"))
    return Credentials(
        None,
        refresh_token=token_json["refresh_token"],
        token_uri=token_json.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=client_blob["client_id"],
        client_secret=client_blob["client_secret"],
        scopes=GMAIL_SCOPES,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Firestore watermark (epoch + tie-breaker id)

def load_watermark() -> Tuple[int, str]:
    """Return (last_epoch_ms, last_msg_id_at_epoch). Backward-compatible with seconds."""
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return 0, ""
    d = doc.to_dict() or {}
    if "last_internal_epoch_ms" in d:
        return int(d.get("last_internal_epoch_ms", 0)), d.get("last_msg_id_at_epoch", "")
    sec = int(d.get("last_internal_epoch", 0))
    return (sec * 1000 if sec > 0 else 0), d.get("last_msg_id_at_epoch", "")

def save_watermark(epoch_ms: int, last_id: str) -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {
            "last_internal_epoch_ms": int(epoch_ms),
            "last_msg_id_at_epoch": last_id,
            "last_internal_epoch": int(epoch_ms // 1000),
        },
        merge=True,
    )


def load_backfill_cursor():
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return None
    d = doc.to_dict() or {}
    return d.get("backfill_before_epoch_ms")

def save_backfill_cursor(before_epoch_ms: Optional[int]):
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {"backfill_before_epoch_ms": before_epoch_ms if before_epoch_ms else None},
        merge=True,
    )

def mark_backfill_done():
    save_backfill_cursor(None)

# Firestore history id helpers
def load_history_id() -> Optional[str]:
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return None
    d = doc.to_dict() or {}
    return d.get("last_history_id")


def save_history_id(history_id: str) -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {"last_history_id": str(history_id)},
        merge=True,
    )

def get_current_history_id(svc) -> Optional[str]:
    """Return the mailbox's current historyId via users.getProfile()."""
    resp = _call_with_retry(lambda: svc.users().getProfile(userId="me").execute(), what="GetProfile")
    return resp.get("historyId")

def fetch_history_ids(svc, start_history_id: str) -> Tuple[set, Optional[str]]:
    """Collect message ids from history.list(startHistoryId=...).
    Returns (set_of_ids, last_history_id_seen)."""
    ids: set = set()
    page_token: Optional[str] = None
    last_history_id = start_history_id

    # Resolve label id once
    label_id = get_label_id(svc, GMAIL_LABEL) or GMAIL_LABEL

    while True:
        def call():
            return svc.users().history().list(
                userId="me",
                startHistoryId=start_history_id,
                pageToken=page_token,
                # include both to catch late labeling into your target label
                historyTypes=["messageAdded", "labelAdded"],
                labelId=label_id,
                maxResults=500,
                fields=("history("
                        "messagesAdded(message/id),"
                        "labelsAdded(message/id)"
                        "),historyId,nextPageToken"),
                quotaUser="gmail-fetcher",
            ).execute()

        resp = _call_with_retry(call, what="HistoryList")

        for h in resp.get("history", []):
            for ma in h.get("messagesAdded", []):
                m = (ma.get("message") or {})
                mid = m.get("id")
                if mid:
                    ids.add(mid)
            for la in h.get("labelsAdded", []):
                m = (la.get("message") or {})
                mid = m.get("id")
                if mid:
                    ids.add(mid)

        last_history_id = resp.get("historyId", last_history_id)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return ids, last_history_id


def load_cooldown_utc() -> Optional[datetime]:
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return None
    s = (doc.to_dict() or {}).get("cooldown_until_utc")
    try:
        return datetime.fromisoformat(s) if s else None
    except Exception:
        return None

def save_cooldown_utc(until_dt_utc: datetime) -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {"cooldown_until_utc": until_dt_utc.astimezone(timezone.utc).isoformat()},
        merge=True,
    )

def clear_cooldown_utc() -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {"cooldown_until_utc": None},
        merge=True,
    )

def _parse_retry_after_abs_utc(e: HttpError) -> Optional[datetime]:
    """
    Returns absolute UTC datetime to wait until, if Gmail provided one.
    Supports: Retry-After header (seconds) OR JSON body 'Retry after 2025-..Z'.
    """
    # Header in seconds
    try:
        ra = e.resp.get('retry-after')
        if ra:
            try:
                secs = float(ra)
                return datetime.now(timezone.utc) + timedelta(seconds=secs)
            except Exception:
                pass
    except Exception:
        pass

    # JSON body absolute time (seen in Gmail responses)
    try:
        body = getattr(e, "content", b"") or b""
        s = body.decode("utf-8", "ignore")
        m = re.search(r"Retry after (\d{4}-\d{2}-\d{2}T[\d:.]+Z)", s)
        if m:
            ts = m.group(1)  # RFC3339 with Z
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass

    return None

# ────────────────────────────────────────────────────────────────────────────────
# Gmail helpers (retry, query, parsing)
class CooldownActive(Exception):
    def __init__(self, until_utc: datetime):
        super().__init__(f"Cooldown until {until_utc.isoformat()}")
        self.until_utc = until_utc

COOLDOWN_PAD_SECS = int(os.environ.get("COOLDOWN_PAD_SECS", "15"))

def save_cooldown_max(until_dt_utc: datetime) -> None:
    """
    Save a cooldown that is the max of the current saved value and `until_dt_utc`,
    with a small padding to avoid boundary races.
    """
    target = (until_dt_utc + timedelta(seconds=COOLDOWN_PAD_SECS)).astimezone(timezone.utc)
    current = load_cooldown_utc()
    if current is None or target > current:
        save_cooldown_utc(target)

def _call_with_retry(req_exec, what: str, max_tries: int = 6):
    delay = 0.5
    last_error = None
    for attempt in range(1, max_tries + 1):
        try:
            return req_exec()
        except HttpError as e:
            last_error = e
            status = getattr(e.resp, "status", None)
            if status in (429, 500, 502, 503, 504):
                until = _parse_retry_after_abs_utc(e)
                if until and datetime.now(timezone.utc) < until:
                    save_cooldown_max(until)
                    logging.warning(
                        "[gmail] %s status=%s attempt=%d/%d; honoring Retry-After until %s UTC → abort run",
                        what, status, attempt, max_tries, until.isoformat(timespec="seconds")
                    )
                    # Do NOT sleep long; abort this run so scheduler + global gate handle the wait.
                    raise CooldownActive(until)
                else:
                    # Fallback exponential backoff with jitter
                    retry_after = 0.0
                    try:
                        retry_after = float(e.resp.get('retry-after', 0))
                    except Exception:
                        pass
                    sleep_for = retry_after if retry_after > 0 else delay + random.random() * 0.2
                    logging.warning("[gmail] %s status=%s attempt=%d/%d; backoff=%.2fs",
                                    what, status, attempt, max_tries, sleep_for)
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 60.0)
                continue
            logging.error("[gmail] %s non-retryable status=%s body=%r",
                          what, status, getattr(e, "content", b"")[:500])
            raise
    if last_error is not None:
        logging.error("[gmail] %s failed after %d attempts. Last error status=%s body=%r",
                      what, max_tries, getattr(last_error.resp, "status", "?"),
                      getattr(last_error, "content", b"")[:1000])
    raise RuntimeError(f"{what} failed after {max_tries} attempts")


def list_page_ids(svc, q: str, page_token: Optional[str]) -> Tuple[List[str], Optional[str]]:
    resp = _call_with_retry(
        svc.users().messages().list(
            userId="me",
            q=q,
            #labelIds=["INBOX"],
            maxResults=500,              # gentle page size
            includeSpamTrash=False,
            pageToken=page_token,
            fields="messages(id),nextPageToken",
            quotaUser="gmail-fetcher"
        ).execute,
        what="ListMessages",
    )
    ids = [m["id"] for m in resp.get("messages", [])]
    return ids, resp.get("nextPageToken")

def fetch_meta_epoch(svc, mid: str) -> int:
    resp = _call_with_retry(
        svc.users().messages().get(
            userId="me", id=mid, format="metadata",
            fields="id,internalDate"
        ).execute,
        what="GetMessage(metadata)",
    )
    return int(resp.get("internalDate", "0"))

def fetch_raw_with_retry(svc, mid: str) -> Tuple[int, bytes]:
    resp = _call_with_retry(
        svc.users().messages().get(
            userId="me", id=mid, format="raw", fields="id,internalDate,raw"
        ).execute,
        what="GetMessage(raw)",
    )
    epoch_ms = int(resp.get("internalDate", "0"))
    eml_bytes = base64.urlsafe_b64decode(resp["raw"])
    return epoch_ms, eml_bytes

def to_text(s: str) -> str:
    if "<html" in s.lower() or "<body" in s.lower():
        try:
            return BeautifulSoup(s, "html.parser").get_text(separator="\n")
        except Exception:
            pass
    return s

def parse_from_raw(eml_bytes: bytes) -> Tuple[Dict[str, str], str]:
    msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
    subject = msg["subject"] or ""
    from_addr = msg["from"] or ""

    text_body = ""
    if msg.is_multipart():
        html_fallback = ""
        for part in msg.walk():
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    text_body = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    text_body = payload.decode(errors="ignore")
                break
            if ctype == "text/html" and not html_fallback:
                try:
                    html_fallback = to_text(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    html_fallback = to_text(payload.decode(errors="ignore"))
        if not text_body and html_fallback:
            text_body = html_fallback
    else:
        ctype = msg.get_content_type()
        try:
            text_body = msg.get_content()
        except Exception:
            payload = msg.get_payload(decode=True) or b""
            text_body = payload.decode(errors="ignore")
        if ctype == "text/html":
            text_body = to_text(text_body)

    return {"subject": subject, "from": from_addr}, text_body

def should_skip(mid: str, epoch_ms: int, last_epoch_ms: int, last_id: str, lookback_secs: int, bronze_exists_fn) -> bool:
    """Implements strict ordering with lookback overlap handling."""
    is_older = (epoch_ms < last_epoch_ms) or (epoch_ms == last_epoch_ms and mid <= last_id)
    if not is_older:
        return False
    # allow within lookback if not already ingested (handles delayed visibility)
    if epoch_ms >= last_epoch_ms - (lookback_secs * 1000):
        # Caller passes bronze_exists; we only skip if bronze already there
        return bronze_exists_fn(mid, epoch_ms)
    # outside lookback and older than watermark → skip
    return True

def should_ingest_backfill(mid: str, epoch_ms: int, bronze_exists_fn) -> bool:
    """Backfill rule: ignore watermark; only skip if we already saved this message."""
    return not bronze_exists_fn(mid, epoch_ms)

def get_label_id(svc, label_name: str) -> Optional[str]:
    resp = _call_with_retry(
        svc.users().labels().list(userId="me").execute,
        what="ListLabels",
    )
    for lab in resp.get("labels", []):
        if lab.get("name") == label_name:
            return lab.get("id")
    return None

from zoneinfo import ZoneInfo
MAILBOX_TZ = os.environ.get("MAILBOX_TZ", "Europe/Stockholm")  # e.g. "Europe/Stockholm"
def list_ids_between(svc, label_name: str, after_epoch: int, before_epoch: int) -> List[str]:
    # after_epoch and before_epoch are provided as seconds (unix epoch)
    # Gmail's `after:`/`before:` with dates is strict and uses day precision
    # (e.g. `before:2025/08/19` means strictly before midnight UTC on 2025-08-19).
    # To capture messages in the half-open interval [after_epoch, before_epoch)
    # when using date-only operators, convert to UTC dates and make the
    # `before:` date one day later so you don't accidentally exclude the
    # intended end-of-day messages.
    tz = ZoneInfo(MAILBOX_TZ)
    
    # Convert the UTC epochs to mailbox-local dates
    after_dt_local  = datetime.fromtimestamp(after_epoch, tz=timezone.utc).astimezone(tz).date()
    before_dt_local = datetime.fromtimestamp(before_epoch, tz=timezone.utc).astimezone(tz).date()


    #Apply your empirically-validated day tweaks (defaults: +1/+1)
    after_dt_local  = after_dt_local  + timedelta(days=1)
    before_dt_local = before_dt_local + timedelta(days=1)


    q = f'label:"{label_name}" after:{after_dt_local.strftime("%Y/%m/%d")} before:{before_dt_local.strftime("%Y/%m/%d")}'
    
    ids, page = [], None
    while True:
        page_ids, page = list_page_ids(svc, q, page)
        ids.extend(page_ids)
        if not page:
            break
        time.sleep(0.5)
    return ids

# def save_cooldown_max(until_dt_utc: datetime) -> None:
#     """Persist the later of the existing cooldown and the new one."""
#     cur = load_cooldown_utc()
#     if cur and cur > until_dt_utc:
#         # keep the longer existing cooldown
#         return
#     save_cooldown_utc(until_dt_utc)


def utc_midnight_ms(ms: int) -> int:
    """Return the epoch ms for 00:00:00 UTC of the day containing ms."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    mid = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(mid.timestamp() * 1000)


# ────────────────────────────────────────────────────────────────────────────────
# GCS writers (bronze & silver)

_gcs = storage.Client()

def bronze_exists(msg_id: str, internal_epoch_ms: int) -> bool:
    b = _gcs.bucket(BUCKET)
    dt_str = datetime.fromtimestamp(internal_epoch_ms / 1000, tz=timezone.utc).date().isoformat()
    path = f"bronze_raw/source=gmail/dt={dt_str}/msgId={msg_id}/message.eml"
    return b.blob(path).exists(_gcs)

def write_bronze(msg_id: str, internal_epoch_ms: int, eml_bytes: bytes, meta: Dict) -> None:
    b = _gcs.bucket(BUCKET)
    dt_str = datetime.fromtimestamp(internal_epoch_ms / 1000, tz=timezone.utc).date().isoformat()
    base = f"bronze_raw/source=gmail/dt={dt_str}/msgId={msg_id}"
    b.blob(f"{base}/message.eml").upload_from_string(eml_bytes, content_type="message/rfc822")
    b.blob(f"{base}/meta.json").upload_from_string(json.dumps(meta, ensure_ascii=False), content_type="application/json")

def silver_exists(msg_id: str, release_date: str) -> bool:
    b = _gcs.bucket(BUCKET)
    path = f"silver_normalized/table=press_releases/release_date={release_date}/msgId={msg_id}.parquet"
    return b.blob(path).exists(_gcs)

def write_silver_unique(row: Dict) -> None:
    # Path includes msgId -> idempotent
    msg_id = row["press_release_id"]
    release_date = row["release_date"]
    b = _gcs.bucket(BUCKET)
    path = f"silver_normalized/table=press_releases/release_date={release_date}/msgId={msg_id}.parquet"
    blob = b.blob(path)
    if blob.exists(_gcs):
        return  # already written

    # write a single-row parquet
    df = pl.DataFrame([row])
    buf = io.BytesIO()
    df.write_parquet(buf)
    blob.upload_from_string(buf.getvalue(), content_type="application/octet-stream")
    logging.info(f"[silver] wrote 1 row → gs://{BUCKET}/{path}")



# ────────────────────────────────────────────────────────────────────────────────
# Orchestration

def run_once():
    try:
        # ── Global cooldown gate ──────────────────────────────────────────
        now_utc = datetime.now(timezone.utc)
        cd = load_cooldown_utc()
        if cd and now_utc < cd:
            logging.warning("[gate] In cooldown until %s UTC; skipping run",
                            cd.isoformat(timespec="seconds"))
            return
        if cd and now_utc >= cd:
            clear_cooldown_utc()
        # ─────────────────────────────────────────────────────────────────

        # Build Gmail client after the gate
        creds = gmail_creds_from_existing_secrets()
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

        # Load state
        last_epoch_ms, last_id = load_watermark()
        start_history_id = load_history_id()
        backfill_cursor = load_backfill_cursor()   # <- NEW: always check this

        # ── A) Backfill one window if there is outstanding work ──────────
        #if backfill_cursor is not None:
        try:
            seen, ingested, next_before = run_backfill_once(svc, backfill_cursor)
            logging.info("[backfill] seen=%d ingested=%d next_before_ms=%s",
                            seen, ingested, next_before)
        except CooldownActive:
            # cooldown was persisted; abort run so scheduler retries later
            raise
        except Exception:
            logging.exception("[backfill] failure; keeping cursor for resume")


        # ── B) Incremental path ──────────────────────────────────────────
        if not start_history_id:
            # First-time bridge only: capture baseline & drain deltas once
            logging.info("[mode] first-run: (optional) handoff after backfill")
            try:
                baseline_history = get_current_history_id(svc)
            except CooldownActive:
                raise
            except HttpError as e:
                if getattr(e.resp, "status", None) == 429:
                    until = _parse_retry_after_abs_utc(e)
                    if until:
                        save_cooldown_max(until)
                        raise CooldownActive(until)
                baseline_history = None
            except Exception:
                baseline_history = None
            logging.info("[init] baseline historyId=%s", baseline_history)

            if baseline_history:
                try:
                    ids_set, new_hist = fetch_history_ids(svc, str(baseline_history))
                    ids = sorted(ids_set)
                    processed = 0
                    for mid in ids:
                        try:
                            epoch_ms = fetch_meta_epoch(svc, mid)
                            if bronze_exists(mid, epoch_ms):
                                continue
                            if should_skip(mid, epoch_ms, last_epoch_ms, last_id,
                                           LOOKBACK_SECS, bronze_exists):
                                continue
                            epoch_ms2, eml_bytes = fetch_raw_with_retry(svc, mid)
                            time.sleep(0.5)
                            headers, text_body = parse_from_raw(eml_bytes)
                            subject = headers.get("subject", "")
                            from_addr = headers.get("from", "")
                            write_bronze(mid, epoch_ms2, eml_bytes, {
                                "id": mid, "from": from_addr, "subject": subject,
                                "internal_epoch_ms": epoch_ms2
                            })
                            dt = datetime.fromtimestamp(epoch_ms2 / 1000, tz=timezone.utc)
                            row = {
                                "press_release_id": mid,
                                "company_name": "",
                                "category": "unknown",
                                "release_date": dt.date().isoformat(),
                                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "title": subject,
                                "full_text": text_body,
                                "source": "gmail",
                                "source_url": "",
                                "parser_version": PARSER_VER,
                                "schema_version": SCHEMA_VER,
                            }
                            write_silver_unique(row)
                            if epoch_ms2 > last_epoch_ms or (epoch_ms2 == last_epoch_ms and mid > (last_id or "")):
                                last_epoch_ms, last_id = epoch_ms2, mid
                                save_watermark(last_epoch_ms, last_id)
                            processed += 1
                        except HttpError as e:
                            if getattr(e.resp, "status", None) == 429:
                                until = _parse_retry_after_abs_utc(e)
                                if until:
                                    save_cooldown_max(until)
                                    raise CooldownActive(until)
                            logging.warning("[handoff] http error id=%s: %s", mid, e)
                            continue
                        except CooldownActive:
                            raise
                        except Exception:
                            logging.exception("[handoff] unexpected for id=%s", mid)
                            continue

                        cd2 = load_cooldown_utc()
                        if cd2 and datetime.now(timezone.utc) < cd2:
                            logging.warning("[gate] Cooldown set mid-run; aborting handoff loop early.")
                            break

                    logging.info("[handoff] processed=%d messages since baseline", processed)
                    save_history_id(new_hist or str(baseline_history))

                except CooldownActive:
                    raise
                except HttpError as e:
                    if getattr(e.resp, "status", None) == 429:
                        until = _parse_retry_after_abs_utc(e)
                        if until:
                            save_cooldown_max(until)
                            raise CooldownActive(until)
                    logging.warning("[handoff] history drain http error: %s", e)
                except Exception:
                    logging.exception("[handoff] unexpected failure during history drain")

            # Done with first-time bridge. Next runs will be purely incremental.
            return

        # Already have a history watermark → normal incremental
        logging.info("[mode] incremental via historyId start=%s", start_history_id)
        ids_set, new_history_id = fetch_history_ids(svc, start_history_id)
        ids = sorted(ids_set)
        total_processed = 0

        for mid in ids:
            try:
                epoch_ms = fetch_meta_epoch(svc, mid)
                if bronze_exists(mid, epoch_ms):
                    continue
                if should_skip(mid, epoch_ms, last_epoch_ms, last_id, LOOKBACK_SECS, bronze_exists):
                    continue
                epoch_ms2, eml_bytes = fetch_raw_with_retry(svc, mid)
                time.sleep(0.5)
                headers, text_body = parse_from_raw(eml_bytes)
                subject = headers.get("subject", "")
                from_addr = headers.get("from", "")
                write_bronze(mid, epoch_ms2, eml_bytes, {
                    "id": mid, "from": from_addr, "subject": subject,
                    "internal_epoch_ms": epoch_ms2
                })
                dt = datetime.fromtimestamp(epoch_ms2 / 1000, tz=timezone.utc)
                row = {
                    "press_release_id": mid,
                    "company_name": "",
                    "category": "unknown",
                    "release_date": dt.date().isoformat(),
                    "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "title": subject,
                    "full_text": text_body,
                    "source": "gmail",
                    "source_url": "",
                    "parser_version": PARSER_VER,
                    "schema_version": SCHEMA_VER,
                }
                write_silver_unique(row)
                if epoch_ms2 > last_epoch_ms or (epoch_ms2 == last_epoch_ms and mid > (last_id or "")):
                    last_epoch_ms, last_id = epoch_ms2, mid
                    save_watermark(last_epoch_ms, last_id)
                total_processed += 1
            except HttpError as e:
                if getattr(e.resp, "status", None) == 429:
                    until = _parse_retry_after_abs_utc(e)
                    if until:
                        save_cooldown_max(until)
                        raise CooldownActive(until)
                logging.warning("[gmail] http error id=%s: %s", mid, e)
                continue
            except CooldownActive:
                raise
            except Exception:
                logging.exception("[gmail] unexpected error processing id=%s", mid)
                continue

            cd2 = load_cooldown_utc()
            if cd2 and datetime.now(timezone.utc) < cd2:
                logging.warning("[gate] Cooldown set mid-run; aborting incremental loop early.")
                break

        if new_history_id:
            save_history_id(new_history_id)

        logging.info("[done] incremental processed=%d new_epoch_ms=%d history_id=%s",
                     total_processed, last_epoch_ms, load_history_id())

    except CooldownActive as ce:
        logging.warning("[gate] Aborting run due to cooldown: %s", ce)
        return


def run_backfill_once(svc, before_ms: Optional[int]) -> Tuple[int, int, Optional[int]]:
    """
    Process at most one date window, returning:
    (messages_seen, messages_ingested, next_before_epoch_ms_or_None)
    """
    # Establish or resume the "before" cursor
    #before_ms = load_backfill_cursor()
    

    if before_ms is None:
        # first ever backfill call: anchor to now or BACKFILL_START_ISO
        if BACKFILL_START_ISO:
            start_dt = datetime.fromisoformat(BACKFILL_START_ISO).replace(tzinfo=timezone.utc)
            before_ms = int(start_dt.timestamp() * 1000)
        else:
            before_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        save_backfill_cursor(before_ms)
    
    # (Optional) guard: must be a positive epoch
    if before_ms <= 0:
        raise ValueError(f"Invalid backfill cursor epoch: {before_ms}")

    # Define the window [after, before)
    window_end_ms = before_ms
    window_start_ms = max(0, window_end_ms - BACKFILL_WINDOW_DAYS * 24 * 3600 * 1000)

    logging.info("[backfill] window %s → %s UTC",
                 datetime.fromtimestamp(window_start_ms/1000, tz=timezone.utc).isoformat(timespec="seconds"),
                 datetime.fromtimestamp(window_end_ms/1000, tz=timezone.utc).isoformat(timespec="seconds"))

    # Collect candidate IDs in this window (by label)
    candidate_ids = list_ids_between(svc, BACKFILL_LABEL, window_start_ms//1000, window_end_ms//1000)
    candidate_ids.sort()  # determinism
    seen = 0
    ingested = 0
    raw_budget = BACKFILL_MAX_RAW_PER_RUN

    # Use your existing epoch+id watermark to preserve strict ordering/idempotency
    #last_epoch_ms, last_id = load_watermark()
    exhausted_budget = False
    for mid in candidate_ids:
        if raw_budget <= 0:
            exhausted_budget = True
            break
        try:
            seen += 1
            epoch_ms = fetch_meta_epoch(svc, mid)

            if not should_ingest_backfill(mid, epoch_ms, bronze_exists):
                continue

            # Only download RAW when we know we need it
            epoch_ms2, eml_bytes = fetch_raw_with_retry(svc, mid)
            time.sleep(0.5)  # gentle pacing

            headers, text_body = parse_from_raw(eml_bytes)
            subject = headers.get("subject", "")
            from_addr = headers.get("from", "")

            write_bronze(mid, epoch_ms2, eml_bytes, {
                "id": mid, "from": from_addr, "subject": subject, "internal_epoch_ms": epoch_ms2
            })

            dt = datetime.fromtimestamp(epoch_ms2 / 1000, tz=timezone.utc)
            row = {
                "press_release_id": mid,
                "company_name": "",
                "category": "unknown",
                "release_date": dt.date().isoformat(),
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "title": subject,
                "full_text": text_body,
                "source": "gmail",
                "source_url": "",
                "parser_version": PARSER_VER,
                "schema_version": SCHEMA_VER,
            }
            write_silver_unique(row)

            # # advance your epoch watermark
            # cur_max_ms, ids_at_max = load_watermark()  # reload if other workers exist; else track locally
            # if epoch_ms2 > cur_max_ms or (epoch_ms2 == cur_max_ms and mid > (ids_at_max or "")):
            #     save_watermark(epoch_ms2, mid)

            ingested += 1
            raw_budget -= 1

        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status == 429:
                until = _parse_retry_after_abs_utc(e)
                if until:
                    save_cooldown_max(until)
                    logging.warning("[backfill] 429; honoring Retry-After until %s UTC → abort run",
                                    until.isoformat(timespec="seconds"))
                    raise CooldownActive(until)
            logging.warning("[backfill] http error id=%s: %s", mid, e)
            continue
        except CooldownActive:
            # bubble up so run_once() can end the run
            raise
        except Exception as e:
            logging.exception("[backfill] unexpected for id=%s: %s", mid, e)
            continue

        # after each message, respect any cooldown set by other calls
        cd = load_cooldown_utc()
        if cd and datetime.now(timezone.utc) < cd:
            logging.warning("[gate] Cooldown set mid-run; aborting backfill loop early.")
            break

    # Compute next cursor: move the "before" boundary back by one window
    if exhausted_budget:
        # Stay on the same window; do NOT move the cursor back yet
        next_before_ms = before_ms
    else:
        # Finished this whole window → move to the next (older) window
        next_before_ms = window_start_ms if window_start_ms > 0 else None
    
    save_backfill_cursor(next_before_ms)
    
    if next_before_ms is None:
        logging.info("[backfill] reached beginning (or configured start); backfill complete.")

    logging.info("[backfill] seen=%d ingested=%d next_before_ms=%s", seen, ingested, next_before_ms)
    return seen, ingested, next_before_ms


if __name__ == "__main__":
    run_once()
