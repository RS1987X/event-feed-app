import os, json, base64, logging, io
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Dict
from bs4 import BeautifulSoup
import polars as pl

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.cloud import secretmanager, firestore, storage




# ────────────────────────────────────────────────────────────────────────────────
# Config (from env)
#PROJECT_ID   = os.environ["PROJECT_ID"]
PROJECT_ID = os.environ.get("PROJECT_ID", "event-feed-app-463206")
BUCKET = os.environ.get("GCS_BUCKET", "event-feed-app-data")
GMAIL_LABEL  = os.environ.get("GMAIL_LABEL", "INBOX")
SCHEMA_VER   = 1
PARSER_VER   = 1
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
LOOKBACK_SECS = int(os.environ.get("LOOKBACK_SECS", "600"))  # 10 minutes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ────────────────────────────────────────────────────────────────────────────────
# Secrets / Auth

def _secret_text(name: str) -> str:
    sm = secretmanager.SecretManagerServiceClient()
    rn = f"projects/{PROJECT_ID}/secrets/{name}/versions/latest"
    return sm.access_secret_version(request={"name": rn}).payload.data.decode()

def gmail_creds_from_existing_secrets() -> Credentials:
    client_json = json.loads(_secret_text("GMAIL_OAUTH_JSON"))
    client_blob = client_json.get("installed") or client_json.get("web") or {}
    client_id = client_blob["client_id"]
    client_secret = client_blob["client_secret"]

    token_json = json.loads(_secret_text("GMAIL_OAUTH_CREDENTIALS"))
    refresh_token = token_json["refresh_token"]
    token_uri = token_json.get("token_uri", "https://oauth2.googleapis.com/token")

    return Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=GMAIL_SCOPES,
    )

# ────────────────────────────────────────────────────────────────────────────────
# Firestore watermark (epoch + tie-breaker id)

def load_watermark() -> Tuple[int, str]:
    """Return (last_epoch_ms, last_msg_id_at_epoch). Backward-compatible with
    older docs that stored seconds in 'last_internal_epoch'."""
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return 0, ""
    d = doc.to_dict() or {}
    # Prefer ms if present, else upgrade from seconds
    if "last_internal_epoch_ms" in d:
        return int(d.get("last_internal_epoch_ms", 0)), d.get("last_msg_id_at_epoch", "")
    # fallback from seconds -> ms
    sec = int(d.get("last_internal_epoch", 0))
    return sec * 1000 if sec > 0 else 0, d.get("last_msg_id_at_epoch", "")

def save_watermark(epoch_ms: int, last_id: str) -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {
            "last_internal_epoch_ms": int(epoch_ms),
            "last_msg_id_at_epoch": last_id,
            # keep old field once (optional), helps manual inspection
            "last_internal_epoch": int(epoch_ms // 1000),
        },
        merge=True,
    )

# ────────────────────────────────────────────────────────────────────────────────
# Gmail helpers

def pick_body_part(payload: Dict) -> Dict:
    """Prefer text/plain, fallback text/html; walk recursively."""
    def walk(p):
        if p.get("mimeType", "").startswith("text/plain"):
            return p
        if p.get("mimeType", "").startswith("text/html"):
            return p
        for c in (p.get("parts") or []):
            r = walk(c)
            if r: return r
    return walk(payload) or {}

def to_text(s: str) -> str:
    if "<html" in s.lower() or "<body" in s.lower():
        try:
            return BeautifulSoup(s, "html.parser").get_text(separator="\n")
        except Exception:
            pass
    return s

# ────────────────────────────────────────────────────────────────────────────────
# GCS writers

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

def append_silver(rows: List[Dict]) -> None:
    if not rows:
        return
    by_date: Dict[str, List[Dict]] = {}
    for r in rows:
        by_date.setdefault(r["release_date"], []).append(r)
    b = _gcs.bucket(BUCKET)
    for release_date, group in by_date.items():
        df = pl.DataFrame(group)
        buf = io.BytesIO()
        df.write_parquet(buf)                # write to in-memory buffer
        parquet_bytes = buf.getvalue()
        part_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        part = f"silver_normalized/table=press_releases/release_date={release_date}/part-{part_ts}.parquet"
        b.blob(part).upload_from_string(parquet_bytes, content_type="application/octet-stream")
        logging.info(f"[silver] wrote {len(group)} rows → gs://{BUCKET}/{part}")

# ────────────────────────────────────────────────────────────────────────────────
# Main run (single pass)

def run_once():
    creds = gmail_creds_from_existing_secrets()
    svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

    last_epoch_ms, last_id = load_watermark()

    # Use a lookback window so delayed-visibility messages are still listed
    from_epoch_secs = max(0, (last_epoch_ms // 1000) - LOOKBACK_SECS)
    q = f"after:{datetime.fromtimestamp(from_epoch_secs, tz=timezone.utc).strftime('%Y/%m/%d')}"
    logging.info(f"[gmail] labelIds=[{GMAIL_LABEL}] q={q!r} (watermark_ms={last_epoch_ms}, lookback={LOOKBACK_SECS}s)")

    # list message IDs
    ids, page = [], None
    while True:
        resp = svc.users().messages().list(
            userId="me",
            labelIds=[GMAIL_LABEL],      # e.g. ["INBOX"]
            q=q,
            maxResults=500,
            includeSpamTrash=False,
            pageToken=page
        ).execute()
        ids.extend(m["id"] for m in resp.get("messages", []))
        page = resp.get("nextPageToken")
        if not page:
            break

    if not ids:
        logging.info("[gmail] no candidates from list()")
        return

    rows = []
    max_epoch_ms, ids_at_max = last_epoch_ms, []

    for mid in ids:
        # Full message
        msg = svc.users().messages().get(userId="me", id=mid, format="full").execute()
        epoch_ms = int(msg.get("internalDate", "0"))

        # Strict ordering: include if (epoch_ms, id) > (last_epoch_ms, last_id)
        should_skip = (epoch_ms < last_epoch_ms) or (epoch_ms == last_epoch_ms and mid <= last_id)

        if should_skip:
            # Within lookback? allow if not already ingested (handles delayed visibility)
            if epoch_ms >= last_epoch_ms - (LOOKBACK_SECS * 1000):
                if bronze_exists(mid, epoch_ms):
                    continue  # already ingested in overlap
                # else: fall through and ingest
            else:
                continue  # outside lookback and older than watermark → skip

        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        subject   = headers.get("subject", "")
        from_addr = headers.get("from", "")

        # body
        part = pick_body_part(msg["payload"])
        data = (part.get("body") or {}).get("data")
        raw_body = base64.urlsafe_b64decode(data).decode(errors="ignore") if data else ""
        text_body = to_text(raw_body)

        # raw .eml for bronze
        raw = svc.users().messages().get(userId="me", id=mid, format="raw").execute()
        eml_bytes = base64.urlsafe_b64decode(raw["raw"])
        write_bronze(mid, epoch_ms, eml_bytes, {
            "id": mid, "from": from_addr, "subject": subject, "internal_epoch_ms": epoch_ms
        })

        dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
        rows.append({
            "press_release_id": mid,
            "company_name": "",
            "category": "unknown",
            "release_date": dt.date().isoformat(),                          # partition key
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "title": subject,
            "full_text": text_body,
            "source": "gmail",
            "source_url": "",
            "parser_version": PARSER_VER,
            "schema_version": SCHEMA_VER,
        })

        # Track new watermark
        if epoch_ms > max_epoch_ms:
            max_epoch_ms, ids_at_max = epoch_ms, [mid]
        elif epoch_ms == max_epoch_ms:
            ids_at_max.append(mid)

    append_silver(rows)
    save_watermark(max_epoch_ms, max(ids_at_max) if ids_at_max else last_id)
    logging.info(f"[done] wrote {len(rows)} releases; new watermark_ms={max_epoch_ms}")

if __name__ == "__main__":
    run_once()
