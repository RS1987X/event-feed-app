import os, json, base64, datetime, logging
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
GMAIL_LABEL  = os.environ.get("GMAIL_LABEL", "PressReleases")
SCHEMA_VER   = 1
PARSER_VER   = 1
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

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
    db = firestore.Client(project=PROJECT_ID)
    doc = db.collection("ingest_state").document("gmail").get()
    if not doc.exists:
        return 0, ""
    d = doc.to_dict() or {}
    return int(d.get("last_internal_epoch", 0)), d.get("last_msg_id_at_epoch", "")

def save_watermark(epoch: int, last_id: str) -> None:
    db = firestore.Client(project=PROJECT_ID)
    db.collection("ingest_state").document("gmail").set(
        {"last_internal_epoch": int(epoch), "last_msg_id_at_epoch": last_id},
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

def write_bronze(msg_id: str, internal_epoch: int, eml_bytes: bytes, meta: Dict) -> None:
    b = _gcs.bucket(BUCKET)
    dt = datetime.datetime.utcfromtimestamp(internal_epoch).date().isoformat()
    base = f"bronze_raw/source=gmail/dt={dt}/msgId={msg_id}"
    b.blob(f"{base}/message.eml").upload_from_string(eml_bytes, content_type="message/rfc822")
    b.blob(f"{base}/meta.json").upload_from_string(json.dumps(meta, ensure_ascii=False), content_type="application/json")

def append_silver(rows: List[Dict]) -> None:
    if not rows: 
        return
    # group into release_date partitions, write one file per date
    by_date: Dict[str, List[Dict]] = {}
    for r in rows:
        by_date.setdefault(r["release_date"], []).append(r)
    b = _gcs.bucket(BUCKET)
    for release_date, group in by_date.items():
        df = pl.DataFrame(group)
        parquet_bytes = df.write_parquet()
        part = f"silver_normalized/table=press_releases/release_date={release_date}/part-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.parquet"
        b.blob(part).upload_from_string(parquet_bytes, content_type="application/octet-stream")
        logging.info(f"[silver] wrote {len(group)} rows → gs://{BUCKET}/{part}")

# ────────────────────────────────────────────────────────────────────────────────
# Main run (single pass)

def run_once():
    creds = gmail_creds_from_existing_secrets()
    svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

    last_epoch, _ = load_watermark()
    query = f"label:{GMAIL_LABEL} after:{last_epoch}"
    logging.info(f"[gmail] query: {query}")

    # list message IDs
    ids = []
    page = None
    while True:
        resp = svc.users().messages().list(userId="me", q=query, pageToken=page, maxResults=500).execute()
        ids += [m["id"] for m in resp.get("messages", [])]
        page = resp.get("nextPageToken")
        if not page: break

    if not ids:
        logging.info("[gmail] no new messages")
        return

    logging.info(f"[gmail] found {len(ids)} messages")

    rows = []
    max_epoch, ids_at_max = last_epoch, []

    for mid in ids:
        # Full message
        msg = svc.users().messages().get(userId="me", id=mid, format="full").execute()
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        subject   = headers.get("subject", "")
        from_addr = headers.get("from", "")
        epoch     = int(msg.get("internalDate", "0")) // 1000

        # watermark tracking
        if epoch > max_epoch:
            max_epoch, ids_at_max = epoch, [mid]
        elif epoch == max_epoch:
            ids_at_max.append(mid)

        # body
        part = pick_body_part(msg["payload"])
        data = part.get("body", {}).get("data")
        raw_body = base64.urlsafe_b64decode(data).decode(errors="ignore") if data else ""
        text_body = to_text(raw_body)

        # raw .eml for bronze
        raw = svc.users().messages().get(userId="me", id=mid, format="raw").execute()
        eml_bytes = base64.urlsafe_b64decode(raw["raw"])
        write_bronze(mid, epoch, eml_bytes, {
            "id": mid, "from": from_addr, "subject": subject, "internal_epoch": epoch
        })

        # silver row
        dt = datetime.datetime.utcfromtimestamp(epoch)
        rows.append({
            "press_release_id": mid,
            "company_name": "",
            "category": "unknown",
            "release_date": dt.date().isoformat(),
            "ingested_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "title": subject,
            "full_text": text_body,
            "source": "gmail",
            "source_url": "",
            "parser_version": PARSER_VER,
            "schema_version": SCHEMA_VER,
        })

    append_silver(rows)
    save_watermark(max_epoch, max(ids_at_max) if ids_at_max else "")
    logging.info(f"[done] wrote {len(rows)} releases; new watermark epoch={max_epoch}")

if __name__ == "__main__":
    run_once()
