#!/usr/bin/env python3
# rss_single_feed_job.py
"""
Hardcoded single-feed Cloud Run Job:
- Fetch RSS (conditional GET with Last-Modified/ETag in Firestore)
- For each item, fetch the article page, extract text
- Write bronze:
    bronze_raw/source=globenewswire/dt=YYYY-MM-DD/msgId=<id>/{content.html.gz, entry.json.gz}
- Write silver:
    silver_normalized/table=press_releases/release_date=YYYY-MM-DD/msgId=<id>.parquet

ENV required:
  PROJECT_ID=event-feed-app-463206
  GCS_BUCKET=event-feed-app-data

Optional ENV:
  USER_AGENT="PressReleaseFetcher/1.0 (+contact@example.com)"
  HTTP_TIMEOUT_SECS=25
  SLEEP_BETWEEN_REQUESTS_SECS=0.7
  MAX_ITEMS_PER_RUN=150
  SCHEMA_VER=1
  PARSER_VER=1
  HTML_MIN_LEN=600
  COOLDOWN_PAD_SECS=10
  RESPECT_MAX_AGE=true  # honor Cache-Control: max-age on feed response

Deps:
  pip install google-cloud-storage google-cloud-firestore requests feedparser beautifulsoup4 pyarrow trafilatura
"""

import os, io, re, json, time, gzip, hashlib, logging
from typing import Optional, Dict
from datetime import datetime, timezone, timedelta

import requests, feedparser
from bs4 import BeautifulSoup
from google.cloud import storage, firestore
import pyarrow as pa
import pyarrow.parquet as pq

# ── Hardcoded feed & source ─────────────────────────────────────────────────────
FEED_URL = "https://www.globenewswire.com/RssFeed/country/United%20States/feedTitle/GlobeNewswire%20-%20News%20from%20United%20States"
SOURCE   = "globenewswire"  # used in GCS paths

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ID     = os.environ.get("PROJECT_ID", "")
BUCKET         = os.environ.get("GCS_BUCKET", "")
USER_AGENT     = os.environ.get("USER_AGENT", "PressReleaseFetcher/1.0 (+contact@example.com)")
TIMEOUT        = int(os.environ.get("HTTP_TIMEOUT_SECS", "25"))
SLEEP_BETWEEN  = float(os.environ.get("SLEEP_BETWEEN_REQUESTS_SECS", "0.7"))
MAX_ITEMS_PER_RUN = int(os.environ.get("MAX_ITEMS_PER_RUN", "150"))
SCHEMA_VER     = int(os.environ.get("SCHEMA_VER", "1"))
PARSER_VER     = int(os.environ.get("PARSER_VER", "1"))
HTML_MIN_LEN   = int(os.environ.get("HTML_MIN_LEN", "600"))
COOLDOWN_PAD_SECS = int(os.environ.get("COOLDOWN_PAD_SECS", "10"))
RESPECT_MAX_AGE   = (os.environ.get("RESPECT_MAX_AGE", "true").lower() in ("1","true","yes"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_gcs = storage.Client(project=PROJECT_ID)
_fs  = firestore.Client(project=PROJECT_ID)

# ── Utils ──────────────────────────────────────────────────────────────────────
class CooldownActive(Exception):
    def __init__(self, until_utc: datetime):
        super().__init__(f"Cooldown until {until_utc.isoformat()}")
        self.until_utc = until_utc

def _short_hash(*parts: str, n=16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:n]

def _gz_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ── Firestore state (per-feed) + cooldown ──────────────────────────────────────
def _feed_state_doc():
    # Key the doc off FEED_URL so it’s stable
    key = _short_hash(FEED_URL, n=16)
    return _fs.collection("ingest_state").document(f"rss_{key}")

def load_feed_state() -> Dict:
    doc = _feed_state_doc().get()
    return doc.to_dict() if doc.exists else {}

def save_feed_state(*, last_modified: Optional[str] = None, etag: Optional[str] = None):
    patch = {}
    if last_modified is not None: patch["last_modified"] = last_modified
    if etag is not None:          patch["etag"] = etag
    if patch:
        _feed_state_doc().set(patch, merge=True)

def load_cooldown_utc() -> Optional[datetime]:
    doc = _fs.collection("ingest_state").document("rss").get()
    if not doc.exists: return None
    s = (doc.to_dict() or {}).get("cooldown_until_utc")
    try:
        return datetime.fromisoformat(s) if s else None
    except Exception:
        return None

def save_cooldown_utc(until_dt_utc: datetime) -> None:
    _fs.collection("ingest_state").document("rss").set(
        {"cooldown_until_utc": until_dt_utc.astimezone(timezone.utc).isoformat()}, merge=True
    )

def clear_cooldown_utc():
    _fs.collection("ingest_state").document("rss").set({"cooldown_until_utc": None}, merge=True)

def save_cooldown_max(until_dt_utc: datetime):
    target = (until_dt_utc + timedelta(seconds=COOLDOWN_PAD_SECS)).astimezone(timezone.utc)
    current = load_cooldown_utc()
    if current is None or target > current:
        save_cooldown_utc(target)

# ── HTTP helpers ────────────────────────────────────────────────────────────────
def polite_headers(extra: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    h = {"User-Agent": USER_AGENT, "Accept-Language": "en", "Connection": "close"}
    if extra: h.update(extra)
    return h

def http_get(url: str, headers: Dict[str,str]) -> requests.Response:
    r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    if r.status_code in (429, 503):
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                secs = float(ra)
                until = datetime.now(timezone.utc) + timedelta(seconds=secs)
                save_cooldown_max(until)
                raise CooldownActive(until)
            except Exception:
                pass
    r.raise_for_status()
    return r

# ── Extraction ─────────────────────────────────────────────────────────────────
def extract_text(html: str, base_url: str) -> str:
    # Try trafilatura first (optional)
    try:
        import trafilatura
        txt = trafilatura.extract(html, url=base_url, include_comments=False, include_tables=False)
        if txt and len(txt.strip()) >= HTML_MIN_LEN:
            return txt.strip()
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","iframe","form","nav","header","footer","aside","button","svg"]):
        tag.decompose()

    selectors = [
        "article",
        '[itemprop="articleBody"]',
        ".article-body,.articleBody,#article-body,#articleBody",
        ".press-release,.pressrelease,.pr-body,#release-body",
        ".entry-content,.content__body,.story-body,.post-content",
        ".gnw-article,.gnw-press-release",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text("\n", strip=True)
            if len(text) >= HTML_MIN_LEN:
                return text

    best = ""
    for div in soup.find_all(["div","section"], limit=400):
        t = div.get_text("\n", strip=True)
        if len(t) > len(best):
            best = t
    return best

# ── GCS writers (msgId in paths) ───────────────────────────────────────────────
def bronze_exists(msg_id: str, day_iso: str) -> bool:
    b = _gcs.bucket(BUCKET)
    path = f"bronze_raw/source={SOURCE}/dt={day_iso}/msgId={msg_id}/content.html.gz"
    return b.blob(path).exists(_gcs)

def write_bronze(*, msg_id: str, day_iso: str, html: str, entry_json: Dict, page_url: str, page_headers: Dict):
    b = _gcs.bucket(BUCKET)
    base = f"bronze_raw/source={SOURCE}/dt={day_iso}/msgId={msg_id}"

    blob_html = b.blob(f"{base}/content.html.gz")
    blob_html.content_type = "text/html"
    blob_html.content_encoding = "gzip"
    blob_html.cache_control = "no-transform,private,max-age=0"
    blob_html.metadata = {"msgId": msg_id, "source": SOURCE, "url": page_url, "ingested_at": _now_utc_iso()}
    blob_html.upload_from_string(_gz_bytes(html.encode("utf-8")),
                                 content_type="text/html",
                                 if_generation_match=0)

    blob_meta = b.blob(f"{base}/entry.json.gz")
    blob_meta.content_type = "application/json"
    blob_meta.content_encoding = "gzip"
    blob_meta.cache_control = "no-transform,private,max-age=0"
    ej = dict(entry_json); ej["msgId"] = msg_id
    blob_meta.upload_from_string(_gz_bytes(json.dumps(ej, ensure_ascii=False).encode("utf-8")),
                                 content_type="application/json",
                                 if_generation_match=0)

def write_silver_parquet(row: Dict):
    schema = pa.schema([
        ("press_release_id", pa.string()),
        ("company_name",     pa.string()),
        ("category",         pa.string()),
        ("release_date",     pa.string()),
        ("ingested_at",      pa.string()),
        ("title",            pa.string()),
        ("full_text",        pa.string()),
        ("source",           pa.string()),
        ("source_url",       pa.string()),
        ("parser_version",   pa.int32()),
        ("schema_version",   pa.int32()),
    ])
    data = {name: [row.get(name)] for name in schema.names}
    table = pa.Table.from_pydict(data, schema=schema)

    msg_id_full = row["press_release_id"]          # e.g. "globenewswire__0d4cc4a6201f3cc3"
    short_id = msg_id_full.split("__", 1)[-1]      # "0d4cc4a6201f3cc3" (no-op if no prefix)
    day = row["release_date"]
    source = row["source"]

    path = f"silver_normalized/table=press_releases/source={source}/release_date={day}/msgId={short_id}.parquet"

    b = _gcs.bucket(BUCKET)
    blob = b.blob(path)
    if blob.exists(_gcs):
        return
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", version="2.6", use_dictionary=False)
    blob.upload_from_string(buf.getvalue(), content_type="application/octet-stream")
    logging.info(f"[silver] wrote 1 row → gs://{BUCKET}/{path}")


# ── Core ───────────────────────────────────────────────────────────────────────
def _msg_id_for_entry(e: dict) -> str:
    raw = e.get("id") or e.get("guid") or e.get("link") or (e.get("title","") + "|" + e.get("link",""))
    return f"{SOURCE}__{_short_hash(raw)}"

def _pub_date_iso(e: dict) -> str:
    try:
        t = e.get("published_parsed") or e.get("updated_parsed")
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc).date().isoformat()
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()

def process_once() -> int:
    if not PROJECT_ID or not BUCKET:
        raise RuntimeError("Set PROJECT_ID and GCS_BUCKET env vars.")

    # cooldown gate
    now_utc = datetime.now(timezone.utc)
    cd = load_cooldown_utc()
    if cd and now_utc < cd:
        logging.warning("[gate] cooldown until %s; skipping run", cd.isoformat(timespec="seconds"))
        return 0
    if cd and now_utc >= cd:
        clear_cooldown_utc()

    # conditional GET for the feed
    state = load_feed_state()
    headers = polite_headers({})
    if state.get("last_modified"): headers["If-Modified-Since"] = state["last_modified"]
    if state.get("etag"):          headers["If-None-Match"]     = state["etag"]

    r = http_get(FEED_URL, headers)
    if r.status_code == 304:
        logging.info("[feed] not modified")
        return 0

    save_feed_state(last_modified=r.headers.get("Last-Modified"), etag=r.headers.get("ETag"))

    if RESPECT_MAX_AGE:
        cc = r.headers.get("Cache-Control","")
        m = re.search(r"max-age=(\d+)", cc)
        if m:
            secs = int(m.group(1))
            if secs >= 30:
                save_cooldown_max(datetime.now(timezone.utc) + timedelta(seconds=secs))

    parsed = feedparser.parse(r.content)
    feed_title = (parsed.feed.get("title") or "rss").strip()

    written = 0
    for e in parsed.entries:
        if written >= MAX_ITEMS_PER_RUN:
            logging.info("[feed] hit MAX_ITEMS_PER_RUN=%d; stopping.", MAX_ITEMS_PER_RUN)
            break

        msg_id = _msg_id_for_entry(e)
        link   = e.get("link","")
        title  = e.get("title","")
        day    = _pub_date_iso(e)

        if bronze_exists(msg_id, day):
            continue

        try:
            page = http_get(link, polite_headers({}))
        except CooldownActive as ce:
            logging.warning("[gate] cooldown (target site): %s", ce)
            raise
        except Exception as ex:
            logging.warning("[fetch] failed url=%s err=%s", link, ex)
            continue

        html = page.text
        text = extract_text(html, page.url)
        if not text or len(text) < 200:
            logging.warning("[extract] short text msgId=%s url=%s", msg_id, page.url)

        entry_json = {
            "msgId": msg_id,
            "title": title,
            "link": link,
            "published": e.get("published"),
            "summary": e.get("summary"),
            "feed_title": feed_title,
            "feed_url": FEED_URL,
            "final_url": page.url,
            "fetched_at": _now_utc_iso(),
            "headers": {k:v for k,v in page.headers.items() if k.lower() in ("content-type","cache-control","last-modified")}
        }
        write_bronze(msg_id=msg_id, day_iso=day, html=html, entry_json=entry_json,
                     page_url=page.url, page_headers=page.headers)

        row = {
            "press_release_id": msg_id,
            "company_name": "",
            "category": "unknown",
            "release_date": day,
            "ingested_at": _now_utc_iso(),
            "title": title,
            "full_text": text,
            "source": SOURCE,
            "source_url": page.url,
            "parser_version": PARSER_VER,
            "schema_version": SCHEMA_VER,
        }
        write_silver_parquet(row)

        written += 1
        time.sleep(SLEEP_BETWEEN)

    logging.info("[done] source=%s items_written=%d", SOURCE, written)
    return written

if __name__ == "__main__":
    try:
        process_once()
    except CooldownActive as ce:
        logging.warning("[gate] abort run due to cooldown until %s", ce.until_utc.isoformat(timespec="seconds"))
