# export_press_releases_for_labeling.py
# Python 3.9+
# pip install: pyarrow gcsfs pandas bs4 lxml

import os
import re
import hashlib
import html as ihtml
import pandas as pd
import pyarrow.dataset as ds
import gcsfs
from bs4 import BeautifulSoup

# -----------------------
# CONFIG (env-overridable)
# -----------------------
# Source table (Parquet on GCS)
GCS_SILVER_ROOT = os.getenv(
    "GCS_SILVER_ROOT",
    "event-feed-app-data/silver_normalized/table=press_releases"
)
# Optional Hive-style partitioning (set to 1 to enable)
USE_PARTITIONING = bool(int(os.getenv("USE_PARTITIONING", "0")))

# Optional date filters (YYYY-MM-DD). Leave empty to export all.
MIN_DATE = os.getenv("MIN_DATE", "")  # e.g. "2024-01-01"
MAX_DATE = os.getenv("MAX_DATE", "")  # e.g. "2025-12-31"

# Output CSV path (local)
OUT_CSV = os.getenv("OUT_CSV", "press_releases_for_labeling.csv")

# If you also want to upload the CSV to GCS, set e.g. gs://bucket/path/file.csv
OUT_GCS = os.getenv("OUT_GCS", "")

# Drop obvious system emails (unsubscribe/verify etc)
FILTER_SYSTEM_EMAILS = bool(int(os.getenv("FILTER_SYSTEM_EMAILS", "1")))

# Minimum body length (post-cleaning); rows below this get dropped
MIN_CONTENT_CHARS = int(os.getenv("MIN_CONTENT_CHARS", "40"))

# -----------------------
# Cleaning helpers
# -----------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
WS_RE    = re.compile(r"\s+")

TITLE_BLOCKLIST = re.compile(
    r"(?i)\b(confirm your subscription|verify your subscription|welcome( as)? (a )?subscriber|"
    r"subscription (activation|request|removal)|email alerting service|validate account|"
    r"stock quote notification)\b"
)
BODY_HINTS_BLOCKLIST = re.compile(
    r"(?i)\b(unsubscribe|manage (your )?subscription|click (the )?link|copy and paste|browser|"
    r"privacy policy|cookie policy|email alerting service|verify (your )?subscription|"
    r"confirm( your)? subscription)\b"
)

def strip_html(text: str) -> str:
    """Remove HTML, URLs, emails; collapse whitespace."""
    if not isinstance(text, str) or not text:
        return ""
    txt = ihtml.unescape(text)
    soup = BeautifulSoup(txt, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    txt = soup.get_text(separator=" ")
    txt = URL_RE.sub(" ", txt)
    txt = EMAIL_RE.sub(" ", txt)
    txt = WS_RE.sub(" ", txt).strip()
    return txt

def stable_id(row: pd.Series) -> str:
    """
    Use existing press_release_id if present; otherwise build a stable hash
    from title + release_date + source_url (or full_text as last resort).
    """
    if "press_release_id" in row and pd.notna(row["press_release_id"]) and str(row["press_release_id"]).strip():
        return str(row["press_release_id"]).strip()
    parts = [
        str(row.get("title", "") or ""),
        str(row.get("release_date", "") or ""),
        str(row.get("source_url", "") or ""),
        str(row.get("full_text", "") or "")
    ]
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
    return f"auto_{h[:16]}"

def write_local_to_gcs(fs: gcsfs.GCSFileSystem, local_path: str, gcs_dest_path: str):
    with fs.open(gcs_dest_path, "wb") as fout, open(local_path, "rb") as fin:
        fout.write(fin.read())

# -----------------------
# Main
# -----------------------
def main():
    fs = gcsfs.GCSFileSystem()

    # Build dataset
    if USE_PARTITIONING:
        import pyarrow as pa
        partitioning = ds.partitioning(
            schema=pa.schema([pa.field("release_date", pa.string())]),
            flavor="hive",
        )
        dataset = ds.dataset(GCS_SILVER_ROOT, filesystem=fs, format="parquet", partitioning=partitioning)
    else:
        dataset = ds.dataset(GCS_SILVER_ROOT, filesystem=fs, format="parquet")

    # Read minimal columns (read extra ones for stable id fallback)
    want_cols = ["press_release_id", "company_name", "title", "full_text", "release_date", "source_url"]
    available = [f.name for f in dataset.schema]
    cols = [c for c in want_cols if c in available]
    table = dataset.to_table(columns=cols)
    df = table.to_pandas()

    # Basic hygiene
    for c in ["title", "full_text", "company_name", "release_date", "source_url"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")

    # Optional date filtering
    if MIN_DATE:
        df = df[df["release_date"] >= MIN_DATE]
    if MAX_DATE:
        df = df[df["release_date"] <= MAX_DATE]

    # Clean title + body
    df["title_clean"] = df["title"].map(strip_html)
    df["content"] = df["full_text"].map(strip_html)

    # Optional system email filtering
    if FILTER_SYSTEM_EMAILS:
        mask_sys = (
            df["title_clean"].str.contains(TITLE_BLOCKLIST, na=False) |
            df["content"].str.contains(BODY_HINTS_BLOCKLIST, na=False)
        )
        if mask_sys.any():
            df = df[~mask_sys].copy()

    # Drop very short bodies
    if MIN_CONTENT_CHARS > 0:
        df = df[df["content"].str.len() >= MIN_CONTENT_CHARS]

    # Build message_id
    df["message_id"] = df.apply(stable_id, axis=1)

    # Final shape
    out = df[["message_id", "company_name", "title_clean", "content"]].rename(
        columns={"title_clean": "title"}
    )

    # Write local CSV
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote {OUT_CSV} — rows: {len(out):,}")

    # Optional upload to GCS
    if OUT_GCS:
        if not OUT_GCS.startswith("gs://"):
            raise ValueError("OUT_GCS must be a gs:// path, e.g. gs://my-bucket/exports/press_releases.csv")
        write_local_to_gcs(fs, OUT_CSV, OUT_GCS)
        print(f"Uploaded to {OUT_GCS}")

if __name__ == "__main__":
    main()
