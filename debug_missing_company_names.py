# debug_missing_company_names.py
import os, json, re
from email.utils import parseaddr
from email import policy
from email.parser import BytesParser

import gcsfs
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

# ---------------- config ----------------
GCS_SILVER_ROOT  = os.getenv("GCS_SILVER_ROOT", "gs://event-feed-app-data/silver_normalized/table=press_releases")
GCS_BRONZE_ROOT  = os.getenv("GCS_BRONZE_ROOT", "gs://event-feed-app-data/bronze_raw/source=gmail")
MAX_TO_SHOW      = int(os.getenv("MAX_TO_SHOW", "25"))

PARTITIONING = ds.partitioning(
    schema=pa.schema([pa.field("release_date", pa.large_string())]),
    flavor="hive",
)

# -------------- helpers ----------------
NOISE = re.compile(r"\b(via|on behalf of|investor relations|press(?: office| team)?)\b", re.I)

def parse_company_from_from_header(s: str) -> tuple[str, str, str]:
    """Return (display_name, email, cleaned_company_name) from a raw From header string."""
    name, email = parseaddr(s or "")
    name = (name or "").strip().strip('"')
    if name:
        name = re.split(r"\b(via|on behalf of)\b", name, 1, flags=re.I)[0].strip(" ,-|")
        name = NOISE.sub("", name).strip(" ,-|")
    if not name and email:
        dom = email.split("@", 1)[-1].split(".")[0]
        name = dom.capitalize()
    return (name or ""), (email or ""), name or ""

def read_meta_from_header(fs: gcsfs.GCSFileSystem, meta_path: str) -> tuple[str, dict]:
    """Try to read a From-like value from meta.json in various shapes."""
    with fs.open(meta_path, "rb") as f:
        meta = json.loads(f.read().decode("utf-8", "ignore"))
    # try a few common shapes
    if isinstance(meta, dict):
        if "from" in meta:                                  # {"from": "..."}
            return str(meta.get("from") or ""), meta
        if "headers" in meta and isinstance(meta["headers"], dict):     # {"headers":{"From":"..."}}
            return str(meta["headers"].get("From", "") or ""), meta
        if "headers" in meta and isinstance(meta["headers"], list):     # {"headers":[{"name":"From","value":"..."}]}
            for h in meta["headers"]:
                if str(h.get("name", "")).lower() == "from":
                    return str(h.get("value", "") or ""), meta
    return "", meta

def find_bronze_msg_dir(fs: gcsfs.GCSFileSystem, msg_id: str) -> str | None:
    """Find path like .../dt=YYYY-MM-DD/msgId=<msg_id>"""
    # wildcard search over dates
    pattern = GCS_BRONZE_ROOT.rstrip("/") + f"/dt=*/msgId={msg_id}"
    matches = fs.glob(pattern)
    return matches[0] if matches else None

# --------------- main -------------------
def main():
    fs = gcsfs.GCSFileSystem()

    # pyarrow + gcsfs: strip scheme for dataset
    silver_base = GCS_SILVER_ROOT.replace("gs://", "")
    dataset = ds.dataset(silver_base, filesystem=fs, format="parquet", partitioning=PARTITIONING)
    cols = [c for c in ["press_release_id","release_date","title","company_name"] if c in [f.name for f in dataset.schema]]
    df = dataset.to_table(columns=cols).to_pandas()

    missing = df[df["company_name"].fillna("").str.strip() == ""].copy()
    print(f"Missing company_name rows in silver: {len(missing):,}")

    if missing.empty:
        return

    # take a small sample for printing
    sample = missing.head(MAX_TO_SHOW)

    print("\n--- Inspecting bronze for missing rows ---")
    for _, row in sample.iterrows():
        msg_id = row["press_release_id"]
        msg_dir = find_bronze_msg_dir(fs, msg_id)
        status = {"msg_dir": bool(msg_dir), "meta": False, "eml": False}
        raw_from = ""
        cand = ("", "", "")

        if msg_dir:
            meta_path = msg_dir + "/meta.json"
            eml_path  = msg_dir + "/message.eml"

            if fs.exists(meta_path):
                try:
                    raw_from, meta_obj = read_meta_from_header(fs, meta_path)
                    status["meta"] = True
                except Exception:
                    pass

            if not raw_from and fs.exists(eml_path):
                try:
                    with fs.open(eml_path, "rb") as f:
                        msg = BytesParser(policy=policy.default).parsebytes(f.read())
                    raw_from = msg.get("from", "") or ""
                    status["eml"] = True
                except Exception:
                    pass

            if raw_from:
                cand = parse_company_from_from_header(raw_from)

        # print a concise one-line report
        print(
            f"- {msg_id} | {row['release_date']} | title='{row['title'][:60]}...' | "
            f"bronze_found={status['msg_dir']} meta={status['meta']} eml={status['eml']} | "
            f"From='{raw_from[:70]}{'...' if len(raw_from)>70 else ''}' | "
            f"parsed_company='{cand[2]}'"
        )

if __name__ == "__main__":
    main()
