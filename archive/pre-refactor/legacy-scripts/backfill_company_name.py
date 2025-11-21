# backfill_company_name_inplace.py

import os, json, re, subprocess, io
from email.utils import parseaddr
from email import policy
from email.parser import BytesParser
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import gcsfs

# ---- Config ----
GCS_BRONZE_ROOT = os.getenv("GCS_BRONZE_ROOT", "gs://event-feed-app-data/bronze_raw/source=gmail")
GCS_SILVER_ROOT = os.getenv("GCS_SILVER_ROOT", "gs://event-feed-app-data/silver_normalized/table=press_releases")

# Normalize GCS paths for pyarrow + gcsfs combo (strip scheme) — still useful for reading dataset
BRONZE_BASE = GCS_BRONZE_ROOT.replace("gs://", "")
SILVER_BASE = GCS_SILVER_ROOT.replace("gs://", "")

# Optional: allow overwriting even if silver already has a non-empty name
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() in ("1", "true", "yes")

PARTITIONING = ds.partitioning(
    schema=pa.schema([pa.field("release_date", pa.string())]),  # ← change here
    flavor="hive",
)
# ---- Utilities ----
def backup_silver_path(src: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    dst = src.rstrip("/") + f"_backup_{ts}"
    print(f"Creating backup copy of {src} -> {dst}")
    subprocess.check_call(["gsutil", "-m", "cp", "-r", src, dst])
    return dst

_NOISE = re.compile(r"\b(via|on behalf of|investor relations|press(?: office| team)?)\b", re.I)

def parse_from_header_value(s: str) -> str:
    name, email = parseaddr(s or "")
    name = (name or "").strip().strip('"')
    if name:
        name = re.split(r"\b(via|on behalf of)\b", name, 1, flags=re.I)[0].strip(" ,-|")
        name = _NOISE.sub("", name).strip(" ,-|")
    if not name and email:
        dom = email.split("@",1)[-1].split(".")[0]
        name = dom.capitalize()
    return name

def company_map_from_bronze(fs: gcsfs.GCSFileSystem) -> pd.DataFrame:
    """Walk dt=*/msgId=*/ and extract company name from meta.json or message.eml."""
    rows = []
    # list dt= folders
    for dt_dir in fs.ls(GCS_BRONZE_ROOT):
        if "/dt=" not in dt_dir:
            continue
        # list msgId= folders under each date
        for msg_dir in fs.ls(dt_dir):
            if "msgId=" not in msg_dir:
                continue
            press_release_id = msg_dir.rsplit("msgId=", 1)[-1].strip("/")
            from_header = ""

            meta_path = msg_dir + "/meta.json"
            if fs.exists(meta_path):
                try:
                    with fs.open(meta_path, "rb") as f:
                        meta = json.loads(f.read().decode("utf-8", "ignore"))
                    # try common shapes: {"from": "..."} or {"headers":{"From":"..."}}
                    if isinstance(meta, dict):
                        if "from" in meta:
                            from_header = str(meta.get("from") or "")
                        elif "headers" in meta and isinstance(meta["headers"], dict):
                            from_header = str(meta["headers"].get("From", "") or "")
                except Exception:
                    pass

            if not from_header:
                eml_path = msg_dir + "/message.eml"
                if fs.exists(eml_path):
                    try:
                        with fs.open(eml_path, "rb") as f:
                            msg = BytesParser(policy=policy.default).parsebytes(f.read())
                        from_header = msg.get("from", "") or ""
                    except Exception:
                        pass

            if from_header:
                company = parse_from_header_value(from_header)
                if company:
                    rows.append((press_release_id, company))

    df = pd.DataFrame(rows, columns=["press_release_id", "company_name_candidate"])
    if not df.empty:
        df = df.drop_duplicates("press_release_id", keep="first")
    return df

# ---- Main ----
def main():
    fs = gcsfs.GCSFileSystem()

    print("Scanning bronze for From: headers…")
    cmap = company_map_from_bronze(fs)
    if cmap.empty:
        print("No candidates found in bronze. Exiting without changes.")
        return
    print(f"Found {len(cmap):,} company_name candidates")

    # Read silver dataset to get release_date + current names (we won't rewrite the dataset)
    # Full canonical schema → forces the columns to exist (missing ones become null)
    silver_schema = pa.schema([
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

    silver_ds = ds.dataset(SILVER_BASE, filesystem=fs, format="parquet", partitioning=PARTITIONING, schema=silver_schema)
    silver_df = silver_ds.to_table().to_pandas()
    if "press_release_id" not in silver_df.columns:
        raise RuntimeError("silver missing 'press_release_id'")
    if "company_name" not in silver_df.columns:
        silver_df["company_name"] = ""

    merged = silver_df.merge(cmap, on="press_release_id", how="left")

    # Decide which rows to update
    blank_name = merged["company_name"].fillna("").str.strip() == ""
    cand_ok = merged["company_name_candidate"].fillna("").str.strip() != ""
    mask = (blank_name & cand_ok) | (FORCE_OVERWRITE & cand_ok)

    updates = merged.loc[mask, ["press_release_id", "release_date", "company_name_candidate"]].copy()
    
    # make sure release_date is plain string (not datetime64)
    updates["release_date"] = updates["release_date"].astype(str)

    if updates.empty:
        print("No rows require updates based on current silver + candidates.")
        return

    print(f"Will update {len(updates):,} individual parquet files in silver")

    # Audit file (what we intend to change)
    updates.rename(columns={"company_name_candidate": "company_name"}, inplace=True)
    updates.to_csv("company_name_backfill_changes.csv", index=False, encoding="utf-8")
    print("Wrote: company_name_backfill_changes.csv")

    # Backup silver folder before patching individual files
    try:
        backup = backup_silver_path(GCS_SILVER_ROOT)
        print(f"Backup created at {backup}")
    except FileNotFoundError:
        raise RuntimeError("gsutil not found on PATH. Install Google Cloud SDK.")

    # Patch each msgId parquet in place
    updated = 0
    skipped_existing_nonblank = 0

    for _, r in updates.iterrows():
        msg_id = r["press_release_id"]
        release_date = str(r["release_date"])
        new_name = str(r["company_name"])

        gs_path = f"{GCS_SILVER_ROOT}/release_date={release_date}/msgId={msg_id}.parquet"

        if not fs.exists(gs_path):
            print(f"WARNING: expected {gs_path} but file missing")
            continue

        # Read, patch, write back
        try:
            with fs.open(gs_path, "rb") as f:
                table = pq.read_table(f)

            # Ensure target column exists and is pa.string()
            if "company_name" not in table.column_names:
                table = table.append_column("company_name", pa.array([None] * table.num_rows, type=pa.string()))

            # If not forcing, skip files that already have a non-empty name
            if not FORCE_OVERWRITE:
                col = table.column(table.schema.get_field_index("company_name")).combine_chunks()
                # treat empty/whitespace as blank
                has_nonblank = any((v is not None) and (str(v).strip() != "") for v in col.to_pylist())
                if has_nonblank:
                    skipped_existing_nonblank += 1
                    continue

            # Replace values (all rows in the file) with a single new name
            new_arr = pa.array([new_name] * table.num_rows, type=pa.string())
            idx = table.schema.get_field_index("company_name")
            table = table.set_column(idx, "company_name", new_arr)

            # (Optional) cast other columns to your canonical types
            schema_fix = pa.schema([
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
            # Cast where columns exist; keep extras as-is
            cast_fields = []
            for f in table.schema:
                tgt = schema_fix.field(f.name) if f.name in schema_fix.names else f
                cast_fields.append(pa.field(f.name, tgt.type, nullable=True))
            table = table.cast(pa.schema(cast_fields))

            buf = io.BytesIO()
            pq.write_table(table, buf, compression="zstd")
            with fs.open(gs_path, "wb") as f:
                f.write(buf.getvalue())
            updated += 1
            if updated % 250 == 0:
                print(f"…updated {updated:,} files so far")

        except Exception as e:
            print(f"ERROR updating {gs_path}: {e}")

    print(f"Done. Updated {updated:,} file(s). Skipped {skipped_existing_nonblank:,} already-nonblank file(s).")
    print(f"Backfill complete: {GCS_SILVER_ROOT} patched in-place")

if __name__ == "__main__":
    main()
