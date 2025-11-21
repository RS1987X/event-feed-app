# inspect_silver_company_names.py
import os
import pyarrow.dataset as ds
import gcsfs
import pandas as pd
import pyarrow as pa

# --- Config ---
GCS_SILVER_ROOT = os.getenv(
    "GCS_SILVER_ROOT",
    "gs://event-feed-app-data/silver_normalized/table=press_releases"
)

PARTITIONING = ds.partitioning(
    schema=pa.schema([pa.field("release_date", pa.string())]),
    flavor="hive",
)

CANONICAL_SCHEMA = pa.schema([
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

def main():
    fs = gcsfs.GCSFileSystem()
    # Strip gs:// for pyarrow+gcsfs
    silver_base = GCS_SILVER_ROOT.replace("gs://", "")

    # Load dataset
    dataset = ds.dataset(
        silver_base,
        filesystem=fs,
        format="parquet",
        partitioning=PARTITIONING,
        schema=CANONICAL_SCHEMA,
    )

    # Select a few cols
    cols = [c for c in ["press_release_id", "release_date", "title", "company_name"] 
            if c in [f.name for f in dataset.schema]]
    table = dataset.to_table(columns=cols)

    # Convert to pandas
    df = table.to_pandas()

    if df.empty:
        print("No rows in silver dataset.")
        return

    sample = df.sample(min(100, len(df)), random_state=42)

    for _, row in sample.iterrows():
        pr_id = (row.get("press_release_id") or "") if isinstance(row, dict) else row["press_release_id"]
        rdate = (row.get("release_date") or "") if isinstance(row, dict) else row["release_date"]
        cname = (row.get("company_name") or "") if isinstance(row, dict) else row["company_name"]
        title = (row.get("title") or "") if isinstance(row, dict) else row["title"]
        print(f"- {pr_id} | {rdate} | {cname} | {str(title)[:80]}...")
    
    # for _, row in sample.iterrows():
    #     print(f"- {row['press_release_id']} | {row['release_date']} | {row['company_name']} | {row['title'][:80]}...")

if __name__ == "__main__":
    main()