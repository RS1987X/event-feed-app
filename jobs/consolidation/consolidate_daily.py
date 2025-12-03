#!/usr/bin/env python3
"""
Daily consolidation job for GCS silver layer.

Runs at 00:05 CET to consolidate yesterday's hourly batch files into
a single optimized file per partition.

Features:
- Reads all silver batch files from yesterday (D-1)
- Deduplicates by press_release_id (keeps last)
- Writes consolidated file with optimal compression
- Verifies integrity before deleting batch files
- Monitors and alerts on anomalies

Safety features:
- Atomic writes (write consolidated before deleting batches)
- Verification (row count check)
- Graceful degradation (keeps batches on error)
- Monitoring metrics

Usage:
    python jobs/consolidation/consolidate_daily.py [--date YYYY-MM-DD] [--dry-run]
    
Environment:
    PROJECT_ID: GCP project
    GCS_BUCKET: Storage bucket name
    CONSOLIDATION_MIN_ROWS: Alert if row count < this (default: 50)
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Config
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "event-feed-app-data")
SILVER_BASE = "silver_normalized/table=press_releases"
CONSOLIDATION_MIN_ROWS = int(os.getenv("CONSOLIDATION_MIN_ROWS", "50"))

# Timezone for "yesterday" calculation
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")


def get_yesterday_date(ref_date: datetime = None) -> str:
    """Get yesterday's date in Stockholm timezone."""
    if ref_date is None:
        ref_date = datetime.now(STOCKHOLM_TZ)
    yesterday = (ref_date - timedelta(days=1)).date()
    return yesterday.isoformat()


def list_silver_batches(
    client: storage.Client,
    bucket_name: str,
    date: str,
    source: str = None
) -> List[str]:
    """List all silver batch files for a given date (and optional source)."""
    bucket = client.bucket(bucket_name)
    
    if source:
        prefix = f"{SILVER_BASE}/source={source}/release_date={date}/"
    else:
        prefix = f"{SILVER_BASE}/"
    
    blobs = bucket.list_blobs(prefix=prefix)
    
    # Filter to actual batch files (not directories, not consolidated)
    batch_files = []
    for blob in blobs:
        if blob.name.endswith(".parquet") and "batch_" in blob.name:
            # Check if this file is for our target date
            if f"release_date={date}/" in blob.name:
                batch_files.append(blob.name)
    
    return sorted(batch_files)


def read_silver_batches(
    client: storage.Client,
    bucket_name: str,
    batch_files: List[str]
) -> pd.DataFrame:
    """Read and concatenate all bronze batch files."""
    if not batch_files:
        return pd.DataFrame()
    
    bucket = client.bucket(bucket_name)
    dfs = []
    
    for file_path in batch_files:
        try:
            blob = bucket.blob(file_path)
            content = blob.download_as_bytes()
            
            # Read parquet from bytes
            table = pq.read_table(pa.BufferReader(content))
            df = table.to_pandas()
            dfs.append(df)
            
            logger.debug(f"Read {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Read {len(combined)} rows from {len(batch_files)} batch files")
    
    return combined


def deduplicate_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Deduplicate records by press_release_id, keeping the last occurrence.
    
    Returns:
        (deduplicated_df, num_duplicates)
    """
    if df.empty:
        return df, 0
    
    initial_count = len(df)
    
    # Sort by ingested_at to ensure we keep the latest version
    if "ingested_at" in df.columns:
        df = df.sort_values("ingested_at")
    
    # Drop duplicates, keeping last
    df_dedup = df.drop_duplicates(subset=["press_release_id"], keep="last")
    
    num_duplicates = initial_count - len(df_dedup)
    
    if num_duplicates > 0:
        logger.info(f"Removed {num_duplicates} duplicate records")
    
    return df_dedup, num_duplicates


def write_consolidated_silver(
    client: storage.Client,
    bucket_name: str,
    df: pd.DataFrame,
    date: str,
    source: str
) -> str:
    """Write consolidated dataframe to silver layer."""
    if df.empty:
        logger.warning(f"No data to write for {source}/{date}")
        return ""
    
    # Build path
    path = f"{SILVER_BASE}/source={source}/release_date={date}/consolidated.parquet"
    
    # Convert to Arrow table with schema
    table = pa.Table.from_pandas(df)
    
    # Write to buffer with optimal compression
    buf = pa.BufferOutputStream()
    pq.write_table(
        table,
        buf,
        compression="zstd",
        compression_level=9,  # Max compression for archival
        version="2.6",
        use_dictionary=False,
        row_group_size=10000,  # Optimize for analytics
    )
    
    # Upload to GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(buf.getvalue().to_pybytes(), content_type="application/octet-stream")
    
    size_mb = len(buf.getvalue().to_pybytes()) / (1024 * 1024)
    logger.info(
        f"Wrote consolidated: {len(df)} rows → gs://{bucket_name}/{path} "
        f"({size_mb:.2f} MB)"
    )
    
    return path


def verify_consolidated(
    client: storage.Client,
    bucket_name: str,
    path: str,
    expected_rows: int
) -> bool:
    """Verify the consolidated file was written correctly."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)
        
        if not blob.exists():
            logger.error(f"Consolidated file does not exist: {path}")
            return False
        
        # Read and verify row count
        content = blob.download_as_bytes()
        table = pq.read_table(pa.BufferReader(content))
        actual_rows = len(table)
        
        if actual_rows != expected_rows:
            logger.error(
                f"Row count mismatch! Expected {expected_rows}, got {actual_rows}"
            )
            return False
        
        logger.info(f"Verification passed: {actual_rows} rows")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def delete_silver_batches(
    client: storage.Client,
    bucket_name: str,
    batch_files: List[str],
    dry_run: bool = False
) -> None:
    """Delete silver batch files after successful consolidation."""
    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(batch_files)} batch files")
        return
    
    bucket = client.bucket(bucket_name)
    
    for file_path in batch_files:
        try:
            blob = bucket.blob(file_path)
            blob.delete()
            logger.debug(f"Deleted {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            # Continue with other files
    
    logger.info(f"Deleted {len(batch_files)} bronze batch files")


def consolidate_source_date(
    client: storage.Client,
    bucket_name: str,
    date: str,
    source: str,
    dry_run: bool = False
) -> dict:
    """
    Consolidate all bronze batches for a specific source and date.
    
    Returns:
        dict with consolidation stats
    """
    logger.info(f"Consolidating {source}/{date}...")
    
    stats = {
        "date": date,
        "source": source,
        "batch_count": 0,
        "total_rows": 0,
        "duplicates": 0,
        "final_rows": 0,
        "size_mb": 0,
        "success": False,
    }
    
    try:
        # 1. List bronze batches
        batch_files = list_silver_batches(client, bucket_name, date, source)
        stats["batch_count"] = len(batch_files)
        
        if not batch_files:
            logger.warning(f"No bronze batches found for {source}/{date}")
            stats["success"] = True  # Not an error, just no data
            return stats
        
        # 2. Read all batches
        df = read_silver_batches(client, bucket_name, batch_files)
        stats["total_rows"] = len(df)
        
        if df.empty:
            logger.warning(f"No data in batches for {source}/{date}")
            stats["success"] = True
            return stats
        
        # 3. Deduplicate
        df_dedup, num_dups = deduplicate_records(df)
        stats["duplicates"] = num_dups
        stats["final_rows"] = len(df_dedup)
        
        # 4. Check for suspiciously low row count
        if stats["final_rows"] < CONSOLIDATION_MIN_ROWS:
            logger.warning(
                f"⚠️  Low row count: {stats['final_rows']} < {CONSOLIDATION_MIN_ROWS} "
                f"for {source}/{date}"
            )
        
        # 5. Write consolidated file
        if not dry_run:
            consolidated_path = write_consolidated_silver(
                client, bucket_name, df_dedup, date, source
            )
            
            # 6. Verify
            if not verify_consolidated(client, bucket_name, consolidated_path, stats["final_rows"]):
                raise RuntimeError("Verification failed")
            
            # 7. Delete bronze batches only after verification passes
            delete_silver_batches(client, bucket_name, batch_files, dry_run=False)
        else:
            logger.info(f"[DRY RUN] Would consolidate {stats['final_rows']} rows")
        
        stats["success"] = True
        logger.info(f"✅ Consolidation complete for {source}/{date}")
        
    except Exception as e:
        logger.error(f"❌ Consolidation failed for {source}/{date}: {e}")
        stats["success"] = False
        stats["error"] = str(e)
    
    return stats


def consolidate_all_sources(
    date: str,
    sources: List[str] = None,
    dry_run: bool = False
) -> List[dict]:
    """
    Consolidate bronze batches for all sources (or specified sources).
    
    Args:
        date: Date to consolidate (YYYY-MM-DD)
        sources: List of sources to consolidate (None = discover from GCS)
        dry_run: If True, don't actually write/delete files
    
    Returns:
        List of consolidation stats per source
    """
    client = storage.Client(project=PROJECT_ID)
    
    # Discover sources if not specified
    if sources is None:
        bucket = client.bucket(GCS_BUCKET)
        prefix = f"{BRONZE_BASE}/"
        
        # List all source= directories
        sources_set = set()
        for blob in bucket.list_blobs(prefix=prefix, delimiter="/"):
            # Look for source= in the path
            if "/source=" in blob.name:
                parts = blob.name.split("/source=")
                if len(parts) > 1:
                    source = parts[1].split("/")[0]
                    sources_set.add(source)
        
        sources = sorted(sources_set)
        logger.info(f"Discovered sources: {sources}")
    
    # Consolidate each source
    all_stats = []
    for source in sources:
        stats = consolidate_source_date(client, GCS_BUCKET, date, source, dry_run)
        all_stats.append(stats)
    
    return all_stats


def print_summary(all_stats: List[dict]) -> None:
    """Print consolidation summary."""
    print("\n" + "=" * 80)
    print("CONSOLIDATION SUMMARY")
    print("=" * 80)
    
    total_batches = sum(s["batch_count"] for s in all_stats)
    total_rows = sum(s["total_rows"] for s in all_stats)
    total_final = sum(s["final_rows"] for s in all_stats)
    total_dups = sum(s["duplicates"] for s in all_stats)
    successes = sum(1 for s in all_stats if s["success"])
    
    print(f"\nProcessed {len(all_stats)} source(s)")
    print(f"  Successes: {successes}/{len(all_stats)}")
    print(f"  Total batches: {total_batches}")
    print(f"  Total rows: {total_rows}")
    print(f"  Duplicates removed: {total_dups}")
    print(f"  Final rows: {total_final}")
    
    print("\nPer-source breakdown:")
    for stats in all_stats:
        status = "✅" if stats["success"] else "❌"
        print(
            f"  {status} {stats['source']}/{stats['date']}: "
            f"{stats['batch_count']} batches → {stats['final_rows']} rows"
        )
        if not stats["success"]:
            print(f"     Error: {stats.get('error', 'Unknown')}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Consolidate daily bronze batches to silver")
    parser.add_argument(
        "--date",
        help="Date to consolidate (YYYY-MM-DD). Default: yesterday (Stockholm TZ)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Specific sources to consolidate (default: all discovered)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    
    args = parser.parse_args()
    
    # Determine target date
    if args.date:
        target_date = args.date
    else:
        target_date = get_yesterday_date()
    
    logger.info(f"Starting consolidation for {target_date}")
    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")
    
    # Run consolidation
    all_stats = consolidate_all_sources(
        date=target_date,
        sources=args.sources,
        dry_run=args.dry_run,
    )
    
    # Print summary
    print_summary(all_stats)
    
    # Exit with error if any consolidation failed
    if any(not s["success"] for s in all_stats):
        logger.error("Some consolidations failed")
        sys.exit(1)
    
    logger.info("✅ Consolidation complete")


if __name__ == "__main__":
    main()
