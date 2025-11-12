#!/usr/bin/env python3
"""
Test the alert system end-to-end with real GCP data.

This script (guidance-only mode):
1. Fetches recent press releases from GCS silver bucket
2. (Optional) applies housekeeping/newsletter gating to remove noise
3. Ensures clean text columns exist (`title_clean`, `full_text_clean`)
4. Runs documents directly through GuidanceChangePlugin via alert detector
5. (Dry-run) prints detected guidance change alerts; (full run) delivers alerts

Usage:
    # Fetch last N days of data and test alerts
    python3 scripts/test_alerts_with_gcp_data.py --days 7
    
    # Dry run (no actual delivery)
    python3 scripts/test_alerts_with_gcp_data.py --days 3 --dry-run
    
    # Test with specific date range
    python3 scripts/test_alerts_with_gcp_data.py --start-date 2025-11-01 --end-date 2025-11-10
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import yaml

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Load environment
from load_env import load_env
load_env()

# Core imports
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import gcsfs

# Pipeline imports
from event_feed_app.config import Settings
from event_feed_app.utils.io import load_from_gcs
from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from event_feed_app.gating.newsletter_quotes_gate import apply_newsletter_gate

# Alert imports
from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.alerts.store import AlertStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_silver_data(start_date: str, end_date: str, max_rows: int = 1000, source: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch press releases from GCS silver bucket using PyArrow dataset with Hive partitioning.
    
    Efficient approach: Uses Arrow dataset API with partition filters to only scan target date partitions.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_rows: Maximum number of rows to fetch
        source: Optional source filter ('gmail', 'globenewswire', etc.)
    
    Returns:
        DataFrame with press releases sorted by release_date descending
    """
    logger.info(f"Fetching press releases from GCS (date range: {start_date} to {end_date}, source={source or 'all'})")
    
    try:
        import gcsfs
        import pyarrow.dataset as ds
        import pyarrow.compute as pc
        from datetime import datetime, timedelta
        
        cfg = Settings()
        base_path = cfg.gcs_silver_root
        
        # Build GCS path (without gs:// scheme for gcsfs compatibility)
        gcs_path = base_path
        if source:
            gcs_path = f"{gcs_path}/source={source}"
        
        logger.info(f"Loading from: gs://{gcs_path}")
        
        # Calculate target date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_back = (end_dt - start_dt).days + 1
        
        # Generate list of target dates (most recent first)
        target_dates = []
        for i in range(days_back):
            date = (end_dt - timedelta(days=i)).strftime('%Y-%m-%d')
            target_dates.append(date)
        
        logger.info(f"Target dates ({len(target_dates)}): {target_dates[:5]}{'...' if len(target_dates) > 5 else ''}")
        
        # Create filesystem
        fs = gcsfs.GCSFileSystem()
        
        # Load dataset with Hive partitioning
        # Note: gcsfs returns paths without gs:// prefix, so we pass base path without scheme
        logger.info("Loading Arrow dataset with Hive partitioning...")
        dataset = ds.dataset(
            gcs_path,
            filesystem=fs,
            partitioning="hive",
            format="parquet"
        )
        
        # Build partition filter for target dates
        # Filter: release_date IN (target_dates)
        filter_expr = ds.field("release_date").isin(target_dates)
        
        # If source not already in path, add source filter
        if not source:
            # Would need to add: ds.field("source") == "gmail", etc.
            # For now, source is pre-filtered via path
            pass
        
        logger.info(f"Scanning partitions with filter: release_date IN {target_dates}")
        
        # Define columns to load (only those that exist in silver schema)
        columns = [
            "press_release_id",
            "company_name", 
            "release_date",
            "title",
            "full_text",
            "source_url",
            "category"
        ]
        
        # Load to Arrow table with partition filter
        table = dataset.to_table(
            filter=filter_expr,
            columns=columns
        )
        
        logger.info(f"Loaded {len(table)} records from Arrow dataset")
        
        if len(table) == 0:
            logger.warning("No data found in target date partitions")
            return pd.DataFrame()
        
        # Convert to pandas
        df = table.to_pandas()
        
        # Ensure full_text and title are filled
        df["full_text"] = df["full_text"].fillna("")
        df["title"] = df["title"].fillna("")
        
        # Sort by release_date descending (most recent first)
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'])
            df = df.sort_values('release_date', ascending=False)
            logger.info(f"Sorted by release_date, latest: {df['release_date'].iloc[0]}")
        
        # Limit to max_rows
        if len(df) > max_rows:
            df = df.head(max_rows).copy()
            logger.info(f"Limited to {max_rows} most recent press releases")
        
        # Convert release_date back to string for consistency with downstream code
        if 'release_date' in df.columns:
            df['release_date'] = df['release_date'].dt.strftime('%Y-%m-%d')
        
        logger.info(f"✓ Fetched {len(df)} press releases from GCS")
        return df
    
    except Exception as e:
        logger.error(f"Failed to fetch from GCS: {e}", exc_info=True)
        logger.info("Ensure you have gcsfs and proper GCP credentials configured")
        raise


def apply_pipeline_gating(df: pd.DataFrame) -> pd.DataFrame:
    """Apply housekeeping and newsletter filters."""
    logger.info(f"Applying gating filters to {len(df)} documents...")
    
    # Housekeeping filter (returns tuple: df, stats)
    before = len(df)
    df_housekeeping, housekeeping_stats = apply_housekeeping_filter(df, title_col="title_clean", body_col="full_text_clean")
    logger.info(f"  Housekeeping: {before} → {len(df_housekeeping)} ({housekeeping_stats.get('gated', 0)} filtered)")
    
    # Newsletter filter (returns tuple: df, stats)
    before = len(df_housekeeping)
    df_final, newsletter_stats = apply_newsletter_gate(df_housekeeping, title_col="title_clean", body_col="full_text_clean")
    logger.info(f"  Newsletter: {before} → {len(df_final)} ({newsletter_stats['gated']} filtered)")
    
    return df_final


#############################
# Guidance Detection ONLY   #
#############################


def process_alerts(
    df: pd.DataFrame,
    dry_run: bool = True,
    min_significance: float = 0.5,
    alert_all: bool = False,
) -> dict:
    """
    Process documents through alert orchestration pipeline.
    
    Args:
        df: DataFrame with press releases
        dry_run: If True, detect but don't deliver
        min_significance: Minimum significance score threshold
    
    Returns:
        Dict with processing statistics
    """
    logger.info(f"\nProcessing {len(df)} documents through alert system...")
    
    # Load plugin YAML config
    config_path = project_root / "src" / "event_feed_app" / "configs" / "significant_events.yaml"
    logger.info(f"Loading plugin config from: {config_path}")
    
    with open(config_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    
    # Extract guidance_change config (events is a list)
    events = yaml_cfg.get("events", [])
    guidance_cfg = next((e for e in events if e.get("key") == "guidance_change"), {})
    logger.info(f"Loaded config with {len(guidance_cfg.get('patterns', {}).get('triggers', []))} trigger patterns")
    
    # Build alert config
    config = {
        "min_significance": min_significance,
        "alert_all_guidance": alert_all,
        "guidance_plugin_config": guidance_cfg,  # Pass to detector for plugin.configure()
        "smtp": {
            "host": os.getenv("SMTP_HOST"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_address": os.getenv("SMTP_FROM_ADDRESS", "alerts@event-feed-app.com"),
        },
        "telegram": {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN")
        }
    }
    # Ensure at least one Telegram user if env provided
    from event_feed_app.alerts.store import AlertStore
    store = AlertStore()
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if tg_chat_id:
        user_id = os.getenv("ALERT_USER_ID", f"tg:{tg_chat_id}")
        prefs = store.get_user_preferences(user_id) or {}
        prefs.update({
            "user_id": user_id,
            "telegram_chat_id": tg_chat_id,
            "email_address": os.getenv("ALERT_EMAIL", ""),
            "alert_types": ["guidance_change"],
            # If alert_all is enabled, set user threshold to 0 to ensure delivery
            "min_significance": 0.0 if alert_all else min_significance,
            "delivery_channels": ["telegram"],
            "active": True,
        })
        store.save_user_preferences(user_id, prefs)
        logger.info(f"Ensured Telegram test user in preferences (user_id={user_id})")

    # Initialize orchestrator with Option A (trust plugin dedup)
    orchestrator = AlertOrchestrator(
        config=config,
        enable_document_dedup=False  # Plugin handles guidance-level dedup
    )
    
    if dry_run:
        logger.info("DRY RUN MODE: Detecting alerts but not delivering")
        
        # Just detect, don't deliver (single batch call for clearer logging)
        docs = df.to_dict("records")
        alerts = orchestrator.detector.detect_alerts(docs)  # type: ignore[arg-type]
        
        logger.info(f"✓ Detected {len(alerts)} alerts")
        
        # Show sample alerts
        for i, alert in enumerate(alerts[:5], 1):
            logger.info(f"\nAlert {i}:")
            logger.info(f"  Company: {alert.get('company_name')}")
            logger.info(f"  Summary: {alert.get('summary')}")
            logger.info(f"  Significance: {alert.get('significance_score'):.3f}")
            logger.info(f"  Metrics: {len(alert.get('metrics', []))}")
        
        return {
            "docs_processed": len(df),
            "alerts_detected": len(alerts),
            "alerts_delivered": 0,
            "delivery_failures": 0,
        }
    else:
        # Full processing with delivery
        docs = df.to_dict("records")
        stats = orchestrator.process_documents(docs)  # type: ignore[arg-type]
        
        logger.info(f"\n{'='*60}")
        logger.info("Alert Processing Results:")
        logger.info(f"  Documents processed: {stats['docs_processed']}")
        logger.info(f"  Alerts detected: {stats['alerts_detected']}")
        logger.info(f"  Alerts deduplicated: {stats['alerts_deduplicated']}")
        logger.info(f"  Alerts delivered: {stats['alerts_delivered']}")
        logger.info(f"  Delivery failures: {stats['delivery_failures']}")
        logger.info(f"{'='*60}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Test alert system with real GCP data")
    parser.add_argument("--days", type=int, help="Fetch last N days of data")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-rows", type=int, default=1000, help="Max rows to fetch")
    parser.add_argument("--source", help="Filter by source (gmail, globenewswire, etc.)")
    parser.add_argument("--dry-run", action="store_true", help="Detect but don't deliver")
    parser.add_argument("--min-significance", type=float, default=0.5, help="Min significance threshold")
    parser.add_argument("--alert-all", action="store_true", help="Alert on any guidance commentary regardless of significance")
    parser.add_argument("--skip-gating", action="store_true", help="Skip housekeeping/newsletter filters")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        # days=1 means "last 1 day" (today only), days=7 means "last 7 days" (today back to 6 days ago)
        start_date = (datetime.now() - timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Default: last 7 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*60}")
    logger.info("Alert System End-to-End Test")
    logger.info(f"{'='*60}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Source filter: {args.source or 'all'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Min significance: {args.min_significance}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Step 1: Fetch data from GCS
        df = fetch_silver_data(start_date, end_date, args.max_rows, source=args.source)
        
        if len(df) == 0:
            logger.warning("No data found for date range")
            return
        
        # Step 2: Apply gating (optional - creates _clean columns)
        if not args.skip_gating:
            df = apply_pipeline_gating(df)
        else:
            # If skipping gating, still need to ensure text columns exist and create _clean variants
            logger.info("Skipping gating; standardizing text columns and creating clean columns...")
            # Standardize full_text column if not present
            if "full_text" not in df.columns:
                if "content" in df.columns:
                    df["full_text"] = df["content"]
                elif "body" in df.columns:
                    df["full_text"] = df["body"]
                else:
                    df["full_text"] = ""
            if "title" not in df.columns:
                if "headline" in df.columns:
                    df["title"] = df["headline"]
                else:
                    df["title"] = ""
            df["title_clean"] = df["title"].fillna("").astype(str)
            df["full_text_clean"] = df["full_text"].fillna("").astype(str)
        
        if len(df) == 0:
            logger.warning("All documents filtered out by gating")
            return
        
        # Step 3: Guidance detection (no category classification)
        logger.info(f"\nRunning guidance change detection on {len(df)} documents...")
        stats = process_alerts(
            df,
            dry_run=args.dry_run,
            min_significance=args.min_significance,
            alert_all=args.alert_all,
        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Test Complete!")
        logger.info(f"  Total documents processed: {len(df)}")
        logger.info(f"  Alerts detected: {stats['alerts_detected']}")
        if not args.dry_run:
            logger.info(f"  Alerts delivered: {stats['alerts_delivered']}")
        logger.info("="*60 + "\n")
        
        if not args.dry_run and stats['alerts_delivered'] > 0:
            logger.info("✓ Check your Telegram/email for alerts!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
