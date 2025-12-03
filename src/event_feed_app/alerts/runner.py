#!/usr/bin/env python3
"""
Canonical CLI to run the guidance-change alerts pipeline end-to-end.

Features:
- Loads press releases from GCS Silver (Hive-partitioned) for a date range
- Optional source filter (e.g., source=gmail)
- Applies gating (housekeeping, newsletter) to reduce noise
- Runs GuidanceChangePlugin via AlertOrchestrator (dry-run or full delivery)

Usage examples:
  # Last 3 days, detect only (no delivery)
  event-alerts-run --days 3 --dry-run

  # Specific date range, full run with delivery
  event-alerts-run --start-date 2025-11-01 --end-date 2025-11-10

  # Limit rows and filter by source partition
  event-alerts-run --days 7 --max-rows 500 --source gmail
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
import yaml

from event_feed_app.config import Settings
from event_feed_app.utils.gcs_io import load_parquet_df_date_range
from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from event_feed_app.gating.newsletter_quotes_gate import apply_newsletter_gate
from event_feed_app.alerts.orchestration import AlertOrchestrator
from event_feed_app.alerts.store import AlertStore

logger = logging.getLogger(__name__)


def _build_dataset_uri(cfg: Settings, source: Optional[str]) -> str:
    """Construct the GCS dataset URI for Silver press releases.

    Settings.gcs_silver_root can be either without or with gs://; both are accepted
    by our I/O utils. Optionally append a Hive partition path for source.
    """
    base = cfg.gcs_silver_root.rstrip("/")
    if source:
        return f"{base}/source={source}"
    return base


def _ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure title/full_text columns exist and create *_clean variants.

    Some sources may have different schemas (content/body). This standardizes expected
    columns for downstream gating and detection.
    """
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
    return df


def fetch_data(start_date: str, end_date: str, max_rows: int, source: Optional[str]) -> pd.DataFrame:
    """Fetch press releases from GCS Silver (consolidated) and Bronze (today's batches).
    
    Strategy:
    - For today's date: Read from bronze batches (not yet consolidated)
    - For historical dates: Read from silver consolidated files
    - Deduplicate across both sources by press_release_id (keep last by ingested_at)
    """
    from datetime import date as dt_date
    from google.cloud import storage
    
    cfg = Settings()
    today = dt_date.today().strftime("%Y-%m-%d")
    
    # Parse date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    today_dt = dt_date.today()
    
    columns: List[str] = [
        "press_release_id",
        "company_name",
        "release_date",
        "title",
        "full_text",
        "source_url",
        "category",
        "ingested_at",  # Need for deduplication
    ]
    
    dfs = []
    
    # Load historical data from silver (consolidated)
    if start_dt < today_dt:
        historical_end = min(end_dt, today_dt - timedelta(days=1))
        silver_uri = _build_dataset_uri(cfg, source)
        logger.info(f"Loading silver (historical): {silver_uri} [{start_date} to {historical_end}]")
        
        df_silver = load_parquet_df_date_range(
            uri=silver_uri,
            start_date=start_date,
            end_date=historical_end.strftime("%Y-%m-%d"),
            date_column="release_date",
            columns=columns,
            max_rows=None,  # Don't limit per layer
            sort_descending=True,
        )
        if len(df_silver) > 0:
            logger.info(f"Loaded {len(df_silver)} rows from silver")
            dfs.append(df_silver)
    
    # Load today's data from bronze (batches not yet consolidated)
    if end_dt >= today_dt:
        # Build bronze URI (replace silver_normalized with bronze)
        bronze_base = cfg.gcs_silver_root.replace("silver_normalized", "bronze")
        bronze_uri = bronze_base.rstrip("/")
        if source:
            bronze_uri = f"{bronze_uri}/source={source}"
        
        # Add today's partition
        bronze_uri_today = f"{bronze_uri}/release_date={today}"
        
        logger.info(f"Loading bronze (today's batches): {bronze_uri_today}")
        try:
            df_bronze = load_parquet_df_date_range(
                uri=bronze_uri_today,
                start_date=today,
                end_date=today,
                date_column="release_date",
                columns=columns,
                max_rows=None,
                sort_descending=True,
            )
            if len(df_bronze) > 0:
                logger.info(f"Loaded {len(df_bronze)} rows from bronze (today)")
                dfs.append(df_bronze)
        except Exception as e:
            logger.warning(f"No bronze data for today (may not exist yet): {e}")
    
    # Combine and deduplicate
    if not dfs:
        logger.warning("No data found in requested date range")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(df)} rows")
    
    # Deduplicate: sort by ingested_at, keep last (most recent ingestion)
    if "ingested_at" in df.columns:
        before_dedup = len(df)
        df = df.sort_values("ingested_at").drop_duplicates(
            subset=["press_release_id"], keep="last"
        )
        logger.info(f"Deduplicated: {before_dedup} → {len(df)} rows")
    
    # Apply max_rows limit after deduplication
    if max_rows and len(df) > max_rows:
        logger.info(f"Limiting to {max_rows} rows (from {len(df)})")
        df = df.head(max_rows)
    
    # Normalize date string and basic fields
    if "release_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["release_date"]):
        df["release_date"] = pd.to_datetime(df["release_date"]).dt.strftime("%Y-%m-%d")
    # Ensure nullable text columns are strings
    if "full_text" in df.columns:
        df["full_text"] = df["full_text"].fillna("")
    else:
        df["full_text"] = ""
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
    else:
        df["title"] = ""
    
    # Drop ingested_at before returning (not needed downstream)
    if "ingested_at" in df.columns:
        df = df.drop(columns=["ingested_at"])
    
    return df


def apply_gating(df: pd.DataFrame) -> pd.DataFrame:
    """Apply housekeeping and newsletter gating to reduce noise."""
    before = len(df)
    df1, stats1 = apply_housekeeping_filter(df, title_col="title_clean", body_col="full_text_clean")
    logger.info(f"Housekeeping: {before} → {len(df1)} (gated={stats1.get('gated', 0)})")
    before2 = len(df1)
    df2, stats2 = apply_newsletter_gate(df1, title_col="title_clean", body_col="full_text_clean")
    logger.info(f"Newsletter:   {before2} → {len(df2)} (gated={stats2.get('gated', 0)})")
    return df2


def run_alerts(df: pd.DataFrame, dry_run: bool, min_significance: float, alert_all: bool) -> dict:
    """Run documents through the alert orchestrator (dry/full)."""
    project_root = Path(__file__).resolve().parents[3]  # .../event-feed-app
    cfg_path = project_root / "src" / "event_feed_app" / "configs" / "significant_events.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    with cfg_path.open("r") as f:
        yaml_cfg = yaml.safe_load(f)

    # Extract the guidance_change block for the plugin
    events = yaml_cfg.get("events", []) or []
    guidance_cfg = next((e for e in events if e.get("key") == "guidance_change"), {})

    alert_config = {
        "min_significance": min_significance,
        "alert_all_guidance": alert_all,
        "guidance_plugin_config": guidance_cfg,
        "smtp": {
            "host": os.getenv("SMTP_HOST"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_address": os.getenv("SMTP_FROM_ADDRESS", "alerts@event-feed-app.com"),
        },
        "telegram": {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        },
    }

    # Ensure a Telegram test user if env provided
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
            "min_significance": 0.0 if alert_all else min_significance,
            "delivery_channels": ["telegram"],
            "active": True,
        })
        store.save_user_preferences(user_id, prefs)
        logger.info(f"Ensured Telegram test user (user_id={user_id})")

    orch = AlertOrchestrator(config=alert_config, enable_document_dedup=False)

    docs = df.to_dict("records")
    if dry_run:
        alerts = orch.detector.detect_alerts(docs)  # type: ignore[arg-type]
        logger.info(f"✓ Detected {len(alerts)} alerts (dry run)")
        return {
            "docs_processed": len(df),
            "alerts_detected": len(alerts),
            "alerts_delivered": 0,
            "delivery_failures": 0,
        }
    else:
        stats = orch.process_documents(docs)  # type: ignore[arg-type]
        logger.info(
            "Processing done: docs=%s detected=%s delivered=%s",
            stats.get("docs_processed"), stats.get("alerts_detected"), stats.get("alerts_delivered")
        )
        return stats


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    p = argparse.ArgumentParser(description="Run guidance-change alerts on GCS Silver data")
    p.add_argument("--days", type=int, help="Fetch last N days of data")
    p.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    p.add_argument("--max-rows", type=int, default=1000, help="Max rows to load")
    p.add_argument("--source", help="Filter by source partition (e.g., gmail)")
    p.add_argument("--dry-run", action="store_true", help="Detect but don't deliver")
    p.add_argument("--min-significance", type=float, default=0.5, help="Min significance threshold")
    p.add_argument("--alert-all", action="store_true", help="Alert on any guidance commentary")
    p.add_argument("--skip-gating", action="store_true", help="Skip housekeeping/newsletter filters")

    args = p.parse_args(argv)

    # Resolve date range
    if args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
    elif args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    logger.info("%s", "=" * 60)
    logger.info("Alerts Runner")
    logger.info("Date range: %s → %s | source: %s | dry_run: %s | min_sig: %.2f",
                start_date, end_date, args.source or "all", args.dry_run, args.min_significance)
    logger.info("%s", "=" * 60)

    # Load data
    df = fetch_data(start_date, end_date, max_rows=args.max_rows, source=args.source)
    if len(df) == 0:
        logger.warning("No documents to process")
        return

    # Standardize text columns and optionally apply gating
    df = _ensure_text_columns(df)
    if not args.skip_gating:
        df = apply_gating(df)
        if len(df) == 0:
            logger.warning("All documents filtered out by gating")
            return

    # Run alerts
    stats = run_alerts(df, dry_run=args.dry_run, min_significance=args.min_significance, alert_all=args.alert_all)

    # Summary
    logger.info("%s", "=" * 60)
    logger.info("Summary: docs=%s detected=%s delivered=%s",
                stats.get("docs_processed"), stats.get("alerts_detected"), stats.get("alerts_delivered"))
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
