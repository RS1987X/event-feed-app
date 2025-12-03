#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/backfill_signals_from_silver.py
"""
Backfill signals by reprocessing ALL historical PRs from silver layer.

This is the CORRECT way to backfill - it:
1. Reads PRs from silver layer (canonical source of truth)
2. Runs current detection logic (with all improvements)
3. Extracts full parametrization with current NLP
4. Writes to new signals/ storage in Parquet format

This will capture historical PRs that weren't detected before due to
earlier detection logic being less sophisticated.
"""

import argparse
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from google.cloud import storage
from event_feed_app.alerts.detector import GuidanceAlertDetector
from event_feed_app.signals.store import SignalStore
from event_feed_app.alerts.runner import fetch_data
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load YAML configuration for detection."""
    config_path = Path("src/event_feed_app/configs/significant_events.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_batch(
    detector: GuidanceAlertDetector,
    docs: List[Dict[str, Any]],
    batch_num: int,
    total_batches: int
) -> tuple[int, int, int]:
    """
    Process a batch of documents through the detector.
    
    Args:
        detector: GuidanceAlertDetector instance
        docs: List of document dicts
        batch_num: Current batch number
        total_batches: Total number of batches
    
    Returns:
        Tuple of (processed_count, detected_count, error_count)
    """
    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(docs)} documents)")
    
    processed = 0
    detected = 0
    errors = 0
    
    for doc in docs:
        try:
            alerts = detector.detect_alerts([doc])
            processed += 1
            
            if alerts:
                detected += len(alerts)
                
        except Exception as e:
            errors += 1
            pr_id = doc.get("press_release_id", "unknown")
            logger.error(f"Error processing {pr_id}: {e}")
    
    return processed, detected, errors


def backfill_date_range(
    start_date: str,
    end_date: str,
    max_rows: int = 50000,
    batch_size: int = 100,
    source: Optional[str] = None,
    write_signals: bool = True,
    dry_run: bool = False
):
    """
    Backfill signals for a date range by reprocessing silver layer PRs.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_rows: Maximum PRs to fetch
        batch_size: Batch size for processing
        source: Filter by source (globenewswire, prnewswire, etc.) or None for all
        write_signals: Whether to write signals to storage
        dry_run: If True, process but don't write
    """
    logger.info(f"{'='*80}")
    logger.info(f"BACKFILLING SIGNALS FROM SILVER LAYER")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Source filter: {source or 'ALL'}")
    logger.info(f"Max rows: {max_rows}")
    logger.info(f"Write signals: {write_signals and not dry_run}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"{'='*80}\n")
    
    # Load configuration
    yaml_config = load_config()
    
    # Extract guidance_change event config from YAML
    guidance_event = next((e for e in yaml_config.get('events', []) if e.get('key') == 'guidance_change'), {})
    
    # Wrap config properly for detector
    config = {
        'write_signals': write_signals and not dry_run,
        'alert_all_guidance': False,  # Use normal significance threshold
        'guidance_plugin_config': guidance_event  # Pass guidance event config to plugin
    }
    
    # Initialize detector with signal writing enabled
    signal_store = SignalStore() if write_signals else None
    detector = GuidanceAlertDetector(
        config=config,
        signal_store=signal_store
    )
    
    # Fetch data from silver layer
    logger.info("Fetching PRs from silver layer...")
    df = fetch_data(
        start_date=start_date,
        end_date=end_date,
        max_rows=max_rows,
        source=source
    )
    
    logger.info(f"Fetched {len(df)} PRs from silver layer")
    
    if df.empty:
        logger.warning("No data fetched!")
        return
    
    # Show distribution by source
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        logger.info(f"\nSource distribution:")
        for src, count in source_counts.items():
            logger.info(f"  {src}: {count}")
    
    # Convert to list of dicts
    docs = df.to_dict('records')
    
    # Process in batches
    total_processed = 0
    total_detected = 0
    total_errors = 0
    
    num_batches = (len(docs) + batch_size - 1) // batch_size
    
    logger.info(f"\nProcessing {len(docs)} documents in {num_batches} batches...")
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        processed, detected, errors = process_batch(
            detector,
            batch,
            batch_num,
            num_batches
        )
        
        total_processed += processed
        total_detected += detected
        total_errors += errors
        
        # Log progress every 10 batches
        if batch_num % 10 == 0:
            logger.info(f"Progress: {total_processed}/{len(docs)} processed, "
                       f"{total_detected} alerts generated, {total_errors} errors")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total PRs processed: {total_processed}")
    logger.info(f"Total alerts generated: {total_detected}")
    logger.info(f"Detection rate: {(total_detected/total_processed*100):.2f}%")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Dry run: {dry_run}")
    
    if write_signals and not dry_run:
        logger.info(f"\n✅ Signals written to gs://event-feed-app-data/signals/guidance_change/")
        logger.info(f"   - Individual events: signals/guidance_change/events/date=YYYY-MM-DD/")
        logger.info(f"   - Aggregated alerts: signals/guidance_change/aggregated/date=YYYY-MM-DD/")
    else:
        logger.info(f"\n⚠️  No signals written (dry_run={dry_run}, write_signals={write_signals})")
    
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill signals by reprocessing historical PRs from silver layer"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Maximum number of PRs to fetch (default: 50000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--source",
        help="Filter by source (globenewswire, prnewswire, etc.)"
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Don't write signals to storage (for testing detection logic)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process but don't write (same as --no-write)"
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Dates must be in YYYY-MM-DD format")
        sys.exit(1)
    
    backfill_date_range(
        start_date=args.start_date,
        end_date=args.end_date,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        source=args.source,
        write_signals=not args.no_write,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
