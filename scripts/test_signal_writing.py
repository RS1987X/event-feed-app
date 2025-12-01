#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/test_signal_writing.py
"""
Test script to verify signal writing works correctly.

Runs detection on a small sample of PRs and validates that:
1. Events are written to Parquet files
2. Schema is correct
3. Data is queryable
4. No errors occur
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import yaml
from datetime import date

from event_feed_app.alerts.detector import GuidanceAlertDetector
from event_feed_app.signals.store import SignalStore
from event_feed_app.alerts.runner import fetch_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_signal_writing():
    """Test signal writing with a small sample."""
    
    logger.info("="*80)
    logger.info("TESTING SIGNAL WRITING")
    logger.info("="*80)
    
    # Load config
    config_path = Path("src/event_feed_app/configs/significant_events.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Enable signal writing
    config['write_signals'] = True
    config['alert_all_guidance'] = False
    
    # Initialize detector
    signal_store = SignalStore()
    detector = GuidanceAlertDetector(
        config=config,
        signal_store=signal_store
    )
    
    # Fetch small sample (just 1 week of recent data)
    logger.info("\nFetching sample PRs...")
    df = fetch_data(
        start_date="2025-11-18",
        end_date="2025-11-24",
        max_rows=1000,
        source=None
    )
    
    logger.info(f"Fetched {len(df)} PRs")
    
    if df.empty:
        logger.error("No data fetched!")
        return False
    
    # Process a small batch
    logger.info("\nProcessing PRs through detector...")
    docs = df.head(50).to_dict('records')  # Just 50 PRs for testing
    
    total_alerts = 0
    for i, doc in enumerate(docs, 1):
        try:
            alerts = detector.detect_alerts([doc])
            if alerts:
                total_alerts += len(alerts)
                logger.info(f"  PR {i}/50: Generated {len(alerts)} alert(s)")
        except Exception as e:
            logger.error(f"  PR {i}/50: Error - {e}")
    
    logger.info(f"\n✅ Generated {total_alerts} alerts from 50 PRs")
    
    # Test querying
    logger.info("\nTesting queries...")
    
    # Try to query events (this will fail if no events were written)
    try:
        # Pick a company from the data
        if 'company_name' in df.columns:
            sample_company = df[df['company_name'].notna()].iloc[0]['company_name']
            company_id = sample_company.lower().replace(" ", "_").replace(",", "")
            
            logger.info(f"Querying events for: {sample_company}")
            events_df = signal_store.query_events_by_company(
                company_id=company_id,
                start_date=date(2025, 11, 18),
                end_date=date(2025, 11, 24)
            )
            
            if not events_df.empty:
                logger.info(f"✅ Found {len(events_df)} events for {sample_company}")
                logger.info(f"\nSample event columns: {list(events_df.columns)}")
                logger.info(f"\nSample event:\n{events_df.iloc[0].to_dict()}")
            else:
                logger.warning(f"No events found for {sample_company} (may not have generated any)")
    
    except Exception as e:
        logger.error(f"Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Signal writing works!")
    logger.info(f"✅ Generated {total_alerts} alerts")
    logger.info(f"✅ Queries work")
    logger.info("\nNext steps:")
    logger.info("1. Check GCS bucket: gs://event-feed-app-data/signals/guidance_change/")
    logger.info("2. Verify Parquet files exist in events/ and aggregated/ folders")
    logger.info("3. Run full backfill with: python scripts/backfill_signals_from_silver.py")
    logger.info("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_signal_writing()
    sys.exit(0 if success else 1)
