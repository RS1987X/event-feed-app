#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# scripts/backfill_signals.py
"""
Backfill signals from existing alert_payloads to new Parquet-based storage.

Reads JSONL files from gs://event-feed-app-data/feedback/alert_payloads/
and transforms them into the new signals schema for queryable analysis.
"""

import argparse
import logging
from datetime import datetime, date
from typing import List, Dict, Any
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from google.cloud import storage
from event_feed_app.signals.store import SignalStore
from event_feed_app.signals.schema import GuidanceEventSchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_alert_payload(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transform old alert payload format to new signal schema.
    
    Args:
        payload: Alert payload dict from JSONL
    
    Returns:
        Tuple of (events list, aggregated alert dict)
    """
    # Extract common fields
    alert_id = payload.get("alert_id")
    press_release_id = payload.get("event_id")  # Note: event_id is actually press_release_id
    company_id = payload.get("company_name", "").lower().replace(" ", "_").replace(",", "")
    company_name = payload.get("company_name", "Unknown")
    detected_at = payload.get("detected_at")
    
    metadata = payload.get("metadata", {})
    press_release_date = metadata.get("release_date")
    source_url = metadata.get("press_release_url")
    
    # Build events from guidance_items
    events = []
    guidance_items = payload.get("guidance_items", [])
    
    for idx, item in enumerate(guidance_items):
        metric = item.get("metric", "unknown")
        period = item.get("period", "UNKNOWN")
        
        # Generate event ID
        event_id = f"{press_release_id}_{metric}_{period}_{idx}"
        
        # Parse value_str to extract numeric values (if possible)
        value_str = item.get("value_str", "")
        value_low, value_high = _parse_value_str(value_str)
        
        # Determine unit from value_str or metric
        unit = _infer_unit(value_str, metric)
        
        # Build context for schema conversion
        context = {
            "event_id": event_id,
            "press_release_id": press_release_id,
            "alert_id": alert_id,
            "company_id": company_id,
            "company_name": company_name,
            "detected_at": detected_at,
            "comparison": {},  # Old payloads don't have full comparison data
            "confidence": item.get("confidence", 0.0),
            "text_snippet": item.get("text_snippet", "")[:200],
            "press_release_date": press_release_date,
            "source_url": source_url,
            "source_type": None,  # Not available in old format
        }
        
        # Build minimal event dict (old payloads lack full parametrization)
        event = {
            "metric": metric,
            "metric_kind": "level",  # Default assumption
            "period": period,
            "value_low": value_low,
            "value_high": value_high,
            "unit": unit,
            "currency": None,
            "basis": "reported",
            "direction_hint": item.get("direction"),
            "_rule_type": None,
            "_rule_name": None,
            "_trigger_source": None,
        }
        
        event_dict = GuidanceEventSchema.event_to_dict(event, context)
        events.append(event_dict)
    
    # Build aggregated alert (simpler for backfill)
    aggregated = {
        "alert_id": alert_id,
        "event_id": press_release_id,
        "company_id": company_id,
        "company_name": company_name,
        "detected_at": detected_at,
        "guidance_count": len(guidance_items),
        "significance_score": payload.get("significance_score", 0.0),
        "summary": payload.get("summary"),
        "metadata": metadata,
        "guidance_items": guidance_items,
    }
    
    return events, aggregated


def _parse_value_str(value_str: str) -> tuple[float | None, float | None]:
    """
    Parse value_str to extract numeric bounds.
    
    Examples:
        "70.9" -> (70.9, 70.9)
        "255.0-270.0" -> (255.0, 270.0)
        "15.0-17.0%" -> (15.0, 17.0)
        "" -> (None, None)
    """
    if not value_str:
        return None, None
    
    # Remove common suffixes
    clean = value_str.replace("%", "").replace("pp", "").strip()
    
    # Check for arrow (change indicator) - take the part after arrow
    if "→" in clean:
        clean = clean.split("→")[-1].strip()
        # Remove parenthetical change
        if "(" in clean:
            clean = clean.split("(")[0].strip()
    
    # Check for range
    if "-" in clean and not clean.startswith("-"):
        parts = clean.split("-")
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return low, high
            except ValueError:
                pass
    
    # Single value
    try:
        val = float(clean)
        return val, val
    except ValueError:
        return None, None


def _infer_unit(value_str: str, metric: str) -> str:
    """Infer unit from value_str or metric name."""
    if not value_str:
        return "text"
    
    if "%" in value_str or "pp" in value_str:
        return "pct"
    
    # Check metric name
    if any(term in metric.lower() for term in ["margin", "growth", "rate"]):
        return "pct"
    
    if any(term in metric.lower() for term in ["revenue", "ebitda", "earnings", "sales"]):
        return "ccy"
    
    # Default
    return "ccy" if value_str and value_str[0].isdigit() else "text"


def backfill_date(bucket_name: str, date_str: str, signal_store: SignalStore, dry_run: bool = False):
    """
    Backfill signals for a specific date.
    
    Args:
        bucket_name: GCS bucket name
        date_str: Date string in YYYY-MM-DD format
        signal_store: SignalStore instance
        dry_run: If True, don't actually write to storage
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Read alert payloads for this date
    blob_path = f"feedback/alert_payloads/signal_type=guidance_change/{date_str}.jsonl"
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        logger.warning(f"No data found for {date_str} at {blob_path}")
        return 0, 0
    
    logger.info(f"Reading {blob_path}...")
    content = blob.download_as_text()
    
    all_events = []
    all_alerts = []
    
    for line_num, line in enumerate(content.strip().split("\n"), 1):
        if not line.strip():
            continue
        
        try:
            payload = json.loads(line)
            events, aggregated = parse_alert_payload(payload)
            
            all_events.extend(events)
            all_alerts.append(aggregated)
            
        except Exception as e:
            logger.error(f"Error parsing line {line_num}: {e}")
            continue
    
    logger.info(f"Parsed {len(all_events)} events and {len(all_alerts)} alerts from {date_str}")
    
    if not dry_run and all_events:
        # Write to signals storage
        partition_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        signal_store.write_events(all_events, partition_date=partition_date)
        
        for alert in all_alerts:
            signal_store.write_aggregated_alert(alert, partition_date=partition_date)
        
        logger.info(f"✅ Backfilled {date_str}: {len(all_events)} events, {len(all_alerts)} alerts")
    
    return len(all_events), len(all_alerts)


def main():
    parser = argparse.ArgumentParser(description="Backfill signals from alert_payloads to new Parquet storage")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--bucket", default="event-feed-app-data", help="GCS bucket name")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write")
    
    args = parser.parse_args()
    
    # Initialize signal store
    signal_store = SignalStore(bucket_name=args.bucket)
    
    # Parse date range
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    logger.info(f"Backfilling signals from {start} to {end} (dry_run={args.dry_run})")
    
    total_events = 0
    total_alerts = 0
    current = start
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        try:
            events, alerts = backfill_date(args.bucket, date_str, signal_store, args.dry_run)
            total_events += events
            total_alerts += alerts
        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}", exc_info=True)
        
        current = current.replace(day=current.day + 1)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Backfill complete!")
    logger.info(f"Total events: {total_events}")
    logger.info(f"Total alerts: {total_alerts}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
