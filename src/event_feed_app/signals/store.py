# SPDX-License-Identifier: MIT
# src/event_feed_app/signals/store.py
"""
Signal storage for structured event data in GCS.

Provides write and query operations for signals stored as Parquet files,
optimized for time-series analysis and cross-company comparisons.
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

from .schema import GuidanceEventSchema, AggregatedAlertSchema

logger = logging.getLogger(__name__)


class SignalStore:
    """
    Storage layer for detected signals in GCS.
    
    Writes individual events and aggregated alerts to Parquet files,
    partitioned by date for efficient querying.
    """
    
    def __init__(
        self,
        bucket_name: str = "event-feed-app-data",
        base_path: str = "signals/guidance_change"
    ):
        """
        Initialize signal store.
        
        Args:
            bucket_name: GCS bucket name
            base_path: Base path within bucket (e.g., "signals/guidance_change")
        """
        self.bucket_name = bucket_name
        self.base_path = base_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Schema handlers
        self.event_schema = GuidanceEventSchema.get_schema()
        self.alert_schema = AggregatedAlertSchema.get_schema()
    
    def write_events(
        self,
        events: List[Dict[str, Any]],
        partition_date: Optional[date] = None
    ) -> str:
        """
        Write individual guidance events to Parquet.
        
        Args:
            events: List of event dicts matching GuidanceEventSchema
            partition_date: Date for partitioning (uses press_release_date from first event if not provided)
        
        Returns:
            GCS path where data was written
        """
        if not events:
            logger.warning("No events to write")
            return ""
        
        # Use press_release_date from first event for partitioning if not specified
        if partition_date is None:
            first_pr_date = events[0].get('press_release_date')
            if isinstance(first_pr_date, str):
                partition_date = datetime.strptime(first_pr_date, "%Y-%m-%d").date()
            elif isinstance(first_pr_date, date):
                partition_date = first_pr_date
            else:
                partition_date = date.today()
        
        partition_str = partition_date.strftime("%Y-%m-%d")
        
        # Convert to DataFrame and validate schema
        df = pd.DataFrame(events)
        
        # Ensure all schema columns exist (fill missing with None)
        for field in self.event_schema:
            if field.name not in df.columns:
                df[field.name] = None
        
        # Reorder columns to match schema
        df = df[[field.name for field in self.event_schema]]
        
        # Convert to PyArrow table with schema enforcement
        table = pa.Table.from_pandas(df, schema=self.event_schema)
        
        # Write to GCS with PR timestamp-based filename for idempotency
        # Extract press_release_date from first event (all events are from same PR)
        path = f"{self.base_path}/events/date={partition_str}"
        pr_date = events[0].get('press_release_date')
        if isinstance(pr_date, str):
            pr_datetime = datetime.strptime(pr_date, "%Y-%m-%d")
        elif isinstance(pr_date, (datetime, date)):
            pr_datetime = pr_date if isinstance(pr_date, datetime) else datetime.combine(pr_date, datetime.min.time())
        else:
            pr_datetime = datetime.utcnow()  # Fallback to current time
        
        # Use PR timestamp for filename (ensures idempotency - same PR = same filename)
        timestamp = pr_datetime.strftime('%Y%m%d_%H%M%S%f')
        blob_name = f"{path}/events_{timestamp}.parquet"
        
        # Write to local buffer first
        import io
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        
        # Upload to GCS
        buffer.seek(0)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(buffer, content_type="application/octet-stream")
        
        gcs_path = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Wrote {len(events)} events to {gcs_path}")
        return gcs_path
    
    def write_aggregated_alert(
        self,
        alert: Dict[str, Any],
        partition_date: Optional[date] = None
    ) -> str:
        """
        Write aggregated alert to Parquet.
        
        Args:
            alert: Alert dict from detector._build_aggregated_alert()
            partition_date: Date for partitioning (uses press_release_date from alert if not provided)
        
        Returns:
            GCS path where data was written
        """
        # Use press_release_date from alert metadata for partitioning if not specified
        if partition_date is None:
            pr_date = alert.get('metadata', {}).get('press_release_date')
            if isinstance(pr_date, str):
                partition_date = datetime.strptime(pr_date, "%Y-%m-%d").date()
            elif isinstance(pr_date, date):
                partition_date = pr_date
            else:
                partition_date = date.today()
        
        partition_str = partition_date.strftime("%Y-%m-%d")
        
        # Convert to schema-compliant dict
        alert_dict = AggregatedAlertSchema.alert_to_dict(alert)
        
        # Create DataFrame
        df = pd.DataFrame([alert_dict])
        
        # Ensure all schema columns exist
        for field in self.alert_schema:
            if field.name not in df.columns:
                df[field.name] = None
        
        # Reorder columns to match schema
        df = df[[field.name for field in self.alert_schema]]
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df, schema=self.alert_schema)
        
        # Write to GCS
        path = f"{self.base_path}/aggregated/date={partition_str}"
        blob_name = f"{path}/alert_{alert.get('alert_id', 'unknown')}.parquet"
        
        import io
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="snappy")
        
        buffer.seek(0)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(buffer, content_type="application/octet-stream")
        
        gcs_path = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Wrote aggregated alert to {gcs_path}")
        return gcs_path
    
    def query_events_by_company(
        self,
        company_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Query all events for a specific company.
        
        Args:
            company_id: Company identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame of matching events
        """
        # Build path pattern
        if start_date and end_date:
            # Query specific date range
            date_list = pd.date_range(start_date, end_date, freq="D")
            paths = [
                f"gs://{self.bucket_name}/{self.base_path}/events/date={d.strftime('%Y-%m-%d')}/*.parquet"
                for d in date_list
            ]
        else:
            # Query all dates
            paths = [f"gs://{self.bucket_name}/{self.base_path}/events/**/*.parquet"]
        
        # Read Parquet files
        try:
            if len(paths) == 1:
                df = pd.read_parquet(paths[0])
            else:
                # Read multiple files and concatenate
                dfs = [pd.read_parquet(p) for p in paths]
                df = pd.concat(dfs, ignore_index=True)
            
            # Filter by company
            df = df[df["company_id"] == company_id]
            
            # Additional date filtering if needed
            if start_date:
                df = df[df["detected_at"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["detected_at"] <= pd.Timestamp(end_date)]
            
            return df.sort_values("detected_at")
        
        except Exception as e:
            logger.error(f"Error querying events: {e}")
            return pd.DataFrame()
    
    def query_events_by_period(
        self,
        period: str,
        metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query all events for a specific period across all companies.
        
        Useful for cross-company comparisons (e.g., "Show all Q3-2025 revenue guidance").
        
        Args:
            period: Period string (e.g., "Q3-2025", "FY2025")
            metric: Optional metric filter
        
        Returns:
            DataFrame of matching events
        """
        # Read all events (TODO: optimize with partition pruning)
        path = f"gs://{self.bucket_name}/{self.base_path}/events/**/*.parquet"
        
        try:
            df = pd.read_parquet(path)
            
            # Filter by period
            df = df[df["period"] == period]
            
            # Optional metric filter
            if metric:
                df = df[df["metric"] == metric]
            
            return df.sort_values(["company_name", "metric"])
        
        except Exception as e:
            logger.error(f"Error querying events: {e}")
            return pd.DataFrame()
    
    def get_company_guidance_history(
        self,
        company_id: str,
        metric: str
    ) -> pd.DataFrame:
        """
        Get time-series of guidance for a specific company/metric.
        
        Useful for tracking how guidance evolved over time.
        
        Args:
            company_id: Company identifier
            metric: Metric name (e.g., "revenue", "ebitda")
        
        Returns:
            DataFrame sorted by period, showing guidance evolution
        """
        df = self.query_events_by_company(company_id)
        
        if df.empty:
            return df
        
        # Filter by metric
        df = df[df["metric"] == metric]
        
        # Select relevant columns
        cols = [
            "period", "value_low", "value_high", "unit", "basis",
            "prior_value_low", "prior_value_high", "delta_pct", "delta_pp",
            "change_label", "detected_at"
        ]
        
        return df[cols].sort_values("period")
