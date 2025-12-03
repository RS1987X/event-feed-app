#!/usr/bin/env python3
"""
Quick DuckDB queries for signals data in GCS.

Examples:
    # Query all events for a date range
    python query_signals.py --query events --start-date 2025-09-01 --end-date 2025-11-30
    
    # Query aggregated alerts
    python query_signals.py --query aggregated --start-date 2025-09-01 --end-date 2025-11-30
    
    # Custom SQL
    python query_signals.py --custom "SELECT company_name, metric, COUNT(*) FROM events GROUP BY 1, 2"
"""

import duckdb
import argparse
from datetime import date, datetime, timedelta


def query_events(start_date: str = None, end_date: str = None, company: str = None):
    """Query individual events from Parquet files."""
    
    # Build GCS path with wildcards for date partitions
    if start_date and end_date:
        # For date range, we'd need to query each partition (DuckDB doesn't support date range in path)
        # Simpler: just use wildcard and filter in WHERE clause
        gcs_path = "gs://event-feed-app-data/signals/guidance_change/events/date=*/*.parquet"
        date_filter = f"WHERE press_release_date BETWEEN '{start_date}' AND '{end_date}'"
    else:
        gcs_path = "gs://event-feed-app-data/signals/guidance_change/events/date=*/*.parquet"
        date_filter = ""
    
    if company:
        company_filter = f"AND company_name ILIKE '%{company}%'" if date_filter else f"WHERE company_name ILIKE '%{company}%'"
    else:
        company_filter = ""
    
    query = f"""
        SELECT 
            press_release_date,
            company_name,
            company_id,
            metric,
            period,
            value_guidance,
            direction,
            confidence
        FROM read_parquet('{gcs_path}', hive_partitioning=1)
        {date_filter}
        {company_filter}
        ORDER BY press_release_date DESC, company_name
    """
    
    conn = duckdb.connect()
    # Install and load GCS extension
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute("SET s3_region='us-central1'")
    
    result = conn.execute(query).fetchdf()
    print(f"\n📊 Found {len(result)} events")
    print(result.to_string())
    return result


def query_aggregated(start_date: str = None, end_date: str = None):
    """Query aggregated alerts from Parquet files."""
    
    gcs_path = "gs://event-feed-app-data/signals/guidance_change/aggregated/date=*/*.parquet"
    
    if start_date and end_date:
        # Note: aggregated alerts partition by detected_at (Dec 1), not press_release_date
        # So we filter by press_release_date in WHERE clause
        date_filter = f"WHERE press_release_date BETWEEN '{start_date}' AND '{end_date}'"
    else:
        date_filter = ""
    
    query = f"""
        SELECT 
            alert_id,
            company_name,
            company_id,
            press_release_date,
            num_guidance_items,
            guidance_summary,
            max_confidence
        FROM read_parquet('{gcs_path}', hive_partitioning=1)
        {date_filter}
        ORDER BY press_release_date DESC
    """
    
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute("SET s3_region='us-central1'")
    
    result = conn.execute(query).fetchdf()
    print(f"\n🔔 Found {len(result)} alerts")
    print(result.to_string())
    return result


def custom_query(sql: str):
    """Run custom SQL query."""
    
    # Replace 'events' and 'aggregated' with actual GCS paths
    sql = sql.replace(
        'events',
        "read_parquet('gs://event-feed-app-data/signals/guidance_change/events/date=*/*.parquet', hive_partitioning=1)"
    )
    sql = sql.replace(
        'aggregated',
        "read_parquet('gs://event-feed-app-data/signals/guidance_change/aggregated/date=*/*.parquet', hive_partitioning=1)"
    )
    
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    conn.execute("SET s3_region='us-central1'")
    
    print(f"\n🔍 Running: {sql}\n")
    result = conn.execute(sql).fetchdf()
    print(result.to_string())
    return result


def main():
    parser = argparse.ArgumentParser(description="Query signals data with DuckDB")
    parser.add_argument('--query', choices=['events', 'aggregated', 'custom'], default='events')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--company', help='Filter by company name (case-insensitive)')
    parser.add_argument('--custom', help='Custom SQL query')
    
    args = parser.parse_args()
    
    if args.query == 'events':
        query_events(args.start_date, args.end_date, args.company)
    elif args.query == 'aggregated':
        query_aggregated(args.start_date, args.end_date)
    elif args.query == 'custom' and args.custom:
        custom_query(args.custom)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
