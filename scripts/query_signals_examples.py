"""
Example queries for analyzing guidance signals.

Demonstrates how to query the signals store to:
1. Track guidance changes for a specific company
2. Compare period-over-period guidance
3. Detect significant changes (material revisions)
4. Aggregate by time period and metric
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as fs
from datetime import date
from typing import List, Optional


def query_all_signals(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Query all guidance signals within a date range.
    
    Args:
        start_date: Start of detection date range
        end_date: End of detection date range (inclusive)
    
    Returns:
        DataFrame with all signals
    """
    gcs = fs.GcsFileSystem()
    
    # Build partition paths
    date_range = pd.date_range(start_date, end_date, freq='D')
    paths = [
        f'event-feed-app-data/signals/guidance_change/events/date={d.strftime("%Y-%m-%d")}/'
        for d in date_range
    ]
    
    # Read all partitions
    tables = []
    for path in paths:
        try:
            dataset = pq.ParquetDataset(path, filesystem=gcs)
            tables.append(dataset.read())
        except Exception as e:
            # Partition might not exist
            continue
    
    if not tables:
        return pd.DataFrame()
    
    # Combine all tables
    import pyarrow as pa
    combined = pa.concat_tables(tables)
    return combined.to_pandas()


def query_company_guidance_history(
    company_id: str,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Get all guidance signals for a specific company.
    
    Useful for tracking guidance evolution over time.
    
    Args:
        company_id: Company identifier
        start_date: Start of press release date range
        end_date: End of press release date range
    
    Returns:
        DataFrame sorted by press_release_date
    """
    df = query_all_signals(start_date, end_date)
    
    if df.empty:
        return df
    
    # Filter by company and sort
    company_df = df[df['company_id'] == company_id].copy()
    company_df = company_df.sort_values('press_release_date')
    
    return company_df[['press_release_date', 'period', 'metric', 'metric_kind',
                       'value_low', 'value_high', 'currency', 'direction_hint',
                       'text_snippet', 'alert_id']]


def find_period_changes(
    company_id: str,
    period: str,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Find all times a company updated guidance for a specific period.
    
    Example: Find all Q3-2025 guidance updates from Company X
    
    Args:
        company_id: Company identifier
        period: Period string (e.g., "Q3-2025", "FY2025")
        start_date: Start of search range
        end_date: End of search range
    
    Returns:
        DataFrame sorted by press_release_date showing guidance evolution
    """
    df = query_company_guidance_history(company_id, start_date, end_date)
    
    if df.empty:
        return df
    
    # Filter to specific period
    period_df = df[df['period'] == period].copy()
    
    # Calculate change metrics if multiple updates
    if len(period_df) > 1:
        period_df['revision_number'] = range(1, len(period_df) + 1)
        period_df['days_since_last'] = period_df['press_release_date'].diff().dt.days
    
    return period_df


def detect_significant_changes(
    df: pd.DataFrame,
    delta_pp_threshold: float = 5.0,
    delta_pct_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Filter signals to only those representing significant changes.
    
    Args:
        df: DataFrame of signals
        delta_pp_threshold: Minimum percentage point change (for margins)
        delta_pct_threshold: Minimum percentage change (for revenue/EBITDA)
    
    Returns:
        DataFrame with only significant changes
    """
    if df.empty:
        return df
    
    significant = df[
        (df['delta_pp'].abs() >= delta_pp_threshold) |
        (df['delta_pct'].abs() >= delta_pct_threshold)
    ].copy()
    
    return significant.sort_values('delta_pct', ascending=False)


def aggregate_by_period_and_metric(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate signals by period and metric for trending analysis.
    
    Args:
        df: DataFrame of signals
    
    Returns:
        Aggregated DataFrame with counts and stats
    """
    if df.empty:
        return df
    
    agg_df = df.groupby(['period', 'metric']).agg({
        'alert_id': 'count',
        'company_id': 'nunique',
        'value_low': ['mean', 'min', 'max'],
        'value_high': ['mean', 'min', 'max']
    }).reset_index()
    
    agg_df.columns = ['period', 'metric', 'event_count', 'company_count',
                      'avg_value_low', 'min_value_low', 'max_value_low',
                      'avg_value_high', 'min_value_high', 'max_value_high']
    
    return agg_df.sort_values('event_count', ascending=False)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Example 1: Query all signals from the backfill
    print("=" * 60)
    print("Example 1: Query all signals from backfill")
    print("=" * 60)
    
    signals = query_all_signals(
        start_date=date(2025, 12, 1),  # Detection date
        end_date=date(2025, 12, 1)
    )
    print(f"\nTotal signals: {len(signals)}")
    print(f"Date range: {signals['press_release_date'].min()} to {signals['press_release_date'].max()}")
    print(f"Unique companies: {signals['company_id'].nunique()}")
    
    # Example 2: Track guidance for a specific company
    print("\n" + "=" * 60)
    print("Example 2: Track guidance for a specific company")
    print("=" * 60)
    
    if not signals.empty:
        # Pick a company with multiple events
        company_counts = signals['company_id'].value_counts()
        if len(company_counts) > 0:
            top_company = company_counts.index[0]
            company_history = query_company_guidance_history(
                company_id=top_company,
                start_date=date(2025, 9, 1),
                end_date=date(2025, 11, 30)
            )
            print(f"\nGuidance history for {top_company}:")
            print(company_history[['press_release_date', 'period', 'metric', 'value_low', 'value_high']])
    
    # Example 3: Aggregate by period and metric
    print("\n" + "=" * 60)
    print("Example 3: Aggregate by period and metric")
    print("=" * 60)
    
    agg = aggregate_by_period_and_metric(signals)
    print("\nTop 10 period/metric combinations:")
    print(agg.head(10)[['period', 'metric', 'event_count', 'company_count']])
    
    # Example 4: Find significant changes
    print("\n" + "=" * 60)
    print("Example 4: Significant changes (delta_pp >= 5 or delta_pct >= 10)")
    print("=" * 60)
    
    significant = detect_significant_changes(signals)
    if not significant.empty:
        print(f"\n{len(significant)} significant changes detected")
        print("\nTop 5 by percentage change:")
        print(significant.head(5)[['company_name', 'period', 'metric', 'delta_pp', 'delta_pct', 'press_release_date']])
    else:
        print("\nNo significant changes in current dataset")
