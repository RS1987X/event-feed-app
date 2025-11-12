# src/event_feed_app/tools/gcs_io.py  (or .../utils/gcs_io.py)
"""
GCS I/O utilities with partition-aware loading support.

This module provides efficient Parquet data loading from GCS with support for:
- Hive partition filtering (e.g., filter by date partitions)
- Column selection (load only needed columns)
- Row limits (avoid loading entire datasets)
- Date range queries (optimized for time-partitioned data)

Key functions:
- load_parquet_df(): Basic loader with column selection
- load_parquet_df_partitioned(): Advanced loader with partition filters
- load_parquet_df_date_range(): Optimized date range loader

Performance: For date-partitioned data with 8,500+ files, using partition
filters can reduce load time from 3+ minutes to <30 seconds by only scanning
target partitions.
"""
from __future__ import annotations
import gcsfs
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import Optional, Sequence, Iterable, Dict, Any, List
from datetime import datetime, timedelta

def _as_gcsfs_path(uri: str) -> str:
    # gcsfs expects paths without the "gs://" scheme for dataset discovery
    return uri[5:] if uri.startswith("gs://") else uri

def load_parquet_df(uri: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(_as_gcsfs_path(uri), filesystem=fs, format="parquet")
    table = dataset.to_table(columns=columns)
    return table.to_pandas()

def load_parquet_df_partitioned(
    uri: str,
    columns: Optional[Sequence[str]] = None,
    partition_filters: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_descending: bool = True
) -> pd.DataFrame:
    """
    Load Parquet data with Hive partition filtering support.
    
    Efficient loader that can filter by partition columns (e.g., release_date) 
    before loading data, avoiding full table scans.
    
    Args:
        uri: GCS path to Parquet dataset (with or without gs:// prefix)
        columns: Optional list of columns to load (None = all columns)
        partition_filters: Dict of partition column filters, e.g.:
            {"release_date": ["2025-11-12", "2025-11-11"]} - load specific dates
            {"source": "gmail"} - filter by source
        max_rows: Optional max number of rows to return after sorting
        sort_by: Optional column name to sort by
        sort_descending: Sort descending (True) or ascending (False)
    
    Returns:
        DataFrame with filtered and optionally sorted data
        
    Example:
        # Load only last 7 days of Gmail press releases
        df = load_parquet_df_partitioned(
            uri="gs://bucket/silver/table=press_releases/source=gmail",
            partition_filters={"release_date": ["2025-11-12", "2025-11-11", ...]},
            columns=["press_release_id", "title", "full_text"],
            max_rows=100,
            sort_by="release_date",
            sort_descending=True
        )
    """
    fs = gcsfs.GCSFileSystem()
    path = _as_gcsfs_path(uri)
    
    # Load dataset with Hive partitioning
    dataset = ds.dataset(path, filesystem=fs, partitioning="hive", format="parquet")
    
    # Build partition filter expression
    filter_expr = None
    if partition_filters:
        filters = []
        for field_name, value in partition_filters.items():
            if isinstance(value, list):
                # IN filter for lists
                filters.append(ds.field(field_name).isin(value))
            else:
                # Equality filter for single values
                filters.append(ds.field(field_name) == value)
        
        # Combine filters with AND
        if len(filters) == 1:
            filter_expr = filters[0]
        else:
            filter_expr = filters[0]
            for f in filters[1:]:
                filter_expr = filter_expr & f
    
    # Load to Arrow table with filters
    table = dataset.to_table(filter=filter_expr, columns=columns)
    
    # Convert to pandas
    df = table.to_pandas()
    
    # Sort if requested
    if sort_by and sort_by in df.columns and len(df) > 0:
        df = df.sort_values(sort_by, ascending=not sort_descending)
    
    # Limit rows if requested
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows).copy()
    
    return df

def load_parquet_df_date_range(
    uri: str,
    start_date: str,
    end_date: str,
    date_column: str = "release_date",
    columns: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
    sort_descending: bool = True
) -> pd.DataFrame:
    """
    Load Parquet data for a specific date range using partition filtering.
    
    Optimized for Hive-partitioned datasets with date partitions.
    Only scans partitions within the date range.
    
    Args:
        uri: GCS path to Parquet dataset
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        date_column: Name of the partition column containing dates (default: "release_date")
        columns: Optional list of columns to load
        max_rows: Optional max rows after sorting
        sort_descending: Sort by date descending (newest first)
    
    Returns:
        DataFrame filtered to date range, sorted by date
        
    Example:
        # Load last 7 days of press releases
        df = load_parquet_df_date_range(
            uri="gs://bucket/silver/table=press_releases/source=gmail",
            start_date="2025-11-06",
            end_date="2025-11-12",
            columns=["press_release_id", "title", "full_text"],
            max_rows=100
        )
    """
    # Generate list of dates in range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days_count = (end_dt - start_dt).days + 1
    
    target_dates = []
    for i in range(days_count):
        date = (start_dt + timedelta(days=i)).strftime('%Y-%m-%d')
        target_dates.append(date)
    
    # Use partition filter
    partition_filters = {date_column: target_dates}
    
    return load_parquet_df_partitioned(
        uri=uri,
        columns=columns,
        partition_filters=partition_filters,
        max_rows=max_rows,
        sort_by=date_column,
        sort_descending=sort_descending
    )

# def load_from_gcs(uri: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
#     """
#     Backwards-compatible reader returning a pandas DataFrame.
#     If columns is None and a legacy function exists, delegate to it.
#     Otherwise read via gcsfs + pyarrow.dataset.
#     """
#     if columns is None and legacy_load_from_gcs:
#         return legacy_load_from_gcs(uri)
#     fs = gcsfs.GCSFileSystem()
#     dataset = ds.dataset(uri, filesystem=fs, format="parquet")
#     table = dataset.to_table(columns=columns)
#     return table.to_pandas()

def load_window_parquet_df(
    uri: str,
    start_utc,
    end_utc,
    time_col: str = "published_utc",
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = load_parquet_df(uri, columns=columns)
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        start_ts = pd.Timestamp(start_utc, tz="UTC")
        end_ts = pd.Timestamp(end_utc, tz="UTC")
        m = (df[time_col] >= start_ts) & (df[time_col] < end_ts)
        df = df.loc[m]
    return df

def overwrite_parquet_gcs(uri: str, df: pd.DataFrame) -> None:
    fs = gcsfs.GCSFileSystem()
    with fs.open(uri, "wb") as f:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)

def append_parquet_gcs(uri: str, new_rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(new_rows)
    if not rows:
        return
    fs = gcsfs.GCSFileSystem()
    new_df = pd.DataFrame(rows)
    if fs.exists(uri):
        old_df = load_parquet_df(uri)
        cols = sorted(set(old_df.columns) | set(new_df.columns))
        old_df = old_df.reindex(columns=cols)
        new_df = new_df.reindex(columns=cols)
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df
    overwrite_parquet_gcs(uri, out)

def ensure_empty_parquet(uri: str, schema: pa.schema) -> None:
    fs = gcsfs.GCSFileSystem()
    if fs.exists(uri):
        return
    empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema],
                                 names=[f.name for f in schema])
    with fs.open(uri, "wb") as f:
        pq.write_table(empty, f)