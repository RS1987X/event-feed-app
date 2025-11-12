
import gcsfs
import pyarrow.dataset as ds
import pandas as pd
from typing import Optional, Sequence, Dict, Any

def load_from_gcs(
    uri: str, 
    columns: Optional[Sequence[str]] = None,
    partition_filters: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load Parquet data from GCS with optional partition filtering.
    
    Args:
        uri: GCS path (e.g., "gs://bucket/path" or "bucket/path")
        columns: Optional list of columns to load
        partition_filters: Optional dict of partition filters, e.g.:
            {"release_date": ["2025-11-12", "2025-11-11"]}
        max_rows: Optional limit on number of rows returned
    
    Returns:
        DataFrame with loaded data
        
    Examples:
        # Load all data
        df = load_from_gcs("gs://bucket/data.parquet")
        
        # Load specific columns only
        df = load_from_gcs("gs://bucket/data.parquet", columns=["id", "title"])
        
        # Load with partition filtering (for Hive-partitioned datasets)
        df = load_from_gcs(
            "gs://bucket/table/source=gmail",
            partition_filters={"release_date": ["2025-11-12"]},
            max_rows=100
        )
    """
    # Strip gs:// prefix if present (gcsfs compatibility)
    path = uri[5:] if uri.startswith("gs://") else uri
    
    fs = gcsfs.GCSFileSystem()
    
    # Determine if we have partitioning based on path structure or explicit filters
    use_hive = partition_filters is not None or "=" in path
    
    if use_hive:
        dataset = ds.dataset(path, filesystem=fs, format="parquet", partitioning="hive")
    else:
        dataset = ds.dataset(path, filesystem=fs, format="parquet")
    
    # Build filter expression if partition filters provided
    filter_expr = None
    if partition_filters:
        filters = []
        for field_name, value in partition_filters.items():
            if isinstance(value, list):
                filters.append(ds.field(field_name).isin(value))
            else:
                filters.append(ds.field(field_name) == value)
        
        if len(filters) == 1:
            filter_expr = filters[0]
        else:
            filter_expr = filters[0]
            for f in filters[1:]:
                filter_expr = filter_expr & f
    
    # Load table
    table = dataset.to_table(filter=filter_expr, columns=columns)
    df = table.to_pandas()
    
    # Apply row limit if specified
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    
    return df


# def load_from_gcs(uri: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
#     """
#     Unified reader used by both the orchestrator and the guidance job.
#     Returns a pandas DataFrame. If 'columns' is provided, reads only those columns.
#     """
#     fs = gcsfs.GCSFileSystem()
#     dataset = ds.dataset(uri, filesystem=fs, format="parquet")
#     table = dataset.to_table(columns=columns)
#     return table.to_pandas()  # keep defaults to avoid dtype surprises

def append_row(path: str, row: dict):
    import csv, os
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)