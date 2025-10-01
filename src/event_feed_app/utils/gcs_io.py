# src/event_feed_app/tools/gcs_io.py  (or .../utils/gcs_io.py)
from __future__ import annotations
import gcsfs
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import Optional, Sequence, Iterable, Dict, Any

def _as_gcsfs_path(uri: str) -> str:
    # gcsfs expects paths without the "gs://" scheme for dataset discovery
    return uri[5:] if uri.startswith("gs://") else uri

def load_parquet_df(uri: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(_as_gcsfs_path(uri), filesystem=fs, format="parquet")
    table = dataset.to_table(columns=columns)
    return table.to_pandas()

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