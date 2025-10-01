
import gcsfs
import pyarrow.dataset as ds
import pandas as pd
from typing import Optional, Sequence

def load_from_gcs(uri: str) -> pd.DataFrame:
    fs = gcsfs.GCSFileSystem()
    dataset = ds.dataset(uri, filesystem=fs, format="parquet")
    return dataset.to_table().to_pandas()


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