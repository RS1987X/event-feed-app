from __future__ import annotations
from typing import Iterable, List, Dict
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs

def _fs() -> pafs.FileSystem:
    # Works with Application Default Credentials
    return pafs.GcsFileSystem()  # type: ignore

def file_exists(path: str) -> bool:
    return _fs().get_file_info(path).type != pafs.FileType.NotFound

def read_parquet(path: str) -> List[Dict]:
    if not file_exists(path): return []
    table = pq.read_table(path, filesystem=_fs())
    return table.to_pylist()

def overwrite_parquet(path: str, rows: Iterable[Dict], schema: pa.schema) -> None:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    with _fs().open_output_stream(path) as f:
        pq.write_table(table, f)

def append_parquet(path: str, new_rows: Iterable[Dict], schema: pa.schema) -> None:
    new_tbl = pa.Table.from_pylist(list(new_rows), schema=schema)
    if not file_exists(path):
        overwrite_parquet(path, new_rows, schema); return
    # small scale: read + concat + overwrite
    old_tbl = pq.read_table(path, filesystem=_fs())
    pq.write_table(pa.concat_tables([old_tbl, new_tbl]), _fs().open_output_stream(path))

def guidance_versions_schema() -> pa.schema:
    return pa.schema([
        pa.field("company_id", pa.string()),
        pa.field("period", pa.string()),
        pa.field("metric", pa.string()),
        pa.field("metric_kind", pa.string()),  # level|growth|margin
        pa.field("basis", pa.string()),        # reported|cc_fx|organic
        pa.field("unit", pa.string()),         # ccy|pct
        pa.field("currency", pa.string()).with_nullable(True),
        pa.field("value_type", pa.string()),   # point|range
        pa.field("value_origin", pa.string()), # numeric
        pa.field("value_low", pa.float64()),
        pa.field("value_high", pa.float64()),
        pa.field("observed_at_utc", pa.timestamp("us")),
        pa.field("source_event_id", pa.string()),
        pa.field("source_url", pa.string()).with_nullable(True),
    ])

def events_schema() -> pa.schema:
    return pa.schema([
        pa.field("event_id", pa.string()),
        pa.field("event_key", pa.string()),
        pa.field("taxonomy_class", pa.string()),
        pa.field("company_id", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("published_utc", pa.timestamp("us")),
        pa.field("first_seen_utc", pa.timestamp("us")),
        pa.field("doc_id", pa.string()),
        pa.field("cluster_id", pa.string()).with_nullable(True),
        pa.field("is_significant", pa.bool_()),
        pa.field("sig_score", pa.float64()),
        pa.field("decision_version", pa.string()),
        pa.field("evidence_snippet", pa.string()),
        pa.field("features_json", pa.string()),
        pa.field("details_json", pa.string()),
    ])
