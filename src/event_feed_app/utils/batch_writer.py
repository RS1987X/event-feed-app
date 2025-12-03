"""
Batch writer for efficient GCS Parquet writes.

Accumulates records in memory and writes them in batches to reduce
file count and improve compression efficiency.
"""
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

logger = logging.getLogger(__name__)


class ParquetBatchWriter:
    """
    Accumulates press releases and writes them in batches to GCS.
    
    Features:
    - In-memory batching (writes when batch_size reached or explicitly flushed)
    - Partitioned writes (by date and source)
    - Atomic file writes (no append needed)
    - Configurable compression and batch size
    
    Usage:
        writer = ParquetBatchWriter(
            bucket_name="event-feed-app-data",
            base_path="bronze/table=press_releases",
            batch_size=100
        )
        
        for pr in press_releases:
            writer.add(pr)
        
        writer.close()  # Flush any remaining records
    """
    
    def __init__(
        self,
        bucket_name: str,
        base_path: str,
        batch_size: int = 100,
        compression: str = "snappy",
        schema: Optional[pa.Schema] = None
    ):
        """
        Initialize batch writer.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            base_path: Base path within bucket (e.g., "bronze/table=press_releases")
            batch_size: Number of records to accumulate before writing
            compression: Parquet compression codec (snappy, zstd, gzip)
            schema: PyArrow schema (optional, inferred from first batch if None)
        """
        self.bucket_name = bucket_name
        self.base_path = base_path.rstrip("/")
        self.batch_size = batch_size
        self.compression = compression
        self.schema = schema
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Batches grouped by partition key (date, source)
        self.batches: Dict[tuple, List[Dict]] = {}
        self.batch_counts: Dict[tuple, int] = {}
        
        logger.info(
            f"Initialized ParquetBatchWriter: bucket={bucket_name}, "
            f"base_path={base_path}, batch_size={batch_size}"
        )
    
    def add(self, record: Dict) -> None:
        """
        Add a record to the batch.
        
        The record must have 'release_date' and 'source' fields for partitioning.
        
        Args:
            record: Dictionary with press release data
        """
        # Extract partition keys
        release_date = record.get("release_date")
        source = record.get("source", "unknown")
        
        if not release_date:
            logger.warning(f"Skipping record without release_date: {record.get('press_release_id')}")
            return
        
        partition_key = (release_date, source)
        
        # Initialize batch for this partition if needed
        if partition_key not in self.batches:
            self.batches[partition_key] = []
            self.batch_counts[partition_key] = 0
        
        # Add to batch
        self.batches[partition_key].append(record)
        
        # Flush if batch size reached
        if len(self.batches[partition_key]) >= self.batch_size:
            self._flush_partition(partition_key)
    
    def _flush_partition(self, partition_key: tuple) -> None:
        """Write a single partition's batch to GCS."""
        batch = self.batches.get(partition_key, [])
        if not batch:
            return
        
        release_date, source = partition_key
        
        # Increment batch counter for this partition
        batch_num = self.batch_counts[partition_key]
        self.batch_counts[partition_key] += 1
        
        # Build path: base_path/source=X/release_date=Y/batch_N_timestamp.parquet
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = (
            f"{self.base_path}/source={source}/release_date={release_date}/"
            f"batch_{batch_num:04d}_{timestamp}.parquet"
        )
        
        try:
            # Convert to PyArrow table
            if self.schema:
                # Use provided schema
                data = {name: [r.get(name) for r in batch] for name in self.schema.names}
                table = pa.Table.from_pydict(data, schema=self.schema)
            else:
                # Infer schema from data
                table = pa.Table.from_pylist(batch)
            
            # Write to buffer
            buf = io.BytesIO()
            pq.write_table(
                table,
                buf,
                compression=self.compression,
                version="2.6",
                use_dictionary=False,  # Better for batched writes
            )
            
            # Upload to GCS
            blob = self.bucket.blob(path)
            blob.upload_from_string(buf.getvalue(), content_type="application/octet-stream")
            
            logger.info(
                f"Wrote batch: {len(batch)} records → gs://{self.bucket_name}/{path} "
                f"({len(buf.getvalue()) / 1024:.1f} KB)"
            )
            
            # Clear batch
            self.batches[partition_key] = []
            
        except Exception as e:
            logger.error(f"Failed to write batch for {partition_key}: {e}")
            raise
    
    def flush(self) -> None:
        """Flush all pending batches to GCS."""
        logger.info(f"Flushing {len(self.batches)} partition(s)...")
        
        for partition_key in list(self.batches.keys()):
            if self.batches[partition_key]:
                self._flush_partition(partition_key)
        
        logger.info("Flush complete")
    
    def close(self) -> None:
        """Flush remaining records and close writer."""
        self.flush()
        logger.info("ParquetBatchWriter closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Default schema for press releases
PRESS_RELEASE_SCHEMA = pa.schema([
    ("press_release_id", pa.string()),
    ("company_name",     pa.string()),
    ("category",         pa.string()),
    ("release_date",     pa.string()),
    ("release_ts_utc",   pa.string()),
    ("ingested_at",      pa.string()),
    ("title",            pa.string()),
    ("full_text",        pa.string()),
    ("source",           pa.string()),
    ("source_url",       pa.string()),
    ("parser_version",   pa.int32()),
    ("schema_version",   pa.int32()),
])
