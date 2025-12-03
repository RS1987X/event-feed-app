# GCS Batch Writing Implementation

## Problem Statement

Current ingestion writes one Parquet file per press release (200-300 files/day):
- Poor compression efficiency (tiny files can't compress well)
- High file overhead (metadata per file)
- Slow partition scans (many small files to list/read)
- No benefit from Parquet's columnar format at single-row scale

## Solution Architecture

Two-layer approach: raw audit trail + consolidated analytics layer.

### 1. Bronze Layer: Hourly Batching
### 1. Bronze Raw: Immutable Audit Trail
- **Purpose**: Preserve original source material for reprocessing
- **Path**: `bronze_raw/source={source}/dt={date}/msgId={id}/`
- **Content**: 
  - Gmail: `.eml` files + `meta.json`
  - RSS: `.html.gz` + `entry.json.gz`
- **Lifecycle**: Never deleted, permanent audit trail


### 2. Silver Layer: Batched Ingestion + Daily Consolidation
- **Purpose**: Efficient analytics-ready data with deduplication
- **Ingestion**: Write batches (snappy compression, fast)
  - Path: `silver_normalized/table=press_releases/source={source}/release_date={date}/batch_*.parquet`
  - Batch Size: 100 records (configurable)
  - File Pattern: `batch_{count:04d}_{timestamp}.parquet`
  - Expected: 10-15 batch files per day (down from 200-300)
- **Consolidation**: Daily merge (ZSTD level 9, maximum compression)
  - Path: `silver_normalized/table=press_releases/source={source}/release_date={date}/consolidated.parquet`
  - Schedule: 00:05 CET daily
  - Process: Read batches → Deduplicate → Write consolidated → Verify → Delete batches
  - Expected: 1 consolidated file per source/date

- **Expected Volume**: 1 consolidated file per source/date
### 3. Smart Loading: Automatic Deduplication
- **Alert Runner**: Reads all Parquet files from silver (batches + consolidated)
- **Deduplication**: By press_release_id, keep last by ingested_at
- **Backward Compatible**: Works seamlessly with existing data

- **Deduplication**: By press_release_id, keep last by ingested_at
- **Backward Compatible**: Works with existing silver-only data

## Implementation Components

### ParquetBatchWriter (`src/event_feed_app/utils/batch_writer.py`)

**Purpose**: Reusable utility for batched Parquet writes to GCS

**Key Features**:
- In-memory batching partitioned by (release_date, source)
- Auto-flush when batch_size threshold reached
- Context manager support for automatic cleanup
- Configurable compression and batch size
- Thread-safe batch counters per partition

**Usage**:
```python
  base_path="silver_normalized/table=press_releases",
    batch_size=100,
    compression="snappy"
)

# Add records (auto-flushes at batch_size)
writer.add({
    "press_release_id": "...",
    "company_name": "...",
    "release_date": "2025-01-15",
    "source": "gmail",
    # ... other fields
})

# Flush remaining records and cleanup
writer.close()
```

**API**:
- `add(record: Dict)`: Add record to batch, auto-flush when full
- `flush()`: Flush all pending batches
- `close()`: Final flush and cleanup
- `_flush_partition(partition_key)`: Internal flush for one partition

### Daily Consolidation Job (`jobs/consolidation/consolidate_daily.py`)

**Purpose**: Consolidate yesterday's bronze batches into silver layer

**Key Features**:
- Auto-discovers sources from GCS bronze layer
- Per-source consolidation with statistics
- Atomic writes with verification before cleanup
- Graceful error handling (keeps batches on failure)
- Monitoring and anomaly detection (< 50 rows alert)
- CLI with dry-run mode for testing

**CLI Usage**:
```bash
# Consolidate all sources for yesterday (default)
python jobs/consolidation/consolidate_daily.py

# Specific date
python jobs/consolidation/consolidate_daily.py --date 2025-01-15

# Specific sources
python jobs/consolidation/consolidate_daily.py --sources gmail rss

# Dry run (no writes or deletes)
python jobs/consolidation/consolidate_daily.py --dry-run
```

**Environment Variables**:
- `PROJECT_ID`: Google Cloud project ID
- `GCS_BUCKET`: GCS bucket name
- `CONSOLIDATION_MIN_ROWS`: Alert threshold (default: 50)

**Process Flow**:
1. **Discover**: List all sources in bronze layer
2. **Read**: Load all batch files for source/date
3. **Deduplicate**: Sort by ingested_at, keep last per press_release_id
4. **Write**: Consolidated file to silver (ZSTD level 9, row_group_size=10000)
5. **Verify**: Check row count matches expected
6. **Delete**: Remove batch files only if verification passes
7. **Monitor**: Log statistics, alert on anomalies

**Output**:
```
=== Consolidation Summary ===
gmail:
  Batch count: 12
  Total rows: 234
  Duplicates: 3
  Final rows: 231
  Status: ✓ Success

rss:
  Batch count: 8
  Total rows: 156
  Duplicates: 1
  Final rows: 155
  Status: ✓ Success
```

### Alert Runner Updates (`src/event_feed_app/alerts/runner.py`)

**Changes**:
- `fetch_data()` now reads from both bronze and silver layers
- Historical dates: Read from silver (consolidated)
- Today's date: Read from bronze (batches not yet consolidated)
- Deduplicates across both layers by press_release_id
- Maintains backward compatibility with existing data

**Logic**:
```python
if date < today:
    read from silver (consolidated)
if date == today:
    read from bronze (batches)

# Combine and deduplicate by press_release_id (keep last by ingested_at)
```

## Safety Mechanisms

### 1. Race Condition Prevention
- **Write → Verify → Delete** pattern (never delete before verification)
- Atomic GCS blob uploads (no partial writes visible)
- No concurrent writes to same partition (single ingestion job per source)

### 2. Data Loss Prevention
- Deduplication keeps `last` record sorted by ingested_at
- Bronze batches preserved on consolidation failure
- Verification step before batch deletion
- Bronze layer acts as audit trail

### 3. Failed Consolidation Handling
- Graceful error handling with exception logging
- No partial deletes (all-or-nothing per source)
- Failed sources logged but don't block other sources
- Batches remain in bronze for retry

### 4. Schema Validation
- Explicit PyArrow schema enforcement in batch writer
- Schema constant `PRESS_RELEASE_SCHEMA` ensures consistency
- Type checking before write

### 5. Verification
- Row count check: sum(batch rows) == consolidated rows (after dedup)
- Fails consolidation if mismatch detected
- Logs expected vs actual counts

### 6. Monitoring
- Per-source statistics (batch_count, total_rows, duplicates, final_rows)
- Anomaly detection (< 50 rows threshold, configurable)
- Success/failure status per source
- Detailed logging at each step

### 7. Idempotency
- Deduplication by press_release_id ensures same result on retry
- Can safely re-run consolidation for same date
- Bronze batches can be deleted and re-consolidated from backups

## Expected Benefits

### Storage Efficiency
- **Before**: 200-300 files/day × ~5KB/file = ~1.5MB/day
- **After**: ~10-15 batches + 1 consolidated = ~15-20 files/day
- **Reduction**: ~90% fewer files

### Compression Improvement
- Tiny files (< 10KB): Poor compression ratio (~1.2x)
- Batched files (~500KB): Better compression (~2-3x)
- Consolidated files (> 1MB): Excellent compression (~5-10x)
- **Expected**: 3-5x better overall compression

### Query Performance
- Fewer files to scan during partition reads
- Better Parquet row group efficiency (10K rows/group)
- Columnar compression benefits at scale
- **Expected**: 2-3x faster data loading

### Operational Benefits
- Reduced GCS API calls (fewer files to list/read)
- Lower egress costs (better compression)
- Easier debugging (single consolidated file per day)
- Bronze layer audit trail for data lineage

## Migration Plan

### Phase 1: Foundation (✅ COMPLETE)
- [x] Create ParquetBatchWriter utility
- [x] Create consolidation job
- [x] Update alert runner for bronze + silver reads
- [x] Commit to feat/gcs-batch-writes branch

### Phase 2: Ingestion Integration (🚧 IN PROGRESS)
- [ ] Update gmail ingestion job to use ParquetBatchWriter
- [ ] Update RSS ingestion job to use ParquetBatchWriter
- [ ] Test end-to-end flow in dev environment
- [ ] Verify batch writes and consolidation

### Phase 3: Deployment (⬜ PENDING)
- [ ] Deploy updated ingestion jobs to Cloud Run
- [ ] Deploy consolidation job to Cloud Scheduler (00:05 CET daily)
- [ ] Configure environment variables (PROJECT_ID, GCS_BUCKET)
- [ ] Monitor first consolidation run

### Phase 4: Verification (⬜ PENDING)
- [ ] Verify compression ratio improvement (expect 3-5x)
- [ ] Verify query performance improvement (expect 2-3x faster)
- [ ] Monitor for data loss or duplication issues
- [ ] Validate alert runner works with bronze + silver

### Phase 5: Backfill (⬜ PENDING)
- [ ] Create backfill script to consolidate existing tiny files
- [ ] Run backfill for historical dates
- [ ] Verify consolidated files match original data
- [ ] Delete old tiny files after verification

## Monitoring Checklist

### Daily Monitoring
- [ ] Check consolidation job logs (00:05 CET)
- [ ] Verify no anomalies (< 50 rows per source)
- [ ] Check for failed sources
- [ ] Monitor GCS storage usage (expect reduction)

### Weekly Monitoring
- [ ] Review compression ratios (bronze vs silver)
- [ ] Check query performance metrics
- [ ] Verify no duplicate alerts
- [ ] Review error logs for patterns

### Monthly Monitoring
- [ ] Calculate storage cost reduction
- [ ] Measure query performance improvement
- [ ] Review deduplication statistics
- [ ] Audit bronze/silver divergence

## Rollback Plan

If issues arise, rollback is straightforward:

1. **Revert ingestion jobs**: Restore old single-file writes
2. **Disable consolidation**: Remove Cloud Scheduler job
3. **Update alert runner**: Revert to silver-only reads
4. **Data recovery**: Bronze batches preserved, can reconstruct silver

## Configuration

### Environment Variables
```bash
export PROJECT_ID="your-gcp-project"
export GCS_BUCKET="your-bucket-name"
export CONSOLIDATION_MIN_ROWS=50  # Alert threshold
```

### Settings (src/event_feed_app/config.py)
- `gcs_silver_root`: Silver layer base path
- Derives bronze path: `gcs_silver_root.replace("silver_normalized", "bronze")`

### Batch Writer Config
- `batch_size`: 100 (adjustable based on ingestion rate)
- `compression`: "snappy" for bronze (fast)
- `schema`: PRESS_RELEASE_SCHEMA (12 fields)

### Consolidation Config
- `compression`: "zstd" level 9 (maximum)
- `row_group_size`: 10000 (optimal for analytics)
- `schedule`: "5 0 * * *" (00:05 CET daily)
- `timezone`: "Europe/Stockholm"

## Next Steps

1. **Integrate ingestion jobs** with ParquetBatchWriter
2. **Test consolidation** with --dry-run first
3. **Deploy to production** with monitoring
4. **Monitor metrics** for 1 week
5. **Run backfill** to consolidate historical data

## References

- Implementation PR: feat/gcs-batch-writes branch
- Related docs: 
  - `docs/GCS_LOADER_MIGRATION.md` (original GCS migration)
  - `docs/DEDUPLICATION_STRATEGY.md` (dedup approach)
- Code:
  - `src/event_feed_app/utils/batch_writer.py`
  - `jobs/consolidation/consolidate_daily.py`
  - `src/event_feed_app/alerts/runner.py`
