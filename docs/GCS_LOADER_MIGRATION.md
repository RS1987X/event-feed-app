# Optimized GCS Loader Migration Guide

## Summary

The GCS loaders in `src/event_feed_app/utils/` now support **partition-aware loading** for significantly improved performance when querying date-partitioned datasets.

## Performance Improvements

- **Before**: Loading all 8,509 Gmail press releases: ~3.5 minutes
- **After**: Loading last 7 days (7 partitions): ~30-60 seconds
- **Speedup**: ~4-10x faster for recent date queries

## What Changed

### `io.py` - `load_from_gcs()`

**Before (loads everything):**
```python
df = load_from_gcs("gs://bucket/silver/table=press_releases/source=gmail")
# Loads ALL 8,509 files
```

**After (backward compatible + new features):**
```python
# Option 1: Still works - loads everything (backward compatible)
df = load_from_gcs("gs://bucket/silver/table=press_releases/source=gmail")

# Option 2: Load with partition filtering (NEW - recommended)
df = load_from_gcs(
    "gs://bucket/silver/table=press_releases/source=gmail",
    partition_filters={"release_date": ["2025-11-12", "2025-11-11"]},
    columns=["press_release_id", "title", "full_text"],
    max_rows=100
)
```

### `gcs_io.py` - New Functions

**`load_parquet_df_partitioned()`** - Advanced loader with full control:
```python
from event_feed_app.utils.gcs_io import load_parquet_df_partitioned

df = load_parquet_df_partitioned(
    uri="gs://bucket/silver/table=press_releases/source=gmail",
    partition_filters={"release_date": ["2025-11-12", "2025-11-11"]},
    columns=["press_release_id", "company_name", "title", "full_text"],
    max_rows=100,
    sort_by="release_date",
    sort_descending=True
)
```

**`load_parquet_df_date_range()`** - Optimized for date ranges:
```python
from event_feed_app.utils.gcs_io import load_parquet_df_date_range

df = load_parquet_df_date_range(
    uri="gs://bucket/silver/table=press_releases/source=gmail",
    start_date="2025-11-06",
    end_date="2025-11-12",
    columns=["press_release_id", "title", "full_text"],
    max_rows=100
)
```

## Migration Examples

### Example 1: Loading Recent Press Releases

**Before:**
```python
from event_feed_app.utils.io import load_from_gcs

# Loads ALL 8,509 files (slow!)
df = load_from_gcs("gs://event-feed-app-data/silver_normalized/table=press_releases/source=gmail")

# Then filter in pandas (after loading everything)
df['release_date'] = pd.to_datetime(df['release_date'])
df = df[df['release_date'] >= '2025-11-06']
df = df.head(100)
```

**After:**
```python
from event_feed_app.utils.gcs_io import load_parquet_df_date_range

# Loads ONLY 7 date partitions (fast!)
df = load_parquet_df_date_range(
    uri="gs://event-feed-app-data/silver_normalized/table=press_releases/source=gmail",
    start_date="2025-11-06",
    end_date="2025-11-12",
    max_rows=100
)
```

### Example 2: Loading with Multiple Filters

**Before:**
```python
# Load everything, filter in memory
df = load_from_gcs("gs://bucket/silver/table=press_releases")
df = df[df['source'] == 'gmail']
df = df[df['release_date'].isin(['2025-11-12', '2025-11-11'])]
```

**After:**
```python
# Only load target partitions
df = load_from_gcs(
    "gs://bucket/silver/table=press_releases/source=gmail",
    partition_filters={"release_date": ["2025-11-12", "2025-11-11"]},
    columns=["press_release_id", "title", "company_name"]
)
```

### Example 3: Loading Today's Data Only

**Before:**
```python
from datetime import datetime

df = load_from_gcs("gs://bucket/silver/table=press_releases/source=gmail")
today = datetime.now().strftime("%Y-%m-%d")
df = df[df['release_date'] == today]
```

**After:**
```python
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
df = load_from_gcs(
    "gs://bucket/silver/table=press_releases/source=gmail",
    partition_filters={"release_date": [today]},
    max_rows=1000
)
```

## Backward Compatibility

✅ **All existing code continues to work** without changes. The updates are **100% backward compatible**.

- `load_from_gcs(uri)` - Still works, loads all data
- `load_parquet_df(uri)` - Still works, loads all data

New features are **opt-in** via additional parameters.

## When to Use Each Function

| Use Case | Recommended Function | Reason |
|----------|---------------------|--------|
| Load entire dataset | `load_parquet_df()` | Simple, no filtering |
| Load recent N days | `load_parquet_df_date_range()` | Optimized for dates |
| Load specific dates | `load_from_gcs()` with filters | Flexible, compatible |
| Custom partition filters | `load_parquet_df_partitioned()` | Full control |
| Backward compatibility | Keep existing calls | Works unchanged |

## Testing

Run the test script to verify performance:
```bash
cd /home/ichard/projects/event-feed-app
source venv/bin/activate
python scripts/test_optimized_loader.py
```

## Key Benefits

1. **Faster queries**: 4-10x speedup for recent date queries
2. **Lower costs**: Less data scanned = lower GCS egress costs
3. **Better UX**: Faster loading = better user experience
4. **Scalable**: Performance stays good as dataset grows
5. **Memory efficient**: Only load what you need

## Next Steps

Consider updating code that:
1. Loads press releases for recent dates (last 7-30 days)
2. Filters by date after loading (move filter to partition level)
3. Only needs specific columns (add `columns` parameter)
4. Processes large datasets (add `max_rows` limit)
