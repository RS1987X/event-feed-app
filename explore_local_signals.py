#!/usr/bin/env python3
"""
Explore local Parquet signal files with DuckDB.

Usage:
    python explore_local_signals.py data/signals_*.parquet
"""

import duckdb
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python explore_local_signals.py <parquet_file_or_pattern>")
    sys.exit(1)

file_pattern = sys.argv[1]

# Connect to DuckDB (in-memory)
conn = duckdb.connect()

print(f"📂 Reading: {file_pattern}\n")

# Show schema
print("=" * 80)
print("SCHEMA")
print("=" * 80)
schema = conn.execute(f"DESCRIBE SELECT * FROM '{file_pattern}'").fetchdf()
print(schema.to_string(index=False))

# Show row count
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
count = conn.execute(f"SELECT COUNT(*) as total_rows FROM '{file_pattern}'").fetchone()[0]
print(f"Total rows: {count}")

# Show sample data
print("\n" + "=" * 80)
print("SAMPLE DATA (first 10 rows)")
print("=" * 80)
sample = conn.execute(f"SELECT * FROM '{file_pattern}' LIMIT 10").fetchdf()
print(sample.to_string(index=False))

# Show metrics distribution
print("\n" + "=" * 80)
print("METRICS DISTRIBUTION")
print("=" * 80)
metrics = conn.execute(f"""
    SELECT 
        metric, 
        COUNT(*) as count,
        AVG(confidence) as avg_confidence
    FROM '{file_pattern}'
    GROUP BY metric
    ORDER BY count DESC
""").fetchdf()
print(metrics.to_string(index=False))

# Show companies
print("\n" + "=" * 80)
print("COMPANIES")
print("=" * 80)
companies = conn.execute(f"""
    SELECT 
        company_name,
        COUNT(*) as guidance_items
    FROM '{file_pattern}'
    GROUP BY company_name
    ORDER BY guidance_items DESC
""").fetchdf()
print(companies.to_string(index=False))

# Interactive mode
print("\n" + "=" * 80)
print("💡 TIP: You can also run custom queries:")
print(f"   duckdb -c \"SELECT * FROM '{file_pattern}' WHERE company_name LIKE '%Tesla%'\"")
print("=" * 80)
