#import pandas as pd

import pyarrow.dataset as ds
import pyarrow as pa
import gcsfs

fs = gcsfs.GCSFileSystem()


print(fs.glob("event-feed-app-data/silver_normalized/table=press_releases/release_date=*/**/*.parquet")[:5])

root = "event-feed-app-data/silver_normalized/table=press_releases"


# Declare partitioning with an explicit schema for release_date
part = ds.partitioning(
    schema=pa.schema([
        pa.field("release_date", pa.large_string()),   # <- force large_string
        # If you also want 'table' as a partition field, add:
        # pa.field("table", pa.string()),
    ]),
    flavor="hive",
)

# Point at the table root (not the date folders)
dataset = ds.dataset(
    root,
    filesystem=fs,
    format="parquet",
    partitioning=part,#"hive",  # picks up table=..., release_date=...
)


# Example 1: Read a few columns only (projection pushdown)
cols = ["release_date", "company_name", "title", "full_text"]  # adjust to your schema
table = dataset.to_table(columns=cols)  # efficient scan
df = table.to_pandas()
#print(df.shape, df.head())



# Save to CSV
df.to_csv("press_releases_sample.csv", index=False)

print("Wrote", df.shape, "rows to press_releases_sample.csv")

# table = dataset.to_table()
# df = table.to_pandas()
# print(df.shape, df.head())