#!/usr/bin/env python3
"""
Reset guidance signals and rerun with new directional constraint logic.
1. Backup current guidance files
2. Clear/reset the current state
3. Rerun today's messages
"""

import gcsfs
import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    fs = gcsfs.GCSFileSystem()
    bucket = "event-feed-app-data"
    today = datetime.now().strftime("%Y-%m-%d")
    
    files = {
        "versions": f"gs://{bucket}/silver/guidance_versions.parquet",
        "current": f"gs://{bucket}/gold/guidance_current.parquet",
    }
    
    backup_dir = f"gs://{bucket}/backups/{today}"
    
    print(f"=" * 70)
    print(f"Guidance Signal Reset - {today}")
    print(f"=" * 70)
    
    # Step 1: Backup existing files
    print(f"\n1. Backing up existing files to: {backup_dir}")
    for name, path in files.items():
        if fs.exists(path):
            backup_path = f"{backup_dir}/guidance_{name}.parquet"
            print(f"   Backing up {name}: {path} -> {backup_path}")
            fs.copy(path, backup_path)
            
            # Show backup info
            info = fs.info(backup_path)
            size_mb = info.get('size', 0) / (1024 * 1024)
            print(f"   ✓ Backup created: {size_mb:.2f} MB")
        else:
            print(f"   ⚠ {name} doesn't exist: {path}")
    
    # Step 2: Clear current files (replace with empty dataframes)
    print(f"\n2. Clearing current guidance state")
    for name, path in files.items():
        if fs.exists(path):
            # Read current file to get schema
            df = pd.read_parquet(path, filesystem=fs)
            print(f"   Current {name}: {len(df)} rows")
            
            # Create empty dataframe with same schema
            empty_df = df.iloc[:0].copy()
            
            # Write empty dataframe
            with fs.open(path, 'wb') as f:
                empty_df.to_parquet(f, index=False)
            
            print(f"   ✓ Cleared {name}: now 0 rows")
    
    print(f"\n3. Next steps:")
    print(f"   Run the guidance job to reprocess today's messages with new logic:")
    print(f"   ")
    print(f"   cd /home/ichard/projects/event-feed-app")
    print(f"   python3 -m event_feed_app.events.guidance_change.guidance_change")
    print(f"")
    print(f"   Or restore from backup if needed:")
    print(f"   python3 restore_guidance_backup.py {today}")
    
    print(f"\n" + "=" * 70)
    print(f"✓ Reset complete!")
    print(f"=" * 70)

if __name__ == "__main__":
    main()
