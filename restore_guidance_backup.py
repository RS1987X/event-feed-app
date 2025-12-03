#!/usr/bin/env python3
"""
Restore guidance signals from backup.
Usage: python3 restore_guidance_backup.py YYYY-MM-DD
"""

import sys
import gcsfs
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 restore_guidance_backup.py YYYY-MM-DD")
        print(f"Example: python3 restore_guidance_backup.py {datetime.now().strftime('%Y-%m-%d')}")
        sys.exit(1)
    
    backup_date = sys.argv[1]
    fs = gcsfs.GCSFileSystem()
    bucket = "event-feed-app-data"
    
    backup_dir = f"gs://{bucket}/backups/{backup_date}"
    
    files = {
        "versions": {
            "backup": f"{backup_dir}/guidance_versions.parquet",
            "target": f"gs://{bucket}/silver/guidance_versions.parquet",
        },
        "current": {
            "backup": f"{backup_dir}/guidance_current.parquet",
            "target": f"gs://{bucket}/gold/guidance_current.parquet",
        },
    }
    
    print(f"=" * 70)
    print(f"Restoring Guidance Signals from {backup_date}")
    print(f"=" * 70)
    
    for name, paths in files.items():
        backup_path = paths["backup"]
        target_path = paths["target"]
        
        if not fs.exists(backup_path):
            print(f"\n✗ Backup not found: {backup_path}")
            continue
        
        print(f"\n{name}:")
        print(f"  From: {backup_path}")
        print(f"  To:   {target_path}")
        
        # Copy backup to target
        fs.copy(backup_path, target_path)
        
        # Verify
        info = fs.info(target_path)
        size_mb = info.get('size', 0) / (1024 * 1024)
        print(f"  ✓ Restored: {size_mb:.2f} MB")
    
    print(f"\n" + "=" * 70)
    print(f"✓ Restore complete!")
    print(f"=" * 70)

if __name__ == "__main__":
    main()
