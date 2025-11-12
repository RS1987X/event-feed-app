#!/usr/bin/env python3
"""
Quick viewer for Press Releases when source URLs are missing.

Usage:
  ./venv/bin/python scripts/view_press_releases.py --id 1980c93f9f819a6d
  ./venv/bin/python scripts/view_press_releases.py --ids 1980c93f9f819a6d 198a92bb04fe9f70
  ./venv/bin/python scripts/view_press_releases.py --file ids.txt

It loads the Silver parquet dataset (same one used by alerts) and prints
Title, Company, Release date, and the cleaned/full text, wrapped to terminal width.

Respects environment loaded by scripts/load_env.py and Settings() for GCS path.
"""
import os
import sys
import argparse
from pathlib import Path
import textwrap

# Project import paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from load_env import load_env  # type: ignore
load_env()

from event_feed_app.config import Settings  # type: ignore
from event_feed_app.utils.io import load_from_gcs  # type: ignore


def read_ids_from_file(path: str):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))


def main():
    ap = argparse.ArgumentParser(description="View press releases by id from Silver store")
    ap.add_argument("--id", dest="id", help="Single press_release_id to show")
    ap.add_argument("--ids", nargs="*", help="Multiple press_release_id values")
    ap.add_argument("--file", help="File with one id per line")
    ap.add_argument("--raw", action="store_true", help="Print raw full_text if available (not cleaned)")
    ap.add_argument("--width", type=int, default=100, help="Wrap width for body text")
    args = ap.parse_args()

    ids = []
    if args.id:
        ids.append(args.id.strip())
    if args.ids:
        ids.extend([x.strip() for x in args.ids if x.strip()])
    if args.file:
        ids.extend(read_ids_from_file(args.file))

    if not ids:
        print("No ids provided. Use --id, --ids, or --file.")
        sys.exit(1)

    # Load the silver dataset
    cfg = Settings()
    print(f"Loading Silver dataset: {cfg.gcs_silver_root}")
    df = load_from_gcs(cfg.gcs_silver_root)

    # Standardize columns we will use
    title_col = "title_clean" if "title_clean" in df.columns else ("title" if "title" in df.columns else None)
    body_col = "full_text_clean" if "full_text_clean" in df.columns else ("full_text" if "full_text" in df.columns else None)

    if "press_release_id" not in df.columns:
        print("Data does not have 'press_release_id' column.")
        sys.exit(2)

    # Filter the requested ids
    dfv = df[df["press_release_id"].isin(ids)].copy()
    if dfv.empty:
        print("No matching press_release_id rows found.")
        sys.exit(3)

    # Print each row
    for _, row in dfv.iterrows():
        print("=" * 120)
        print(f"press_release_id: {row.get('press_release_id','')}")
        print(f"company:          {row.get('company_name','')}")
        print(f"release_date:     {row.get('release_date','')}")
        if title_col:
            print(f"title:            {row.get(title_col, '')}")
        src = (row.get("source_url") or row.get("url") or row.get("link") or "")
        print(f"source_url:       {src}")
        print()
        body = row.get("full_text" if args.raw and "full_text" in row else (body_col or ""), "")
        if isinstance(body, str) and body:
            print(wrap(body, width=args.width))
        else:
            print("<no body content>")
        print()


if __name__ == "__main__":
    main()
