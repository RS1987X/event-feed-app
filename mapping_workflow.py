#!/usr/bin/env python3
# mapping_workflow.py
#
# pip install pandas gcsfs
#
# Usage:
#   # A) Create the review sheet (from your clustering output)
#   python mapping_workflow.py make-review \
#     --clusters unsup_clusters/2025-08-25T1145Z/cluster_summaries.csv \
#     --out      unsup_clusters/2025-08-25T1145Z/cluster_mapping_review.csv
#
#   # (Human edits proposed_label / confidence / notes)
#
#   # B) Convert the reviewed CSV to JSON (what your pipeline consumes)
#   python mapping_workflow.py to-json \
#     --review unsup_clusters/2025-08-25T1145Z/cluster_mapping_review.csv \
#     --out    unsup_clusters/2025-08-25T1145Z/cluster_mapping.json \
#     --taxonomy-version 1.0 \
#     --k 38
#
#   # C) Upload JSON to GCS
#   python mapping_workflow.py upload \
#     --local unsup_clusters/2025-08-25T1145Z/cluster_mapping.json \
#     --gcs   gs://event-feed-app-data/mappings/cluster_mapping.json

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

REVIEW_COLUMNS = [
    "cluster", "size", "topic_hint", "sample_titles_representative",
    "proposed_label", "confidence", "notes"
]

def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

def _read_csv_local_or_gcs(path: str) -> pd.DataFrame:
    """
    Read a CSV from local filesystem or GCS (gs://...) using gcsfs.
    """
    if path.startswith("gs://"):
        try:
            import gcsfs  # lazily import so local-only users don't need it
        except ImportError:
            raise SystemExit("[error] reading from GCS requires `pip install gcsfs`")
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            return pd.read_csv(f)
    else:
        return pd.read_csv(path)

def _write_csv_local(path: str, df: pd.DataFrame, overwrite: bool = False) -> None:
    if (not overwrite) and os.path.exists(path):
        raise SystemExit(f"[abort] {path} already exists. Use --overwrite to replace.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)

def _write_json_local(path: str, payload: dict, overwrite: bool = False) -> None:
    if (not overwrite) and os.path.exists(path):
        raise SystemExit(f"[abort] {path} already exists. Use --overwrite to replace.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _upload_to_gcs(local_path: str, gcs_path: str) -> None:
    if not os.path.exists(local_path):
        raise SystemExit(f"[error] local file not found: {local_path}")
    if not gcs_path.startswith("gs://"):
        raise SystemExit(f"[error] gcs path must start with gs:// (got {gcs_path})")
    try:
        import gcsfs
    except ImportError:
        raise SystemExit("[error] uploading to GCS requires `pip install gcsfs`")
    fs = gcsfs.GCSFileSystem()
    with open(local_path, "rb") as fin, fs.open(gcs_path, "wb") as fout:
        fout.write(fin.read())

# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_make_review(clusters_csv: str, out_csv: str, overwrite: bool) -> None:
    """
    Build a human-friendly review CSV with empty columns reviewers can fill.
    """
    df = _read_csv_local_or_gcs(clusters_csv)

    required = {"cluster", "size", "topic_hint", "sample_titles_representative"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[error] {clusters_csv} missing columns: {sorted(missing)}")

    out = df.loc[:, ["cluster", "size", "topic_hint", "sample_titles_representative"]].copy()
    out["proposed_label"] = ""  # reviewer fills from your taxonomy
    out["confidence"] = ""      # 1/2/3 (weak/ok/strong)
    out["notes"] = ""           # short rationale or caveat
    out = out.sort_values("size", ascending=False).reset_index(drop=True)

    _write_csv_local(out_csv, out, overwrite=overwrite)
    print(f"[ok] wrote review sheet: {out_csv}  ({len(out)} clusters)")
    print("Next: open it, fill `proposed_label` (and optionally `confidence`, `notes`).")

def cmd_to_json(review_csv: str, out_json: str,
                taxonomy_version: str, cluster_k: Optional[int],
                overwrite: bool) -> None:
    """
    Convert the reviewed CSV into the mapping JSON your pipeline consumes.
    Only rows with a non-empty proposed_label are included.
    """
    df = pd.read_csv(review_csv, dtype={"cluster": int})
    need = set(REVIEW_COLUMNS)
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"[error] {review_csv} missing columns: {sorted(missing)}")

    df = df[df["proposed_label"].astype(str).str.strip() != ""].copy()
    if df.empty:
        raise SystemExit("[error] no rows with proposed_label found. Fill the review file first.")

    payload = {
        "run_ts": _utc_ts(),
        "taxonomy_version": str(taxonomy_version),
        "cluster_k": int(cluster_k) if cluster_k is not None else None,
        "clusters": [
            {"cluster": int(r.cluster), "label": str(r.proposed_label).strip()}
            for r in df.itertuples(index=False)
        ]
    }

    _write_json_local(out_json, payload, overwrite=overwrite)
    print(f"[ok] wrote mapping json: {out_json}  ({len(payload['clusters'])} labeled clusters)")
    print("Tip: version this file (e.g., .../cluster_mapping_v1.json) before upload.")

def cmd_upload(local_json: str, gcs_path: str) -> None:
    _upload_to_gcs(local_json, gcs_path)
    print(f"[ok] uploaded {local_json} → {gcs_path}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Human-in-the-loop workflow: review → mapping JSON → upload to GCS"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # make-review
    ap_a = sub.add_parser("make-review", help="Create cluster_mapping_review.csv from cluster_summaries.csv")
    ap_a.add_argument("--clusters", required=True,
                      help="Path to cluster_summaries.csv (local or gs://)")
    ap_a.add_argument("--out", required=True,
                      help="Output path for review CSV (local)")
    ap_a.add_argument("--overwrite", action="store_true")

    # to-json
    ap_b = sub.add_parser("to-json", help="Convert reviewed CSV to cluster_mapping.json")
    ap_b.add_argument("--review", required=True,
                      help="Reviewed cluster_mapping_review.csv (local)")
    ap_b.add_argument("--out", required=True,
                      help="Output JSON path (local)")
    ap_b.add_argument("--taxonomy-version", default="1.0")
    ap_b.add_argument("--k", type=int, default=None,
                      help="Optional cluster_k to embed for traceability")
    ap_b.add_argument("--overwrite", action="store_true")

    # upload
    ap_c = sub.add_parser("upload", help="Upload mapping JSON to GCS")
    ap_c.add_argument("--local", required=True, help="Local cluster_mapping.json")
    ap_c.add_argument("--gcs", required=True, help="Destination gs://bucket/path/cluster_mapping.json")

    args = ap.parse_args()

    if args.cmd == "make-review":
        cmd_make_review(args.clusters, args.out, args.overwrite)
    elif args.cmd == "to-json":
        cmd_to_json(args.review, args.out, args.taxonomy_version, args.k, args.overwrite)
    elif args.cmd == "upload":
        cmd_upload(args.local, args.gcs)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
