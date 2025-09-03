# src/event_feed_app/evaluation/assignments_vs_labels_cli.py
# usage:
#   python -m event_feed_app.evaluation.assignments_vs_labels_cli \
#       --labels data/labeling/labeling_pool_2025-09-01.csv \
#       --assignments-dir artifacts/assignments \
#       [--outdir artifacts/eval]

import os, glob, json, argparse
from datetime import datetime, timezone
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from event_feed_app.config import Settings
from event_feed_app.taxonomy.adapters import load as load_taxonomy

LABEL_COL_CANDIDATES = [
    "true_cid", "label_cid", "cid", "category", "label",
    "true_label", "final_label", "gold_cid", "gold"
]
ID_COL_CANDIDATES = ["press_release_id", "id", "press_id"]

def find_latest_assignments(assignments_dir: str) -> str:
    pattern = os.path.join(assignments_dir, "assignments_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No assignments_*.csv found in {assignments_dir}")
    return max(files, key=os.path.getmtime)

def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present. Looked for: {candidates}")

def build_name2cid_from_assignments(assign_df: pd.DataFrame) -> Dict[str, str]:
    """
    Fallback: if adapter loading fails or names are custom, learn mapping from the assignments file.
    Supports both 'final_category_name' and old 'category_name'.
    """
    mapping = {}
    name_col = "final_category_name" if "final_category_name" in assign_df.columns else (
        "category_name" if "category_name" in assign_df.columns else None
    )
    cid_col = (
        "final_cid" if "final_cid" in assign_df.columns else
        ("cid" if "cid" in assign_df.columns else None)
    )
    if name_col and cid_col:
        pairs = (assign_df[[name_col, cid_col]]
                 .dropna()
                 .astype(str)
                 .drop_duplicates())
        for _, row in pairs.iterrows():
            mapping[row[name_col].strip().lower()] = row[cid_col]
    return mapping

def normalize_label(x, allowed_cids: set, local_name2cid: Dict[str, str]) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s_low = s.lower()
    # already a CID (case-sensitive)
    if s in allowed_cids:
        return s
    # by lowercase CID match (rare)
    for c in allowed_cids:
        if c.lower() == s_low:
            return c
    # by name -> cid
    if s_low in local_name2cid:
        return local_name2cid[s_low]
    return None

def main(assignments_dir: str, labels_csv: str, outdir: Optional[str] = None):
    cfg = Settings()
    outdir = outdir or cfg.eval_output_dir
    os.makedirs(outdir, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    latest_path = find_latest_assignments(assignments_dir)
    print(f"[info] Using assignments: {latest_path}")

    df_pred = pd.read_csv(latest_path)
    df_true = pd.read_csv(labels_csv)

    # Identify join columns
    id_pred_col = pick_col(df_pred, ID_COL_CANDIDATES)
    id_true_col = pick_col(df_true, ID_COL_CANDIDATES)

    # Predicted CID column in assignments (your orchestrator writes 'final_cid')
    pred_cid_col = "final_cid" if "final_cid" in df_pred.columns else (
        "cid" if "cid" in df_pred.columns else None
    )
    if pred_cid_col is None:
        raise KeyError("Could not find predicted label column in assignments ('final_cid' or 'cid').")

    # Truth column in labeling pool
    true_col = pick_col(df_true, LABEL_COL_CANDIDATES)

    # ----- Taxonomy order + name→cid mapping via adapters -----
    try:
        tax = load_taxonomy(cfg.taxonomy_version, "en")
        cid_order = tax.ordered_cids
        name2cid = {c.name.strip().lower(): c.cid for c in tax.categories}
    except Exception:
        # fallback if adapters can’t load
        cid_order = sorted(set(df_pred[pred_cid_col].dropna().astype(str)))
        name2cid = build_name2cid_from_assignments(df_pred)

    allowed_cids = set(cid_order)

    # Normalize truth → CID
    df_true = df_true.copy()
    df_true["true_cid_norm"] = df_true[true_col].apply(
        lambda v: normalize_label(v, allowed_cids, name2cid)
    )

    # Filter out rows with unknown/unmappable truths
    n_total_truth = len(df_true)
    df_true_valid = df_true[~df_true["true_cid_norm"].isna()].copy()
    if len(df_true_valid) < n_total_truth:
        print(f"[warn] Dropping {n_total_truth - len(df_true_valid)} rows with unmappable labels.")

    # Join on press_release_id
    merged = df_true_valid.merge(
        df_pred, left_on=id_true_col, right_on=id_pred_col, how="inner", suffixes=("_true", "_pred")
    )
    if merged.empty:
        raise RuntimeError("No overlapping press_release_id between assignments and labeling pool.")

    y_true = merged["true_cid_norm"].astype(str)
    y_pred = merged[pred_cid_col].astype(str)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Stable per-class report ordering (taxonomy order first)
    present_labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda c: (cid_order.index(c) if c in cid_order else 10**9, c)
    )
    cls_report = classification_report(
        y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0
    )

    # Confusion matrix (counts)
    conf = (
        pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["pred"])
        .reindex(index=present_labels, columns=present_labels, fill_value=0)
    )

    root = os.path.splitext(os.path.basename(latest_path))[0]  # e.g., assignments_2025-09-02T135241Z
    out_prefix = f"{root}__eval_{ts}"

    report_json = {
        "assignments_file": latest_path,
        "labels_file": labels_csv,
        "n_truth_total": int(n_total_truth),
        "n_truth_mappable": int(len(df_true_valid)),
        "n_joined": int(len(merged)),
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": cls_report,
    }

    json_path = os.path.join(outdir, f"{out_prefix}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    conf_csv = os.path.join(outdir, f"{out_prefix}_confusion.csv")
    conf.to_csv(conf_csv)

    # Misclassified sample
    errs_cols = [c for c in [
        id_true_col, pred_cid_col, "true_cid_norm",
        "title", "title_clean", "snippet", "final_category_name"
    ] if c in merged.columns]
    errs = merged.loc[y_true != y_pred, errs_cols]
    errs_csv = os.path.join(outdir, f"{out_prefix}_errors_sample.csv")
    errs.head(200).to_csv(errs_csv, index=False)

    # Console summary
    print(f"[ok] Joined on press_release_id: {len(merged)} rows")
    print(f"[ok] Accuracy={acc:.4f}  F1(micro)={f1_micro:.4f}  F1(macro)={f1_macro:.4f}  F1(weighted)={f1_weighted:.4f}")
    print(f"[ok] Wrote: {json_path}")
    print(f"[ok] Wrote: {conf_csv}")
    print(f"[ok] Wrote (sample errors): {errs_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments-dir", default="./outputs", help="Directory containing assignments_*.csv")
    ap.add_argument("--labels", required=True, help="Path to labeling_pool_YYYY-MM-DD.csv")
    ap.add_argument("--outdir", default=None, help="Override output dir (defaults to Settings.eval_output_dir)")
    args = ap.parse_args()
    main(args.assignments_dir, args.labels, args.outdir)
