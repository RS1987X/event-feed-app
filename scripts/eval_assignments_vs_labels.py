# save as: eval_assignments_vs_labels.py
# usage:    python eval_assignments_vs_labels.py --labels labeling_pool_2025_09_01.csv [--assignments-dir .]

import os, glob, json, argparse, datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- Try to import your taxonomy for clean name<->cid mapping (optional but preferred)
NAME2CID = {}
CID_ORDER = []
try:
    from event_taxonomies import l1_v3_en
    CID_ORDER = [cid for cid, _, _ in l1_v3_en.L1_CATEGORIES]
    NAME2CID = {name.strip().lower(): cid for cid, name, _ in l1_v3_en.L1_CATEGORIES}
except Exception:
    pass

LABEL_COL_CANDIDATES = [
    "true_cid"
]
ID_COL_CANDIDATES = ["press_release_id", "id", "press_id"]

def find_latest_assignments(assignments_dir: str) -> str:
    pattern = os.path.join(assignments_dir, "assignments_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No assignments_*.csv found in {assignments_dir}")
    # pick newest by mtime (simplest & robust)
    return max(files, key=os.path.getmtime)

def pick_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present. Looked for: {candidates}")

def build_name2cid_from_assignments(assign_df: pd.DataFrame) -> dict[str, str]:
    """
    Fallback: learn name→CID mapping from the assignments file.
    Prefers rules-first columns (final_*). Returns lowercase name -> CID.
    """
    mapping: dict[str, str] = {}

    def add_pairs(name_col: str, cid_col: str):
        if name_col in assign_df.columns and cid_col in assign_df.columns:
            pairs = (
                assign_df[[name_col, cid_col]]
                .dropna()
                .astype(str)
                .drop_duplicates()
            )
            for _, row in pairs.iterrows():
                mapping[row[name_col].strip().lower()] = row[cid_col].strip()

    # Preferred (your current schema)
    add_pairs("final_category_name", "final_cid")

    # Fallbacks (older exports / other tools)
    add_pairs("category_name", "final_cid")
    add_pairs("category_name", "cid")

    # Optional future fallbacks if you ever add these columns
    add_pairs("ml_category_name", "ml_pred_cid")
    add_pairs("rule_category_name", "rule_pred_cid")

    return mapping

def normalize_label(x, allowed_cids: set, local_name2cid: dict):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s_low = s.lower()
    # already a CID?
    if s in allowed_cids:
        return s
    # by lowercase match
    if s_low in allowed_cids:
        # (rare) if labels are lowercase CIDs
        for c in allowed_cids:
            if c.lower() == s_low:
                return c
    # by name -> cid
    if s_low in local_name2cid:
        return local_name2cid[s_low]
    return None

def main(assignments_dir: str, labels_csv: str, out_prefix: str = None):
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H%M%SZ")
    latest_path = find_latest_assignments(assignments_dir)
    print(f"[info] Using assignments: {latest_path}")

    df_pred = pd.read_csv(latest_path)
    df_true = pd.read_csv(labels_csv)

    # Identify columns
    id_pred_col = pick_col(df_pred, ID_COL_CANDIDATES)
    id_true_col = pick_col(df_true, ID_COL_CANDIDATES)

    # Predicted CID column (prefer rules-first / shipped label)
    pred_cid_candidates = ["final_cid", "rule_pred_cid", "ml_pred_cid", "cid", "category"]
    pred_cid_col = pick_col(df_pred, pred_cid_candidates)
    print(f"[info] Using predicted column: {pred_cid_col}")

    # Warn if ML differs from final (useful sanity check)
    if "final_cid" in df_pred.columns and "ml_pred_cid" in df_pred.columns:
        mismatch = (df_pred["final_cid"].astype(str) != df_pred["ml_pred_cid"].astype(str)).sum()
        if mismatch:
            print(f"[warn] {mismatch} rows where ML disagrees with final_cid (rules-first).")

    # Truth column in pool
    true_col = pick_col(df_true, LABEL_COL_CANDIDATES)

    # Allowed CIDs & name->cid mapping
    cid_order = CID_ORDER.copy() if CID_ORDER else []
    if not cid_order:
        # fallback: derive allowed cids from assignments
        cid_order = sorted(set(df_pred[pred_cid_col].dropna().astype(str)))
    allowed_cids = set(cid_order)

    local_name2cid = dict(NAME2CID)
    if not local_name2cid:
        local_name2cid = build_name2cid_from_assignments(df_pred)


    # Normalize truth → CID
    df_true = df_true.copy()
    df_true["true_cid_norm"] = df_true[true_col].apply(lambda v: normalize_label(v, allowed_cids, local_name2cid))

    # Filter out rows with unknown/unmappable truths
    n_total_truth = len(df_true)
    df_true_valid = df_true[~df_true["true_cid_norm"].isna()].copy()
    if len(df_true_valid) < n_total_truth:
        print(f"[warn] Dropping {n_total_truth - len(df_true_valid)} rows from pool with unmappable labels.")

    # Join on press_release_id
    merged = (df_true_valid
              .merge(df_pred, left_on=id_true_col, right_on=id_pred_col, how="inner", suffixes=("_true", "_pred")))

    if merged.empty:
        raise RuntimeError("No overlapping press_release_id between assignments and labeling pool.")

    # Prepare y_true / y_pred
    y_true = merged["true_cid_norm"].astype(str)
    y_pred = merged[pred_cid_col].astype(str)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Restrict labels in report to those present in either y_true or union with taxonomy (to get stable ordering)
    present_labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda c: (cid_order.index(c) if c in cid_order else 999, c)
    )
    cls_report = classification_report(y_true, y_pred, labels=present_labels, output_dict=True, zero_division=0)

    # Confusion matrix (counts)
    conf = (pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["pred"])
            .reindex(index=present_labels, columns=present_labels, fill_value=0))

    # Outputs
    if out_prefix is None:
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

    os.makedirs("eval_outputs", exist_ok=True)
    json_path = os.path.join("eval_outputs", f"{out_prefix}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    conf_csv = os.path.join("eval_outputs", f"{out_prefix}_confusion.csv")
    conf.to_csv(conf_csv)

    # Save misclassified sample for quick eyeballing
    errs = merged.loc[y_true != y_pred, [
        id_true_col, pred_cid_col, "true_cid_norm",
        *(c for c in ["title", "title_clean", "snippet"] if c in merged.columns),
        *(c for c in ["category_name"] if c in merged.columns),
        *(c for c in ["cat_sim","cat_margin","p1_emb","p1_lsa","entropy_fused"] if c in merged.columns),
    ]]
    errs_csv = os.path.join("eval_outputs", f"{out_prefix}_errors_sample.csv")
    errs.head(200).to_csv(errs_csv, index=False)

    # Console summary
    print(f"[ok] Joined on press_release_id: {len(merged)} rows")
    print(f"[ok] Accuracy={acc:.4f}  F1(micro)={f1_micro:.4f}  F1(macro)={f1_macro:.4f}  F1(weighted)={f1_weighted:.4f}")
    print(f"[ok] Wrote: {json_path}")
    print(f"[ok] Wrote: {conf_csv}")
    print(f"[ok] Wrote (sample errors): {errs_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments-dir", default=".", help="Directory containing assignments_*.csv")
    ap.add_argument("--labels", required=True, help="Path to labeling_pool_YYYY_MM_DD.csv")
    args = ap.parse_args()
    main(args.assignments_dir, args.labels)
