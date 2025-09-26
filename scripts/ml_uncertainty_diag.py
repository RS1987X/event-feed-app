# ml_uncertainty_diag.py
# Usage:
#   python ml_uncertainty_diag.py --assign ./outputs/assignments_YYYY-MM-DDTHHMMSSZ.csv \
#                                 --labels data/labeling/labeling_pool_2025-09-01.csv \
#                                 [--outdir .]

import os, argparse
import pandas as pd
import numpy as np

LABEL_COL_CANDIDATES = [
    "true_cid","label_cid","cid","category","label","true_label","final_label","gold_cid","gold"
]
ID_COL_CANDIDATES = ["press_release_id","id","press_id"]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns are present. Looked for: {candidates}")

def main(assign_path: str, labels_path: str, outdir: str = "."):
    os.makedirs(outdir, exist_ok=True)
    assign = pd.read_csv(assign_path)
    labels = pd.read_csv(labels_path)

    # Identify join + label cols
    id_pred_col = pick_col(assign, ID_COL_CANDIDATES)
    id_true_col = pick_col(labels, ID_COL_CANDIDATES)
    true_col = pick_col(labels, LABEL_COL_CANDIDATES)

    # If your labels are names, map them to cids using assignments' mapping
    name_map = (assign[["final_category_name","final_cid"]]
                .dropna().astype(str).drop_duplicates())
    name2cid = {r.final_category_name.strip().lower(): r.final_cid for _, r in name_map.iterrows()}

    def norm_truth(x):
        if pd.isna(x): return None
        s = str(x).strip()
        # already a CID
        if s in set(assign["final_cid"].astype(str)): return s
        # try lowercase name → cid
        return name2cid.get(s.lower(), None)

    labels = labels.copy()
    labels["true_cid_norm"] = labels[true_col].apply(norm_truth)

    merged = labels.dropna(subset=["true_cid_norm"]).merge(
        assign, left_on=id_true_col, right_on=id_pred_col, how="inner", suffixes=("_true","_pred")
    )
    if merged.empty:
        raise RuntimeError("No overlap between assignments and labeling pool.")

    # Focus on ML-decided rows for the diagnostic
    m = merged.copy()
    if "final_source" in m.columns:
        m = m[m["final_source"].eq("ml")].copy()

    # Required diagnostics columns
    for need in ["final_cid","true_cid_norm","ml_conf_fused","ml_margin_fused","ml_entropy_fused"]:
        if need not in m.columns:
            # some runs might have slightly different names
            if need == "ml_entropy_fused" and "ml_entropy_fused" not in m.columns and "entropy_fused" in m.columns:
                m["ml_entropy_fused"] = m["entropy_fused"]
            else:
                raise KeyError(f"Missing column in assignments: {need}")

    m["is_correct"] = (m["final_cid"].astype(str) == m["true_cid_norm"].astype(str))

    # ----- Overall: correct vs wrong -----
    overall = (m.groupby("is_correct")[["ml_conf_fused","ml_margin_fused","ml_entropy_fused"]]
                 .describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
    overall_path = os.path.join(outdir, "ml_uncertainty_overall.csv")
    overall.to_csv(overall_path)

    # Small, readable summary for console
    def small_summary(df, mask, title):
        sub = df[mask]
        print(f"\n=== {title} (n={len(sub)}) ===")
        for col in ["ml_conf_fused","ml_margin_fused","ml_entropy_fused"]:
            if col in sub:
                s = sub[col].describe()
                print(f"{col}: mean={s['mean']:.6f}  p25={s['25%']:.6f}  p50={s['50%']:.6f}  p75={s['75%']:.6f}")

    small_summary(m, m["is_correct"],       "ML-correct")
    small_summary(m, ~m["is_correct"],      "ML-wrong")

    # ----- Admissions-only breakdown -----
    adm_mask = m["true_cid_norm"].isin(["admission_listing","admission_delisting"]) | \
               m["final_cid"].isin(["admission_listing","admission_delisting"])

    adm = m[adm_mask].copy()
    if not adm.empty:
        adm_overall = (adm.groupby(["is_correct","final_cid"])[["ml_conf_fused","ml_margin_fused","ml_entropy_fused"]]
                         .describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
        adm_path = os.path.join(outdir, "ml_uncertainty_admissions.csv")
        adm_overall.to_csv(adm_path)

        print("\n=== Admissions (rows involving listing/delisting) ===")
        print(adm_overall.reset_index().to_string(index=False)[:1200])  # truncate a bit

    # Optional: quick threshold sweep on margin to see separability
    cuts = np.linspace(0.0, 0.05, 11)
    rows = []
    for c in cuts:
        pred_correct = m["ml_margin_fused"] >= c
        precision = float((m["is_correct"] & pred_correct).sum()) / max(pred_correct.sum(), 1)
        recall    = float((m["is_correct"] & pred_correct).sum()) / max(m["is_correct"].sum(), 1)
        rows.append(dict(margin_cut=c, precision_at_cut=precision, recall_at_cut=recall,
                         kept_frac=float(pred_correct.mean())))
    sweep = pd.DataFrame(rows)
    sweep_path = os.path.join(outdir, "ml_margin_threshold_sweep.csv")
    sweep.to_csv(sweep_path, index=False)

    print(f"\n[ok] Wrote overall CSV: {overall_path}")
    if not adm.empty:
        print(f"[ok] Wrote admissions CSV: {adm_path}")
    print(f"[ok] Wrote margin-threshold sweep: {sweep_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign", required=True, help="Path to assignments CSV")
    ap.add_argument("--labels", required=True, help="Path to labeling pool CSV")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current folder)")
    args = ap.parse_args()
    main(args.assign, args.labels, args.outdir)
