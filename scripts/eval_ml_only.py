# scripts/eval_from_labels.py
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# optional: use your taxonomy ordering if available
try:
    from event_taxonomies import l1_v3_en
    L1_CATEGORIES = l1_v3_en.L1_CATEGORIES
    ORDERED_CIDS = [cid for cid, _, _ in L1_CATEGORIES]
except Exception:
    L1_CATEGORIES = None
    ORDERED_CIDS = None

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate predictions already stored in a labeled CSV (no ML).")
    ap.add_argument("--csv", default="data/labeling_pool_2025-09-01.csv",
                    help="Path to labeled CSV containing true and predicted categories.")
    ap.add_argument("--true-col", default="true_cid",
                    help="Column name for ground-truth labels.")
    ap.add_argument("--pred-col", default="pred_cid",
                    help="Column name for predicted labels (e.g., pred_cid, final_cid, pred_cid_ml).")
    ap.add_argument("--top3-col", default="top3_cids",
                    help="Optional column with JSON list of top-3 candidate cids.")
    ap.add_argument("--outdir", default="eval_outputs",
                    help="Directory to write confusion matrix and misclassifications.")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.true_col not in df.columns:
        raise ValueError(f"Missing true label column '{args.true_col}' in {args.csv}")
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing pred label column '{args.pred_col}' in {args.csv}")

    y_true = df[args.true_col].astype(str)
    y_pred = df[args.pred_col].astype(str)

    # If you want to restrict to known taxonomy cids, do so (otherwise keep all)
    if ORDERED_CIDS is not None:
        mask_valid = y_true.isin(ORDERED_CIDS)
        df = df.loc[mask_valid].reset_index(drop=True)
        y_true = df[args.true_col].astype(str)
        y_pred = df[args.pred_col].astype(str)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n=== Eval from labels ===")
    print(f"File: {args.csv}")
    print(f"True: {args.true_col} | Pred: {args.pred_col}")
    print(f"Docs: {len(df)} | Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f} | Weighted-F1: {f1_weighted:.4f}\n")

    # Full report
    print(classification_report(y_true, y_pred, digits=4))

    # Per-language accuracy if a language column exists
    if "language" in df.columns:
        lang_acc = (
            pd.DataFrame({"lang": df["language"].astype(str), "ok": (y_true.values == y_pred.values).astype(int)})
            .groupby("lang")["ok"].mean().sort_values(ascending=False)
        )
        print("\nPer-language Accuracy:")
        print(lang_acc.round(4))

    # Top-3 accuracy if a top3 column exists
    top3_acc = None
    if args.top3_col in df.columns:
        def parse_top3(x):
            try:
                if isinstance(x, str):
                    return json.loads(x)
                if isinstance(x, (list, tuple)):
                    return list(x)
            except Exception:
                pass
            return []
        top3 = df[args.top3_col].apply(parse_top3).tolist()
        top3_hit = [t in cands for t, cands in zip(y_true.tolist(), top3)]
        if len(top3_hit) == len(df):
            top3_acc = float(np.mean(top3_hit))
            print(f"\nTop-3 Accuracy (from '{args.top3_col}'): {top3_acc:.4f}")

    # Confusion matrix (ordered by taxonomy if available)
    labels_order = ORDERED_CIDS if ORDERED_CIDS is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    cm_path = os.path.join(args.outdir, f"confusion_matrix_{args.pred_col}.csv")
    cm_df.to_csv(cm_path, encoding="utf-8")
    print(f"\n[ok] Saved confusion matrix to {cm_path}")

    # Misclassifications
    miss_mask = y_true.values != y_pred.values
    miss_cols = [c for c in ["press_release_id","release_date","company_name","title","title_clean","full_text_clean","snippet"] if c in df.columns]
    misses = df.loc[miss_mask, miss_cols].copy()
    misses["true_cid"] = y_true[miss_mask].values
    misses["pred_cid"] = y_pred[miss_mask].values
    if top3_acc is not None:
        misses[args.top3_col] = df.loc[miss_mask, args.top3_col]
    miss_path = os.path.join(args.outdir, f"misclassified_{args.pred_col}.csv")
    misses.to_csv(miss_path, index=False, encoding="utf-8")
    print(f"[ok] Saved misclassifications to {miss_path}")

if __name__ == "__main__":
    main()
