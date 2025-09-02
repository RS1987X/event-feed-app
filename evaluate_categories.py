# evaluate_categories.py
import argparse, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def load(assign_csv, labels_csv):
    pred = pd.read_csv(assign_csv)
    truth = pd.read_csv(labels_csv)
    key = "press_release_id" if "press_release_id" in pred.columns and "press_release_id" in truth.columns \
          else None
    if key is None:
        # fallback: risky, but lets you try when id is missing
        pred["_key"]  = pred["title"].str[:160] + "||" + pred["snippet"].str[:160]
        truth["_key"] = truth["title"].str[:160] + "||" + truth["snippet"].str[:160]
        key = "_key"
    df = pred.merge(truth[[key, "true_cid"]], on=key, how="inner")
    return df

def eval_basic(df, only_auto=False):
    if only_auto:
        df = df[df["decision"]=="auto_assign"].copy()
        if df.empty:
            print("[EVAL] No auto_assign rows to evaluate.")
            return
        print(f"[EVAL] Evaluating {len(df)} auto_assign rows")

    y_true = df["true_cid"].astype(str).values
    y_pred = df["cid"].astype(str).values

    # Overall
    acc = (y_true == y_pred).mean()
    print(f"[EVAL] Accuracy: {acc:.3f}  (n={len(df)})")
    print("\n[EVAL] Classification report (macro avg is key on class imbalance)\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix (top-10 labels by support to keep width down)
    top_labels = pd.Series(y_true).value_counts().head(10).index.tolist()
    mask = np.isin(y_true, top_labels)
    cm = confusion_matrix(pd.Series(y_true)[mask], pd.Series(y_pred)[mask], labels=top_labels)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in top_labels], columns=[f"P:{l}" for l in top_labels])
    print("\n[EVAL] Confusion matrix (top-10 by support):\n")
    print(cm_df)

def sweep_thresholds(df, target_precision=0.95):
    """Tune CAT_ASSIGN_MIN_SIM/CAT_LOW_MARGIN using labels."""
    tmp = df.copy()
    tmp["correct"] = (tmp["cid"] == tmp["true_cid"]).astype(int)

    sims = np.linspace(0.50, 0.90, 17)
    margins = np.linspace(0.00, 0.10, 11)

    best = None
    for s in sims:
        for m in margins:
            mask = (tmp["cat_sim"] >= s) & (tmp["cat_margin"] >= m)
            if mask.sum() == 0:
                continue
            prec = tmp.loc[mask, "correct"].mean()
            cov  = mask.mean()
            if prec >= target_precision:
                cand = (prec, cov, s, m)
                best = max(best, cand) if best else cand

    if best:
        prec, cov, s, m = best
        print(f"[EVAL] Thresholds achieving ≥{target_precision:.0%} precision:")
        print(f"  CAT_ASSIGN_MIN_SIM={s:.2f}  CAT_LOW_MARGIN={m:.2f}  → precision={prec:.3f} coverage={cov:.1%}")
    else:
        print("[EVAL] No thresholds hit the target precision. Consider lowering the target or improving queries.")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", default="assignments.csv")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--target_precision", type=float, default=0.95)
    ap.add_argument("--only_auto", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.assignments):
        raise FileNotFoundError(f"Assignments not found: {args.assignments}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    df = load(args.assignments, args.labels)
    print(f"[EVAL] Joined {len(df)} rows before filtering unlabeled.")

    # --- filter unlabeled and normalize ---
    # treat NaN / "" / "nan" / "none" as unlabeled
    tc = (df["true_cid"].astype(str).str.strip().str.lower()
            .replace({"nan": "", "none": ""}))
    df = df.assign(true_cid=tc)
    df = df[df["true_cid"] != ""].copy()

    if df.empty:
        print("[EVAL] No ground-truth labels found in labels.csv (column true_cid is blank).")
        print("       Fill some rows in labels.csv (use the CID ids, e.g. 'dividend', 'earnings_report') and rerun.")
        raise SystemExit(0)

    print(f"[EVAL] Using {len(df)} labeled rows after filtering blanks.")
    eval_basic(df, only_auto=args.only_auto)
    sweep_thresholds(df, target_precision=args.target_precision)