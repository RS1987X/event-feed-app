# tools/calibrate_kwex.py
from __future__ import annotations
import argparse, json, random, os
from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# --- your app imports ---
from event_feed_app.config import Settings
from event_feed_app.utils.io import load_from_gcs
from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from event_feed_app.gating.newsletter_quotes_gate import apply_newsletter_gate
from event_feed_app.preprocessing.dedup_utils import dedup_df
from event_feed_app.preprocessing.company_name_remover import clean_df_from_company_names
from event_feed_app.taxonomy.keywords.keyword import keywords
from event_feed_app.taxonomy.rules_engine import classify_batch

# knobs to tune
from event_feed_app.taxonomy.rules.keywords_generic import DEFAULT_KWEX_CFG, KwExConfig


# ---------------- Metrics ----------------
def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for c in labels:
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


# ---------------- Rules-only prediction (matches your orchestrator’s logic) ----------------
def predict_rules_only(df: pd.DataFrame, conf_thr: float) -> List[str]:
    def is_rules_final(row) -> bool:
        det = row.get("rule_pred_details") or {}
        if det.get("lock"):
            return row.get("rule_pred_cid") != "other_corporate_update"
        if det.get("needs_review_low_sim", False):
            return False
        conf = float(det.get("rule_conf", 0.0))
        return (row.get("rule_pred_cid") not in (None, "", "other_corporate_update")) and (conf >= conf_thr)

    mask = df.apply(is_rules_final, axis=1)
    pred = []
    for i, ok in enumerate(mask):
        if ok:
            pred.append(df.loc[i, "rule_pred_cid"])
        else:
            pred.append("other_corporate_update")
    return pred


# ---------------- Labels ----------------
def _load_labels(path: str | None, id_col: str, cid_col: str) -> pd.DataFrame:
    """
    Load labels from CSV/Parquet.
    Returns a dataframe with columns: press_release_id, label_cid
    """
    if path is None:
        env = os.getenv("KWEX_LABELS_CSV")
        if not env:
            raise FileNotFoundError(
                "No labels file provided. Pass --labels_csv or set KWEX_LABELS_CSV."
            )
        path = env

    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")

    if path.endswith(".parquet"):
        lab = pd.read_parquet(path)
    else:
        lab = pd.read_csv(path)

    for c in (id_col, cid_col):
        if c not in lab.columns:
            raise ValueError(f"Labels file {path} missing required column: {c}")

    lab = lab[[id_col, cid_col]].dropna()
    lab[id_col] = lab[id_col].astype(str)
    lab[cid_col] = lab[cid_col].astype(str).str.strip()
    return lab.rename(columns={id_col: "press_release_id", cid_col: "label_cid"})


# ---------------- Data prep ----------------
def load_calibration_df(
    cfg: Settings,
    labels_path: str | None,
    labels_id_col: str,
    labels_cid_col: str,
    ignore_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Loads dataset from GCS, applies housekeeping + newsletter gates, then merges labels.
    Returns a dataframe with at least: press_release_id, title_clean, full_text_clean, label_cid
    """

    # Load + housekeeping
    df = load_from_gcs(cfg.gcs_silver_root)
    df, _ = apply_housekeeping_filter(df)

    # Clean company names (to reduce false hits)
    df = clean_df_from_company_names(df)

    # Dedup
    df, _ = dedup_df(
        df,
        id_col="press_release_id",
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",
    )

    # Drop newsletters/quote dumps
    df, _ = apply_newsletter_gate(
        df, title_col="title_clean", body_col="full_text_clean", drop=True
    )

    # Load labels and merge
    labels = _load_labels(labels_path, labels_id_col, labels_cid_col)
    df["press_release_id"] = df["press_release_id"].astype(str)
    df = df.merge(labels, on="press_release_id", how="left")

    if ignore_unlabeled:
        df = df.dropna(subset=["label_cid"])
    if len(df) == 0:
        raise SystemExit("No labeled rows after gates/merge; check --labels_csv and columns.")

    df = df.reset_index(drop=True)
    return df


# ---------------- Tuning ----------------
def sample_trial(rnd=None) -> Dict:
    if rnd is None:
        rnd = random  # use global RNG
    return {
        "weights.any": rnd.uniform(0.6, 1.2),
        "weights.context": rnd.uniform(0.25, 0.55),
        "weights.must_all": rnd.uniform(1.8, 2.5),
        "weights.must_any": rnd.uniform(1.0, 1.6),
        "weights.treat_must_any_as_boolean": rnd.random() < 0.6,
        "competition.alpha": rnd.uniform(0.15, 0.35),
        "competition.hit_cap": rnd.uniform(4.0, 7.0),
        "competition.min_competitor_score": rnd.uniform(0.0, 1.2),
        "competition.mode": "strength",
        "review.score_threshold": rnd.uniform(0.8, 1.4),
        "review.total_terms_threshold": rnd.randint(2, 4),
        # ("pair", "incidents_controversies", "product_launch_partnership"): rnd.uniform(0.8, 2.5),
        # ("pair", "product_launch_partnership", "incidents_controversies"): rnd.uniform(0.4, 1.2),
        # ("pair", "personnel_management_change", "earnings_report"): rnd.uniform(1.0, 2.5),
        # ("pair", "personnel_management_change", "agm_egm_governance"): rnd.uniform(1.0, 2.5),
        # ("pair", "equity_actions_non_buyback", "capital_raise_rights_issue"): rnd.uniform(0.8, 1.8),
    }

def apply_trial_to_cfg(params: Dict, cfg: KwExConfig) -> None:
    # weights
    cfg.weights.any = float(params["weights.any"])
    if "review.total_terms_threshold" in params:
        cfg.review.total_terms_threshold = int(params["review.total_terms_threshold"])
    cfg.weights.context = float(params["weights.context"])
    cfg.weights.must_all = float(params["weights.must_all"])
    cfg.weights.must_any = float(params["weights.must_any"])
    cfg.weights.treat_must_any_as_boolean = bool(params["weights.treat_must_any_as_boolean"])
    # competition
    cfg.competition.alpha = float(params["competition.alpha"])
    cfg.competition.hit_cap = float(params["competition.hit_cap"])
    cfg.competition.min_competitor_score = float(params["competition.min_competitor_score"])
    cfg.competition.mode = str(params["competition.mode"])
    # review
    cfg.review.score_threshold = float(params["review.score_threshold"])
    # pair multipliers
    pairs = {}
    for k, v in params.items():
        if isinstance(k, tuple) and k and k[0] == "pair":
            _, a, b = k
            pairs[(a, b)] = float(v)
    cfg.competition.pair_multipliers.clear()
    cfg.competition.pair_multipliers.update(pairs)


def evaluate_cfg(df_labeled: pd.DataFrame, conf_thr: float) -> float:
    # run rules
    df_rules = classify_batch(
        df_labeled.copy(),
        text_cols=("title_clean", "full_text_clean"),
        keywords=keywords,
    )
    y_true = df_rules["label_cid"].astype(str).tolist()
    y_pred = predict_rules_only(df_rules, conf_thr=conf_thr)
    return macro_f1(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser(description="Calibrate keywords+exclusions knobs via random search.")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--conf_thr", type=float, default=0.0, help="Rules-first confidence threshold.")
    ap.add_argument("--out_json", type=str, default="kwex_tuned.json", help="Where to save tuned config.")

    # NEW: explicit labels
    ap.add_argument("--labels_csv", type=str, default=None,
                    help="Path to labeling CSV/Parquet. If omitted, uses $KWEX_LABELS_CSV.")
    ap.add_argument("--labels_id_col", type=str, default="press_release_id",
                    help="ID column name in the labeling file.")
    ap.add_argument("--labels_cid_col", type=str, default="true_cid",
                    help="Ground-truth class column name in the labeling file.")
    ap.add_argument("--ignore_unlabeled", action="store_true", default=True,
                    help="Drop rows without labels after merge (default True).")
    ap.add_argument("--no-ignore_unlabeled", dest="ignore_unlabeled", action="store_false")

    args = ap.parse_args()

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # 3) in main(), build a single RNG and pass it to each trial
    if args.seed is None:
        rng = random.Random()      # system entropy → non-deterministic
    else:
        rng = random.Random(args.seed)  # deterministic


    cfg = Settings()
    df = load_calibration_df(
        cfg=cfg,
        labels_path=args.labels_csv,
        labels_id_col=args.labels_id_col,
        labels_cid_col=args.labels_cid_col,
        ignore_unlabeled=args.ignore_unlabeled,
    )

    if len(df) == 0:
        raise SystemExit("No labeled rows available for calibration.")

    best = (-1.0, None, None)  # (score, params, snapshot)
    for t in range(args.trials):
        params = sample_trial(rnd=rng)   # <-- FIX: no args.seed + t
        # mutate the singleton config in-place
        apply_trial_to_cfg(params, DEFAULT_KWEX_CFG)
        score = evaluate_cfg(df, conf_thr=args.conf_thr)
        if score > best[0]:
            snap = {
                "weights": asdict(DEFAULT_KWEX_CFG.weights),
                "competition": {
                    **asdict(DEFAULT_KWEX_CFG.competition),
                    "pair_multipliers": {
                        f"{a}|{b}": m
                        for (a, b), m in DEFAULT_KWEX_CFG.competition.pair_multipliers.items()
                    },
                },
                "gating": asdict(DEFAULT_KWEX_CFG.gating),
                "review": asdict(DEFAULT_KWEX_CFG.review),
                "confidence": asdict(DEFAULT_KWEX_CFG.confidence),
            }
            best = (score, params, snap)

        if (t + 1) % 10 == 0:
            print(f"[{t+1}/{args.trials}] best macro-F1={best[0]:.4f}")

    print("\n=== BEST ===")
    print(json.dumps(best[2], indent=2))
    with open(args.out_json, "w") as f:
        json.dump(best[2], f, indent=2)
    print(f"Saved tuned config → {args.out_json}")


if __name__ == "__main__":
    main()
