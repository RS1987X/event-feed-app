from __future__ import annotations
"""
Random-search tuner for pipeline-level knobs from config.py, with caching.

Cache contents (under --cache_dir):
  - df.parquet: columns needed for rules-first gating + eval (id, rule fields, lang_conf, label)
  - S_emb.npy:  standardized embedding sims (docs x categories)
  - S_lsa.npy:  standardized tfidf sims (docs x categories)
  - base_cids.json: ordered list of category ids

Example:
  python -m src.event_feed_app.tools.tune_pipeline \
    --labels_csv data/labeling/labeling_pool_2025-09-01.csv \
    --cache_dir .cache/tune_pipeline \
    --rebuild_cache \
    --trials 60 \
    --limit_n 800 \
    --out_json src/event_feed_app/configs/pipeline_tuned.json
"""

import argparse, json, os, random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from event_feed_app.config import Settings
from event_feed_app.utils.io import load_from_gcs
from event_feed_app.gating.housekeeping_gate import apply_housekeeping_filter
from event_feed_app.gating.newsletter_quotes_gate import apply_newsletter_gate
from event_feed_app.preprocessing.dedup_utils import dedup_df
from event_feed_app.preprocessing.company_name_remover import clean_df_from_company_names
from event_feed_app.preprocessing.filters import filter_to_labeled  # not used here but kept for parity

from event_feed_app.taxonomy.keywords.keyword import keywords
from event_feed_app.taxonomy.rules_engine import classify_batch

from event_feed_app.taxonomy.rules_engine import classify_batch
import event_feed_app.taxonomy.rules.keywords_generic          # noqa: F401
import event_feed_app.taxonomy.rules.product_launch_partn  # noqa: F401
import event_feed_app.taxonomy.rules.admissions          # noqa: F401
import event_feed_app.taxonomy.rules.agm  # noqa: F401
import event_feed_app.taxonomy.rules.debt          # noqa: F401
import event_feed_app.taxonomy.rules.regulatory  # noqa: F401
import event_feed_app.taxonomy.rules.incidents          # noqa: F401
import event_feed_app.taxonomy.rules.invite  # noqa: F401
import event_feed_app.taxonomy.rules.mna          # noqa: F401
import event_feed_app.taxonomy.rules.orders  # noqa: F401
import event_feed_app.taxonomy.rules.pdmr          # noqa: F401
import event_feed_app.taxonomy.rules.personnel  # noqa: F401
import event_feed_app.taxonomy.rules.ratings          # noqa: F401
import event_feed_app.taxonomy.rules.rd_clinical_update          # noqa: F401
import event_feed_app.taxonomy.rules.earnings          # noqa: F401


from event_feed_app.taxonomy.adapters import load as load_taxonomy
from event_feed_app.taxonomy.ops import align_texts
from event_feed_app.representation.tfidf_texts import texts_lsa
from event_feed_app.representation.prompts import (
    doc_prompt, cat_prompts_e5_query, PromptStyle
)
from event_feed_app.models.embeddings import get_embedding_model, encode_texts
from event_feed_app.utils.lang import setup_langid, detect_language_batch, normalize_lang
from event_feed_app.utils.math import (
    softmax_rows, l2_normalize_rows, apply_abtt, fit_abtt, rowwise_standardize
)

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

# --------------- Helpers ---------------
def _extract_rule_field(val, key: str, default=None):
    """rule_pred_details may be a dict or JSON string depending on IO backend."""
    if isinstance(val, dict):
        return val.get(key, default)
    if isinstance(val, str):
        try:
            d = json.loads(val)
            return d.get(key, default)
        except Exception:
            return default
    return default

def _ensure_id_column(df: pd.DataFrame, target: str = "press_release_id") -> pd.DataFrame:
    """Normalize to a guaranteed ID column name, or synthesize one."""
    candidates = [target, "press_release_id", "id", "doc_id", "message_id"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        df = df.copy()
        df[target] = np.arange(len(df))
        return df
    if found != target:
        df = df.rename(columns={found: target})
    return df

# --------------- Final labels (rules-first + optional abstain) ---------------
def compute_final_labels_from_cached(
    df: pd.DataFrame,
    base_cids: List[str],
    S_emb: np.ndarray,
    S_lsa: np.ndarray,
    cfg: Settings,
) -> List[str]:
    # Temps & fuse (we stored standardized scores; apply temp via division)
    P_emb = softmax_rows(S_emb / max(cfg.temp_emb, 1e-6), T=1.0)
    P_lsa = softmax_rows(S_lsa / max(cfg.temp_lsa, 1e-6), T=1.0)
    P_fused = cfg.fuse_alpha * P_emb + (1.0 - cfg.fuse_alpha) * P_lsa

    best_idx  = np.argmax(P_fused, axis=1)
    best_prob = P_fused[np.arange(len(P_fused)), best_idx]
    if P_fused.shape[1] >= 2:
        part = np.partition(P_fused, -2, axis=1)
        second_best = part[:, -2]
        margin = best_prob - second_best
    else:
        second_best = np.zeros_like(best_prob)
        margin = np.zeros_like(best_prob)

    ml_pred_cid = [base_cids[i] for i in best_idx]

    # rules-first gating (same logic as orchestrator)
    def is_rules_final(row) -> bool:
        det = row.get("rule_pred_details")
        locked = bool(_extract_rule_field(det, "lock", False))
        if locked:
            return row.get("rule_pred_cid") != "other_corporate_update"
        if bool(_extract_rule_field(det, "needs_review_low_sim", False)):
            return False
        conf = float(_extract_rule_field(det, "rule_conf", 0.0) or 0.0)
        cid  = row.get("rule_pred_cid")
        return (cid not in (None, "", "other_corporate_update")) and (conf >= cfg.rule_conf_thr)

    mask_rules_final = df.apply(is_rules_final, axis=1).to_numpy()

    # Abstain (only applied to ML-sourced rows)
    final = []
    lc = df["lang_conf"].astype(float).fillna(1.0).to_numpy()
    for i in range(len(df)):
        if mask_rules_final[i]:
            final.append(df.loc[df.index[i], "rule_pred_cid"])
        else:
            cid_ml = ml_pred_cid[i]
            if cfg.abstain_enable:
                if (best_prob[i] < cfg.abstain_min_prob) or \
                   (margin[i] < cfg.abstain_min_margin) or \
                   (lc[i] < cfg.abstain_min_lang_conf):
                    final.append(cfg.abstain_other_cid)
                else:
                    final.append(cid_ml)
            else:
                final.append(cid_ml)
    return final

# --------------- Cache builder/loader ---------------
def _paths(cache_dir: str) -> Dict[str, str]:
    return {
        "raw_df": os.path.join(cache_dir, "raw_df.parquet"),
        "df": os.path.join(cache_dir, "df.parquet"),
        "S_emb": os.path.join(cache_dir, "S_emb.npy"),
        "S_lsa": os.path.join(cache_dir, "S_lsa.npy"),
        "cids": os.path.join(cache_dir, "base_cids.json"),
    }

def cache_exists(cache_dir: str) -> bool:
    p = _paths(cache_dir)
    return all(os.path.exists(v) for v in p.values())

def build_cache(
    cfg: Settings,
    labels_csv: str,
    labels_id_col: str,
    labels_cid_col: str,
    cache_dir: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    p = _paths(cache_dir)

    # 1) Load + housekeeping (from GCS once)
    df = load_from_gcs(cfg.gcs_silver_root)
    
    #save raw cache
    p = _paths(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(p["raw"], index=False)
    print(f"[cache] wrote raw (pre-housekeeping) snapshot → {p['raw']}")
    
    df, _ = apply_housekeeping_filter(df)
    df = clean_df_from_company_names(df)
    df, _ = dedup_df(
        df,
        id_col="press_release_id",
        title_col="title_clean",
        body_col="full_text_clean",
        ts_col="release_date",
    )

    # Ensure we have a normalized ID column
    df = _ensure_id_column(df, target="press_release_id")

    # 2) Gate newsletters/quotes
    df, _ = apply_newsletter_gate(df, title_col="title_clean", body_col="full_text_clean", drop=True)

    # 3) Keep only labeled (explicit labels file)
    labels = pd.read_csv(labels_csv) if labels_csv.endswith(".csv") else pd.read_parquet(labels_csv)
    if labels_id_col not in labels.columns or labels_cid_col not in labels.columns:
        raise ValueError(f"Labels file missing columns: {labels_id_col}, {labels_cid_col}")
    labels = labels[[labels_id_col, labels_cid_col]].dropna()
    labels[labels_id_col] = labels[labels_id_col].astype(str)

    df = df.copy()
    df["__id"] = df["press_release_id"].astype(str)
    df = df.merge(labels, left_on="__id", right_on=labels_id_col, how="inner")
    df.rename(columns={labels_cid_col: "label"}, inplace=True)
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No labeled rows remain after gates.")

    # 4) Rules pass (independent of tuning knobs)
    df = classify_batch(df, text_cols=("title_clean", "full_text_clean"), keywords=keywords)

    # 5) Language detection (independent of tuning knobs)
    langid = setup_langid(cfg.lang_whitelist)
    langs, lang_confs = detect_language_batch(df["title_clean"], df["full_text_clean"], langid=langid)
    df["language"] = langs
    df["lang_conf"] = lang_confs

    # 6) Category texts
    tax = load_taxonomy(cfg.taxonomy_version, "en")
    base_cids = tax.ordered_cids
    with open(p["cids"], "w") as f:
        json.dump(base_cids, f)

    cat_texts_lsa_default = texts_lsa(tax)
    cat_texts_emb = cat_prompts_e5_query(tax)

    # 7) Embeddings & ABTT (compute once)
    docs_for_emb = [
        doc_prompt(t, b, style=PromptStyle.E5_PASSAGE)
        for t, b in zip(df["title_clean"], df["full_text_clean"])
    ]
    model = get_embedding_model(cfg.emb_model_name, cfg.max_seq_tokens)
    X = encode_texts(model, docs_for_emb, cfg.emb_batch_size)  # (N, d)
    Q = encode_texts(model, cat_texts_emb, cfg.emb_batch_size) # (C, d)

    mu, P = fit_abtt(X, k=getattr(cfg, "emb_abtt_k", 2))
    X_d = apply_abtt(X, mu, P)
    Q_d = apply_abtt(Q, mu, P)

    Xn = l2_normalize_rows(X_d)
    Qn = l2_normalize_rows(Q_d)
    sims_emb = Xn @ Qn.T

    # 8) TF-IDF sims (multi-lingual) then row-wise standardize both matrices
    langs_present = set(normalize_lang(x) for x in df["language"].astype(str).fillna("und"))
    lang_to_translated_cats = {}
    for lang in langs_present:
        try:
            localized = load_taxonomy(cfg.taxonomy_version, lang)
            lang_to_translated_cats[lang] = align_texts(tax, localized)
        except Exception:
            pass

    rep_keep_idx = list(range(len(df)))  # keep all docs by default

    from event_feed_app.models.tfidf import multilingual_tfidf_cosine
    sims_lsa = multilingual_tfidf_cosine(
        doc_df=df,
        cat_texts_lsa_default=cat_texts_lsa_default,
        rep_keep_idx=rep_keep_idx,
        lang_to_translated_cats=lang_to_translated_cats,
    )

    S_emb = rowwise_standardize(sims_emb)
    S_lsa = rowwise_standardize(sims_lsa)

    # 9) Save cache
    keep_cols = [
        "press_release_id", "rule_pred_cid", "rule_pred_details",
        "language", "lang_conf", "label"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]  # guard
    df_min = df[keep_cols].copy()
    df_min.to_parquet(p["df"], index=False)
    np.save(p["S_emb"], S_emb)
    np.save(p["S_lsa"], S_lsa)

    return df_min, S_emb, S_lsa, base_cids

def load_cache(cache_dir: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    p = _paths(cache_dir)
    df = pd.read_parquet(p["df"])
    S_emb = np.load(p["S_emb"])
    S_lsa = np.load(p["S_lsa"])
    with open(p["cids"], "r") as f:
        base_cids = json.load(f)
    return df, S_emb, S_lsa, base_cids

def load_raw_cache(cache_dir: str) -> pd.DataFrame:
    p = _paths(cache_dir)
    df = pd.read_parquet(p["raw_df"])

    return df

# --------------- Random search ---------------
def sample_trial(rng: random.Random, tune_abstain: bool) -> Dict[str, float]:
    params = {
        # Fusion + temps
        "fuse_alpha":      rng.uniform(0.4, 0.8),
        "temp_emb":        rng.uniform(0.35, 0.8),
        "temp_lsa":        rng.uniform(0.6, 1.0),

        # thresholds (kept for parity; not used in compute_final_labels here)
        "cat_assign_min_sim": rng.uniform(0.04, 0.09),
        "cat_low_margin":     rng.uniform(0.001, 0.02),

        # rules-first
        "rule_conf_thr":   rng.uniform(0.65, 0.9),
    }
    if tune_abstain:
        params.update({
            "abstain_enable":        True,
            "abstain_min_prob":      rng.uniform(0.08, 0.16),
            "abstain_min_margin":    rng.uniform(0.02, 0.06),
            "abstain_min_lang_conf": rng.uniform(0.35, 0.55),
            "abstain_require_agreement": rng.random() < 0.5,  # recorded but not used in compute_final_labels_from_cached
        })
    else:
        params["abstain_enable"] = False
    return params

def apply_params_to_settings(base: Settings, params: Dict) -> Settings:
    return Settings.from_overrides(**params)

# --------------- CLI main ---------------
def main():
    ap = argparse.ArgumentParser(description="Tune pipeline knobs with caching.")
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--labels_id_col", type=str, default="press_release_id")
    ap.add_argument("--labels_cid_col", type=str, default="true_cid")

    ap.add_argument("--cache_dir", type=str, default=".cache/tune_pipeline")
    ap.add_argument("--rebuild_cache", action="store_true", default=False)

    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit_n", type=int, default=0, help="Sample N rows from cache for faster tuning. 0=all.")
    ap.add_argument("--tune_abstain", action="store_true", default=False)

    ap.add_argument("--out_json", type=str, default="src/event_feed_app/configs/pipeline_tuned.json")
    args = ap.parse_args()

    rng = random.Random() if args.seed is None else random.Random(args.seed)
    cfg = Settings()

    if args.rebuild_cache or not cache_exists(args.cache_dir):
        print(f"[cache] building cache under: {args.cache_dir}")
        df, S_emb, S_lsa, base_cids = build_cache(
            cfg, args.labels_csv, args.labels_id_col, args.labels_cid_col, args.cache_dir
        )
    else:
        print(f"[cache] loading cache from: {args.cache_dir}")
        df, S_emb, S_lsa, base_cids = load_cache(args.cache_dir)

    # Optionally subsample for speed (consistent across S_* arrays)
    if args.limit_n and len(df) > args.limit_n:
        idx = df.sample(n=args.limit_n, random_state=42).index
        df = df.loc[idx].reset_index(drop=True)
        S_emb = S_emb[idx, :]
        S_lsa = S_lsa[idx, :]

    best_score, best_params = -1.0, None
    for t in range(1, args.trials + 1):
        params = sample_trial(rng, tune_abstain=args.tune_abstain)
        trial_cfg = apply_params_to_settings(cfg, params)
        y_pred = compute_final_labels_from_cached(df, base_cids, S_emb, S_lsa, trial_cfg)
        score = macro_f1(df["label"].astype(str).tolist(), y_pred)
        if score > best_score:
            best_score, best_params = score, params
        if (t % 5 == 0) or (t == args.trials):
            print(f"[{t}/{args.trials}] best macro-F1 so far = {best_score:.4f}")

    if best_params is None:
        raise SystemExit("No trials produced a score. Check cache/data.")
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n=== BEST PARAMS ===")
    print(json.dumps(best_params, indent=2))
    print(f"Saved tuned pipeline config → {args.out_json}")
    print("\n# To apply (bash):")
    for k, v in best_params.items():
        print(f'export {k.upper()}={str(v).lower() if isinstance(v,bool) else v}')

if __name__ == "__main__":
    main()
