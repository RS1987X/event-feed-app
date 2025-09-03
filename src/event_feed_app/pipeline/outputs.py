import json
import numpy as np
import pandas as pd
from typing import Dict
from event_feed_app.config import Settings
#from src.event_feed_app.taxonomy.categories import L1_CATEGORIES
#from event_feed_app.pipeline.consensus import consensus_by_dup_group, neighbor_consistency_scores
from typing import Dict, Sequence
# ---------- Output builders ----------
def build_output_df(
    base_df: pd.DataFrame,
    best_prob: np.ndarray,
    margin: np.ndarray,
    P_emb: np.ndarray,
    P_lsa: np.ndarray,
    P_fused: np.ndarray,
    top3_idx: np.ndarray,
    cfg: Settings,
    consensus_payload: Dict[str, np.ndarray],
    base_cids: Sequence[str],
    run_id: str,
) -> pd.DataFrame:

    id_cols = [c for c in ["press_release_id","release_date","company_name","source","source_url"]
               if c in base_df.columns]
    out = base_df[id_cols].copy() if id_cols else pd.DataFrame(index=base_df.index)

    # Final assignment
    out["final_cid"]          = base_df["final_cid"]
    out["final_category_name"]= base_df["final_category_name"]
    out["final_source"]       = base_df["final_source"]
    out["final_decision"] = base_df["final_decision"]
    out["ml_decision"]    = base_df["ml_decision"]

    # Rules provenance
    for c in ["rule_id","rule_conf","rule_lock","rule_needs_review","rule_pred_cid"]:
        if c in base_df.columns:
            out[c] = base_df[c]

    # ML diagnostics (fused)
    out["ml_pred_cid"]        = base_df["ml_pred_cid"]
    out["ml_conf_fused"]      = np.round(best_prob, 4)
    out["ml_margin_fused"]    = np.round(margin, 4)
    out["ml_entropy_emb"]     = base_df["ml_entropy_emb"]
    out["ml_entropy_lsa"]     = base_df["ml_entropy_lsa"]
    out["ml_entropy_fused"]   = base_df["ml_entropy_fused"]
    out["ml_p1_emb"]          = base_df["ml_p1_emb"]
    out["ml_p1_lsa"]          = base_df["ml_p1_lsa"]
    out["ml_agreement_emb_lsa"]= base_df["ml_agreement_emb_lsa"]
    out["ml_top3_cids"]       = base_df["ml_top3_cids"]
    out["ml_top3_probs"]      = base_df["ml_top3_probs"]
    out["ml_version"]         = base_df["ml_version"]

    # Text
    out["title"]   = base_df["title_clean"]
    out["snippet"] = base_df["full_text_clean"].astype(str).str.slice(0, cfg.snippet_chars)

    # Group consensus exports
    for k, v in consensus_payload.items():
        if k == "P_fused_g":
            continue
        if k == "best_idx_g":
            out["final_cid_group"] = [base_cids[i] for i in v]
        elif k == "best_prob_g":
            out["ml_conf_fused_group"] = np.round(v, 4)
        elif k == "margin_g":
            out["ml_margin_fused_group"] = np.round(v, 4)
        elif k == "entropy_fused_g":
            out["ml_entropy_fused_group"] = np.round(v, 4)
        else:
            out[k] = v

    # Run metadata
    out["run_id"]           = run_id
    out["taxonomy_version"] = cfg.taxonomy_version
    return out

def summarize_run(out: pd.DataFrame, neighbor_consistency: np.ndarray, cfg: Settings, run_id: str,) -> dict:
    from collections import Counter
    N = len(out)

    ml_dec = out.get("ml_decision")
    if ml_dec is None:
        ml_dec = pd.Series([], dtype=str)
    dec_counts = Counter(ml_dec.tolist())

    cat_counts = Counter(out["final_cid"].tolist()) if "final_cid" in out else Counter()

    share_final_by_rules = float((out["final_source"] == "rules").mean()) if "final_source" in out and N else 0.0

    return {
        "run_id": run_id,
        "taxonomy_version": cfg.taxonomy_version,
        "timestamp_utc": run_id,
        "n_docs": N,
        "use_subset": int(cfg.use_subset),
        "emb_model": cfg.emb_model_name,
        "fuse_alpha": cfg.fuse_alpha,
        "temp_emb": cfg.temp_emb,
        "temp_lsa": cfg.temp_lsa,

        "agreement_rate_emb_lsa": float(np.mean(out["ml_agreement_emb_lsa"])) if N and "ml_agreement_emb_lsa" in out else 0.0,
        "avg_margin_fused": float(np.mean(out["ml_margin_fused"])) if N and "ml_margin_fused" in out else 0.0,
        "avg_entropy_fused": float(np.mean(out["ml_entropy_fused"])) if N and "ml_entropy_fused" in out else 0.0,

        f"avg_neighbor_consistency_k{cfg.knn_k}": float(np.mean(neighbor_consistency)) if len(neighbor_consistency) else 0.0,
        f"p10_neighbor_consistency_k{cfg.knn_k}": float(np.percentile(neighbor_consistency, 10)) if len(neighbor_consistency) else 0.0,

        "rate_auto_assign": dec_counts.get("auto_assign", 0) / N if N else 0.0,
        "rate_needs_review_margin": dec_counts.get("needs_review_margin", 0) / N if N else 0.0,
        "rate_needs_review_low_sim": dec_counts.get("needs_review_low_sim", 0) / N if N else 0.0,

        "share_final_by_rules": share_final_by_rules,

        "category_counts_json": json.dumps(dict(cat_counts), ensure_ascii=False),
        "decision_counts_json": json.dumps(dict(Counter(out.get("final_decision", pd.Series([], dtype=str)).tolist())), ensure_ascii=False),
    }
