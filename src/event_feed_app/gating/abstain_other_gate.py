# src/event_feed_app/gating/abstain_other_gate.py
from __future__ import annotations
import pandas as pd

def apply_abstain_other_gate(
    df: pd.DataFrame,
    *,
    other_cid: str,
    min_prob: float,
    min_margin: float,
    min_lang_conf: float,
    require_emb_lsa_agreement: bool = False,
) -> int:
    """
    Mutates df in-place:
      - Only rows with final_source == 'ml' are eligible.
      - Routes rows failing any threshold to `other_cid`.
    Returns number of rows routed.
    """
    # Eligible: only ML-sourced (never override locked rules)
    mask_ml = df["final_source"].eq("ml")

    # Conditions
    cond_prob   = df["ml_conf_fused"]   < float(min_prob)
    cond_margin = df["ml_margin_fused"] < float(min_margin)
    cond_lang   = df["lang_conf"].fillna(0.0) < float(min_lang_conf)

    if require_emb_lsa_agreement:
        cond_disagree = ~df["ml_agreement_emb_lsa"].astype(bool)
    else:
        cond_disagree = pd.Series(False, index=df.index)

    # Don't bother if already OTHER
    already_other = df["ml_pred_cid"].astype(str).eq(other_cid) | df["final_cid"].astype(str).eq(other_cid)

    #route_mask = mask_ml & ~already_other & (cond_prob | cond_margin | cond_lang | cond_disagree)
    route_mask = mask_ml & ~already_other & (cond_prob | cond_margin | cond_disagree)
    
    n = int(route_mask.sum())
    if n:
        df.loc[route_mask, "final_cid"]    = other_cid
        df.loc[route_mask, "final_source"] = "abstain_gate"
    return n
