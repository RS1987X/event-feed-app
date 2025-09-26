# event_taxonomies/taxonomy_rules_classifier.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Import all rules so they self-register
from .rules import REGISTRY
from .rules.context import Ctx, norm
# Side-effect imports to populate REGISTRY:
from .rules import invite, agm, flagging, ratings, pdmr, debt, orders, personnel, incidents, keywords_generic, mna, admissions  # noqa: F401

def classify_press_release(title: str, body: str, keywords: Dict[str, List[str]]) -> Tuple[str, Dict]:
    """
    Returns: (category_id, details_dict)
    details: {"rule": "...", "hits": {cid: [terms...]}, "notes": "...",
              "needs_review_low_sim": bool, "rule_conf": float, "lock": bool}
    """
    ctx = Ctx(
        title=title or "",
        body=body or "",
        text_norm=norm(f"{title or ''} {body or ''}"),
        text_original=f"{title or ''}\n{body or ''}",
    )

    for rule in sorted(REGISTRY, key=lambda r: r.priority):
        ok, cid, det = rule.run(ctx, keywords)
        if ok and cid:
            return cid, {
                "rule": det.rule,
                "hits": det.hits,
                "notes": det.notes,
                "needs_review_low_sim": det.needs_review_low_sim,
                "rule_conf": det.rule_conf,
                "lock": det.lock,
            }

    # Fallback (should not happen because keywords_generic is last)
    return "other_corporate_update", {
        "rule": "fallback",
        "hits": {},
        "needs_review_low_sim": True,
        "rule_conf": 0.20,
        "lock": False,
    }

def classify_batch(df, text_cols=("title", "body"), keywords: Dict[str, List[str]] = None,
                   title_col: Optional[str] = None, body_col: Optional[str] = None):
    """
    df: pandas DataFrame
    text_cols: (title, body/snippet) column names if title_col/body_col not provided
    Returns a copy with 'rule_pred_cid' and 'rule_pred_details'.
    """
    if keywords is None:
        raise ValueError("Provide the keywords dictionary.")
    if title_col is None or body_col is None:
        title_col, body_col = text_cols
    out = df.copy()
    preds, details = [], []
    for t, b in zip(out[title_col].fillna(""), out[body_col].fillna("")):
        cid, det = classify_press_release(t, b, keywords)
        preds.append(cid)
        details.append(det)
    out["rule_pred_cid"] = preds
    out["rule_pred_details"] = details
    return out
