# event_feed_app/taxonomy/taxonomy_rules_classifier.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pkgutil
import importlib

import pandas as pd  # only needed for classify_batch

# Import registry + context/norm from your rules package
from .rules import REGISTRY
from .rules.context import Ctx
try:
    # If you kept a separate normalizer
    from .rules.context import norm  # type: ignore
except Exception:
    # Fallback: simple normalizer
    def norm(s: str) -> str:
        return (s or "").lower().replace("\n", " ").strip()

# --- Auto-discover and import all rule modules so they self-register ---
def _eager_import_rules():
    """
    Import every submodule in event_feed_app.taxonomy.rules so that any
    module-level '@register' calls run and populate REGISTRY.
    Skips private helpers like _*, 'context', 'constants', and '__init__'.
    """
    pkg_name = f"{__package__}.rules"  # event_feed_app.taxonomy.rules
    pkg = importlib.import_module(pkg_name)
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.ispkg:
            continue
        name = m.name
        if name.startswith("_") or name in {"context", "constants", "__init__"}:
            continue
        importlib.import_module(f"{pkg_name}.{name}")

# Ensure REGISTRY is populated on import
_eager_import_rules()

def classify_press_release(
    title: str,
    body: str,
    keywords: Dict[str, List[str]],
) -> Tuple[str, Dict]:
    """
    Returns: (category_id, details_dict)
    details: {
        "rule": "...",
        "hits": {cid: [terms...]},
        "notes": "...",
        "needs_review_low_sim": bool,
        "rule_conf": float,
        "lock": bool
    }
    """
    ctx = Ctx(
        title=title or "",
        body=body or "",
        text_norm=norm(f"{title or ''} {body or ''}"),
        text_original=f"{title or ''}\n{body or ''}",
    )

    # Walk rules by ascending priority; first decisive match wins.
    for rule in sorted(REGISTRY, key=lambda r: getattr(r, "priority", 1000)):
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

    # Fallback (last resort)
    return "other_corporate_update", {
        "rule": "fallback",
        "hits": {},
        "needs_review_low_sim": True,
        "rule_conf": 0.20,
        "lock": False,
    }

def classify_batch(
    df: pd.DataFrame,
    text_cols: tuple[str, str] = ("title", "body"),
    keywords: Dict[str, List[str]] | None = None,
    title_col: Optional[str] = None,
    body_col: Optional[str] = None,
) -> pd.DataFrame:
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
    preds: List[str] = []
    details_list: List[Dict] = []

    for t, b in zip(out[title_col].fillna(""), out[body_col].fillna("")):
        cid, det = classify_press_release(t, b, keywords)
        preds.append(cid)
        details_list.append(det)

    out["rule_pred_cid"] = preds
    out["rule_pred_details"] = details_list
    return out
