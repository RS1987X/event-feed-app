# kwex_utils.py
from __future__ import annotations
import json, os, logging
from typing import Dict, Tuple
from event_feed_app.taxonomy.rules.keywords_generic import DEFAULT_KWEX_CFG

log = logging.getLogger(__name__)

def _to_float(x, default): 
    try: return float(x)
    except Exception: return default

def _to_int(x, default): 
    try: return int(x)
    except Exception: return default

def apply_kwex_snapshot(snapshot: Dict) -> None:
    """Mutate DEFAULT_KWEX_CFG in-place from a calibrator snapshot dict."""
    cfg = DEFAULT_KWEX_CFG

    w = snapshot.get("weights", {}) or {}
    c = snapshot.get("competition", {}) or {}
    g = snapshot.get("gating", {}) or {}
    r = snapshot.get("review", {}) or {}
    cf = snapshot.get("confidence", {}) or {}

    # --- weights ---
    cfg.weights.must_all  = _to_float(w.get("must_all",  cfg.weights.must_all),  cfg.weights.must_all)
    cfg.weights.must_any  = _to_float(w.get("must_any",  cfg.weights.must_any),  cfg.weights.must_any)
    cfg.weights.any       = _to_float(w.get("any",       cfg.weights.any),       cfg.weights.any)
    cfg.weights.context   = _to_float(w.get("context",   cfg.weights.context),   cfg.weights.context)
    if "treat_must_any_as_boolean" in w:
        cfg.weights.treat_must_any_as_boolean = bool(w["treat_must_any_as_boolean"])

    # --- competition ---
    cfg.competition.alpha                = _to_float(c.get("alpha", cfg.competition.alpha), cfg.competition.alpha)
    cfg.competition.hit_cap              = _to_float(c.get("hit_cap", cfg.competition.hit_cap), cfg.competition.hit_cap)
    cfg.competition.min_competitor_score = _to_float(c.get("min_competitor_score", cfg.competition.min_competitor_score),
                                                     cfg.competition.min_competitor_score)
    if "mode" in c:
        cfg.competition.mode = str(c["mode"])

    # pair multipliers: snapshot uses "a|b": m
    pm = c.get("pair_multipliers", {}) or {}
    if isinstance(pm, dict):
        cfg.competition.pair_multipliers.clear()
        for k, v in pm.items():
            if isinstance(k, str) and "|" in k:
                a, b = k.split("|", 1)
                cfg.competition.pair_multipliers[(a, b)] = _to_float(v, 1.0)

    # --- gating / review ---
    if "exclude_is_hard_veto" in g:
        cfg.gating.exclude_is_hard_veto = bool(g["exclude_is_hard_veto"])

    if "score_threshold" in r:
        cfg.review.score_threshold = _to_float(r["score_threshold"], cfg.review.score_threshold)
    if "total_terms_threshold" in r:
        cfg.review.total_terms_threshold = _to_int(r["total_terms_threshold"], cfg.review.total_terms_threshold)

    # --- confidence (optional) ---
    for k in ("per_strong_hit_bump","per_competitor_deduction","min_conf","max_conf"):
        if k in cf:
            setattr(cfg.confidence, k, _to_float(cf[k], getattr(cfg.confidence, k)))
    for k in ("base_key","fallback_key"):
        if k in cf:
            setattr(cfg.confidence, k, str(cf[k]))

    # light clamps (keeps things sane if a weird JSON slips in)
    cfg.competition.alpha = max(0.0, cfg.competition.alpha)
    cfg.competition.hit_cap = max(0.0, cfg.competition.hit_cap)
    cfg.confidence.min_conf = max(0.0, min(cfg.confidence.min_conf, cfg.confidence.max_conf))

    log.info("Applied KWEX tuned snapshot: weights=%s comp(alpha=%.3f,hit_cap=%.2f,mode=%s) review(score_thr=%.2f,terms=%d)",
             cfg.weights.__dict__,
             cfg.competition.alpha, cfg.competition.hit_cap, cfg.competition.mode,
             cfg.review.score_threshold, cfg.review.total_terms_threshold)

def load_kwex_snapshot(path: str) -> None:
    """Load JSON file and apply in-place."""
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"KWEX tuned config not found: {path}")
    with open(path, "r") as f:
        snap = json.load(f)
    apply_kwex_snapshot(snap)
