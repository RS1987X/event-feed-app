# event_taxonomies/rules/keywords_generic.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Pattern
from collections import defaultdict
from dataclasses import dataclass, field
import re

from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm, any_present
from .constants import (
    RULE_CONF_BASE,
    CID_TIEBREAK_ORDER,
    SUPPRESSORS_BY_CATEGORY,
    CURRENCY_AMOUNT,
)

# ---------- Config dataclasses ----------

@dataclass
class ScoreWeights:
    must_all: float = 2.2
    must_any: float = 1.3   # if treat_must_any_as_boolean=True, this weight is used as a boolean bump
    any: float = 1.0
    context: float = 0.3
    treat_must_any_as_boolean: bool = True  # base scoring uses counts; competitor strength uses boolean

@dataclass
class GatingKnobs:
    exclude_is_hard_veto: bool = True  # keep current behavior
    # If you ever want soft excludes: exclude_penalty: float = 2.5

@dataclass
class BoostRule:
    category: str
    # One of the two matchers can be used; both allowed.
    contains_any: Tuple[str, ...] = ()
    regex: Optional[Pattern] = None
    delta: float = 0.5  # how much to add if condition matched

@dataclass
class CompetitionKnobs:
    alpha: float = 0.28         # penalty per weighted competitor hit
    hit_cap: float = 5.0        # cap competitor strength
    min_competitor_score: float = 8.0  # competitor must have > this to count
    # Optional per competitor multiplier: ("target_cid","competitor_cid") -> multiplier
    pair_multipliers: dict[Tuple[str, str], float] = field(default_factory=dict)
    # Modes: "strength" (current), "count" (flat per active competitor). Keep "strength".
    mode: str = "strength"

@dataclass
class ReviewKnobs:
    score_threshold: float = 1.2
    total_terms_threshold: int = 2

@dataclass
class ConfidenceKnobs:
    base_key: str = "keywords+exclusions"
    fallback_key: str = "fallback"
    strong_hit_cap: int = 3
    competitor_count_cap: int = 3
    per_strong_hit_bump: float = 0.06
    per_competitor_deduction: float = 0.08
    min_conf: float = 0.20
    max_conf: float = 0.85

@dataclass
class KwExConfig:
    # ✅ Use default_factory for all instance-typed fields
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    gating: GatingKnobs = field(default_factory=GatingKnobs)
    competition: CompetitionKnobs = field(default_factory=CompetitionKnobs)
    review: ReviewKnobs = field(default_factory=ReviewKnobs)
    confidence: ConfidenceKnobs = field(default_factory=ConfidenceKnobs)
    boosts: Tuple[BoostRule, ...] = field(
        default_factory=lambda: (
            BoostRule(category="debt_bond_issue", regex=CURRENCY_AMOUNT, delta=0.5),
            BoostRule(category="capital_raise_rights_issue", regex=CURRENCY_AMOUNT, delta=0.5),
            BoostRule(category="equity_actions_non_buyback", contains_any=("total number of shares and votes",), delta=0.5),
        )
    )

DEFAULT_KWEX_CFG = KwExConfig()

import json, os
def _apply_tuned_json(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    with open(path, "r") as f:
        blob = json.load(f)

    w = blob.get("weights", {})
    DEFAULT_KWEX_CFG.weights.must_all = float(w.get("must_all", DEFAULT_KWEX_CFG.weights.must_all))
    DEFAULT_KWEX_CFG.weights.must_any = float(w.get("must_any", DEFAULT_KWEX_CFG.weights.must_any))
    DEFAULT_KWEX_CFG.weights.any = float(w.get("any", DEFAULT_KWEX_CFG.weights.any))
    DEFAULT_KWEX_CFG.weights.context = float(w.get("context", DEFAULT_KWEX_CFG.weights.context))
    DEFAULT_KWEX_CFG.weights.treat_must_any_as_boolean = bool(
        w.get("treat_must_any_as_boolean", DEFAULT_KWEX_CFG.weights.treat_must_any_as_boolean)
    )

    c = blob.get("competition", {})
    DEFAULT_KWEX_CFG.competition.alpha = float(c.get("alpha", DEFAULT_KWEX_CFG.competition.alpha))
    DEFAULT_KWEX_CFG.competition.hit_cap = float(c.get("hit_cap", DEFAULT_KWEX_CFG.competition.hit_cap))
    DEFAULT_KWEX_CFG.competition.min_competitor_score = float(
        c.get("min_competitor_score", DEFAULT_KWEX_CFG.competition.min_competitor_score)
    )
    DEFAULT_KWEX_CFG.competition.mode = str(c.get("mode", DEFAULT_KWEX_CFG.competition.mode))

    # pair multipliers "a|b": m
    pm = c.get("pair_multipliers", {}) or {}
    DEFAULT_KWEX_CFG.competition.pair_multipliers.clear()
    for k, v in pm.items():
        if "|" in k:
            a, b = k.split("|", 1)
            DEFAULT_KWEX_CFG.competition.pair_multipliers[(a, b)] = float(v)

    g = blob.get("gating", {})
    DEFAULT_KWEX_CFG.gating.exclude_is_hard_veto = bool(
        g.get("exclude_is_hard_veto", DEFAULT_KWEX_CFG.gating.exclude_is_hard_veto)
    )

    r = blob.get("review", {})
    DEFAULT_KWEX_CFG.review.score_threshold = float(
        r.get("score_threshold", DEFAULT_KWEX_CFG.review.score_threshold)
    )
    DEFAULT_KWEX_CFG.review.total_terms_threshold = int(
        r.get("total_terms_threshold", DEFAULT_KWEX_CFG.review.total_terms_threshold)
    )

    cf = blob.get("confidence", {})
    DEFAULT_KWEX_CFG.confidence.per_strong_hit_bump = float(
        cf.get("per_strong_hit_bump", DEFAULT_KWEX_CFG.confidence.per_strong_hit_bump)
    )
    DEFAULT_KWEX_CFG.confidence.per_competitor_deduction = float(
        cf.get("per_competitor_deduction", DEFAULT_KWEX_CFG.confidence.per_competitor_deduction)
    )
    DEFAULT_KWEX_CFG.confidence.min_conf = float(cf.get("min_conf", DEFAULT_KWEX_CFG.confidence.min_conf))
    DEFAULT_KWEX_CFG.confidence.max_conf = float(cf.get("max_conf", DEFAULT_KWEX_CFG.confidence.max_conf))

_apply_tuned_json(os.getenv("KWEX_CONFIG_JSON", "kwex_tuned.json"))

# ------------------------
# Helpers   
# ------------------------

def _kw_get_list(kdef: Any, key: str) -> List[str]:
    """Return kdef[key] if kdef is a dict with a list value; else []."""
    if isinstance(kdef, dict):
        v = kdef.get(key, [])
        return v if isinstance(v, list) else []
    return []

def _keyword_fallback_score(
    title: str,
    body: str,
    keywords: Dict[str, Any],
    cids_tiebreak_order: List[str],
    suppression_by_category: Dict[str, List[str]],
    cfg: KwExConfig = DEFAULT_KWEX_CFG,
) -> Tuple[str, Dict[str, Any]]:

    text = f"{title or ''} {body or ''}"

    # 1) Collect raw hits & base scores
    raw_hits: Dict[str, Dict[str, List[str]]] = {}
    base_score: Dict[str, float] = {}

    for cid, kdef in keywords.items():
        any_terms = _kw_get_list(kdef, "any") if isinstance(kdef, dict) else (kdef or [])
        must_all  = _kw_get_list(kdef, "must_all")
        must_any  = _kw_get_list(kdef, "must_any")
        ctx_terms = _kw_get_list(kdef, "context")
        exclude   = _kw_get_list(kdef, "exclude")

        any_hits      = any_present(text, any_terms)  if any_terms else []
        must_all_hits = any_present(text, must_all)   if must_all else []
        must_any_hits = any_present(text, must_any)   if must_any else []
        ctx_hits      = any_present(text, ctx_terms)  if ctx_terms else []
        exclude_hits  = any_present(text, exclude)    if exclude else []

        # gating
        failed_all = bool(must_all) and (len(must_all_hits) < len([t for t in must_all if t]))
        failed_any = bool(must_any) and (len(must_any_hits) == 0)

        if failed_all or failed_any or (cfg.gating.exclude_is_hard_veto and exclude_hits):
            score = 0.0
            any_hits = must_all_hits = must_any_hits = ctx_hits = []
        else:
            w = cfg.weights
            must_any_component = (1 if w.treat_must_any_as_boolean and must_any_hits else len(must_any_hits))
            score = (
                w.must_all * len(must_all_hits) +
                w.must_any * must_any_component +
                w.any * len(any_hits) +
                w.context * len(ctx_hits)
            )

            # domain boosts
            for br in cfg.boosts:
                if br.category != cid:
                    continue
                matched = False
                if br.contains_any and any_present(text, list(br.contains_any)):
                    matched = True
                if (br.regex is not None) and br.regex.search(text):
                    matched = True
                if matched:
                    score += br.delta

        raw_hits[cid] = {
            "any": any_hits,
            "must_all": must_all_hits,
            "must_any": must_any_hits,
            "context": ctx_hits,
            "exclude": exclude_hits,
        }
        base_score[cid] = score

    # 2) Competitor penalties
    penalty: Dict[str, float] = defaultdict(float)

    def _hit_strength(v: Dict[str, List[str]]) -> float:
        # Mirror scoring but must_any as boolean
        return (
            cfg.weights.must_all * len(v.get("must_all", [])) +
            1.0 * (1 if v.get("must_any") else 0) +
            cfg.weights.any * len(v.get("any", [])) +
            cfg.weights.context * len(v.get("context", []))
        )

    comp_cfg = cfg.competition
    for cid, s in base_score.items():
        if s <= 0:
            continue
        for comp in suppression_by_category.get(cid, []):
            cs = base_score.get(comp, 0.0)
            if cs <= comp_cfg.min_competitor_score:
                continue
            if comp_cfg.mode == "count":
                strength = 1.0
            else:  # "strength"
                rh = raw_hits.get(comp, {})
                strength = min(comp_cfg.hit_cap, _hit_strength(rh))
            mult = comp_cfg.pair_multipliers.get((cid, comp), 1.0)
            penalty[cid] += comp_cfg.alpha * strength * mult

    final_score = {
        cid: max(0.0, base_score[cid] - penalty.get(cid, 0.0))
        for cid in base_score
    }

    # 3) Pick best
    pr_index = {c: i for i, c in enumerate(cids_tiebreak_order)}
    scored = [(s, cid) for cid, s in final_score.items() if s > 0]
    if scored:
        scored.sort(key=lambda x: (-x[0], pr_index.get(x[1], 10_000)))
        best_score, best_cid = scored[0]
    else:
        best_score, best_cid = 0.0, "other_corporate_update"

    # 4) Review flag
    total_terms_hit = sum(
        len(v["any"]) + len(v["must_all"]) + len(v["must_any"]) + len(v["context"])
        for v in raw_hits.values()
    )
    needs_review = (
        best_score < cfg.review.score_threshold and
        total_terms_hit <= cfg.review.total_terms_threshold
    )

    # 5) Confidence
    conf_knobs = cfg.confidence
    conf = RULE_CONF_BASE[conf_knobs.base_key]
    if best_cid != "other_corporate_update":
        strong_hits = (
            len(raw_hits[best_cid]["any"]) +
            len(raw_hits[best_cid]["must_all"]) +
            (1 if raw_hits[best_cid]["must_any"] else 0)
        )
        competitor_count = sum(
            1 for c in suppression_by_category.get(best_cid, [])
            if base_score.get(c, 0.0) > 0
        )
        conf += conf_knobs.per_strong_hit_bump * min(conf_knobs.strong_hit_cap, strong_hits)
        conf -= conf_knobs.per_competitor_deduction * min(conf_knobs.competitor_count_cap, competitor_count)
        conf = max(conf_knobs.min_conf, min(conf_knobs.max_conf, conf))
    else:
        conf = RULE_CONF_BASE[conf_knobs.fallback_key]

    # 6) Flatten best hits
    if best_cid != "other_corporate_update":
        hits = raw_hits[best_cid]
        best_hits = list(dict.fromkeys(hits["must_all"] + hits["must_any"] + hits["any"] + hits["context"]))
        hits_dict = {best_cid: best_hits}
    else:
        hits_dict = {}

    return best_cid, {
        "hits": hits_dict,
        "needs_review_low_sim": needs_review,
        "rule_conf": conf,
    }

# ------------------------
# Rule
# ------------------------
from .constants import CID_TIEBREAK_ORDER as TIEBREAK  # back-compat alias
@register
class KeywordsGenericRule:
    """
    LAST-LINE fallback scorer using keyword schema:
      - supports: any / must_all / must_any / context / exclude
      - competitor penalties and priority tie-breaks
      - confidence + weak-signal review flag
    """
    name = "keywords+exclusions"
    priority = 10_000  # ensure this runs after all high-signal rules

    def run(self, ctx: Ctx, keywords: Dict[str, Any]):
        best_cid, scored = _keyword_fallback_score(
            ctx.title, ctx.body, keywords, TIEBREAK , SUPPRESSORS_BY_CATEGORY
        )

        det = Details(
            rule=self.name,
            hits=scored["hits"],
            rule_conf=scored["rule_conf"],
            lock=False,
            needs_review_low_sim=scored["needs_review_low_sim"],
        )

        # If *nothing* scored, still return the explicit fallback
        if best_cid == "other_corporate_update" and not det.needs_review_low_sim:
            # keep a faint review flag for truly empty cases
            det.needs_review_low_sim = True

        return True, best_cid, det
