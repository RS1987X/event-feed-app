# event_taxonomies/rules/keywords_generic.py
from __future__ import annotations
from typing import Dict, List
from . import register, Details
from .context import Ctx, norm, any_present
from .constants import (
    PRIORITY, EXCLUSION_GROUPS, RULE_CONF_BASE,
    CURRENCY_AMOUNT, EXCHANGE_CUES,
)

def _category_hits(text: str, category_terms: List[str]) -> List[str]:
    return any_present(text, category_terms)

@register
class KeywordsFallback:
    name = "keywords+exclusions"
    priority = 1000  # last

    def run(self, ctx: Ctx, keywords: Dict[str, List[str]]):
        text = f"{ctx.title or ''} {ctx.body or ''}"
        text_norm = norm(text)

        all_hits = {cid: _category_hits(text, terms) for cid, terms in keywords.items()}

        # Capital raise reinforcement
        if all_hits.get("capital_raise_rights_issue"):
            reinforce = any_present(text, ["subscription price", "units", "warrants", "prospectus"]) or bool(CURRENCY_AMOUNT.search(text))
            if not reinforce:
                all_hits["capital_raise_rights_issue"] = []

        # # Listing/Delisting soft cues (no-op placeholders, per original)
        # if all_hits.get("admission_listing") and not EXCHANGE_CUES.search(text_norm):
        #     pass
        # if all_hits.get("admission_delisting") and not any_present(text, ["delisting", "avnotering", "trading halt", "suspension", "trading suspended"]):
        #     pass

        # Equity actions non-buyback: boost “total number of shares and votes”
        if any_present(text, ["total number of shares and votes"]):
            all_hits.setdefault("equity_actions_non_buyback", []).append("total number of shares and votes")

        scored = []
        for cid in PRIORITY:
            hits = all_hits.get(cid, [])
            if not hits:
                continue
            comps = EXCLUSION_GROUPS.get(cid, [])
            competing_hits = sum(len(all_hits.get(c, [])) for c in comps)
            score = len(hits) - 0.6 * competing_hits
            scored.append((score, cid, hits))

        if not scored:
            return False, None, Details(self.name, {}, 0.0, False)

        scored.sort(reverse=True)
        best_score, best_cid, best_hits = scored[0]
        total_hits = sum(len(v) for v in all_hits.values())
        comp_hits_best = sum(len(all_hits.get(c, [])) for c in EXCLUSION_GROUPS.get(best_cid, []))
        needs_review = (best_score < 1) and (total_hits <= 2)

        conf = RULE_CONF_BASE["keywords+exclusions"]
        conf += 0.08 * min(3, len(best_hits))
        conf -= 0.06 * min(3, comp_hits_best)
        conf = max(0.20, min(0.85, conf))

        return True, best_cid, Details(
            rule=self.name,
            hits={best_cid: best_hits},
            rule_conf=conf,
            lock=False,
            needs_review_low_sim=needs_review,
        )
