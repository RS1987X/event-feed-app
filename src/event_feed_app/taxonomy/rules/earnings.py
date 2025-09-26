# event_feed_app/taxonomy/rules/earnings.py
from __future__ import annotations
import re
from typing import Dict, Any, List
from .registry import register, Details
from .context import Ctx, norm, any_present_morph, proximity_cooccurs_morph
from .constants import RULE_CONF_BASE, EARNINGS_TITLE_TERMS, FIN_CTX


# Optional soft guard: generic “update/advisory” titles
GENERIC_UPDATE_HINT = re.compile(r"(?i)\b(advisory|service update|trading update)\b")

RULE_CONF_STRONG = 0.85


@register
class EarningsHiSigRule:
    """
    High-signal rule for earnings_report.

    Triggers when:
      A) Title contains a strong earnings term (morph-aware), OR
      B) Body has an earnings term near financial context (within 10 tokens).

    With hard veto for invites/webcasts/IR events (to avoid 'invitation to presentation of the report').
    """
    name = "hi_sig_earnings_v1"
    priority = 120  # ahead of generic keyword rules

    def run(self, ctx: Ctx, keywords: Dict[str, Any]):
        title = ctx.title or ""
        body  = ctx.body or ""
        text  = f"{title} {body}"
        tnorm = norm(text)


        # 1) Strong A: title contains an earnings term
        title_hit = any_present_morph(title, EARNINGS_TITLE_TERMS, min_stem=3)
        strongA = bool(title_hit)

        # 2) Strong B: (body/title) earnings term co-occurs with financial context
        strongB, pairB = proximity_cooccurs_morph(text, EARNINGS_TITLE_TERMS, FIN_CTX, window=10, min_stem=3)

        if not (strongA or strongB):
            # avoid overfiring on generic updates
            if GENERIC_UPDATE_HINT.search(tnorm):
                return False, "", None
            return False, "", None

        # Collect hits for explainability
        hits: List[str] = []
        hits.extend([h.lower() for h in title_hit][:2])
        if strongB and pairB != ("", ""):
            hits.extend([pairB[0].lower(), pairB[1].lower()])

        # Dedup keep-order
        seen = set(); uniq = []
        for h in hits:
            if h and h not in seen:
                seen.add(h); uniq.append(h)

        details = Details(
            rule=self.name,
            hits={"earnings_report": uniq},
            rule_conf=RULE_CONF_STRONG,
            lock=True,  # rules-first should trust this
            needs_review_low_sim=False,
        )
        return True, "earnings_report", details
