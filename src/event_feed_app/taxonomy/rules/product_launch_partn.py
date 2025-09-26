# event_feed_app/taxonomy/rules/product_launch_partnership_rule.py
from __future__ import annotations

import re
from typing import Dict, Any, List

from .registry import register, Details
from .context import (
    Ctx, norm,
    any_present_morph, any_present_inflected, proximity_cooccurs_morph,
    title_pair_close, body_corroborates,   # <-- new helpers
)
from .constants import (
    LAUNCH_PHRASES, DEAL_NOUNS, EN_VERB_STEMS, EN_ENDINGS,
    SVNO_VERB_STEMS, SVNO_ENDINGS, SUPPORT_CTX, PRODUCT_TERMS,
    _HARD_VETO_RE, _SOFT_VETO_HINTS,
)

@register
class ProductLaunchPartnershipRule:
    """
    High-signal rule for product_launch_partnership.

    Triggers on:
      (Title-strong) launch × (deal OR product) within a short window in *title*, or
      (Title-anchor) launch term in title AND body corroborates (launch × deal/product),
      or
      (Body-strong) body-only strong patterns:
         A) launch × deal proximity
         B) (sign|enter|form|agree)+inflection AND a deal noun present
         C) launch × product proximity (solo launches)

    Hard-vetoes governance/advisory/regulatory-only/M&A/transparency.
    """
    name = "hi_sig_product_launch_partnership_v5"
    priority = 120  # run before generic keyword fallback (10_000)

    def run(self, ctx: Ctx, keywords: Dict[str, Any]):
        text_orig = f"{ctx.title or ''} {ctx.body or ''}"
        tnorm = norm(text_orig)

        # 0) Hard veto
        if _HARD_VETO_RE.search(tnorm):
            return False, "", None

        # ---- Title-first evidence -------------------------------------------------
        # Tight proximity in TITLE
        tA, tPairA = proximity_cooccurs_morph(ctx.title or "", LAUNCH_PHRASES, DEAL_NOUNS,
                                              window=8, min_stem=4)
        tC, tPairC = proximity_cooccurs_morph(ctx.title or "", LAUNCH_PHRASES, PRODUCT_TERMS,
                                              window=8, min_stem=4)
        title_strong = tA or tC

        # Anchor: any launch term in TITLE + looser BODY corroboration
        title_anchor = bool(any_present_morph(ctx.title or "", LAUNCH_PHRASES, min_stem=4))
        body_ok = (
            body_corroborates(ctx, LAUNCH_PHRASES, DEAL_NOUNS, window=16, min_stem=4) or
            body_corroborates(ctx, LAUNCH_PHRASES, PRODUCT_TERMS, window=16, min_stem=4)
        )

        # ---- Body-only strong patterns (kept for recall, but lower precedence) ----
        # A) launch × deal proximity (body/text)
        bA, bPairA = proximity_cooccurs_morph(ctx.body or "", LAUNCH_PHRASES, DEAL_NOUNS,
                                              window=12, min_stem=4)

        # B) deal verb (restricted endings) + deal noun (anywhere)
        en_verbs  = any_present_inflected(text_orig, EN_VERB_STEMS, EN_ENDINGS)
        svno_verbs = any_present_inflected(text_orig, SVNO_VERB_STEMS, SVNO_ENDINGS)
        has_deal_noun = bool(any_present_morph(text_orig, DEAL_NOUNS, min_stem=4))
        bB = (bool(en_verbs) or bool(svno_verbs)) and has_deal_noun

        # C) launch × product proximity (solo launches)
        bC, bPairC = proximity_cooccurs_morph(ctx.body or "", LAUNCH_PHRASES, PRODUCT_TERMS,
                                              window=10, min_stem=4)

        # ---- Decision logic with graded confidence/locking ------------------------
        reason = None
        conf = 0.0
        lock = True  # high-signal rule → lock when it fires

        if title_strong:
            conf = 0.85
            reason = "title_proximity"
        elif title_anchor and body_ok:
            conf = 0.80
            reason = "title_anchor_body_corr"
        elif bA or bB or bC:
            conf = 0.78
            reason = "body_only_strong"
        else:
            # Soft veto: advisory/update phrasing without strong signals
            if _SOFT_VETO_HINTS.search(tnorm):
                return False, "", None
            return False, "", None

        # ---- Hits for explainability ---------------------------------------------
        hits: List[str] = []

        # include whichever pairs matched (title first)
        if tA and tPairA != ("", ""):
            hits.extend(list(tPairA))
        if tC and tPairC != ("", ""):
            hits.extend(list(tPairC))
        if bA and bPairA != ("", ""):
            hits.extend(list(bPairA))
        if bC and bPairC != ("", ""):
            hits.extend(list(bPairC))

        # verbs + nouns + supportive context
        hits.extend(en_verbs)
        hits.extend(svno_verbs)
        hits.extend(any_present_morph(text_orig, DEAL_NOUNS,     min_stem=4)[:3])
        hits.extend(any_present_morph(text_orig, PRODUCT_TERMS,  min_stem=4)[:3])
        hits.extend(any_present_morph(text_orig, SUPPORT_CTX,    min_stem=4)[:3])

        # de-dup, keep order
        seen = set()
        uniq = []
        for h in hits:
            h = (h or "").strip().lower()
            if h and h not in seen:
                seen.add(h)
                uniq.append(h)

        details = Details(
            rule=self.name,
            hits={"product_launch_partnership": uniq, "reason": [reason] if reason else []},
            rule_conf=conf,
            lock=lock,
            needs_review_low_sim=False,
        )
        return True, "product_launch_partnership", details
