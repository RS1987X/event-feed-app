# event_taxonomies/rules/flagging.py
from __future__ import annotations
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm, any_present
from .constants import (
    FLAGGING_TERMS,
    FLAG_NUMERIC,
    PATENT_NUMBER,
    RULE_CONF_BASE,
)

def _kw_terms_flat(kdef):
    """Return a flat list of terms whether kdef is a list or a dict schema."""
    if isinstance(kdef, list):
        return kdef
    terms = []
    if isinstance(kdef, dict):
        for k in ("any", "must", "must_any", "context"):
            v = kdef.get(k, [])
            if isinstance(v, list):
                terms += v
    return terms


@register
class RegulatoryEventsRule:
    name = "legal regulatory compliance events"
    priority = -80

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        tnorm = norm(text)

        # --- Primary TR-1 / major holdings gate ---
        hits = any_present(text, FLAGGING_TERMS)
        if hits:
            return True, "legal_regulatory_compliance", Details(
                self.name,
                {"legal_regulatory_compliance": hits},
                RULE_CONF_BASE[self.name],
                lock=True,
            )

        # --- Numeric + context variant (e.g., “9.72%” + voting rights) ---
        if FLAG_NUMERIC.search(tnorm) and any_present(
            text,
            ["voting rights", "major holdings", "threshold", "notification of major holdings", "tr-1"],
        ):
            return True, "legal_regulatory_compliance", Details(
                self.name,
                {"legal_regulatory_compliance": ["%+context"]},
                RULE_CONF_BASE[self.name],
                lock=True,
            )

        # --- Regulatory approvals / patents (still routed to LRC) ---
        # Uses the provided keywords (list or dict schema).
        lrc_keydef = keywords.get("legal_regulatory_compliance") if isinstance(keywords, dict) else None
        if lrc_keydef is not None:
            lrc_terms = _kw_terms_flat(lrc_keydef)
            lrc_hits = any_present(text, lrc_terms)

            # Strong anchors for true approvals (avoid generic MAR/legal boilerplate)
            key_anchors = [
                "fda approval", "ema approval", "chmp opinion", "ce mark",
                "medicare reimbursement", "patent granted", "patent approval",
            ]
            has_anchor = any(k in lrc_hits for k in key_anchors)
            has_patent_no = bool(PATENT_NUMBER.search(text))

            if lrc_hits or has_patent_no:
                if has_anchor or has_patent_no:
                    bump = 0.02 if has_patent_no else 0.0
                    conf = min(1.0, RULE_CONF_BASE[self.name] + bump)
                    hits_all = list(set(lrc_hits + (["patent_number"] if has_patent_no else [])))
                    return True, "legal_regulatory_compliance", Details(
                        "regulatory approval/patent",
                        {"legal_regulatory_compliance": hits_all},
                        conf,
                        lock=True,
                    )

        return False, None, Details(self.name, {}, 0.0, False)
