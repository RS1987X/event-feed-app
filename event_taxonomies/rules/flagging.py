# event_taxonomies/rules/flagging.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, norm, any_present
from .constants import FLAGGING_TERMS, FLAG_NUMERIC, PATENT_NUMBER, RULE_CONF_BASE

@register
class FlaggingRule:
    name = "flagging (legal/numeric)"
    priority = -80

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        hits = any_present(text, FLAGGING_TERMS)
        if hits:
            return True, "legal_regulatory_compliance", Details(self.name, {"legal_regulatory_compliance": hits}, RULE_CONF_BASE[self.name], lock=True)

        # Numeric + context variant
        if FLAG_NUMERIC.search(norm(text)) and any_present(text, ["voting rights", "major holdings", "threshold", "notification of major holdings", "tr-1"]):
            return True, "legal_regulatory_compliance", Details(self.name, {"legal_regulatory_compliance": ["%+context"]}, RULE_CONF_BASE[self.name], lock=True)

        # Regulatory approvals / patents (still under LRC)
        if "legal_regulatory_compliance" in keywords:
            lrc_hits = any_present(text, keywords["legal_regulatory_compliance"])
            if lrc_hits or PATENT_NUMBER.search(text):
                key_anchors = ["fda approval", "ema approval", "chmp opinion", "ce mark", "medicare reimbursement", "patent granted", "patent approval"]
                if any(k in lrc_hits for k in key_anchors) or PATENT_NUMBER.search(text):
                    bump = 0.02 if PATENT_NUMBER.search(text) else 0.0
                    conf = min(1.0, RULE_CONF_BASE["regulatory approval/patent"] + bump)
                    hits = list(set(lrc_hits + (["patent_number"] if PATENT_NUMBER.search(text) else [])))
                    return True, "legal_regulatory_compliance", Details("regulatory approval/patent", {"legal_regulatory_compliance": hits}, conf, lock=True)

        return False, None, Details(self.name, {}, 0.0, False)
