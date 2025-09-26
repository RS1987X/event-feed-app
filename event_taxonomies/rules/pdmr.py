# event_taxonomies/rules/pdmr.py
from __future__ import annotations
import re
from . import register, Details
from .context import Ctx, norm, any_present
from .constants import RULE_CONF_BASE

CORE_PDMR = re.compile(r"\b(managers[’'] transaction|mar article 19|article 19 mar|insynshandel)\b", re.IGNORECASE)

@register
class PdmrRule:
    name = "MAR Article 19 / PDMR"
    priority = -70

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        if "pdmr_managers_transactions" in keywords:
            k_hits = any_present(text, keywords["pdmr_managers_transactions"])
            if k_hits:
                return True, "pdmr_managers_transactions", Details(self.name, {"pdmr_managers_transactions": k_hits}, RULE_CONF_BASE[self.name], lock=True)
        m = CORE_PDMR.search(norm(text))
        if m:
            return True, "pdmr_managers_transactions", Details(self.name, {"pdmr_managers_transactions": [m.group(0)]}, RULE_CONF_BASE[self.name], lock=True)
        return False, None, Details(self.name, {}, 0.0, False)
