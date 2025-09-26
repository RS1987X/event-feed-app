from __future__ import annotations
import re
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm
from .constants import CURRENCY_AMOUNT, RULE_CONF_BASE, DEBT_TERMS


@register
class DebtRule:
    name = "debt security + currency/amount"
    priority = -50

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        if DEBT_TERMS.search(norm(text)) and CURRENCY_AMOUNT.search(text):
            return True, "debt_bond_issue", Details(self.name, {"debt_bond_issue": ["debt+amount"]}, RULE_CONF_BASE[self.name], lock=True)
        return False, None, Details(self.name, {}, 0.0, False)
