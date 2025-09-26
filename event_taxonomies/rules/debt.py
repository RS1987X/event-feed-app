# event_taxonomies/rules/debt.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, norm, rx_any
from .constants import CURRENCY_AMOUNT, RULE_CONF_BASE
import re

DEBT_TERMS = re.compile(
    r"(?i)\b(?:"
    r"bond(?:[\s_-]+)issue|"
    r"notes|"
    r"note(?:[\s_-]+)issue|"
    r"senior(?:[\s_-]+)unsecured|"
    r"convertible|"
    r"loan(?:[\s_-]+)agreement|"
    r"tap(?:[\s_-]+)issue|"
    r"prospectus"
    r")\b"
)

@register
class DebtRule:
    name = "debt security + currency/amount"
    priority = -50

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        if DEBT_TERMS.search(norm(text)) and CURRENCY_AMOUNT.search(text):
            return True, "debt_bond_issue", Details(self.name, {"debt_bond_issue": ["debt+amount"]}, RULE_CONF_BASE[self.name], lock=True)
        return False, None, Details(self.name, {}, 0.0, False)
