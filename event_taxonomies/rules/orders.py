# event_taxonomies/rules/orders.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, norm, rx_any
from .constants import RULE_CONF_BASE

ORDERS_TERMS = rx_any(tuple(["order", "contract awarded", "framework agreement", "tender"]))
PLACE_OR_AMT = rx_any(tuple(["sek", "eur", "usd", "msek", "meur", "factory", "plant", "delivery", "supply"]))

@register
class OrdersRule:
    name = "orders/contracts + context"
    priority = -40

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        t = norm(text)
        if ORDERS_TERMS.search(t) and PLACE_OR_AMT.search(t):
            return True, "orders_contracts", Details(self.name, {"orders_contracts": ["order+context"]}, RULE_CONF_BASE[self.name], lock=False)
        return False, None, Details(self.name, {}, 0.0, False)
