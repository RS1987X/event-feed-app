from __future__ import annotations
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm, any_present, any_present_morph, any_present_inflected, proximity_cooccurs_morph
from .constants import (
    RULE_CONF_BASE, CURRENCY_AMOUNT,
    ORDER_ACTION_STEMS_EN, ORDER_ACTION_STEMS_NORDIC, ORDER_FIXED_PHRASES_EN,
    ORDER_OBJECT_NOUNS, ORDER_CONTEXT_TERMS, ORDER_EXCLUSION_PHRASES
)

EN_VERB_SUFFIX = r"(?:|s|ed|ing)"
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"

@register
class OrdersRule:
    name = "orders/contracts + context"
    priority = -40

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        tnorm = norm(text)

        if any_present_morph(tnorm, ORDER_EXCLUSION_PHRASES):
            if not any_present_morph(tnorm, ["contract", "framework agreement", "ramavtal", "rammeaftale", "rammeavtale", "call-off", "purchase order", "upphandling", "udbud", "anbud"]):
                return False, None, Details(self.name, {}, 0.0, False)

        action_hits = (
            any_present(text, ORDER_FIXED_PHRASES_EN)
            + any_present_inflected(text, ORDER_ACTION_STEMS_EN, EN_VERB_SUFFIX)
            + any_present_inflected(text, ORDER_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX)
        )
        object_hits = any_present_morph(text, ORDER_OBJECT_NOUNS)
        context_hits = (["currency_amount"] if CURRENCY_AMOUNT.search(text) else []) + any_present_morph(text, ORDER_CONTEXT_TERMS)

        if not action_hits or not object_hits:
            return False, None, Details(self.name, {}, 0.0, False)

        prox_ok, prox_pair = proximity_cooccurs_morph(
            text,
            ORDER_FIXED_PHRASES_EN + ORDER_ACTION_STEMS_EN + ORDER_ACTION_STEMS_NORDIC,
            ORDER_OBJECT_NOUNS,
            window=15
        )
        if not prox_ok:
            contains_fixed_object = bool(any_present(text, ["purchase order", "call-off order"]))
            if not contains_fixed_object:
                return False, None, Details(self.name, {}, 0.0, False)

        has_context = bool(context_hits)
        if not has_context:
            return False, None, Details(self.name, {}, 0.0, False)

        det = []
        det += list(dict.fromkeys(action_hits))
        det += list(dict.fromkeys(object_hits))
        if prox_ok and prox_pair != ("",""):
            det += list(prox_pair)
        det += list(dict.fromkeys(context_hits))
        det = list(dict.fromkeys(det))

        return True, "orders_contracts", Details(self.name, {"orders_contracts": det}, RULE_CONF_BASE[self.name], lock=False)
