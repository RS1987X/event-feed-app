# event_taxonomies/rules/ratings.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, any_present, proximity_cooccurs
from .constants import AGENCIES, RATING_ACTIONS, RATING_NOUNS, GUIDANCE_WORDS, RULE_CONF_BASE

@register
class RatingsRule:
    name = "agency + rating action/noun"
    priority = -60

    def run(self, ctx: Ctx, keywords):
        text = f"{ctx.title} {ctx.body}"
        a_hits = any_present(text, AGENCIES)
        if not a_hits:
            return False, None, Details(self.name, {}, 0.0, False)
        b_hits = any_present(text, RATING_ACTIONS + RATING_NOUNS)
        if not b_hits:
            return False, None, Details(self.name, {}, 0.0, False)
        if any_present(text, GUIDANCE_WORDS) and not any_present(text, RATING_NOUNS):
            return False, None, Details(self.name, {}, 0.0, False)
        prox_ok, prox_pair = proximity_cooccurs(text, AGENCIES, RATING_ACTIONS + RATING_NOUNS, window=12)
        details = list(set(a_hits + b_hits + (list(prox_pair) if prox_ok else [])))
        return True, "credit_ratings", Details(self.name, {"credit_ratings": details}, RULE_CONF_BASE[self.name], lock=True)
