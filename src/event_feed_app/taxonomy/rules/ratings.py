from __future__ import annotations
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, any_present, any_present_morph, any_present_inflected, proximity_cooccurs_morph
from .constants import (
    RULE_CONF_BASE,
    AGENCIES, RATING_NOUNS, RATING_ACTION_STEMS_EN, RATING_ACTION_STEMS_SV,
    FIXED_ACTION_PHRASES, RATING_TERMS_FOR_PROX, GUIDANCE_WORDS
)

EN_VERB_SUFFIX = r"(?:|s|ed|ing)"
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"

@register
class CreditRatingsRule:
    name = "agency + rating action/noun"
    priority = -70

    def run(self, ctx: Ctx, keywords):
        title, body = ctx.title or "", ctx.body or ""
        text = f"{title} {body}"

        a_hits = any_present(text, AGENCIES)
        if not a_hits:
            return False, None, Details(self.name, {}, 0.0, False)

        b_hits = (
            any_present(text, FIXED_ACTION_PHRASES)
            + any_present_inflected(text, RATING_ACTION_STEMS_EN, EN_VERB_SUFFIX)
            + any_present_inflected(text, RATING_ACTION_STEMS_SV, SV_VERB_SUFFIX)
            + any_present_morph(text, RATING_NOUNS)
        )
        if not b_hits:
            return False, None, Details(self.name, {}, 0.0, False)

        has_guidance = bool(any_present(text, GUIDANCE_WORDS) or any_present_morph(text, ["prognos", "utsikt"]))
        has_rating_noun = bool(any_present_morph(text, RATING_NOUNS))
        if has_guidance and not has_rating_noun:
            return False, None, Details(self.name, {}, 0.0, False)

        prox_ok, prox_pair = proximity_cooccurs_morph(text, AGENCIES, RATING_TERMS_FOR_PROX, window=12)

        details = a_hits + b_hits + (list(prox_pair) if prox_ok else [])
        details = list(dict.fromkeys(details))

        return True, "credit_ratings", Details(self.name, {"credit_ratings": details}, RULE_CONF_BASE[self.name], lock=True)
