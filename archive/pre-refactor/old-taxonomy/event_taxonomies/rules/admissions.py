# event_taxonomies/rules/admissions.py
from . import register, Details
from .context import Ctx, norm, any_present, rx
import re

EXCHANGE_CUES = rx([
    "nasdaq", "first north", "euronext", "london stock exchange",
    "oslo børs", "stockholmsbörsen", "cboe", "otc", "børsen", "ngm"
])

DELIST_STRICT = rx([
    "delist", "delisting", "removed from trading", "last day of trading", "avnotering"
])

LIST_STRICT = rx([
    "admission to trading", "admitted to trading", "listing on", "listing of", "first day of trading",
])

@register
class AdmissionsRule:
    name = "admissions (list/delist)"
    priority = 250  # before generic fallback

    def run(self, ctx: Ctx, keywords):
        t = f"{ctx.title or ''} {ctx.body or ''}"
        tn = norm(t)

        has_exchange = bool(EXCHANGE_CUES.search(tn))

        # --- Delisting: require strict phrase + (exchange or regulator) ---
        if DELIST_STRICT.search(tn) and has_exchange:
            return True, "admission_delisting", Details(
                rule=self.name, hits={"admission_delisting": ["delist+exchange"]},
                rule_conf=0.90, lock=True
            )

        # Guard against “listing” language being mistaken as delist
        if LIST_STRICT.search(tn) and not DELIST_STRICT.search(tn):
            return True, "admission_listing", Details(
                rule=self.name, hits={"admission_listing": ["list+exchange" if has_exchange else "list"]},
                rule_conf=0.85 if has_exchange else 0.75, lock=True
            )

        return False, None, Details(self.name, {}, 0.0, False)
