# event_taxonomies/rules/agm.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, norm, rx_any
from .constants import RULE_CONF_BASE

AGM_NOTICE = rx_any(tuple(["notice of annual general meeting", "kallelse till årsstämma", "indkaldelse", "kokouskutsu"]))
AGM_OUTCOME = rx_any(tuple(["resolutions at the annual", "proceedings of annual", "beslut vid årsstämman", "vedtagelser"]))

@register
class AgmRule:
    name = "agm notice/outcome"
    priority = -90

    def run(self, ctx: Ctx, keywords):
        text = norm(f"{ctx.title} {ctx.body}")
        if AGM_NOTICE.search(text) or AGM_OUTCOME.search(text):
            return True, "agm_egm_governance", Details(self.name, {}, RULE_CONF_BASE[self.name], lock=True)
        return False, None, Details(self.name, {}, 0.0, False)
