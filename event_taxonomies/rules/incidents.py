# event_taxonomies/rules/incidents.py
from __future__ import annotations
from . import register, Details
from .context import Ctx, norm
from .constants import RECALL, CYBER, RULE_CONF_BASE

@register
class IncidentsRule:
    name = "recall/cyber cues"
    priority = 200  # runs before fallback but after other high-signal rules

    def run(self, ctx: Ctx, keywords):
        text_norm = norm(f"{ctx.title} {ctx.body}")
        if RECALL.search(text_norm) or CYBER.search(text_norm):
            return True, "incidents_controversies", Details(self.name, {}, RULE_CONF_BASE[self.name], lock=False)
        return False, None, Details(self.name, {}, 0.0, False)
