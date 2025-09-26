from __future__ import annotations
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm
from .constants import RECALL, CYBER, RULE_CONF_BASE

@register
class IncidentsRule:
    name = "recall/cyber cues"
    priority = 200

    def run(self, ctx: Ctx, keywords):
        text = norm(f"{ctx.title} {ctx.body}")
        if RECALL.search(text) or CYBER.search(text):
            return True, "incidents_controversies", Details(self.name, {}, RULE_CONF_BASE[self.name], lock=False)
        return False, None, Details(self.name, {}, 0.0, False)
