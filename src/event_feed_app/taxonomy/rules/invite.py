from __future__ import annotations
import re
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx
from .constants import RULE_CONF_BASE, EARNINGS_INVITE_RE, CMD_INVITE_RE, INV_MEETING_RE


@register
class InviteRule:
    name = "invitation routing"
    priority = -100  # run first

    def run(self, ctx: Ctx, keywords):
        t = ctx.title or ""
        if EARNINGS_INVITE_RE.search(t):
            return True, "earnings_report", Details(self.name, {}, RULE_CONF_BASE[self.name], lock=False)
        if CMD_INVITE_RE.search(t) or INV_MEETING_RE.search(t):
            return True, "other_corporate_update", Details(self.name, {}, RULE_CONF_BASE[self.name], lock=False)
        return False, None, Details(self.name, {}, 0.0, False)
