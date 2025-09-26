# event_feed_app/taxonomy/rules/agm.py
from __future__ import annotations
import re
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm
from .constants import RULE_CONF_BASE, AGM_EGM_NOTICE, AGM_EGM_OUTCOME, AGM_EGM_ABBR_SOFT



@register
class AgmRule:
    name = "agm notice/outcome"
    priority = -90  # run early

    def run(self, ctx: Ctx, keywords):
        text = norm(f"{ctx.title} {ctx.body}")

        # Strong matches -> lock
        if AGM_EGM_NOTICE.search(text) or AGM_EGM_OUTCOME.search(text):
            return True, "agm_egm_governance", Details(
                self.name, {}, RULE_CONF_BASE[self.name], lock=True
            )

        # Backoff on clear AGM/EGM abbreviations in title-only patterns (non-locking)
        if AGM_EGM_ABBR_SOFT.search(norm(ctx.title or "")):
            return True, "agm_egm_governance", Details(
                self.name, {}, max(0.80, RULE_CONF_BASE[self.name] - 0.10), lock=False
            )

        return False, None, Details(self.name, {}, 0.0, False)
