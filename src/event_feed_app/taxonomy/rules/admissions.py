# event_feed_app/taxonomy/rules/admissions.py
from __future__ import annotations

import re
from typing import Optional

from .registry import register, Details
from .context import Ctx, norm
from .constants import RULE_CONF_BASE, LIST_STRICT,DELIST_STRICT, EXCHANGE_CUES, WEAK_LISTED_STATUS

# --- Policy toggle ---
# Require an exchange/venue mention as a NECESSARY anchor.
EXCHANGE_REQUIRED = True


def _first_match(pat: re.Pattern, text: str) -> Optional[str]:
    m = pat.search(text)
    if not m:
        return None
    # Return the matched phrase (not the whole text)
    return m.group(0)

def _has_anchor(text: str) -> bool:
    return bool(EXCHANGE_CUES.search(text)) if EXCHANGE_REQUIRED else True

@register
class AdmissionsRule:
    """
    Route announcements to admission_listing / admission_delisting with high precision.

    Logic:
      - NECESSARY: an exchange/venue mention (EXCHANGE_CUES), if EXCHANGE_REQUIRED=True
      - SUFFICIENT: a strong list OR delist cue (LIST_STRICT / DELIST_STRICT)
      - Explicitly ignores weak "is listed on ..." status-lines without a strong cue
    """
    name = "admission/delisting routing"
    priority = -45  # run early, but after hard veto rules if any

    def run(self, ctx: Ctx, keywords):
        text = norm(f"{ctx.title or ''} {ctx.body or ''}")

        if not _has_anchor(text):
            return False, None, None

        # Ignore weak status mentions unless a strong cue is present
        weak = bool(WEAK_LISTED_STATUS.search(text))

        cue_delist = _first_match(DELIST_STRICT, text)
        cue_list   = _first_match(LIST_STRICT, text)
        exch       = _first_match(EXCHANGE_CUES, text)

        if cue_delist:
            return True, "admission_delisting", Details(
                rule=self.name,
                hits={"exchange": [exch] if exch else [], "cue": [cue_delist]},
                rule_conf=RULE_CONF_BASE.get(self.name, 0.88),
                lock=False,
            )

        if cue_list and (not weak or WEAK_LISTED_STATUS.search(cue_list) is None):
            return True, "admission_listing", Details(
                rule=self.name,
                hits={"exchange": [exch] if exch else [], "cue": [cue_list]},
                rule_conf=RULE_CONF_BASE.get(self.name, 0.88),
                lock=False,
            )

        return False, None, None
