# event_taxonomies/rules/invite.py
import re
from . import register, Details
from .context import Ctx

INVITE_WORDS = r"(?:invitation|invite[sd]?|reminder|please\s+join\s+us|save\s+the\s+date)"
EARNINGS_TIME = r"(?:q[1-4]\b|quarter|h[12]\b|half[-\s]?year|full[-\s]?year|year[-\s]?end|fy(?:\s?\d{2,4})?)"
EARNINGS_NOUNS = r"(?:earnings|results|interim)"
EVENT_NOUNS   = r"(?:call|presentation|webcast|report|meeting)"
CMD_WORDS     = r"(?:capital\s+markets?\s+day|investor\s+day)"
INV_MEETING   = r"(?:investor(?:s|’s|'s)?\s+meeting)"

EARNINGS_INVITE_RE = re.compile(
    rf"(?i)\b{INVITE_WORDS}\b.*?\b(?:{EARNINGS_TIME}|{EARNINGS_NOUNS})\b.*?\b{EVENT_NOUNS}\b"
)
CMD_INVITE_RE = re.compile(rf"(?i)\b(?:{INVITE_WORDS}\b.*?\b{CMD_WORDS}\b|{CMD_WORDS}\b)")
INV_MEETING_RE = re.compile(rf"(?i)\b(?:{INVITE_WORDS}\b.*?\b{INV_MEETING}\b|{INV_MEETING}\b)")

@register
class InviteRouter:
    name = "invitation routing"
    priority = -100  # run first

    def run(self, ctx: Ctx, keywords):
        t = ctx.title
        if EARNINGS_INVITE_RE.search(t):
            return True, "earnings_report", Details(self.name, {}, 0.80, False)
        if CMD_INVITE_RE.search(t) or INV_MEETING_RE.search(t):
            return True, "other_corporate_update", Details(self.name, {}, 0.80, False)
        return False, None, Details(self.name, {}, 0.0, False)
