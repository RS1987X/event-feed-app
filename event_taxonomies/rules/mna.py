from __future__ import annotations
import re
from . import register, Details
from .context import Ctx, norm
from .constants import RULE_CONF_BASE

# Core M&A action phrases
MNA_CORE = re.compile(
    r"(?i)\b("
    r"acquisition|acquire[sd]?|to acquire|buy(?:ing)?|takeover|merger|merge[sd]?|"
    r"combination|amalgamation|scheme of arrangement|reverse takeover|"
    r"public (?:cash )?offer (?:to|for) (?:the )?shareholders|tender offer|"
    r"recommended (?:cash )?offer|offer document|merger agreement|"
    r"letters? of intent|binding offer|SPA\b|share purchase agreement"
    r")\b"
)

# Exclude buyback-ish language (to avoid clashes)
BUYBACK_EXCLUDE = re.compile(
    r"(?i)\b(buy-?back|repurchas(?:e|ing)|own shares|treasury shares|"
    r"share repurchase programme|authorization to repurchase)\b"
)

@register
class MnaRule:
    name = "mna (transactional)"
    priority = -35  # after personnel/orders/debt, before fallback

    def run(self, ctx: Ctx, keywords):
        t = norm(f"{ctx.title} {ctx.body}")
        if MNA_CORE.search(t) and not BUYBACK_EXCLUDE.search(t):
            return True, "mna", Details(self.name, {"mna": ["mna_core"]}, rule_conf=0.88, lock=False)
        return False, None, Details(self.name, {}, 0.0, False)
