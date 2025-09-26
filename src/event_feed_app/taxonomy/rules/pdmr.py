# event_feed_app/taxonomy/rules/pdmr.py
from __future__ import annotations
import re
from .registry import register, Details  # instead of `from . import register, Details`
from .context import Ctx, norm, any_present
from .constants import RULE_CONF_BASE, CURRENCY_AMOUNT

def _kw_get_list(kdef, key):
    """Return kdef[key] if dict+list, else []"""
    if isinstance(kdef, dict):
        v = kdef.get(key, [])
        return v if isinstance(v, list) else []
    return []

@register
class PdmrRule:
    name = "MAR Article 19 / PDMR"
    # Run before debt/orders but after invites/AGM/flagging.
    # Your previous build used -65; keep that ordering stable.
    priority = -65

    def run(self, ctx: Ctx, keywords):
        title, body = ctx.title or "", ctx.body or ""
        text = f"{title} {body}"
        tnorm = norm(text)

        # Support both schemas:
        # - dict schema with must_any/any/context/exclude
        # - legacy: keywords["pdmr_managers_transactions"] is a flat list -> treat as "any"
        kdef = keywords.get("pdmr_managers_transactions", {}) if isinstance(keywords, dict) else {}

        # Legacy flat-list compatibility (acts like "any")
        if isinstance(kdef, list) and kdef:
            flat_hits = any_present(text, kdef)
            if flat_hits:
                return True, "pdmr_managers_transactions", Details(
                    self.name,
                    {"pdmr_managers_transactions": flat_hits},
                    RULE_CONF_BASE[self.name],
                    lock=True,
                )
            # fall through to strict/backoff gates

        # Strict schema gates
        must_any = _kw_get_list(kdef, "must_any") or [
            "managers' transaction", "managers’ transaction",
            "pdmr", "person discharging managerial responsibilities",
            "insynshandel", "ledningens transaktioner",
        ]
        any_terms = _kw_get_list(kdef, "any")
        ctx_terms = _kw_get_list(kdef, "context")
        exclude   = _kw_get_list(kdef, "exclude")

        # Hard excludes if you maintain any
        if exclude and any_present(tnorm, exclude):
            return False, None, Details(self.name, {}, 0.0, False)

        # 1) Strong gate via MUST_ANY
        must_any_hits = any_present(tnorm, must_any)
        if must_any_hits:
            det = list(dict.fromkeys(
                must_any_hits
                + any_present(tnorm, any_terms)
                + any_present(tnorm, ctx_terms)
            ))
            return True, "pdmr_managers_transactions", Details(
                self.name,
                {"pdmr_managers_transactions": det},
                RULE_CONF_BASE[self.name],
                lock=True,
            )

        # 2) Backoff: "article 19" near MAR + transaction details
        #    Avoid matching generic MAR disclaimers without transaction detail.
        article19_rx = re.compile(r"\barticle\s*19\b", re.IGNORECASE)
        mar_rx       = re.compile(r"\bmar\b|\bmarket abuse regulation\b", re.IGNORECASE)

        sentences = re.split(r"(?<=[\.\!\?])\s+", text)
        has_article19_near_mar = any(article19_rx.search(s) and mar_rx.search(s) for s in sentences)
        if not has_article19_near_mar:
            return False, None, Details(self.name, {}, 0.0, False)

        # PDMR-ish details: ISIN or "N shares"/price or explicit currency amounts
        isin_rx   = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
        shares_rx = re.compile(r"\b\d[\d\.,]*\s+(?:shares|aktier|osakkeita)\b", re.IGNORECASE)
        price_rx  = re.compile(r"\bprice\s+\d[\d\.,]*\b", re.IGNORECASE)

        has_detail = bool(
            isin_rx.search(text) or
            shares_rx.search(text) or
            price_rx.search(text) or
            CURRENCY_AMOUNT.search(text)
        )
        if not has_detail:
            return False, None, Details(self.name, {}, 0.0, False)

        # Build details from precise cues only (don’t dump every MAR token)
        det = ["article 19 + MAR", "transaction_detail"]
        det += any_present(tnorm, any_terms) + any_present(tnorm, ctx_terms)
        det = list(dict.fromkeys(det))

        return True, "pdmr_managers_transactions", Details(
            self.name,
            {"pdmr_managers_transactions": det},
            RULE_CONF_BASE[self.name],
            lock=True,
        )
