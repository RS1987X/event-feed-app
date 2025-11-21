# event_taxonomies/rules/constants.py
from __future__ import annotations
import re
from .context import rx_any  # memoized, safe to import here

# ---- Credit Ratings ----
AGENCIES = ["moody’s", "moody's", "fitch", "s&p", "standard & poor’s", "standard & poor's"]
RATING_NOUNS = [
    "credit rating", "issuer rating", "bond rating", "instrument rating",
    "ratingförändring", "rating outlook", "watch placement", "rating action",
]
RATING_ACTIONS = [
    "affirmed", "downgrade", "upgrade",
    "outlook revised", "outlook to positive", "outlook to negative", "outlook stable",
    "placed on watch",
]
GUIDANCE_WORDS = ["guidance", "prognos", "utsikter", "outlook for"]

# ---- Personnel ----
PERSONNEL_ROLES = [
    "ceo", "cfo", "coo", "chair", "chairman", "chairperson", "board member",
    "styrelseledamot", "vd", "finanschef",
]
PERSONNEL_MOVES = [
    "appointed", "appointment", "resigns", "resignation", "steps down",
    "leaves", "leaves board", "avgår", "tillsätts", "utträder", "udnævnelse", "fratræder",
]

# ---- Flagging (TR-1) ----
FLAGGING_TERMS = [
    "major shareholder", "shareholding notification", "notification of major holdings",
    "flaggningsanmälan", "notification under chapter 9", "securities market act",
    "värdepappersmarknadslagen", "section 30", "chapter 9", "voting rights", "threshold", "tr-1",
]
FLAG_NUMERIC = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s?%\b")  # e.g., 9.72%

# ---- Money / exchanges ----
CURRENCY_AMOUNT = re.compile(
    r"\b(?:(?:m|bn)\s?)?(sek|nok|dkk|eur|usd|msek|mnok|mdkk|meur|musd)\b"
    r"|\b\d+(?:[\.,]\d+)?\s?(m|bn)\s?(sek|eur|usd)\b",
    re.IGNORECASE,
)

EXCHANGE_CUES = rx_any(tuple([
    "admission to trading on", "listing on nasdaq", "first north",
    "euronext", "otc", "first day of trading",
]))

# ---- Incidents ----
RECALL = rx_any(tuple(["product recall", "recall of", "återkallar", "tilbagekaldelse"]))
CYBER  = rx_any(tuple(["cyber attack", "ransomware", "databrud", "intrång", "data breach"]))

# ---- Patents ----
PATENT_NUMBER = re.compile(r"\b(?:us|ep)\s?\d{6,}\b", re.IGNORECASE)

# ---- Confidence bases (shared) ----
RULE_CONF_BASE = {
    "flagging (legal/numeric)":        0.95,
    "regulatory approval/patent":      0.90,
    "agency + rating action/noun":     0.90,
    "MAR Article 19 / PDMR":           0.95,
    "debt security + currency/amount": 0.90,
    "orders/contracts + context":      0.85,
    "role + move + name":              0.90,
    "agm notice/outcome":              0.90,
    "invitation routing":              0.80,
    "keywords+exclusions":             0.60,
    "recall/cyber cues":               0.80,

    # NEW:
    "mna (transactional)":             0.85,   # tune as you validate precision

    "fallback":                        0.20,
}

# ---- Priority & exclusions (shared by fallback scorer) ----
PRIORITY = [
    "legal_regulatory_compliance",
    "credit_ratings",
    "earnings_report",
    "dividend",
    "share_buyback",
    "capital_raise_rights_issue",
    "admission_listing",
    "admission_delisting",
    "debt_bond_issue",

    # NEW: place M&A here — after debt, before softer ops news
    "mna",

    "equity_actions_non_buyback",
    "agm_egm_governance",
    "product_launch_partnership",
    "orders_contracts",
    "labor_workforce",
    "incidents_controversies",
]
EXCLUSION_GROUPS = {
    "earnings_report": ["dividend", "share_buyback", "capital_raise_rights_issue", "credit_ratings"],
    "dividend": ["earnings_report", "share_buyback", "capital_raise_rights_issue"],
    "share_buyback": ["earnings_report", "dividend", "capital_raise_rights_issue", "mna"],  # add mna
    "credit_ratings": ["earnings_report", "dividend", "share_buyback"],
    "capital_raise_rights_issue": ["equity_actions_non_buyback", "share_buyback", "dividend"],
    "equity_actions_non_buyback": ["capital_raise_rights_issue", "mna"],  # add mna
    "admission_listing": ["admission_delisting"],
    "admission_delisting": ["admission_listing"],
    "legal_regulatory_compliance": ["credit_ratings"],
    # NEW:
    "mna": ["share_buyback", "equity_actions_non_buyback", "product_launch_partnership"],
}
