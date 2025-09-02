# taxonomy_classifier.py
# Usage:
#   from event_taxonomy.keyword_rules import keywords
#   from taxonomy_classifier import classify_press_release, classify_batch
#
#   cid, details = classify_press_release(title, body, keywords)
#   df_out = classify_batch(df, text_cols=("title","snippet"), keywords=keywords)

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
# from event_taxonomies.keyword_rules import keywords

# ------------------------
# Text utils
# ------------------------

def norm(s: str) -> str:
    return (s or "").lower()

def tokenize(s: str) -> List[str]:
    # minimally unicode-aware for nordic letters
    return re.findall(r"[a-z0-9åäöæøéèüïóáç\-]+", norm(s))

def rx(words: List[str]) -> re.Pattern:
    words = [w for w in words if w]
    if not words:
        # match nothing if list empty
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

def any_present(text: str, terms: List[str]) -> List[str]:
    """Return list of terms present as whole-phrase matches (case-insensitive)."""
    t = norm(text)
    hits = []
    for w in terms:
        if not w:
            continue
        pat = r"\b" + re.escape(w.lower()) + r"\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(w.lower())
    return hits

def proximity_cooccurs(text: str, set_a: List[str], set_b: List[str], window: int = 12) -> Tuple[bool, Tuple[str, str]]:
    """True if any term from set_a occurs within `window` tokens of any term from set_b (either order)."""
    toks = tokenize(text)
    idx = {i: tok for i, tok in enumerate(toks)}

    def first_token_positions(terms: List[str]) -> Dict[str, List[int]]:
        pos = {}
        for term in terms:
            tt = tokenize(term)
            if not tt:
                continue
            first = tt[0]
            pos[term] = [i for i, tok in idx.items() if tok == first]
        return pos

    pos_a = first_token_positions(set_a)
    pos_b = first_token_positions(set_b)

    for ta, ia_list in pos_a.items():
        for tb, ib_list in pos_b.items():
            for ia in ia_list:
                for ib in ib_list:
                    if abs(ia - ib) <= window:
                        return True, (ta, tb)
    return False, ("", "")

def detect_person_name(original_text: str) -> Optional[str]:
    """
    Heuristic: detect a 'Name Surname' (supports nordic chars).
    Run on ORIGINAL (non-lowercased) text.
    """
    pat = r"\b[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+(?:\s+[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+)+\b"
    m = re.search(pat, original_text)
    return m.group(0) if m else None

# ------------------------
# Confidence & locking config
# ------------------------

RULE_CONF_BASE = {
    "flagging (legal/numeric)":        0.95,
    "regulatory approval/patent":      0.90,
    "agency + rating action/noun":     0.90,  # credit ratings
    "MAR Article 19 / PDMR":           0.95,
    "debt security + currency/amount": 0.90,
    "orders/contracts + context":      0.85,
    "role + move + name":              0.90,  # personnel
    "agm notice/outcome":              0.90,
    "invitation routing":              0.80,
    "keywords+exclusions":             0.60,  # will be adjusted by hits/competition
    "recall/cyber cues":               0.80,
    "fallback":                        0.20,
}

LOCK_RULES = {
    "flagging (legal/numeric)",
    "regulatory approval/patent",
    "agency + rating action/noun",
    "MAR Article 19 / PDMR",
    "debt security + currency/amount",
    "role + move + name",
    "agm notice/outcome",
}

# ------------------------
# High-signal building blocks
# ------------------------

AGENCIES = ["moody’s", "moody's", "fitch", "s&p", "standard & poor’s", "standard & poor's"]
RATING_NOUNS = [
    "credit rating", "issuer rating", "bond rating", "instrument rating",
    "ratingförändring", "rating outlook", "watch placement", "rating action"
]
RATING_ACTIONS = [
    "affirmed", "downgrade", "upgrade",
    "outlook revised", "outlook to positive", "outlook to negative", "outlook stable",
    "placed on watch"
]
GUIDANCE_WORDS = ["guidance", "prognos", "utsikter", "outlook for"]  # to avoid mixing with ratings

PERSONNEL_ROLES = [
    "ceo", "cfo", "coo", "chair", "chairman", "chairperson", "board member",
    "styrelseledamot", "vd", "finanschef"
]
PERSONNEL_MOVES = [
    "appointed", "appointment", "resigns", "resignation", "steps down",
    "leaves", "leaves board", "avgår", "tillsätts", "utträder", "udnævnelse", "fratræder"
]

FLAGGING_TERMS = [
    "major shareholder", "shareholding notification", "notification of major holdings",
    "flaggningsanmälan", "notification under chapter 9", "securities market act",
    "värdepappersmarknadslagen", "section 30", "chapter 9", "voting rights", "threshold", "tr-1"
]
FLAG_NUMERIC = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s?%\b")  # e.g., 9.72%

CURRENCY_AMOUNT = re.compile(
    r"\b(?:(?:m|bn)\s?)?(sek|nok|dkk|eur|usd|msek|mnok|mdkk|meur|musd)\b|\b\d+(?:[\.,]\d+)?\s?(m|bn)\s?(sek|eur|usd)\b",
    re.IGNORECASE
)

EXCHANGE_CUES = rx([
    "admission to trading on", "listing on nasdaq", "first north",
    "euronext", "otc", "first day of trading"
])

RECALL = rx(["product recall", "recall of", "återkallar", "tilbagekaldelse"])
CYBER  = rx(["cyber attack", "ransomware", "databrud", "intrång", "data breach"])

PATENT_NUMBER = re.compile(r"\b(?:us|ep)\s?\d{6,}\b", re.IGNORECASE)

# ------------------------
# Structured category helpers
# ------------------------

def credit_ratings_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    a_hits = any_present(text, AGENCIES)
    if not a_hits:
        return False, []
    b_hits = any_present(text, RATING_ACTIONS + RATING_NOUNS)
    if not b_hits:
        return False, []
    if any_present(text, GUIDANCE_WORDS) and not any_present(text, RATING_NOUNS):
        return False, []
    prox_ok, prox_pair = proximity_cooccurs(text, AGENCIES, RATING_ACTIONS + RATING_NOUNS, window=12)
    details = list(set(a_hits + b_hits + (list(prox_pair) if prox_ok else [])))
    return True, details

def personnel_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    role_hits = any_present(text, PERSONNEL_ROLES)
    move_hits = any_present(text, PERSONNEL_MOVES)
    name_hit = detect_person_name(f"{title}\n{body}")
    if role_hits and move_hits and name_hit:
        prox_ok, prox_pair = proximity_cooccurs(text, PERSONNEL_ROLES, PERSONNEL_MOVES, window=20)
        details = list(set(role_hits + move_hits))
        if prox_ok:
            details += list(prox_pair)
        details.append(f"name:{name_hit}")
        return True, details
    return False, []

def flagging_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    hits = any_present(text, FLAGGING_TERMS)
    if hits:
        return True, hits
    # Numeric + context variant
    if FLAG_NUMERIC.search(norm(text)) and any_present(text, ["voting rights", "major holdings", "threshold", "notification of major holdings", "tr-1"]):
        return True, ["%+context"]
    return False, []

def pdmr_rule(title: str, body: str, keywords: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    # Prefer your keyword file if it has a PDMR category; else look for core cues
    text = f"{title} {body}"
    if "pdmr_managers_transactions" in keywords:
        k_hits = any_present(text, keywords["pdmr_managers_transactions"])
        if k_hits:
            return True, k_hits
    core = rx(["managers’ transaction", "managers' transaction", "mar article 19", "article 19 mar", "insynshandel"])
    m = core.search(norm(text))
    return (bool(m), [m.group(0)] if m else [])

def debt_strict_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    debt_terms = rx(["bond issue", "notes", "note issue", "senior unsecured", "convertible", "loan agreement", "tap issue", "prospectus"])
    if debt_terms.search(norm(text)) and CURRENCY_AMOUNT.search(text):
        return True, ["debt+amount"]
    return False, []

def orders_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    orders_terms = rx(["order", "contract awarded", "framework agreement", "tender"])
    place_or_amt = rx(["sek", "eur", "usd", "msek", "meur", "factory", "plant", "delivery", "supply"])
    if orders_terms.search(norm(text)) and place_or_amt.search(norm(text)):
        return True, ["order+context"]
    return False, []

def agm_routing_rule(title: str, body: str) -> Tuple[bool, str]:
    text = norm(f"{title} {body}")
    agm_notice = rx(["notice of annual general meeting", "kallelse till årsstämma", "indkaldelse", "kokouskutsu"])
    agm_outcome = rx(["resolutions at the annual", "proceedings of annual", "beslut vid årsstämman", "vedtagelser"])
    if agm_notice.search(text) or agm_outcome.search(text):
        return True, "agm_egm_governance"
    return False, ""

def invite_routing_rule(title: str) -> Tuple[bool, str]:
    t = norm(title)
    inv_earn = rx(["invitation to presentation of q", "invitation to presentation of the half-year", "year-end presentation"])
    inv_cmd  = rx(["capital markets day", "investor day", "save the date"])
    if inv_earn.search(t):
        return True, "earnings_report"
    if inv_cmd.search(t):
        return True, "other_corporate_update"
    return False, ""

# ------------------------
# Orchestration
# ------------------------

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
    "equity_actions_non_buyback",
    "agm_egm_governance",
    "product_launch_partnership",
    "orders_contracts",
    "labor_workforce",
    "incidents_controversies",
    # fallback: other_corporate_update
]

EXCLUSION_GROUPS = {
    "earnings_report": ["dividend", "share_buyback", "capital_raise_rights_issue", "credit_ratings"],
    "dividend": ["earnings_report", "share_buyback", "capital_raise_rights_issue"],
    "share_buyback": ["earnings_report", "dividend", "capital_raise_rights_issue"],
    "credit_ratings": ["earnings_report", "dividend", "share_buyback"],
    "capital_raise_rights_issue": ["equity_actions_non_buyback", "share_buyback", "dividend"],
    "equity_actions_non_buyback": ["capital_raise_rights_issue"],
    "admission_listing": ["admission_delisting"],
    "admission_delisting": ["admission_listing"],
    "legal_regulatory_compliance": ["credit_ratings"],
}

def _category_hits(text: str, category_terms: List[str]) -> List[str]:
    return any_present(text, category_terms)

@dataclass
class ClassificationDetails:
    rule: str
    hits: Dict[str, List[str]]
    notes: str = ""
    needs_review_low_sim: bool = False

def classify_press_release(title: str, body: str, keywords: Dict[str, List[str]]) -> Tuple[str, Dict]:
    """
    Returns: (category_id, details_dict)
    details: {"rule": "...", "hits": {cid: [terms...]}, "notes": "...", "needs_review_low_sim": bool,
              "rule_conf": float, "lock": bool}
    """
    original_text = f"{title or ''}\n{body or ''}"
    text = f"{title or ''} {body or ''}"

    # 0) Special routing: invitations and AGMs
    ok, cat = invite_routing_rule(title)
    if ok:
        return cat, {"rule": "invitation routing", "hits": {}, "rule_conf": RULE_CONF_BASE["invitation routing"], "lock": False}

    ok, cat = agm_routing_rule(title, body)
    if ok:
        return cat, {"rule": "agm notice/outcome", "hits": {}, "rule_conf": RULE_CONF_BASE["agm notice/outcome"], "lock": True}

    # 1) High-precision legal/flagging/reg approvals
    fr_ok, fr_hits = flagging_rule(title, body)
    if fr_ok:
        return "legal_regulatory_compliance", {
            "rule": "flagging (legal/numeric)",
            "hits": {"legal_regulatory_compliance": fr_hits},
            "rule_conf": RULE_CONF_BASE["flagging (legal/numeric)"],
            "lock": True
        }

    # Regulatory approvals / patents (fallback under LRC):
    if "legal_regulatory_compliance" in keywords:
        lrc_hits = any_present(text, keywords["legal_regulatory_compliance"])
        if lrc_hits or PATENT_NUMBER.search(text):
            key_anchors = ["fda approval", "ema approval", "chmp opinion", "ce mark", "medicare reimbursement", "patent granted", "patent approval"]
            if any(k in lrc_hits for k in key_anchors) or PATENT_NUMBER.search(text):
                bump = 0.02 if PATENT_NUMBER.search(text) else 0.0
                return "legal_regulatory_compliance", {
                    "rule": "regulatory approval/patent",
                    "hits": {"legal_regulatory_compliance": list(set(lrc_hits + (["patent_number"] if PATENT_NUMBER.search(text) else [])))},
                    "rule_conf": min(1.0, RULE_CONF_BASE["regulatory approval/patent"] + bump),
                    "lock": True
                }

    # 2) Credit ratings co-occurrence
    cr_ok, cr_details = credit_ratings_rule(title, body)
    if cr_ok:
        # small bump if proximity matched (already contained via details length; optional)
        return "credit_ratings", {
            "rule": "agency + rating action/noun",
            "hits": {"credit_ratings": cr_details},
            "rule_conf": RULE_CONF_BASE["agency + rating action/noun"],
            "lock": True
        }

    # 3) PDMR (MAR Article 19)
    pd_ok, pd_hits = pdmr_rule(title, body, keywords)
    if pd_ok:
        return "pdmr_managers_transactions", {
            "rule": "MAR Article 19 / PDMR",
            "hits": {"pdmr_managers_transactions": pd_hits},
            "rule_conf": RULE_CONF_BASE["MAR Article 19 / PDMR"],
            "lock": True
        }

    # 4) Debt strict (debt term + currency/amount)
    db_ok, db_hits = debt_strict_rule(title, body)
    if db_ok:
        return "debt_bond_issue", {
            "rule": "debt security + currency/amount",
            "hits": {"debt_bond_issue": db_hits},
            "rule_conf": RULE_CONF_BASE["debt security + currency/amount"],
            "lock": True
        }

    # 5) Orders/contracts (term + context)
    od_ok, od_hits = orders_rule(title, body)
    if od_ok:
        return "orders_contracts", {
            "rule": "orders/contracts + context",
            "hits": {"orders_contracts": od_hits},
            "rule_conf": RULE_CONF_BASE["orders/contracts + context"],
            "lock": False
        }

    # 6) Personnel (role + move + name)
    pm_ok, pm_details = personnel_rule(title, body)
    if pm_ok:
        return "personnel_management_change", {
            "rule": "role + move + name",
            "hits": {"personnel_management_change": pm_details},
            "rule_conf": RULE_CONF_BASE["role + move + name"],
            "lock": True
        }

    # 7) Keyword presence + exclusions (generic scoring)
    all_hits = {cid: _category_hits(text, terms) for cid, terms in keywords.items()}
    # Strengthen capital_raise by requiring pricing/units/prospectus/currency when available
    if all_hits.get("capital_raise_rights_issue"):
        reinforce = any_present(text, ["subscription price", "units", "warrants", "prospectus"]) or bool(CURRENCY_AMOUNT.search(text))
        if not reinforce:
            all_hits["capital_raise_rights_issue"] = []

    # Listing/Delisting: add exchange cues for confidence (left as soft hints)
    if all_hits.get("admission_listing") and not EXCHANGE_CUES.search(norm(text)):
        pass
    if all_hits.get("admission_delisting") and not any_present(text, ["delisting", "avnotering", "trading halt", "suspension", "trading suspended"]):
        pass

    # Equity actions non-buyback: boost “total number of shares and votes”
    if any_present(text, ["total number of shares and votes"]):
        all_hits.setdefault("equity_actions_non_buyback", []).append("total number of shares and votes")

    scored = []
    for cid in PRIORITY:
        hits = all_hits.get(cid, [])
        if not hits:
            continue
        comps = EXCLUSION_GROUPS.get(cid, [])
        competing_hits = sum(len(all_hits.get(c, [])) for c in comps)
        score = len(hits) - 0.6 * competing_hits
        scored.append((score, cid, hits))
    if scored:
        scored.sort(reverse=True)
        best_score, best_cid, best_hits = scored[0]
        total_hits = sum(len(v) for v in all_hits.values())
        # recompute competitors for the best cid
        comp_hits_best = sum(len(all_hits.get(c, [])) for c in EXCLUSION_GROUPS.get(best_cid, []))
        needs_review = (best_score < 1) and (total_hits <= 2)  # weak signal heuristic

        # confidence scaling around the base for keywords+exclusions
        conf = RULE_CONF_BASE["keywords+exclusions"]
        conf += 0.08 * min(3, len(best_hits))              # more anchors → higher
        conf -= 0.06 * min(3, comp_hits_best)              # competitors → lower
        conf = max(0.20, min(0.85, conf))                  # clamp

        return best_cid, {
            "rule": "keywords+exclusions",
            "hits": {best_cid: best_hits},
            "needs_review_low_sim": needs_review,
            "rule_conf": conf,
            "lock": False
        }

    # 8) Incidents/controversies extra cues (recall/cyber)
    if RECALL.search(norm(text)) or CYBER.search(norm(text)):
        return "incidents_controversies", {
            "rule": "recall/cyber cues",
            "hits": {},
            "rule_conf": RULE_CONF_BASE["recall/cyber cues"],
            "lock": False
        }

    # 9) Fallback
    return "other_corporate_update", {
        "rule": "fallback",
        "hits": {},
        "needs_review_low_sim": True,
        "rule_conf": RULE_CONF_BASE["fallback"],
        "lock": False
    }

# ------------------------
# Batch helper (optional)
# ------------------------

def classify_batch(df, text_cols=("title", "body"), keywords: Dict[str, List[str]] = None,
                   title_col: Optional[str] = None, body_col: Optional[str] = None):
    """
    df: pandas DataFrame
    text_cols: (title, body/snippet) column names if title_col/body_col not provided
    Returns a copy with 'pred_cid' and 'pred_details'.
    """
    if keywords is None:
        raise ValueError("Provide the keywords dictionary.")
    if title_col is None or body_col is None:
        title_col, body_col = text_cols
    out = df.copy()
    preds, details = [], []
    for t, b in zip(out[title_col].fillna(""), out[body_col].fillna("")):
        cid, det = classify_press_release(t, b, keywords)
        preds.append(cid)
        details.append(det)
    out["pred_cid"] = preds
    out["pred_details"] = details
    return out
