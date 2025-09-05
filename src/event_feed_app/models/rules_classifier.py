# taxonomy_classifier.py
# Usage:
#   from event_taxonomy.keyword_rules import keywords
#   from taxonomy_classifier import classify_press_release, classify_batch
#
#   cid, details = classify_press_release(title, body, keywords)
#   df_out = classify_batch(df, text_cols=("title","snippet"), keywords=keywords)

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
# from event_taxonomies.keyword_rules import keywords
#from event_feed_app.models.flair_ner import detect_person_name_flair, ner_tag_sentences

from event_feed_app.models.spacy_ner import detect_person_name_spacy# as detect_person_name_flair
from event_feed_app.models.spacy_ner import ner_tag_sentences

# ------------------------
# Text utils
# ------------------------

import unicodedata
from functools import lru_cache

def _normalize_quotes(s: str) -> str:
    return (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
    )

def norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = _normalize_quotes(s)
    return s.casefold()

_TOKEN_RX = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)
def tokenize(s: str) -> List[str]:
    return _TOKEN_RX.findall(norm(s))

@lru_cache(maxsize=4096)
def _build_morph_phrase_pattern(phrase: str, min_stem: int = 3) -> str:
    """
    Build a regex that matches the phrase allowing inflectional endings on each *word*.
    - Adds \w* to tokens length>=min_stem.
    - Preserves token order.
    - Allows spaces OR hyphens between tokens.
    """
    toks = tokenize(phrase)
    if not toks:
        # If tokenization fails (symbols-only), fall back to literal with word boundaries
        esc = re.escape(_normalize_quotes(unicodedata.normalize("NFKC", phrase)))
        return rf"\b{esc}\b"

    parts = []
    for t in toks:
        esc = re.escape(t)
        if len(t) >= min_stem:
            parts.append(esc + r"\w*")
        else:
            parts.append(esc)  # avoid overmatching very short tokens

    # Between tokens allow spaces or hyphens
    #inner = r"(?:[\s\-]+)".join(parts)
    # in _build_morph_phrase_pattern
    inner = r"(?:[\s\-_]+)".join(parts)  # allow underscore too
    return rf"\b{inner}\b"

def any_present_morph(text: str, terms: List[str], min_stem: int = 3) -> List[str]:
    """
    Return the *canonical terms* (as provided) that match with morphological flexibility.
    Example: 'utsikt' will match 'utsikterna', 'utsikter', etc.
    """
    t = norm(text)
    hits: List[str] = []
    for term in terms:
        if not term:
            continue
        pat = _build_morph_phrase_pattern(term, min_stem=min_stem)
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(term.casefold())
    # dedupe, keep order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

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
        tw = norm(w)  # normalize the term too
        pat = r"\b" + re.escape(tw) + r"\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(tw)
    # order-preserving dedupe
    seen = set(); out = []
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return out

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

def proximity_cooccurs_morph(text: str, set_a: List[str], set_b: List[str], window: int = 12, min_stem: int = 3) -> Tuple[bool, Tuple[str, str]]:
    """
    Morph-aware proximity: a phrase matches if each token in the phrase
    equals the text token OR the text token startswith the phrase token (for tokens >= min_stem).
    Allows you to pass stems like 'bekräft' and still match 'bekräftade'.
    """
    toks = tokenize(text)

    def positions_morph(terms: List[str]) -> Dict[str, List[int]]:
        pos: Dict[str, List[int]] = {}
        for term in terms:
            ptoks = tokenize(term)
            if not ptoks:
                continue
            m = len(ptoks)
            hits: List[int] = []
            for i in range(0, len(toks) - m + 1):
                ok = True
                for j, pt in enumerate(ptoks):
                    tt = toks[i + j]
                    if len(pt) >= min_stem:
                        if not tt.startswith(pt):
                            ok = False
                            break
                    else:
                        if tt != pt:
                            ok = False
                            break
                if ok:
                    hits.append(i)
            if hits:
                pos[term] = hits
        return pos

    pos_a = positions_morph(set_a)
    pos_b = positions_morph(set_b)

    for ta, ia_list in pos_a.items():
        for tb, ib_list in pos_b.items():
            for ia in ia_list:
                for ib in ib_list:
                    if abs(ia - ib) <= window:
                        return True, (ta, tb)
    return False, ("", "")

# from flair.models import SequenceTagger
# from flair.data import Sentence

# _tagger = SequenceTagger.load("flair/ner-multi")  # downloads once

# def detect_person_name_flair(text: str) -> Optional[str]:
#     sent = Sentence(text)
#     _tagger.predict(sent)
#     for span in sent.get_spans("ner"):
#         if span.get_label("ner").value in {"PER","PERSON"}:
#             return span.text
#     return None

# def detect_person_name(original_text: str) -> Optional[str]:
#     """
#     Heuristic: detect a 'Name Surname' (supports nordic chars).
#     Run on ORIGINAL (non-lowercased) text.
#     """
#     pat = r"\b[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+(?:\s+[A-ZÅÄÖÆØ][a-zåäöæøéèüïóáç\-]+)+\b"
#     m = re.search(pat, original_text)
#     return m.group(0) if m else None

# Controlled endings
EN_VERB_SUFFIX = r"(?:|s|ed|ing)"                 # affirms/affirmed/affirming
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"        # bekräftar/bekräftade/bekräftat, höjer/höjde/höjt, ...

def any_present_inflected(text: str, stems: List[str], suffix_group: str) -> List[str]:
    """
    Match stems with a *restricted* set of endings (suffix_group).
    Examples:
        EN: stem 'affirm' with (?:|s|ed|ing) matches affirms/affirmed/affirming
        SV: stem 'bekräft' with (?:|ar|ade|at|er|te|t) matches bekräftar/bekräftade/...
    Returns the stems (lowercased) that matched. Order-preserving de-dup.
    """
    t = norm(text)
    hits: List[str] = []
    for stem in stems:
        if not stem:
            continue
        pat = rf"\b{re.escape(stem)}{suffix_group}\b"
        if re.search(pat, t, flags=re.IGNORECASE):
            hits.append(stem.lower())
    seen = set()
    out: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


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
    "mna high-signal":                  0.90,  # M&A with ORG gating
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
    "mna high-signal",
}

# ------------------------
# High-signal building blocks
# ------------------------

# ----- Credit ratings vocab -----

AGENCIES = ["moody's", "moodys", "moody's ratings", "fitch", "fitch ratings", "s&p", "standard & poor's","s&p global ratings","standard and poor's"]

RATING_NOUNS = [
    "credit rating", "kreditbetyg", "issuer rating", "bond rating", "instrument rating",
    "ratingförändring", "rating outlook", "watch placement", "rating action",
    "long-term issuer rating", "lt issuer rating",
    "corporate family rating", "cfr",
    "probability of default rating", "pdr",
    "insurance financial strength rating", "ifs rating", "ifsr",
    "senior unsecured rating", "subordinated rating", "hybrid notes rating",
    "mtn programme rating", "euro medium term notes rating",
    "mtn program rating", "emtn program rating",          # US spelling variants
    "stable outlook", "negative outlook", "positive outlook",
    "outlook stable", "outlook negative", "outlook positive",  # reversed order
    "issuer credit rating", "issue rating",
    "recovery rating",
    "rating_action", "icr",                               # abbreviations
    "national scale rating", "nsr",
    "periodic review", "credit opinion", "rating report", # if you want reviews as credit_ratings
    "creditwatch negative", "creditwatch positive", "creditwatch developing",
]

# English verb stems (use restricted suffixes)
RATING_ACTION_STEMS_EN = ["affirm", "downgrade", "upgrade", "revise", "assign", "maintain", "withdraw","reaffirm","confirm", "raise", "lower", "rate"]

# Swedish verb stems (restricted suffixes)
RATING_ACTION_STEMS_SV = ["bekräft", "höj", "sänk", "ändr", "revider","tilldel","behåll","återkall","bibehåll"]

# Multi-word phrases that aren't just a verb inflection
FIXED_ACTION_PHRASES = [
    "placed on watch",   # common literal
    "place on watch",    # captures some headlines
    "put on watch",      # occasional variant
    "outlook changed to",
    "changed the outlook to",
    "outlook maintained",
    "outlook remains",
]

FIXED_ACTION_PHRASES += [
    "placed on creditwatch", "removed from creditwatch",
    "on creditwatch negative", "remains on creditwatch", "kept on creditwatch",
    "outlook revised to", "outlook changed to",
    "outlook unchanged", "unchanged outlook", "outlook remains unchanged",
]

RATING_TERMS_FOR_PROX = FIXED_ACTION_PHRASES + RATING_ACTION_STEMS_EN + RATING_ACTION_STEMS_SV + RATING_NOUNS



# ----- Orders & Contracts vocab -----

# Verbs (English) as stems → use EN_VERB_SUFFIX
ORDER_ACTION_STEMS_EN = [
    "win", "secure", "receive", "book", "award", "sign", "enter", "ink", "land"
]
# Common fixed English phrases (literal)
ORDER_FIXED_PHRASES_EN = [
    "wins order", "wins contract", "secures order", "secures contract",
    "received an order", "received a contract", "books order", "booked order",
    "awarded a contract", "contract awarded", "purchase order",
    "call-off order", "call off order", "call-off from framework",
    "enter into agreement", "signs contract", "signed a contract"
]

# Nordic stems → reuse SV_VERB_SUFFIX (good enough for DA/NO too)
# (We bias toward stems that won’t overmatch normal prose.)
ORDER_ACTION_STEMS_NORDIC = [
    # Swedish
    "tilldel", "teckn", "ingå", "avrop", "beställ", "erhåll", "vinn",
    # Danish / Norwegian close cognates
    "tildel", "indgå", "rammeaftal", "rammeavtal", "vind", "inngå",
    # Finnish (rough stems; still useful)
    "voitt", "saa tilauk", "puitesopim", "sopimukse"
]

# Objects (nouns) & procurement context nouns — use any_present_morph
ORDER_OBJECT_NOUNS = [
    "order", "contract",
    "framework agreement", "ramavtal", "rammeaftale", "rammeavtale", "puitesopimus",
    "call-off", "call-off order", "purchase order",
    "tender", "upphandling", "udbud", "anbud", "tilaus", "sopimus"
]

# Context anchors (units, value words, fulfillment verbs)
ORDER_CONTEXT_TERMS = [
    # value words
    "order value", "ordervärde", "kontraktsvärde", "valued at", "worth", "value of",
    # fulfillment / scope
    "deliver", "delivery", "deliveries", "supply", "supplies", "ship", "install", "commission",
    "customer", "client", "end customer", "end-client", "project", "site", "factory", "plant",
    # coarse units
    "units", "buses", "trucks", "locomotives", "turbines", "mw", "mwh", "kv", "tonnes", "barrels", "bbl"
]

# M&A / equity tender exclusions
ORDER_EXCLUSION_PHRASES = [
    "tender offer", "public tender offer", "takeover bid", "recommended cash offer"
]

# ----- Personnel vocab -----

# Roles (morph-friendly nouns)
PERSONNEL_ROLES = [
    # English
    "ceo", "chief executive officer", "group ceo",
    "cfo", "chief financial officer",
    "coo", "chief operating officer",
    "cto", "chief technology officer",
    "cio", "chief information officer",
    "chair", "chairperson", "chairman", "chair of the board",
    "board member", "board director", "non-executive director", "ned",
    "managing director", "md",

    # Nordic
    "vd", "verkställande direktör",
    "finanschef", "ekonomichef","finansdirektör", 
    "styrelseledamot", "styrelseordförande", "ordförande",
]

# Move verbs (use restricted suffixes)
PERSONNEL_MOVE_STEMS_EN = [
    "appoint", "resign", "step", "leave", "join",
    "retire", "promot", "elect", "re-elect", "dismiss", "terminate",
]
PERSONNEL_MOVE_STEMS_NORDIC = [
    # Swedish
    "utse", "utnämn", "tillsätt", "avgå", "lämn", "frånträd",
    "tillträ", "rekryter", "anställ", "befordr", "väljs", "omväljs",
    # Danish / Norwegian (reuse SV suffix set)
    "udnævn", "udnævn", "fratræd", "træder", "tiltræd", "ansæt",
    "utnev", "fratr", "tiltr", "ansett",
]

# Fixed phrases (literal)
PERSONNEL_FIXED_PHRASES = [
    "appointed as", "elected as", "joins as", "steps down as",
    "to step down as", "leaves the board", "leaves board",
    "assumes the role of", "assumes the position of",
    "new ceo", "interim ceo", "acting ceo", "interim cfo", "acting cfo",
]

# Nomination/AGM proposal (exclusion if it's just a proposal)
PERSONNEL_PROPOSAL_NOUNS = [
    "nomination committee", "valberedningen", "valberedning",
    "valgkomité", "valgkomite", "valgkomiteen",
]
PERSONNEL_PROPOSAL_STEMS = ["propos", "föresl"]  # proposes / föreslår
AGM_TERMS = ["annual general meeting", "extraordinary general meeting", "årsstämma", "extra bolagsstämma"]

# ===== M&A vocab =====

# Verbs (EN) → restricted suffixes
MNA_ACTION_STEMS_EN = [
    "acquir",     # acquire/acquires/acquired/acquiring
    "merg",       # merge/merged/merging
    "combin",     # combine/combination
    "offer",      # offer/offered/offering (paired with MNA nouns)
    "bid",        # bid/bids/bidded/bidding
    "purchas",    # purchase/purchased/purchasing
    "sell",       # sell/sold/selling
    "divest",     # divest/divested/divesting
    "dispos",     # dispose/disposed/disposal
    "spin",       # spin/spun/spinning (use with fixed phrases too)
    "agree"       # agrees/agreed (paired with agreement nouns)
]

# Verbs (Nordic) → reuse SV suffixes for SV/DA/NO
MNA_ACTION_STEMS_NORDIC = [
    # Swedish
    "förvärv", "köp", "uppköp", "lägg", "bud", "försälj", "sälj",
    "avyttr", "fusion", "samgå", "sammanslag",
    # Danish/Norwegian cognates
    "opkøb", "overtagelsestilbud", "overtakelsestilbud",
    "oppkjøp", "fusjon", "sammenslå",
]

# Fixed phrases that are strong M&A anchors (literal)
MNA_FIXED_PHRASES = [
    "public tender offer", "recommended public offer", "recommended public cash offer",
    "takeover bid", "mandatory offer", "cash offer", "exchange offer",
    "merger agreement", "business combination agreement", "combination agreement",
    "share purchase agreement", "asset purchase agreement", "sale and purchase agreement",
    "scheme of arrangement",
    "offer to the shareholders of", "to acquire all shares", "to acquire the remaining shares",
    "sale of", "spin-off", "spins off", "split-off", "carve-out",
]

# Nouns/objects (morph-friendly) that define deal types
MNA_OBJECT_NOUNS = [
    # English
    "acquisition", "merger", "business combination", "combination",
    "offer", "tender offer", "takeover bid", "public offer", "cash offer", "exchange offer",
    "divestment", "disposal", "sale", "spin-off", "demerger", "split-off", "carve-out",
    "minority stake", "majority stake",
    # Swedish / Nordic
    "förvärv", "uppköp", "offentligt uppköpserbjudande", "kontantbud", "budpliktsbud",
    "försäljning", "avyttring", "fusion", "samgående", "sammanslagning",
]

# Consideration / context terms (morph-friendly)
MNA_CONTEXT_TERMS = [
    "offer price", "equity value", "enterprise value", "ev", "valuation", "implies a premium",
    "premium of", "cash consideration", "share consideration", "mixed consideration",
    "acceptance period", "acceptance level", "acceptance rate", "unconditional", "settlement",
    "timetable", "completion", "closing", "conditions to completion", "regulatory approvals",
    "merger clearance", "antitrust approval", "competition approval",
    # common per-share/unit wording (in addition to currency regex)
    "per share", "per aktie",
]

# Helpful regex for explicit per-share phrasing (in addition to morph terms)
PER_SHARE_RX = re.compile(r"\b(per[-\s]?share|per\s+aktie)\b", re.IGNORECASE)

# RATING_ACTIONS = [
#     # English lemmas/phrases
#     "affirm", "downgrade", "upgrade", "outlook revise",
#     "outlook stable", "outlook positive", "outlook negative", "place on watch",

#     # Swedish stems/lemmas (short but distinctive)
#     "bekräft",    # bekräftar, bekräftade, bekräftat
#     "höj",        # höjer, höjde, höjt
#     "sänk",       # sänker, sänkte, sänkt
#     "ändr",       # ändrar, ändrade, ändrat
#     "revider",    # reviderar, reviderade, reviderat
#     "bevakning",  # placerar på bevakning
# ]

GUIDANCE_WORDS = ["guidance", "prognos", "utsikter", "outlook for"]  # to avoid mixing with ratings

# PERSONNEL_ROLES = [
#     "ceo", "cfo", "coo", "chair", "chairman", "chairperson", "board member",
#     "styrelseledamot", "vd", "finanschef"
# ]
# PERSONNEL_MOVES = [
#     "appointed", "appointment", "resigns", "resignation", "steps down",
#     "leaves", "leaves board", "avgår", "tillsätts", "utträder", "udnævnelse", "fratræder"
# ]

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

    # Agencies: keep exact (brands shouldn't morph)
    a_hits = any_present(text, AGENCIES)
    if not a_hits:
        return False, []

    # Actions/nouns:
    # - Fixed phrases: literal match
    # - EN/SV stems: restricted suffix sets
    # - Nouns: morph-aware (safe to allow \w* on nouns)
    b_hits = (
        any_present(text, FIXED_ACTION_PHRASES)
        + any_present_inflected(text, RATING_ACTION_STEMS_EN, EN_VERB_SUFFIX)
        + any_present_inflected(text, RATING_ACTION_STEMS_SV, SV_VERB_SUFFIX)
        + any_present_morph(text, RATING_NOUNS)
    )
    if not b_hits:
        return False, []

    # Guidance exclusion (incl. Swedish inflections)
    has_guidance = bool(
        any_present(text, GUIDANCE_WORDS) or
        any_present_morph(text, ["prognos", "utsikt"])
    )
    has_rating_noun = bool(any_present_morph(text, RATING_NOUNS))
    if has_guidance and not has_rating_noun:
        return False, []

    # Proximity: morph-aware so stems work (optional to enforce)
    prox_ok, prox_pair = proximity_cooccurs_morph(
        text,
        AGENCIES,
        RATING_TERMS_FOR_PROX,
        window=12
    )

    # Order-preserving de-dup for stable outputs
    details = a_hits + b_hits + (list(prox_pair) if prox_ok else [])
    details = list(dict.fromkeys(details))

    return True, details

def personnel_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    original = f"{title}\n{body}"

    # Exclude nomination proposals for AGMs (these are governance, not confirmed moves)
    if (any_present_morph(text, PERSONNEL_PROPOSAL_NOUNS)
        and (any_present_inflected(text, PERSONNEL_PROPOSAL_STEMS, EN_VERB_SUFFIX)
             or any_present_inflected(text, PERSONNEL_PROPOSAL_STEMS, SV_VERB_SUFFIX))
        and any_present_morph(text, AGM_TERMS)):
        return False, []

    role_hits = any_present_morph(text, PERSONNEL_ROLES)

    move_hits = (
        any_present(text, PERSONNEL_FIXED_PHRASES)
        + any_present_inflected(text, PERSONNEL_MOVE_STEMS_EN, EN_VERB_SUFFIX)
        + any_present_inflected(text, PERSONNEL_MOVE_STEMS_NORDIC, SV_VERB_SUFFIX)
    )

    if not (role_hits and move_hits):
        return False, []

    name_hit = detect_person_name_spacy(original)

    if not name_hit:
        return False, []

    # Proximity between role and move (morph-aware)
    prox_ok, prox_pair = proximity_cooccurs_morph(text, PERSONNEL_ROLES,
                                                  PERSONNEL_MOVE_STEMS_EN + PERSONNEL_MOVE_STEMS_NORDIC + PERSONNEL_FIXED_PHRASES,
                                                  window=24)

    if not prox_ok:
        return False, []

    # Ensure the person's name is near a role or a move (extra precision)
    # Find first token index of the detected name
    toks = tokenize(text)
    name_tok = tokenize(name_hit)[0] if name_hit else ""
    name_idxs = [i for i, t in enumerate(toks) if t == name_tok]

    def _positions_morph(terms: List[str], min_stem: int = 3) -> List[int]:
        hits = []
        for term in terms:
            ptoks = tokenize(term)
            if not ptoks:
                continue
            m = len(ptoks)
            for i in range(0, len(toks) - m + 1):
                ok = True
                for j, pt in enumerate(ptoks):
                    tt = toks[i + j]
                    if len(pt) >= min_stem:
                        if not tt.startswith(pt):
                            ok = False; break
                    else:
                        if tt != pt:
                            ok = False; break
                if ok:
                    hits.append(i)
        return hits

    anchor_idxs = _positions_morph(PERSONNEL_ROLES) + _positions_morph(PERSONNEL_MOVE_STEMS_EN + PERSONNEL_MOVE_STEMS_NORDIC + PERSONNEL_FIXED_PHRASES)
    near_name = any(abs(i - j) <= 30 for i in name_idxs for j in anchor_idxs) if name_idxs and anchor_idxs else False
    if not near_name:
        return False, []

    # Build details (order-preserving)
    details = []
    details += list(dict.fromkeys(role_hits))
    details += list(dict.fromkeys(move_hits))
    if prox_ok and prox_pair != ("", ""):
        details += list(prox_pair)
    details.append(f"name:{name_hit}")
    details = list(dict.fromkeys(details))
    return True, details

# --- M&A rule with Flair sentence-level ORG gating ---

def mna_rule_flair(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title or ''} {body or ''}"

    # Cheap prefilter: don’t run Flair unless the PR looks M&A-ish
    cheap_action = (
        any_present(text, MNA_FIXED_PHRASES) or
        any_present_inflected(text, MNA_ACTION_STEMS_EN, EN_VERB_SUFFIX) or
        any_present_inflected(text, MNA_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX)
    )
    cheap_object = any_present_morph(text, MNA_OBJECT_NOUNS)
    if not (cheap_object and cheap_action):
        return False, []

    # Sentence tagging + NER
    sents = ner_tag_sentences(text)

    def _hits(s: str):
        action = (any_present(s, MNA_FIXED_PHRASES) +
                  any_present_inflected(s, MNA_ACTION_STEMS_EN, EN_VERB_SUFFIX) +
                  any_present_inflected(s, MNA_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX))
        obj = any_present_morph(s, MNA_OBJECT_NOUNS)
        ctx = []
        if CURRENCY_AMOUNT.search(s): ctx.append("currency_amount")
        ctx += any_present_morph(s, MNA_CONTEXT_TERMS)
        if PER_SHARE_RX.search(s): ctx.append("per_share")
        strong = bool(any_present(s, MNA_FIXED_PHRASES))
        return action, obj, ctx, strong

    # Pass 1: same-sentence requirement incl. ORG count
    for sent in sents:
        s = sent["text"]
        orgs = [e for e in sent["entities"] if e["type"] == "ORG"]
        if not orgs:
            continue
        a, o, c, strong = _hits(s)
        if not (o and (a or strong) and (c or strong)):
            continue

        # require enough ORGs (2 for acquisition/merger combos; 1 ok for tender/spin/sale titles)
        needs_two = any_present_morph(s, ["acquisition", "merger", "business combination", "combination"])
        if len(orgs) < (2 if needs_two else 1):
            continue

        details = list(dict.fromkeys(a + o + c))
        return True, details

    # Pass 2: adjacent-sentence backoff (combine ORGs + cues across s_i and s_{i+1})
    for i in range(len(sents) - 1):
        s1, s2 = sents[i]["text"], sents[i+1]["text"]
        orgs = [e for e in sents[i]["entities"] + sents[i+1]["entities"] if e["type"] == "ORG"]
        if not orgs:
            continue
        a1, o1, c1, strong1 = _hits(s1)
        a2, o2, c2, strong2 = _hits(s2)
        a = a1 + a2; o = o1 + o2; c = c1 + c2
        strong = strong1 or strong2
        if o and (a or strong) and (c or strong) and len(orgs) >= 2:
            details = list(dict.fromkeys(a + o + c))
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

def pdmr_rule(title: str, body: str, keywords: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Strict PDMR gate:
      - require at least one MUST_ANY hit (e.g., "managers' transaction", "PDMR", "insynshandel"), OR
      - allow "article 19" only if MAR is nearby (same sentence / short window) AND there's a PDMR-like detail (ISIN or shares/price).
    Context/any terms do NOT gate; they only add detail when already gated-in.
    """
    text = f"{title or ''} {body or ''}"
    tnorm = norm(text)

    kdef = keywords.get("pdmr_managers_transactions", {}) if isinstance(keywords, dict) else {}
    must_any   = _kw_get_list(kdef, "must_any") or [
        "managers' transaction", "managers’ transaction",
        "pdmr", "person discharging managerial responsibilities",
        "insynshandel", "ledningens transaktioner",
    ]
    any_terms  = _kw_get_list(kdef, "any")
    ctx_terms  = _kw_get_list(kdef, "context")
    exclude    = _kw_get_list(kdef, "exclude")

    # Hard excludes (if you maintain any)
    if exclude and any_present(tnorm, exclude):
        return False, []

    # 1) Strong gate via MUST_ANY
    must_any_hits = any_present(tnorm, must_any)
    if must_any_hits:
        details = list(dict.fromkeys(must_any_hits + any_present(tnorm, any_terms) + any_present(tnorm, ctx_terms)))
        return True, details

    # 2) Backoff: "article 19" near MAR + transaction details
    #    Avoid generic MAR disclaimers.
    article19_rx = re.compile(r"\barticle\s*19\b", re.IGNORECASE)
    mar_rx       = re.compile(r"\bmar\b|\bmarket abuse regulation\b", re.IGNORECASE)

    # quick sentence split; use your existing splitter if you prefer
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    has_article19_near_mar = False
    for s in sentences:
        if article19_rx.search(s) and mar_rx.search(s):
            has_article19_near_mar = True
            break

    if not has_article19_near_mar:
        return False, []

    # PDMR-ish detail: ISIN or "N shares"/price
    isin_rx   = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
    shares_rx = re.compile(r"\b\d[\d\.,]*\s+(?:shares|aktier|osakkeita)\b", re.IGNORECASE)
    price_rx  = re.compile(r"\bprice\s+\d[\d\.,]*\b", re.IGNORECASE)

    has_detail = bool(isin_rx.search(text) or shares_rx.search(text) or price_rx.search(text) or CURRENCY_AMOUNT.search(text))
    if not has_detail:
        return False, []

    # Build details from precise cues only (don’t dump every MAR token)
    details = ["article 19 + MAR", "transaction_detail"]
    details += any_present(tnorm, any_terms) + any_present(tnorm, ctx_terms)
    details = list(dict.fromkeys(details))
    return True, details


def debt_strict_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    debt_terms = rx(["bond issue", "notes", "note issue", "senior unsecured", "convertible", "loan agreement", "tap issue", "prospectus"])
    if debt_terms.search(norm(text)) and CURRENCY_AMOUNT.search(text):
        return True, ["debt+amount"]
    return False, []

def orders_rule(title: str, body: str) -> Tuple[bool, List[str]]:
    text = f"{title} {body}"
    tnorm = norm(text)

    # 1) Fast exclusion: M&A tender offers (unless there are clear contract/order nouns)
    if any_present_morph(tnorm, ORDER_EXCLUSION_PHRASES):
        # If it's truly procurement, there should still be contract/order nouns present.
        if not any_present_morph(tnorm, ["contract", "framework agreement", "ramavtal", "rammeaftale", "rammeavtale", "call-off", "purchase order", "upphandling", "udbud", "anbud"]):
            return False, []

    # 2) Gather signals
    action_hits = (
        any_present(text, ORDER_FIXED_PHRASES_EN)
        + any_present_inflected(text, ORDER_ACTION_STEMS_EN, EN_VERB_SUFFIX)
        + any_present_inflected(text, ORDER_ACTION_STEMS_NORDIC, SV_VERB_SUFFIX)
    )
    object_hits = any_present_morph(text, ORDER_OBJECT_NOUNS)
    context_hits = (
        (["currency_amount"] if CURRENCY_AMOUNT.search(text) else [])
        + any_present_morph(text, ORDER_CONTEXT_TERMS)
    )

    if not action_hits or not object_hits:
        return False, []

    # 3) Proximity between (action) and (object) keeps precision high
    prox_ok, prox_pair = proximity_cooccurs_morph(
        text,
        # use the *phrases we matched on* to be faithful; fallback to base lists otherwise
        set_a=(ORDER_FIXED_PHRASES_EN + ORDER_ACTION_STEMS_EN + ORDER_ACTION_STEMS_NORDIC),
        set_b=ORDER_OBJECT_NOUNS,
        window=15
    )
    if not prox_ok:
        # allow strong fixed phrases that already contain the object (e.g., "purchase order")
        contains_fixed_object = bool(any_present(text, ["purchase order", "call-off order"]))
        if not contains_fixed_object:
            return False, []

    # 4) Require some context: money, units, delivery verbs, or customer/plant/site
    has_context = bool(context_hits)
    if not has_context:
        return False, []

    # 5) Build details (order-preserving, with anchors)
    details = []
    details += list(dict.fromkeys(action_hits))
    details += list(dict.fromkeys(object_hits))
    if prox_ok and prox_pair != ("", ""):
        details += list(prox_pair)
    details += list(dict.fromkeys(context_hits))
    details = list(dict.fromkeys(details))

    return True, details


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

from typing import Any
from collections import defaultdict

def _kw_get_list(kdef: Any, key: str) -> List[str]:
    """Return kdef[key] if dict+list, else []"""
    if isinstance(kdef, dict):
        v = kdef.get(key, [])
        return v if isinstance(v, list) else []
    return []

# use this helper you already wrote (extend it a bit below)
def _kw_terms_flat(kdef: Any) -> List[str]:
    if isinstance(kdef, list):
        return kdef
    terms: List[str] = []
    for k in ("any", "must", "must_any", "context"):  # include must_any (see #3)
        v = kdef.get(k, []) if isinstance(kdef, dict) else []
        if isinstance(v, list):
            terms += v
    return terms



from typing import Any, Dict, List, Tuple, Any
from collections import defaultdict

def keyword_fallback_rule(
    title: str,
    body: str,
    keywords: Dict[str, Any],
    priority_order: List[str],
    exclusion_groups: Dict[str, List[str]],
) -> Tuple[str, Dict]:
    """
    Scoring:
      score = 2.0*|must_all_hits| + 1.5*|must_any_hits| + 1.0*|any_hits| + 0.5*|context_hits|
      - If 'must_all' exists, ALL must appear; if 'must_any' exists, at least ONE must appear.
      - If any 'exclude' term hits, the category is vetoed (score = 0).
      - Subtract 0.6 per competing category (from EXCLUSION_GROUPS) that also scored > 0.
      - Tie-break by higher score, then by PRIORITY order.
    Returns (best_cid, details dict) or ("other_corporate_update", {...}) if nothing scores.
    """
    text = f"{title or ''} {body or ''}"

    # 1) Collect raw hits & base scores
    raw_hits: Dict[str, Dict[str, List[str]]] = {}
    base_score: Dict[str, float] = {}

    for cid, kdef in keywords.items():
        # Support both dict schema and legacy list (treated as "any")
        any_terms   = _kw_get_list(kdef, "any") if isinstance(kdef, dict) else kdef
        must_all    = _kw_get_list(kdef, "must_all")
        must_any    = _kw_get_list(kdef, "must_any")
        ctx_terms   = _kw_get_list(kdef, "context")
        exclude     = _kw_get_list(kdef, "exclude")

        any_hits       = any_present(text, any_terms)     if any_terms else []
        must_all_hits  = any_present(text, must_all)      if must_all else []
        must_any_hits  = any_present(text, must_any)      if must_any else []
        ctx_hits       = any_present(text, ctx_terms)     if ctx_terms else []
        exclude_hits   = any_present(text, exclude)       if exclude else []

        # gating
        failed_all = bool(must_all) and (len(must_all_hits) < len([t for t in must_all if t]))
        failed_any = bool(must_any) and (len(must_any_hits) == 0)

        if failed_all or failed_any or exclude_hits:
            score = 0.0
            any_hits = must_all_hits = must_any_hits = ctx_hits = []
        else:
            score = (
                2.0 * len(must_all_hits)
                + 1.5 * len(must_any_hits)
                + 1.0 * len(any_hits)
                + 0.5 * len(ctx_hits)
            )

            # Optional tiny domain boosts
            if cid in ("debt_bond_issue", "capital_raise_rights_issue") and CURRENCY_AMOUNT.search(text):
                score += 0.5
            if cid == "equity_actions_non_buyback" and any_present(text, ["total number of shares and votes"]):
                score += 0.5

        raw_hits[cid] = {
            "any": any_hits,
            "must_all": must_all_hits,
            "must_any": must_any_hits,
            "context": ctx_hits,
            "exclude": exclude_hits,
        }
        base_score[cid] = score

    # 2) Light competitor penalties
    penalty: Dict[str, float] = defaultdict(float)
    for cid, score in base_score.items():
        if score <= 0:
            continue
        competitors = exclusion_groups.get(cid, [])
        active_competitors = sum(1 for c in competitors if base_score.get(c, 0) > 0)
        penalty[cid] = 0.6 * active_competitors

    final_score = {cid: base_score[cid] - penalty.get(cid, 0.0) for cid in base_score}

    # 3) Pick best: max score, tie-break by PRIORITY
    scored = [(s, cid) for cid, s in final_score.items() if s > 0]
    if not scored:
        return "other_corporate_update", {
            "rule": "keywords+exclusions",
            "hits": {},
            "needs_review_low_sim": True,
            "rule_conf": RULE_CONF_BASE["fallback"],
            "lock": False,
        }

    pr_index = {c: i for i, c in enumerate(priority_order)}
    scored.sort(key=lambda x: (-x[0], pr_index.get(x[1], 10_000)))
    best_score, best_cid = scored[0]

    # Weak-signal heuristic
    total_terms_hit = sum(
        len(v["any"]) + len(v["must_all"]) + len(v["must_any"]) + len(v["context"])
        for v in raw_hits.values()
    )
    needs_review = (best_score < 1.0) and (total_terms_hit <= 2)

    # Confidence
    conf = RULE_CONF_BASE["keywords+exclusions"]
    strong_hits = len(raw_hits[best_cid]["any"]) + len(raw_hits[best_cid]["must_all"]) + len(raw_hits[best_cid]["must_any"])
    competitor_count = int(penalty.get(best_cid, 0) / 0.6)
    conf += 0.08 * min(3, strong_hits)
    conf -= 0.06 * min(3, competitor_count)
    conf = max(0.20, min(0.85, conf))

    # Build details (order-preserving, no duplication)
    hits = raw_hits[best_cid]
    best_hits = list(dict.fromkeys(
        hits["must_all"] + hits["must_any"] + hits["any"] + hits["context"]
    ))

    return best_cid, {
        "rule": "keywords+exclusions",
        "hits": {best_cid: best_hits},
        "needs_review_low_sim": needs_review,
        "rule_conf": conf,
        "lock": False,
    }


# ------------------------
# Orchestration
# ------------------------

# Put high-signal / locked classes early so they win ties in the fallback scorer
PRIORITY = [
    "legal_regulatory_compliance",
    "credit_ratings",
    "pdmr_managers_transactions",   # add: allow keyword fallback if high-signal rule didn't fire
    "mna",                          # add: allow keyword fallback if high-signal rule didn't fire
    "earnings_report",
    "share_buyback",
    "dividend",
    "capital_raise_rights_issue",
    "debt_bond_issue",
    "admission_listing",
    "admission_delisting",
    "equity_actions_non_buyback",
    "agm_egm_governance",
    "orders_contracts",
    "product_launch_partnership",
    "personnel_management_change",  # add: allow keyword fallback if name detection misses
    "labor_workforce",
    "incidents_controversies",
    # fallback: other_corporate_update
]

# Competitor sets derived from your confusion matrix (bidirectional where it matters)
EXCLUSION_GROUPS = {
    # Existing
    "earnings_report": [
        "dividend", "share_buyback", "capital_raise_rights_issue", "credit_ratings",
        "debt_bond_issue",                 # add (biggest new conflict)
        "product_launch_partnership",      # add (moderate conflict)
    ],
    "dividend": ["earnings_report", "share_buyback", "capital_raise_rights_issue"],
    "share_buyback": ["earnings_report", "dividend", "capital_raise_rights_issue", "equity_actions_non_buyback"],  # add
    "credit_ratings": ["earnings_report", "dividend", "share_buyback"],
    "capital_raise_rights_issue": ["equity_actions_non_buyback", "share_buyback", "dividend"],
    "equity_actions_non_buyback": ["capital_raise_rights_issue", "share_buyback"],  # add

    "admission_listing": ["admission_delisting"],
    "admission_delisting": ["admission_listing"],

    # New from matrix
    "debt_bond_issue": ["earnings_report", "legal_regulatory_compliance"],  # add second based on smaller collision
    "product_launch_partnership": ["legal_regulatory_compliance", "earnings_report"],  # add both directions
    "legal_regulatory_compliance": ["credit_ratings", "product_launch_partnership", "debt_bond_issue"],  # expand
    "orders_contracts": ["labor_workforce"],  # add
    "labor_workforce": ["orders_contracts"],  # add

    "mna": ["personnel_management_change"],               # add
    "personnel_management_change": ["mna"],               # add

    # Keep governance separate to avoid dragging personnel into AGM nomination PRs
    "agm_egm_governance": [],
}

def _category_hits(text: str, category_terms: List[str]) -> List[str]:
    return any_present(text, category_terms)

@dataclass
class ClassificationDetails:
    rule: str
    hits: Dict[str, List[str]]
    notes: str = ""
    needs_review_low_sim: bool = False

def classify_press_release(title: str, body: str, keywords: Dict[str, Any]) -> Tuple[str, Dict]:
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
        lrc_hits = any_present(text, _kw_terms_flat(keywords["legal_regulatory_compliance"]))
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

    # 4b) M&A high-signal (sentence-level ORG gating)
    mna_ok, mna_hits = mna_rule_flair(title, body)
    if mna_ok:
        return "mna", {
            "rule": "mna high-signal",
            "hits": {"mna": mna_hits},
            "rule_conf": RULE_CONF_BASE.get("mna high-signal", 0.90),
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
    best_cid, details = keyword_fallback_rule(
        title, body, keywords, PRIORITY, EXCLUSION_GROUPS
    )
    if best_cid != "other_corporate_update" or details["needs_review_low_sim"]:
        return best_cid, details

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

def classify_batch(df, text_cols=("title", "body"), keywords: Dict[str, Any] = None,
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
    out["rule_pred_cid"] = preds
    out["rule_pred_details"] = details
    return out
