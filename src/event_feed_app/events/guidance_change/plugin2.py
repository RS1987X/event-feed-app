from __future__ import annotations
import re, json, hashlib, math
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional, Pattern, List, cast
from datetime import datetime, timezone

from ..base import EventPlugin
from event_feed_app.taxonomy.rules.textutils import (
    any_present_morph,
    proximity_cooccurs_morph,
    proximity_cooccurs_morph_spans
)

from event_feed_app.utils.text import norm, tokenize

_TOKEN_RX = re.compile(r"[a-z0-9åäöæøéèüïóáç\-]+", re.IGNORECASE)
# ---------------- Metric hints & regexes ----------------
FIN_METRICS = {
    # text -> (metric, metric_kind, unit)
    "revenue": ("revenue","level","ccy"),
    "net sales": ("net_sales","level","ccy"),
    "sales": ("net_sales","level","ccy"),
    "ebitda": ("ebitda","level","ccy"),
    "ebit": ("ebit","level","ccy"),
    "ebita": ("ebita","level","ccy"),
    "operating margin": ("operating_margin","margin","pct"),
    #"margin": ("operating_margin","margin","pct"),
    "ebita margin": ("ebita_margin","margin","pct"),
    "organic growth": ("organic_growth","growth","pct"),
    "like-for-like": ("organic_growth","growth","pct"),
    "lfl": ("organic_growth","growth","pct"),
    # sv
    "omsättning": ("revenue","level","ccy"),
    "försäljning": ("net_sales","level","ccy"),
    "rörelsemarginal": ("operating_margin","margin","pct"),
    "organisk tillväxt": ("organic_growth","growth","pct"),
    "ebita marginal": ("ebita_margin","margin","pct"),
}

FIN_METRICS.update({
    # Already have: "omsättning": ("revenue","level","ccy"),
    "omsättningen": ("revenue","level","ccy"),

    # Already have: "försäljning": ("net_sales","level","ccy"),
    "försäljningen": ("net_sales","level","ccy"),

    # EBIT / operating margin Swedish variants
    "rörelseresultat":  ("ebit","level","ccy"),
    "rörelseresultatet":("ebit","level","ccy"),
    "rörelsemarginalen":("operating_margin","margin","pct"),

    # EBITA margin definite
    "ebita-marginalen": ("ebita_margin","margin","pct"),
})

FIN_METRICS.update({
    # Revenue synonyms
    "nettoomsättning":      ("revenue", "level", "ccy"),
    "nettoomsättningen":    ("revenue", "level", "ccy"),
    # If acceptable for your corpus, include these two. If “intäkter” is too broad for you, omit them.
    "intäkter":             ("revenue", "level", "ccy"),
    "intäkterna":           ("revenue", "level", "ccy"),

    # Operating margin variants
    "rörelsemarginal":      ("operating_margin", "margin", "pct"),   # <- missing base form
    "rörelsemarginalen":    ("operating_margin", "margin", "pct"),   # (you already had this)

    # EBIT margin used as operating margin in Swedish PRs
    "ebit-marginal":        ("operating_margin", "margin", "pct"),
    "ebit marginal":        ("operating_margin", "margin", "pct"),
    "ebit-marginalen":      ("operating_margin", "margin", "pct"),

    # EBITDA/EBITA margin (you already had ebita-marginalen)
    "ebitda-marginal":      ("ebitda_margin", "margin", "pct"),
    "ebitda marginal":      ("ebitda_margin", "margin", "pct"),
    "ebitda-marginalen":    ("ebitda_margin", "margin", "pct"),
    "ebita-marginal":       ("ebita_margin",  "margin", "pct"),
    "ebita marginal":       ("ebita_margin",  "margin", "pct"),
    # you already have:
    # "ebita-marginalen":   ("ebita_margin",  "margin", "pct"),
})

APPROX_WORDS = (
    r"(?:~|≈|"                        # symbols
    r"about|around|approximately|approx(?:\.|imately)?|circa|c\.?|"  # EN
    r"omkring|runt|ungefär|cirka|ca\.?|c:a)"                          # SV
)

PERCENT_RANGE = re.compile(r"""
    (?P<approx>
        about|around|approximately|approx\.?|close\sto|near|circa|c\.|~|
        omkring|runt|ungefär|cirka|ca\.?|c\:a|≈
    )?\s*
    (?P<a>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
    (?P<unit>%|percent|procent)\s?
    (?:
        (?:-|–|to)\s?
        (?P<b>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
        (?P=unit)
    )?
""", re.I | re.X)


NUM_GROUP = r"\d{1,3}(?:[ ,.\u00A0\u202F\u2009’']\d{3})*(?:[.,]\d{1,2})?"

# Add Swedish scale tokens to the trailing scale group
_SCALE = r"(?:million|mn|m|billion|bn|b|thousand|k|mkr|mnkr|mdr|mdkr|tkr)"

CCY_NUMBER = re.compile(
    r"(?P<ccy>SEK|EUR|USD|NOK|DKK)\s?"
    rf"(?P<num>{NUM_GROUP})"
    rf"(?:\s*(?P<scale>{_SCALE}))?",
    re.I
)

CCY_RANGE = re.compile(
    r"(?P<ccy>SEK|EUR|USD|NOK|DKK)\s?"
    rf"(?P<a>{NUM_GROUP})\s?"
    r"(?:-|–|to)\s?"
    rf"(?P<b>{NUM_GROUP})"
    rf"(?:\s*(?P<scale>{_SCALE}))?",
    re.I
)

import os
def _dprint(*args, **kwargs):
    if os.environ.get("GUIDANCE_DEBUG") == "1":
        print(*args, **kwargs)

def _apply_scale(value: float, scale: str | None) -> float:
    """Normalize to *millions* of currency."""
    if not scale:
        return value
    s = scale.strip().lower()
    # strip trivial trailing punctuation
    s = s.rstrip('.:;,')
    # millions
    if s in {"m", "mn", "million", "mkr", "mnkr"}:
        return value
    # billions -> millions
    if s in {"bn", "billion", "b", "mdr", "mdkr"}:
        return value * 1000.0
    # thousands -> millions
    if s in {"k", "thousand", "tkr"}:
        return value / 1000.0
    return value


PERIOD = re.compile(
    r"\b((FY|FY-)?(?P<y>20\d{2})|(Q(?P<q>[1-4])[-\s]?(?P<y2>20\d{2}))|(H(?P<h>[12])[-\s]?(?P<y3>20\d{2}))|full[-\s]?year|helår|kvartal\s?Q?[1-4])\b",
    re.I
)
BASIS_ORGANIC = re.compile(r"(organic|like[-\s]?for[-\s]?like|organisk|jämförbar|lfl)", re.I)
BASIS_CCFX   = re.compile(r"(constant currency|cc[-\s]?fx|fast valuta|exkl\.\s*valuta)", re.I)

# ---------------- Helpers ----------------
def _compile_trigger_list(patterns_cfg: Dict[str, Any]) -> List[tuple[str, re.Pattern]]:
    """
    Build a list of (label, compiled_regex) from YAML patterns.
    Labels look like 'en_any[0]', 'sv_any[2]', or 'any[1]'.
    Falls back to a tiny default list if YAML is empty.
    """
    pats: List[tuple[str, re.Pattern]] = []
    for group in ("en_any", "sv_any", "any"):
        v = (patterns_cfg or {}).get(group)
        if isinstance(v, list):
            for i, p in enumerate(v):
                label = f"{group}[{i}]"
                pats.append((label, re.compile(p, re.I)))
    if not pats:
        defaults = [r"guidance", r"profit warning", r"nedskrivning", r"impairment"]
        pats = [(f"default[{i}]", re.compile(p, re.I)) for i, p in enumerate(defaults)]
    return pats


def _is_num(x):
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def _norm_period(text: str) -> str:
    m = PERIOD.search(text)
    if not m:
        return "UNKNOWN"
    g = m.groupdict()
    if g.get("q") and g.get("y2"): return f"Q{g['q']}-{g['y2']}"
    if g.get("h") and g.get("y3"): return f"H{g['h']}-{g['y3']}"
    y = g.get("y")
    if y: return f"FY{y}"
    if re.search(r"full[-\s]?year|helår", text, re.I): return "FY-UNKNOWN"
    return "PERIOD-UNKNOWN"


def _detect_basis(s: str) -> str:
    """Return basis tag or default 'reported' instead of implicit None."""
    if BASIS_ORGANIC.search(s):
        return "organic"
    if BASIS_CCFX.search(s):
        return "cc_fx"
    return "reported"
def _to_float(x: str) -> float:
    s = str(x).strip()
    if not s:
        raise ValueError("empty numeric string")

    # normalize spaces & apostrophes used as thousands separators
    s = (s.replace("\u00A0","").replace("\u202F","").replace("\u2009","")
           .replace(" ", "").replace("’","").replace("'",""))

    # sign
    sign = 1.0
    if s and s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]

    has_dot = "." in s
    has_com = "," in s

    if has_dot and has_com:
        # whichever comes later is the decimal mark; the other is thousands
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_dot > last_com:
            # '.' decimal, ',' thousands
            s = s.replace(",", "")
        else:
            # ',' decimal, '.' thousands
            s = s.replace(".", "").replace(",", ".")
        return sign * float(s)

    if has_com and not has_dot:
        parts = s.split(",")
        # if last group is exactly 3 digits and previous groups are <=3, treat as thousands
        if len(parts[-1]) == 3 and all(1 <= len(p) <= 3 for p in parts[:-1]):
            s = "".join(parts)
        else:
            s = s.replace(",", ".")
        return sign * float(s)

    if has_dot and not has_com:
        parts = s.split(".")
        # same heuristic for '.' thousands
        if len(parts[-1]) == 3 and all(1 <= len(p) <= 3 for p in parts[:-1]):
            s = "".join(parts)
        # else: '.' already decimal
        return sign * float(s)

    # digits only
    return sign * float(s)

def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0

def _event_id(parts: Iterable[str]) -> str:
    return hashlib.blake2s("|".join(str(p) for p in parts).encode("utf-8"), digest_size=16).hexdigest()

def _find_block_range(text: str, rx: re.Pattern) -> Optional[tuple[int,int]]:
    """Find a disclaimer block start; return (start, end) range (end = len(text))."""
    m = rx.search(text)
    if not m:
        return None
    s, _ = m.span()
    return (s, len(text))

def _choose_metric_for_number(ms: int, me: int, sent_text: str,
                              metric_spans: list[tuple[int,int,str]],
                              token_window: int = 8, char_window: int = 120):
    """
    Pick the most plausible metric for a number/range at [ms,me) inside `sent_text`.
    Heuristics:
      1) nearest metric to the LEFT within small window,
      2) else nearest RIGHT if a linking preposition ('in|for|of|i|för|på') is between
         the number and the metric or it's very close in tokens,
      3) else global nearest by char distance.
    Returns (metric_key, (start,end)) or (None, None) if no metric spans.
    """
    if not metric_spans:
        return None, None

    # Partition by side
    left  = [(k, (a,b)) for (a,b,k) in metric_spans if b <= ms]
    right = [(k, (a,b)) for (a,b,k) in metric_spans if a >= me]

    # Helper: token distance in the substring (cheap)
    def _tokdist(sub: str) -> int:
        from event_feed_app.utils.text import tokenize
        return len(tokenize(sub))

    # 1) Prefer nearest on the LEFT within small char/token window
    if left:
        k, (a,b) = min(left, key=lambda x: ms - x[1][1])  # smallest gap on the left
        if (ms - b) <= char_window and _tokdist(sent_text[b:ms]) <= token_window:
            return k, (a,b)

    # 2) RIGHT with preposition link or very close
    if right:
        k, (a,b) = min(right, key=lambda x: x[1][0] - me)
        between = sent_text[me:a]
        if (re.search(r"\b(in|for|of|i|för|på)\b", between, re.I)
            or _tokdist(between) <= token_window
            or (a - me) <= char_window):
            return k, (a,b)

    # 3) Fallback: global nearest by char distance
    k, (a,b) = min(((k,(a,b)) for (a,b,k) in metric_spans),
                   key=lambda x: min(abs(ms - x[1][0]), abs(me - x[1][1])))
    return k, (a,b)

# --- earnings-like label helper (strings, lists, delimited strings) ---
_EARNINGS_LABELS = {"earnings_report","earnings","results","q1","q2","q3","q4"}
_DELIMS = re.compile(r"[,\|;]")

def _has_earnings_label(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (list, tuple, set)):
        toks = [str(t).strip().lower() for t in val]
        return any(t in _EARNINGS_LABELS for t in toks)
    s = str(val).strip().lower()
    if not s:
        return False
    if s in _EARNINGS_LABELS:
        return True
    toks = [t.strip() for t in _DELIMS.split(s) if t.strip()]
    return any(t in _EARNINGS_LABELS for t in toks)

def _rx_match_nothing() -> re.Pattern:
    return re.compile(r"$^")

def _compile_union_group(patterns_cfg: Dict[str, Any], group_name: str,
                         fallback: Optional[List[str]] = None) -> re.Pattern:
    """
    Compile a union regex from patterns.<group_name> with subkeys en_any/sv_any/any.
    """
    grp = (patterns_cfg or {}).get(group_name) or {}
    parts: List[str] = []
    for lang_key in ("en_any", "sv_any", "any"):
        seq = grp.get(lang_key)
        if isinstance(seq, list):
            parts.extend([f"(?:{p})" for p in seq])
    if not parts and fallback:
        parts = [f"(?:{p})" for p in fallback]
    return re.compile("|".join(parts), re.I) if parts else _rx_match_nothing()

#SENT_SPLIT = re.compile(r"(?<=[\.\?\!\:\;])\s+|\n+")
#NEGATION_RX = re.compile(r"\b(no|not|never|deny|denies|denied|false|rumor|speculation|inte|nekar|rykte)\b", re.I)
#COMPANY_PRONOUNS = re.compile(r"\b(we|our|the company|vi|bolaget)\b", re.I)
# Negations around triggers like "denies profit warning", "no plans to update guidance"
NEGATION_RX = re.compile(
    r"\b("
    r"deny|denies|denied|refute|refutes|refuted|"
    r"no\s+(?:plans?|intention)s?\s+to|"
    r"not\s+(?:planning|intending|issuing|updating|withdrawing)|"
    r"without\s+(?:issuing|changing|updating)|"
    r"does\s+not\s+(?:intend|expect|plan)"
    r"|nekar|förnekar|inga\s+planer|inte\s+(?:avsikt|förväntar|planerar)"
    r")\b",
    re.I
)

# Company-binding pronouns/mentions used to avoid macro/analyst false positives
COMPANY_PRONOUNS = re.compile(
    r"\b(we|our|the\s+company|the\s+group|vi|vår(?:a|t)?|bolaget|koncernen)\b",
    re.I
)

# # --- Sentence splitting: don't break on decimals like 1.700 or 2.3 ---
# # Split on: period/question/exclamation/semicolon that are NOT part of a number, or on newlines.
# _SPLIT_RX = re.compile(r"(?:(?<!\d)[.!?;](?!\d))|\n+")

# # --- Sentence splitting: don't break on decimals like 1.700 or 2.3 ---
# # Also don't split after common abbreviations such as "approx.", "ca.", or "c."
# _SPLIT_RX = re.compile(
#     r"(?:(?<!\d)"           # not part of a number
#     r"(?<!\bapprox)"        # don't split after 'approx.'
#     r"(?<!\bca)"            # don't split after 'ca.'
#     r"(?<!\bc)"             # don't split after 'c.'
#     r"[.!?;](?!\d))"        # the actual splitter punctuation
#     r"|\n+",
#     re.I
# )

# # Split on '.', '!' or '?' when the '.' is NOT a decimal dot.
# # Allow optional closing quotes/parens before whitespace or end of string.
# _SPLIT_RX = re.compile(
#     r"(?:\.(?!\d)|[!?])(?=[)\]\"'”’»]*\s|$)"
# )

# Split on '.', '!' or '?' when the '.' is NOT a decimal dot,
# but do NOT split after common abbreviations (approx., ca., c., e.g., i.e.).
_SPLIT_RX = re.compile(
    r"(?:(?<!\bapprox)(?<!\bca)(?<!\bc)(?<!\be\.g)(?<!\bi\.e)\.(?!\d)|[!?])(?=[)\]\"'”’»]*\s|$)",
    re.I,
)

def _sentences_with_spans(text: str) -> list[tuple[int,int,str]]:
    out = []
    s = 0
    for m in _SPLIT_RX.finditer(text):
        e = m.start()
        seg = text[s:e].strip()
        if seg:
            out.append((s, e, seg))
        s = m.end()
    tail = text[s:].strip()
    if tail:
        out.append((s, len(text), tail))
    return out

def _span_in_sentence(span: tuple[int,int], sent_span: tuple[int,int]) -> bool:
    (a, b), (s, e) = span, sent_span
    return a >= s and b <= e

def _norm_period_sentence(s: str) -> str:
    m = PERIOD.search(s)
    if not m:
        return "UNKNOWN"
    g = m.groupdict()
    if g.get("q") and g.get("y2"): return f"Q{g['q']}-{g['y2']}"
    if g.get("h") and g.get("y3"): return f"H{g['h']}-{g['y3']}"
    y = g.get("y")
    if y: return f"FY{y}"
    if re.search(r"full[-\s]?year|helår", s, re.I): return "FY-UNKNOWN"
    return "PERIOD-UNKNOWN"

def _period_from_near_span(text, span, win=160):
    s, e = span
    lo = max(0, s - win); hi = min(len(text), e + win)
    seg = text[lo:hi]
    m = re.search(r'\bFY[-\s]?((?:20|19)\d{2})\b', seg, re.I)
    if m: return f"FY{m.group(1)}"
    m = re.search(r'\b(Q[1-4])[-\s]?((?:20|19)\d{2})\b', seg, re.I)
    if m: return f"{m.group(1).upper()}-{m.group(2)}"
    m = re.search(r'\b(H[12])[-\s]?((?:20|19)\d{2})\b', seg, re.I)
    if m: return f"{m.group(1).upper()}-{m.group(2)}"
    m = re.search(r'\b((?:20|19)\d{2})\b', seg, re.I)
    if m: return f"FY{m.group(1)}"
    return None

def _period_from_doc(doc):
    return doc.get("period_doc") or doc.get("period")

# Compile a union regex from a subgroup dict with en_any/sv_any/any arrays
def _compile_union_from_dict(sub: dict | None, fallback: list[str] | None = None) -> re.Pattern:
    sub = sub or {}
    parts: list[str] = []
    for k in ("en_any","sv_any","any"):
        v = sub.get(k)
        if isinstance(v, list):
            parts.extend([f"(?:{p})" for p in v])
    if not parts and fallback:
        parts = [f"(?:{p})" for p in fallback]
    return re.compile("|".join(parts), re.I) if parts else _rx_match_nothing()



# ---------------- Plugin ----------------
class GuidanceChangePlugin(EventPlugin):
    event_key = "guidance_change"
    taxonomy_class = "earnings_report"

    def __init__(self):
        """Initialize fields and compiled regex holders."""
        self.event_key = "guidance_change"
        # YAML / runtime config
        self.cfg: dict = {}
        self.thresholds: dict = {}

        # Strong trigger regexes (kept regex-based)
        # List of (label, compiled_pattern) for provenance + combined union
        self.trigger_regexes: list[tuple[str, re.Pattern]] = []
        self.trigger_rx: re.Pattern = _rx_match_nothing()

        # Directional / cue / disclaimers / invites
        self.rx_fwd_cue: re.Pattern = _rx_match_nothing()   # kept for back-compat
        self.rx_dir_up:  re.Pattern = _rx_match_nothing()
        self.rx_dir_dn:  re.Pattern = _rx_match_nothing()
        self.rx_invite:  re.Pattern = _rx_match_nothing()
        self.rx_disclaimer_blk: re.Pattern = _rx_match_nothing()
        self.rx_disclaimer_inl: re.Pattern = _rx_match_nothing()

        # Suppressors
        self.rx_sup_analyst: re.Pattern = _rx_match_nothing()
        self.rx_sup_macro:   re.Pattern = _rx_match_nothing()
        self.rx_sup_ma:      re.Pattern = _rx_match_nothing()
        self.rx_sup_debt:    re.Pattern = _rx_match_nothing()

        # Proximity engine (defaults; overridden by YAML thresholds.proximity)
        self.prox_mode: str = "token"      # "token" | "char" | "both" | "either"
        self.prox_chars: int = 120         # back-compat / char gating
        self.prox_tokens: int = 25         # token window (will be tightened by YAML)
        self.allow_cross_sentence: bool = False
        self.look_left_sentences: int = 0
        self.look_right_sentences: int = 0

        # Unchanged markers from thresholds.proximity
        self.unchanged_markers: dict[str, list[str]] = {"en": [], "sv": []}

        # Literal term lists (used by token proximity & direction)
        self.fwd_cue_terms: list[str] = []
        self.dir_up_terms:  list[str] = []
        self.dir_dn_terms:  list[str] = []
        #self.margin_prefixes: list[str] = []
        self.margin_prefixes = ("operating","gross","ebit","ebitda","net","rörelse","brutto")
        # Metrics: literal phrases to scan for (built from METRIC_MAP)
        self.fin_metric_terms: list[str] = []

        # NEW: parsed rule sets from YAML
        self.prox_rules: list[dict] = []     # patterns.proximity_triggers
        self.numeric_rules: list[dict] = []  # patterns.metric_numeric_triggers
        # safe defaults until configure() is called
        self.expected_forward_terms = ()
        self.expected_past_terms = ()
        #self.rx_expected_forward = None
        #self.rx_expected_past = None
        self.rx_expected_forward = _rx_match_nothing()
        self.rx_expected_past = _rx_match_nothing()
        self.rx_expect_family = re.compile(r"\b(expect\w*|förvänt\w*)\b", re.I)

        self.text_stub_rules: list[dict] = []

    # ---------------- Gate ----------------
    def gate(self, doc: Mapping[str, Any]) -> bool:
        """
        Gate intentionally passes all documents through (returns True).
        
        The original gate was too restrictive (required earnings labels OR aggressive 
        trigger phrases), blocking many legitimate guidance announcements. For precision 
        guidance detection, the detect() method's internal logic is sufficient.
        
        See guidance_change.py line 596-597 where the original job also bypasses gate.
        """
        return True  # Pass all documents through; detect() handles precision filtering

    def configure(self, cfg: Optional[dict]) -> None:
        """
        Ingest YAML config, compile regexes, and normalize proximity/numeric rules.
        Reads everything from the event node matching self.event_key ("guidance_change").
        """
        import re
        
        # -------- Root + event node --------
        self.cfg = cfg or {}
        
        # Accept both root config and single-event block
        if self.cfg.get("key") == self.event_key:
            ev = self.cfg
        else:
            events = {e.get("key"): e for e in (self.cfg.get("events") or [])}
            ev = events.get(self.event_key, {})
            raw = self.cfg.get("events") or {}
            if isinstance(raw, dict):
                ev = raw.get(self.event_key, ev)
            else:
                ev = next((e for e in (raw or []) if (e or {}).get("key") == self.event_key), ev)

        self.thresholds  = ev.get("thresholds", {}) or {}
        patterns_cfg     = ev.get("patterns",   {}) or {}
        terms_cfg        = ev.get("terms",      {}) or {}
        tokens_cfg       = ev.get("tokens",     {}) or {}

        # # Blocks under the event node
        # self.thresholds  = ev.get("thresholds", {}) or {}
        # patterns_cfg     = ev.get("patterns",   {}) or {}
        # terms_cfg        = ev.get("terms",      {}) or {}
        # tokens_cfg       = ev.get("tokens",     {}) or {}

        # -------- Strong triggers (regex) --------
        self.trigger_regexes = _compile_trigger_list(patterns_cfg.get("triggers") or patterns_cfg)
        self.trigger_rx = (
            re.compile("|".join(f"(?:{rx.pattern})" for _, rx in self.trigger_regexes), re.I)
            if self.trigger_regexes else _rx_match_nothing()
        )

        # -------- Proximity tuning --------
        prox = (self.thresholds.get("proximity") or {})
        self.prox_mode   = str(prox.get("mode", "token")).lower()
        self.prox_chars  = int(prox.get("fwd_cue_metric_chars",
                            self.thresholds.get("fwd_cue_metric_window_chars", 120)))
        self.prox_tokens = int(prox.get("fwd_cue_metric_tokens", 25))
        self.allow_cross_sentence = bool(prox.get("allow_cross_sentence", False))
        self.look_left_sentences  = int(prox.get("look_left_sentences", 0))
        self.look_right_sentences = int(prox.get("look_right_sentences", 0))

        # Unchanged markers (prefer thresholds-level; fallback to proximity subkey if present)
        self.unchanged_markers["en"] = list(
            self.thresholds.get("unchanged_markers_en")
            or prox.get("unchanged_markers_en")
            or []
        )
        self.unchanged_markers["sv"] = list(
            self.thresholds.get("unchanged_markers_sv")
            or prox.get("unchanged_markers_sv")
            or []
        )

        # -------- Forward-looking cues (literal) --------
        fwd_cfg = (patterns_cfg.get("fwd_cue") or {})
        fwd_en  = list((fwd_cfg.get("terms") or {}).get("en") or [])
        fwd_sv  = list((fwd_cfg.get("terms") or {}).get("sv") or [])
        tune_terms = list((terms_cfg or {}).get("fwd_cue_terms") or [])
        self.fwd_cue_terms = [str(x) for x in (fwd_en + fwd_sv + tune_terms)]

        # Compile a single union regex for “forward cues” (used by past-tense suppression guards)
        def _wb(term: str) -> str:
            return r'(?<!\w)' + re.escape(term).replace(r'\ ', r'\s+') + r'(?!\w)'

        self.rx_fwd_cues = (
            re.compile("|".join(_wb(t) for t in self.fwd_cue_terms), re.I)
            if self.fwd_cue_terms else _rx_match_nothing()
        )
        # Back-compat alias if other code references rx_fwd_cue
        self.rx_fwd_cue = self.rx_fwd_cues

        # -------- Directional / invite / disclaimers (regex unions) --------
        self.rx_dir_up         = _compile_union_group(patterns_cfg, "dir_up")
        self.rx_dir_dn         = _compile_union_group(patterns_cfg, "dir_dn")
        self.rx_invite         = _compile_union_group(patterns_cfg, "invite")
        self.rx_disclaimer_blk = _compile_union_group(patterns_cfg, "disclaimer_block")
        self.rx_disclaimer_inl = _compile_union_group(patterns_cfg, "disclaimer_inline")

        # Suppressors
        sups = patterns_cfg.get("suppressors") or {}
        self.rx_sup_analyst = _compile_union_from_dict(sups.get("analyst"))
        self.rx_sup_macro   = _compile_union_from_dict(sups.get("macro"))
        self.rx_sup_ma      = _compile_union_from_dict(sups.get("ma"))
        self.rx_sup_debt    = _compile_union_from_dict(sups.get("debt"))

        # -------- Term lists (with sensible defaults) --------
        def _get_terms(key: str, fallback: list[str]) -> list[str]:
            vals = terms_cfg.get(key)
            return [str(x) for x in vals] if isinstance(vals, list) and vals else fallback

        self.margin_prefixes = _get_terms(
            "margin_allowed_prefixes",
            ["operating", "op.", "EBIT", "EBITDA", "EBITA", "rörelse", "ebit", "ebitda", "ebita"],
        )
        self.dir_up_terms = _get_terms(
            "dir_up_terms",
            ["improve","improving","increase","increasing","expand","grow","strengthen","better",
            "förbättra","förbättras","öka","ökande","stärka","bättre"],
        )
        self.dir_dn_terms = _get_terms(
            "dir_dn_terms",
            ["decline","declining","decrease","decreasing","contract","pressure","headwinds","worse",
            "försämra","försämras","minska","minskande","press","motvind","sämre"],
        )

        # -------- Metrics (literal phrases) --------
        self.fin_metric_terms = sorted(set(FIN_METRICS.keys()) | {"like for like"})

        # # -------- Proximity rules (verbs/metrics + unchanged tokens) --------
        # def _resolve_tokens(val):
        #     if isinstance(val, str) and val.startswith("@"):
        #         key = val[1:]
        #         return list(self.thresholds.get(key) or prox.get(key) or [])
        #     return list(val or [])
        
        # --- widen @token resolution to terms/patterns too ---
        def _resolve_tokens(val):
            if isinstance(val, str) and val.startswith("@"):
                key = val[1:]
                return list(self.thresholds.get(key)
                            or (self.thresholds.get("proximity") or {}).get(key)
                            or terms_cfg.get(key, [])
                            or patterns_cfg.get(key, [])
                            or [])
            return list(val or [])
        
        self.prox_rules = []
        prox_src = (patterns_cfg.get("guidance_proximity_rules")
         or patterns_cfg.get("proximity_triggers")
         or patterns_cfg.get("proximity_rules")
         or [])
        for rule in prox_src:
            r = dict(rule)
            r["name"]  = r.get("name") or "unnamed_prox_rule"
            r["type"]  = r.get("type") or "guidance_change"
            r["lang"]  = r.get("lang") or "en"
            r["verbs"] = _resolve_tokens(r.get("verbs"))

            gm = r.get("guidance_markers")
            if gm is None:
                gm = r.get("metrics")  # legacy key
            r["guidance_markers"] = _resolve_tokens(gm)

            r["unchanged_tokens"]    = _resolve_tokens(r.get("unchanged_tokens"))
            r["window_tokens"]       = int(r.get("window_tokens") or self.prox_tokens)
            r["same_sentence"]       = bool(r.get("same_sentence", True))
            r["require_period_token"] = bool(r.get("require_period_token", False))
            self.prox_rules.append(r)

        # -------- Numeric (statement-only) rules --------
        self.numeric_rules = []
        num_src = (patterns_cfg.get("numeric_guidance_rules")
        or patterns_cfg.get("metric_numeric_triggers")
        or patterns_cfg.get("numeric_triggers")
        or [])
        for rule in num_src:
            r = dict(rule)
            r["name"] = r.get("name") or "unnamed_numeric_rule"
            r["type"] = r.get("type") or "guidance_stated"
            r["lang"] = r.get("lang") or "en"

            gm = r.get("guidance_markers")
            if gm is None:
                gm = r.get("metrics")  # legacy key
            r["guidance_markers"] = _resolve_tokens(gm)

            r["window_tokens"]  = int(r.get("window_tokens") or self.prox_tokens)
            r["same_sentence"]  = bool(r.get("same_sentence", True))
            r["require_number"] = bool(r.get("require_number", True))
            self.numeric_rules.append(r)

        # -------- “expected” tokens (forward vs. past) --------
        self.expected_forward_terms = tuple(tokens_cfg.get("expected_forward_en", []))
        self.expected_past_terms    = tuple(tokens_cfg.get("expected_past_en", []))
        def _compile_expected_union(terms):
            terms = [t for t in (terms or []) if t]
            if not terms:
                return _rx_match_nothing()
            parts = [re.escape(t).replace(r"\ ", r"\s+") for t in terms]
            return re.compile("(?:%s)" % "|".join(parts), re.I)

        self.rx_expected_forward = _compile_expected_union(self.expected_forward_terms)
        self.rx_expected_past    = _compile_expected_union(self.expected_past_terms)
        
        self.rx_expect_family       = re.compile(r"\b(expect\w*|förvänt\w*)\b", re.I)

        # --- compile text-stub rules from YAML ---
        self.text_stub_rules = []
        stub_src = patterns_cfg.get("guidance_text_stub_rules") or []

        def _wb_term(t: str) -> str:
            # word-boundary-ish, tolerate spaces vs hyphens
            return r'(?<!\w)' + re.escape(t).replace(r'\ ', r'[\s-]+') + r'(?!\w)'

        for rule in stub_src:
            r = dict(rule or {})
            markers = r.get("markers") or []
            rx = re.compile("|".join(_wb_term(m) for m in markers), re.I) if markers else _rx_match_nothing()
            r["rx"] = rx
            r["same_sentence"] = bool(r.get("same_sentence", True))
            r["require_period_token"] = bool(r.get("require_period_token", True))
            self.text_stub_rules.append(r)


    # in the plugin class
    def detect(self, doc):
        # use v3 by default
        yield from self._detect_v3(doc)
        
        # if getattr(self, "_use_v2", False):
        #     yield from self._detect_v2(doc)
        # else:
        #     yield from self._detect_v1(doc)
# '
#     # ------ 1) Gate: cheap prefilter ------
#     def gate(self, doc: Mapping[str, Any]) -> bool:
#         # a) fast path via classifier labels
#         if _has_earnings_label(doc.get("l1_class")):      return True
#         if _has_earnings_label(doc.get("rule_pred_cid")): return True
#         if _has_earnings_label(doc.get("rule_label")):    return True

#         # b) YAML-driven strong triggers (title + lead)
#         head = f"{doc.get('title','')} {doc.get('body','')[:800]}"
#         return bool(self.trigger_rx.search(head))
  
    # ------ 2) Detect numeric/text candidates ------
    def _detect_v1(self, doc: Mapping[str, Any]) -> Iterable[dict]:
        """
        Extract earnings-guidance candidates from a document (title/body). A document
        passes the gate if it has either (A) a strong trigger or (B) a forward-looking
        cue that is proximate to a metric mention. On success, numeric candidates
        (percent/currency) are yielded first; a tight text-only fallback can be emitted
        if no numeric candidate is found.

        Parameters
        ----------
        doc : Mapping[str, Any]
            Expected keys:
            - "title": str
            - "body": str
            - "source_type": Optional[str] in {"issuer_pr","exchange","regulator",...}

        Yields
        ------
        dict
            Candidate objects with:
            - metric : str                  # e.g. "revenue", "operating_margin", "guidance"
            - metric_kind : str             # "level" | "margin" | "growth" | "text"
            - unit : str                    # "ccy" | "pct" | "text"
            - currency : Optional[str]
            - value_type : str              # "point" | "range" | "na"
            - value_low, value_high : float | None
            - basis : str                   # e.g. "reported"
            - period : str                  # normalized period
            - direction_hint : Optional[str]# "up" | "down" | None
            - _trigger_source  : "strong" | "fwd_cue"
            - _trigger_label   : Optional[str]
            - _trigger_pattern : Optional[str]
            - _trigger_match   : Optional[str]
            - _trigger_span    : Optional[Tuple[int,int]]

        Pipeline
        --------
        1) Build text
        text = title + "\\n" + body; head = title + first ~400 chars of body.

        2) Strong trigger (regex-based)
        trig = self._find_trigger(text). If found, suppress when a negation occurs
        within ±60 chars of the trigger span. A surviving trigger alone passes the gate.

        3) Cue–metric proximity (when no strong trigger)
        - Token proximity:
            (ok_tok, prox) = proximity_cooccurs_morph_spans(
                text, self.fwd_cue_terms, self.metric_terms,
                window=self.prox_tokens, min_stem=3
            )
            Here, cues and metrics are **literal phrases** (not regex). Matching is
            token-by-token with simple morphological tolerance (prefix match for tokens
            of length ≥ min_stem); selected short/polysemous tokens are forced to match
            exactly (EXACT_ONLY).
        - Char proximity (center-to-center):
            If `prox` exists, compute `char_ok` by comparing the centers of the cue
            span and the metric span and require distance ≤ self.prox_chars.
        - Char-only fallback:
            If `prox` is None and `prox_mode ∈ {"char","either"}` (optionally "both"),
            rerun the proximity helper with an effectively unlimited token window to
            find the closest cue–metric pair and re-check the char threshold. If it
            passes (and there is no strong trigger), adopt this pair as cue provenance.
        - Gate decision (when no strong trigger):
            * "token" → require ok_tok
            * "char"  → require char_ok
            * "both"  → require ok_tok AND char_ok
            * otherwise (e.g. "either") → ok_tok OR char_ok
        - Provenance:
            Build `cue` from the `prox` object that passed gating (term + exact span).

        Sentence constraints & suppressors (applied to the *cue sentence*):
            * Same-sentence with the metric unless `allow_cross_sentence` is True;
            if allowed, enforce the configured sentence window (`look_left/right`).
            * Suppress invites/disclaimers (inline and trailing block).
            * Suppress macro-only outlook for non-issuer sources unless a company pronoun appears.
            * Suppress third-party analyst voice for non-issuer sources.
            * Add friction for bare "will/kommer att" unless accompanied by guidance
            keywords or concrete numbers/percentages.

        4) Document context
        period = _norm_period(text)
        basis  = _detect_basis(text) or "reported"

        5) Metric hints & direction
        - Scan METRIC_MAP keys using `\\b...\\b` and treat spaces/hyphens as equivalent.
        - Direction hint: prefer `any_present_morph(text, dir_up_terms/dir_dn_terms)`
            (morph lists) and fall back to `rx_dir_up`/`rx_dir_dn` unions.

        6) Extraction passes
        a) Percent values (growth/margins)
            - For each PERCENT_RANGE: find the sentence; skip M&A/debt/analyst sentences;
                require a metric hint in the **same sentence** with a guard for generic
                "margin"; derive a sentence-level period (with a limited fallback for
                withdraw/suspend headlines); compute point/range (±band for approx);
                yield for metrics with unit == "pct".
        b) Currency values (levels)
            - For each CCY_NUMBER: find the sentence; apply suppressors; require a **level**
                metric in the same sentence and a concrete sentence period; parse currency
                and number; yield for metrics with unit == "ccy".
        c) Text-only fallback
            - If nothing yielded: require any metric hint anywhere, a concrete document
                period, a non-null direction_hint, and no invite in `head`. For non-issuer
                sources, also require a sentence containing
                (guidance|outlook|forecast|prognos|utsikter) AND a company pronoun. Yield a
                guidance stub (unit == "text").

        Notes & assumptions
        -------------------
        - self.fwd_cue_terms and self.metric_terms are **literal phrases**.
        - Tokenization includes hyphens; regex scans normalize space/hyphen via `[\\s-]+`.
        - Token proximity distance uses start token indices; char proximity uses **center-to-center**
        span distance.
        - Provenance prefers strong triggers; otherwise it uses the **exact** cue occurrence
        that passed proximity.

        Complexity
        ----------
        - Token proximity ~ O(N * M) where N = #text tokens and M = total pattern tokens
        across cue/metric lists; numeric extraction adds linear regex scans.
        """

        text = f"{doc.get('title','')}\n{doc.get('body','')}"
        head = f"{doc.get('title','')} {doc.get('body','')[:400]}"

        # ---------------- Strong trigger (regex-based) ----------------
        trig = self._find_trigger(text)
        if trig:
            ss, se = trig["span"]
            ctx_s = max(0, ss - 60); ctx_e = min(len(text), se + 60)
            if NEGATION_RX.search(text[ctx_s:ctx_e]):
                trig = None  # e.g., "no impairment" near trigger

        # ---------------- Tokenization & sentences (once) ----------------
        toks_spans = [(m.group(0), m.span()) for m in _TOKEN_RX.finditer(norm(text))]
        toks = [t for t,_ in toks_spans]
        sents = list(_sentences_with_spans(text))

        def _sent_idx(span):
            for k,(ss,se,_) in enumerate(sents):
                if _span_in_sentence(span,(ss,se)): return k
            return None

        PERIOD_TOKENS = {"fy","full-year","h1","h2","q1","q2","q3","q4"}

        def _has_number_near(toks, i, window=8):
            lo, hi = max(0, i-window), min(len(toks), i+window+1)
            seg = " ".join(toks[lo:hi])
            return bool(PERCENT_RANGE.search(seg) or CCY_NUMBER.search(seg))

        def _has_period_near(toks, i, window=8):
            lo, hi = max(0, i-window), min(len(toks), i+window+1)
            for t in toks[lo:hi]:
                if t in PERIOD_TOKENS: return True
                if len(t) == 4 and t.isdigit() and t.startswith(("20","19")): return True
            return False

        # ---------------- Proximity rules from YAML ----------------
        prox_hit: Optional[Dict[str, Any]] = None  # {"rule": rule_dict, "meta": proximity_meta}
        best_d = 10**9

        for r in self.prox_rules:
            ok, meta = (False, None)

            # Case A: verbs + metrics (e.g., raise/lower/reaffirm near guidance)
            if r.get("verbs"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["verbs"], set_b=r["guidance_markers"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            # Case B: metric + unchanged markers (verb-less: "guidance unchanged")
            elif r.get("unchanged_tokens"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["guidance_markers"], set_b=r["unchanged_tokens"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            if ok:
                # Optional: enforce period token for "initiate" rules
                if r.get("require_period_token"):
                    if not meta:
                        continue
                    m0, _ = meta["metric_span"]
                    j = next((i for i, (_,sp) in enumerate(toks_spans) if sp[0] <= m0 < sp[1]), None)
                    if j is None or not _has_period_near(toks, j, window=r["window_tokens"]):
                        continue

                # Keep closest by token distance
                if meta and meta.get("token_dist", 10**9) < best_d:
                    prox_hit = {"rule": r, "meta": meta}
                    best_d = int(meta.get("token_dist", best_d))

        # Optional char proximity if you still support char/either/both
        char_ok: bool = False
        if prox_hit and prox_hit.get("meta"):
            cs, ce = prox_hit["meta"]["cue_span"]; ms, me = prox_hit["meta"]["metric_span"]
            cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
            char_ok = abs(cue_c - met_c) <= self.prox_chars

        # ---------------- Numeric gate for statement-only guidance ----------------
        numeric_gate_ok = False
        if not trig and not prox_hit and self.numeric_rules:
            for r in self.numeric_rules:
                # Find any metric token and require a number near it
                for i, (tok, sp) in enumerate(toks_spans):
                    if any(tok.startswith(m) for m in r["metrics"]):
                        if (not r["require_number"]) or _has_number_near(toks, i, window=r["window_tokens"]):
                            numeric_gate_ok = True
                            break
                if numeric_gate_ok:
                    break
    # ---------------- Legacy fallback: fwd-cue ↔ metric proximity ----------------
        # Covers cases like: "expects revenue to be EUR ..." (no 'guidance' token)
        if not trig and not prox_hit and not numeric_gate_ok and self.fwd_cue_terms and self.fin_metric_terms:
            ok_legacy, meta_legacy = proximity_cooccurs_morph_spans(
                text, self.fwd_cue_terms, self.fin_metric_terms,
                window=self.prox_tokens, min_stem=3
            )
            if ok_legacy and meta_legacy:
                prox_hit = {
                    "rule": {"name": "fwd_cue_near_metric", "type": "guidance_change", "lang": "any"},
                    "meta": meta_legacy,
                }
                # keep your char gating consistent
                cs, ce = meta_legacy["cue_span"]; ms, me = meta_legacy["metric_span"]
                cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
                char_ok = abs(cue_c - met_c) <= self.prox_chars
        
        # ---------------- Gate decision ----------------
        if not trig:
            mode = self.prox_mode
            prox_ok = (prox_hit is not None) if mode=="token" else \
                    (char_ok)              if mode=="char"  else \
                    (prox_hit and char_ok) if mode=="both"  else \
                    (prox_hit or char_ok)
            if not (prox_ok or numeric_gate_ok):
                return

        # ---------------- Provenance + sentence suppressors ----------------
        cue = None
        cue_sent = None

        if prox_hit:
            # Provenance from the *actual* proximity cue
            cue = {
                "label":   prox_hit["rule"]["name"],
                "pattern": prox_hit["meta"]["cue_term"],
                "match":   text[prox_hit["meta"]["cue_span"][0]:prox_hit["meta"]["cue_span"][1]],
                "span":    prox_hit["meta"]["cue_span"],
            }
            cue_i = _sent_idx(prox_hit["meta"]["cue_span"])
            met_i = _sent_idx(prox_hit["meta"]["metric_span"])
            if cue_i is None or met_i is None:
                return
            if not self.allow_cross_sentence:
                if cue_i != met_i:
                    return
                cue_sent = sents[cue_i]
            else:
                if not (-self.look_left_sentences <= (met_i-cue_i) <= self.look_right_sentences):
                    return
                cue_sent = sents[cue_i]

            ss, se, sent = cue_sent
            # Inline suppressors
            if self.rx_invite.search(sent) or self.rx_disclaimer_inl.search(sent):
                return
            blk = _find_block_range(text, self.rx_disclaimer_blk)
            if blk and (ss >= blk[0]):
                return
            # Macro/analyst suppressors for non-issuer sources
            st = doc.get("source_type")
            if self.rx_sup_macro.search(sent) and st not in {"issuer_pr","exchange","regulator"}:
                if not COMPANY_PRONOUNS.search(sent):
                    return
            if self.rx_sup_analyst.search(sent) and st not in {"issuer_pr","exchange","regulator"}:
                return
            # Bare "will/kommer att" friction unless numbers/keywords nearby
            if re.search(r"\b(will|kommer\s+att)\b", sent, re.I):
                if not (re.search(r"\b(guidance|outlook|forecast|target|aim|mål)\b", sent, re.I)
                        or PERCENT_RANGE.search(sent) or CCY_NUMBER.search(sent)):
                    return

        # ---------------- Document context ----------------
        period = _norm_period(text)
        basis = _detect_basis(text) or "reported"

        # ---------------- Metric hints & direction ----------------
        hits: list[Tuple[str, Tuple[str,str,str]]] = []
        for k, v in FIN_METRICS.items():
            k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
            if k_rx.search(text):
                hits.append((k, v))

        # Direction (prefer morph term lists, fallback to regex unions)
        direction_hint = None
        if any_present_morph(text, self.dir_up_terms, min_stem=3):
            direction_hint = "up"
        elif any_present_morph(text, self.dir_dn_terms, min_stem=3):
            direction_hint = "down"
        else:
            direction_hint = "up" if self.rx_dir_up.search(text) else ("down" if self.rx_dir_dn.search(text) else None)

        # If the proximity rule explicitly indicates unchanged, normalize to "flat"
        if prox_hit and prox_hit["rule"]["type"] in {"guidance_unchanged", "guidance_reaffirm"}:
            direction_hint = "flat"

        # Keep handy for provenance on yields
        prov = trig or cue
        trigger_source = "strong" if trig else ("fwd_cue" if prox_hit else ("numeric_gate" if numeric_gate_ok else "none"))
        yielded = False

        # ---------------- Extraction passes (unchanged from your logic) ----------------
        # 1) Percent (growth/margin)
        for m in PERCENT_RANGE.finditer(text):
            ms, me = m.span()
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                continue
            ss, se, s = sent

            # Suppressors in-sentence
            st = doc.get("source_type")
            if self.rx_sup_ma.search(s) or self.rx_sup_debt.search(s) or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"}):
                continue

            # Require a metric hint in the SAME sentence (guard generic 'margin')
            local_hits = []
            for k, v in FIN_METRICS.items():
                k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
                if k_rx.search(s):
                    if k.strip().lower() == "margin":
                        # ensure a valid prefix (operating/EBIT/EBITDA/...)
                        mpos = s.lower().find('margin')
                        left = s[max(0, mpos-40):mpos]
                        if not any(re.search(rf"\b{re.escape(p)}\b", left, re.I) for p in self.margin_prefixes):
                            continue
                    local_hits.append((k, v))
            if not local_hits:
                continue

            # Require period unless withdraw/suspend in headline
            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                head_lower = head.lower()
                is_withdraw_suspend = bool(re.search(r"(withdraw|withdraws|withdrew|withdrawn|suspend|suspends|suspended)", head_lower))
                if is_withdraw_suspend:
                    pass  # limited headline fallback already allowed
                elif period != "UNKNOWN" and prox_hit:
                    period_local = period  # NEW: doc-level fallback when we have a validated fwd-cue/metric match
                else:
                    continue

            a = _to_float(m.group("a"))
            b = m.group("b")
            approx = bool((m.groupdict() or {}).get("approx"))
            low, high = (min(a, _to_float(b)), max(a, _to_float(b))) if b else (a, a)
            for _, (metric, kind, unit) in local_hits:
                if unit != "pct":
                    continue
                if b is None and approx:
                    band = float(self.thresholds.get("approx_band_pp_margin" if kind=="margin" else "approx_band_pp_growth",
                                                    0.5 if kind=="margin" else 1.0))
                    low, high = a - band, a + band
                yield {
                    "metric": metric, "metric_kind": kind, "unit": "pct", "currency": None,
                    "value_type": "range" if (b or approx) else "point",
                    "value_low": low, "value_high": high,
                    "basis": basis, "period": period_local,
                    "direction_hint": direction_hint,
                    "_trigger_source": trigger_source,
                    "_trigger_label":   prov["label"]   if prov else None,
                    "_trigger_pattern": prov["pattern"] if prov else None,
                    "_trigger_match":   prov["match"]   if prov else None,
                    "_trigger_span":    prov["span"]    if prov else None,
                    "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                    "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
                }
                yielded = True

        # Track spans of ranges to avoid double-counting in CCY_NUMBER
        range_spans: list[tuple[int,int]] = []

        # 2a) currency RANGES (levels)
        for m in CCY_RANGE.finditer(text):
            ms, me = m.span()
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                continue
            ss, se, s = sent

            st = doc.get("source_type")
            if self.rx_sup_ma.search(s) or self.rx_sup_debt.search(s) or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"}):
                continue

            # level metric in SAME sentence
            local_hits = []
            for k, v in FIN_METRICS.items():
                if v[2] != "ccy":
                    continue
                if re.search(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", s, re.I):
                    local_hits.append((k, v))
            if not local_hits:
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period != "UNKNOWN" and prox_hit:
                    period_local = period  # doc-level fallback with validated cue
                else:
                    continue

            # CCY_RANGE block
            ccy   = m.group("ccy").upper()
            scale = (m.groupdict() or {}).get("scale")
            a = _apply_scale(_to_float(m.group("a")), scale)
            b = _apply_scale(_to_float(m.group("b")), scale)
            low, high = (min(a, b), max(a, b))

            for _, (metric, kind, unit) in local_hits:
                yield {
                    "metric": metric, "metric_kind": kind, "unit": "ccy", "currency": ccy,
                    "value_type": "range", "value_low": low, "value_high": high,
                    "basis": basis, "period": period_local,
                    "direction_hint": direction_hint,
                    "_trigger_source": trigger_source,
                    "_trigger_label":   prov["label"]   if prov else None,
                    "_trigger_pattern": prov["pattern"] if prov else None,
                    "_trigger_match":   prov["match"]   if prov else None,
                    "_trigger_span":    prov["span"]    if prov else None,
                    "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                    "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
                }
                yielded = True

            range_spans.append((ms, me))

        # 2) Currency (levels)
        def _in_any_range_span(span):
            s, e = span
            for rs, re_ in range_spans:
                if s >= rs and e <= re_:
                    return True
            return False
        
        for m in CCY_NUMBER.finditer(text):
            ms, me = m.span()
            if _in_any_range_span((ms, me)):
                continue  # already covered by a range

            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                continue
            ss, se, s = sent

            st = doc.get("source_type")
            if self.rx_sup_ma.search(s) or self.rx_sup_debt.search(s) or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"}):
                continue

            # level metric in SAME sentence
            local_hits = []
            for k, v in FIN_METRICS.items():
                if v[2] != "ccy":
                    continue
                if re.search(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", s, re.I):
                    local_hits.append((k, v))
            if not local_hits:
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period != "UNKNOWN" and prox_hit:
                    period_local = period  # doc-level fallback
                else:
                    continue

            ccy = m.group("ccy").upper()
            scale = m.groupdict().get("scale")
            val = _apply_scale(_to_float(m.group("num")), scale)

            for _, (metric, kind, unit) in local_hits:
                yield {
                    "metric": metric, "metric_kind": kind, "unit": "ccy", "currency": ccy,
                    "value_type": "point", "value_low": val, "value_high": val,
                    "basis": basis, "period": period_local,
                    "direction_hint": direction_hint,
                    "_trigger_source": trigger_source,
                    "_trigger_label":   prov["label"]   if prov else None,
                    "_trigger_pattern": prov["pattern"] if prov else None,
                    "_trigger_match":   prov["match"]   if prov else None,
                    "_trigger_span":    prov["span"]    if prov else None,
                    "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                    "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
                }
                yielded = True

        # ---------------- Text-only stubs ----------------
        # A) Unchanged or Initiate (explicit via proximity rules) with known period
        if (not yielded) and prox_hit and period != "UNKNOWN":
            if prox_hit["rule"]["type"] in {"guidance_unchanged", "guidance_reaffirm", "guidance_initiate"}:
                # For initiate, you may later route via thresholds.initiate_new_period == always
                yield {
                    "metric": "guidance",
                    "metric_kind": "text",
                    "unit": "text",
                    "currency": None,
                    "value_type": "na",
                    "value_low": None,
                    "value_high": None,
                    "basis": basis,
                    "period": period,
                    "direction_hint": ("flat" if prox_hit["rule"]["type"] in {"guidance_unchanged","guidance_reaffirm"} else None),
                    "_trigger_source":  "fwd_cue",
                    "_trigger_label":   cue["label"]   if cue else None,
                    "_trigger_pattern": cue["pattern"] if cue else None,
                    "_trigger_match":   cue["match"]   if cue else None,
                    "_trigger_span":    cue["span"]    if cue else None,
                    "_rule_type":       prox_hit["rule"]["type"],
                    "_rule_name":       prox_hit["rule"]["name"],
                }
                return  # stub emitted; stop here

        # B) Original tight fallback: metric + period + direction + not invite
        if (not yielded
            and hits
            and period != "UNKNOWN"
            and direction_hint is not None
            and not self.rx_invite.search(head)):
            # for wires/news: need a sentence with (guidance|outlook|forecast|prognos|utsikter) AND a company pronoun
            if doc.get("source_type") not in {"issuer_pr","exchange","regulator"}:
                ok_textonly = False
                for _, _, s in sents:
                    if re.search(r"\b(guidance|outlook|forecast|prognos|utsikter)\b", s, re.I) and COMPANY_PRONOUNS.search(s):
                        ok_textonly = True
                        break
                if not ok_textonly:
                    return
            yield {
                "metric": "guidance",
                "metric_kind": "text",
                "unit": "text",
                "currency": None,
                "value_type": "na",
                "value_low": None,
                "value_high": None,
                "basis": basis,
                "period": period,
                "direction_hint": direction_hint,
                "_trigger_source":  trigger_source,
                "_trigger_label":   ((trig or cue) or {}).get("label"),
                "_trigger_pattern": ((trig or cue) or {}).get("pattern"),
                "_trigger_match":   ((trig or cue) or {}).get("match"),
                "_trigger_span":    ((trig or cue) or {}).get("span"),
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }


        
    # --- v2 wrapper: same gate/provenance as v1, modular extraction ---
    def _detect_v2(self, doc):
        """
        Behavior-preserving refactor of _detect_v1:
        - Keeps the exact gate/provenance logic from v1.
        - Delegates numeric extraction to small helpers.
        """
        import re

        text = f"{doc.get('title','')}\n{doc.get('body','')}"
        head = f"{doc.get('title','')} {doc.get('body','')[:400]}"

        # ---------------- Strong trigger (regex-based) ----------------
        trig = self._find_trigger(text)
        if trig:
            ss, se = trig["span"]
            ctx_s = max(0, ss - 60); ctx_e = min(len(text), se + 60)
            if NEGATION_RX.search(text[ctx_s:ctx_e]):
                trig = None  # e.g., "no impairment" near trigger

        # ---------------- Tokenization & sentences (once) ----------------
        toks_spans = [(m.group(0), m.span()) for m in _TOKEN_RX.finditer(norm(text))]
        toks = [t for t,_ in toks_spans]
        sents = list(_sentences_with_spans(text))

        def _sent_idx(span):
            for k,(ss,se,_) in enumerate(sents):
                if _span_in_sentence(span,(ss,se)): return k
            return None

        PERIOD_TOKENS = {"fy","full-year","h1","h2","q1","q2","q3","q4"}

        def _has_number_near(toks, i, window=8):
            lo, hi = max(0, i-window), min(len(toks), i+window+1)
            seg = " ".join(toks[lo:hi])
            return bool(PERCENT_RANGE.search(seg) or CCY_NUMBER.search(seg))

        def _has_period_near(toks, i, window=8):
            lo, hi = max(0, i-window), min(len(toks), i+window+1)
            for t in toks[lo:hi]:
                if t in PERIOD_TOKENS: return True
                if len(t) == 4 and t.isdigit() and t.startswith(("20","19")): return True
            return False

        # ---------------- Proximity rules from YAML ----------------
        prox_hit = None  # {"rule": rule_dict, "meta": proximity_meta}
        best_d = 10**9

        for r in self.prox_rules:
            ok, meta = (False, None)

            # Case A: verbs + metrics (e.g., raise/lower/reaffirm near guidance)
            if r.get("verbs"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["verbs"], set_b=r["guidance_markers"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            # Case B: metric + unchanged markers (verb-less: "guidance unchanged")
            elif r.get("unchanged_tokens"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["guidance_markers"], set_b=r["unchanged_tokens"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            if ok:
                # Optional: enforce period token for "initiate" rules
                if r.get("require_period_token"):
                    if not meta or not meta.get("metric_span"):
                        continue
                    m0, _ = meta["metric_span"]
                    j = next((i for i, (_,sp) in enumerate(toks_spans) if sp[0] <= m0 < sp[1]), None)
                    if j is None or not _has_period_near(toks, j, window=r["window_tokens"]):
                        continue

                # Keep closest by token distance
                if meta and meta.get("token_dist") is not None and meta["token_dist"] < best_d:
                    prox_hit = {"rule": r, "meta": meta}
                    best_d = meta["token_dist"]

        # Optional char proximity (same as v1)
        char_ok = False
        if prox_hit:
            cs, ce = prox_hit["meta"]["cue_span"]; ms, me = prox_hit["meta"]["metric_span"]
            cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
            char_ok = abs(cue_c - met_c) <= self.prox_chars

        # ---------------- Numeric gate for statement-only guidance ----------------
        numeric_gate_ok = False
        if not trig and not prox_hit and self.numeric_rules:
            for r in self.numeric_rules:
                for i, (tok, sp) in enumerate(toks_spans):
                    if any(tok.startswith(m) for m in r["guidance_markers"]):
                        if (not r["require_number"]) or _has_number_near(toks, i, window=r["window_tokens"]):
                            numeric_gate_ok = True
                            break
                if numeric_gate_ok:
                    break
        
        # Optional bare "guidance + period" text gate (no numbers/verbs)  # NEW
        text_gate_span: Optional[Tuple[int,int]] = None
        if not trig and not prox_hit and not numeric_gate_ok:  # NEW
            for i, (tok, sp) in enumerate(toks_spans):
                if tok.startswith("guidance") and _has_period_near(toks, i, window=8):
                    text_gate_span = sp
                    break


        # Legacy fwd-cue ↔ metric fallback (unchanged)
        if not trig and not prox_hit and not numeric_gate_ok and self.fwd_cue_terms and self.fin_metric_terms:
            ok_legacy, meta_legacy = proximity_cooccurs_morph_spans(
                text, self.fwd_cue_terms, self.fin_metric_terms,
                window=self.prox_tokens, min_stem=3
            )
            if ok_legacy and meta_legacy:
                prox_hit = {
                    "rule": {"name": "fwd_cue_near_metric", "type": "guidance_change", "lang": "any"},
                    "meta": meta_legacy,
                }
                cs, ce = meta_legacy["cue_span"]; ms, me = meta_legacy["metric_span"]
                cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
                char_ok = abs(cue_c - met_c) <= self.prox_chars

        # ---------------- Gate decision (unchanged) ----------------
        if not trig:
            mode = self.prox_mode
            prox_ok = (prox_hit is not None) if mode=="token" else \
                    (char_ok)              if mode=="char"  else \
                    (prox_hit and char_ok) if mode=="both"  else \
                    (prox_hit or char_ok)
            
            if not (prox_ok or numeric_gate_ok or text_gate_span):   # NEW: include text_gate_span
                return

        # === NEW: even if trig=True, don’t proceed without a real gate ===
        if not (prox_hit or numeric_gate_ok or text_gate_span):
            return
        
        # ---------------- Provenance + sentence suppressors (unchanged) ----------------
        cue = None
        cue_sent = None

        if prox_hit:
            cue = {
                "label":   prox_hit["rule"]["name"],
                "pattern": prox_hit["meta"]["cue_term"],
                "match":   text[prox_hit["meta"]["cue_span"][0]:prox_hit["meta"]["cue_span"][1]],
                "span":    prox_hit["meta"]["cue_span"],
            }
            cue_i = _sent_idx(prox_hit["meta"]["cue_span"])
            met_i = _sent_idx(prox_hit["meta"]["metric_span"])
            if cue_i is None or met_i is None:
                return
            if not self.allow_cross_sentence:
                if cue_i != met_i:
                    return
                cue_sent = sents[cue_i]
            else:
                if not (-self.look_left_sentences <= (met_i-cue_i) <= self.look_right_sentences):
                    return
                cue_sent = sents[cue_i]

            ss, se, sent = cue_sent
            if self.rx_invite.search(sent) or self.rx_disclaimer_inl.search(sent):
                return
            blk = _find_block_range(text, self.rx_disclaimer_blk)
            if blk and (ss >= blk[0]):
                return
            st = doc.get("source_type")
            if self.rx_sup_macro.search(sent) and st not in {"issuer_pr","exchange","regulator"}:
                if not COMPANY_PRONOUNS.search(sent):
                    return
            if self.rx_sup_analyst.search(sent) and st not in {"issuer_pr","exchange","regulator"}:
                return
            if re.search(r"\b(will|kommer\s+att)\b", sent, re.I):
                if not (re.search(r"\b(guidance|outlook|forecast|target|aim|mål)\b", sent, re.I)
                        or PERCENT_RANGE.search(sent) or CCY_NUMBER.search(sent)):
                    return

            #cue_term = (prox_hit["meta"].get("cue_term") or "")
            cue_term = (prox_hit.get("meta", {}).get("cue_term") or "")
            _dprint("[guidance_v2 cue_term]", f"{cue_term=}")

            # FIXED: keep the 'past' check inside the if block
            emode = None
            if self.rx_expect_family.search(cue_term):
                emode = self._expected_mode(sent, cue_term=cue_term)
                if emode == "past":
                    return

            # if self.rx_expect_family.search(cue_term):
            #     emode = self._expected_mode(sent, cue_term=cue_term)
            #     if emode == "past":
            #         return  # suppress only if the *cue itself* is past-looking

        # If we tripped the bare text gate, emit a stub and stop
        if text_gate_span:
            period_local = _period_from_near_span(text, text_gate_span) or _period_from_doc(doc) or "UNKNOWN"
            yield {
                "metric": "guidance",
                "metric_kind": "text",
                "unit": "text",
                "currency": None,
                "value_type": "na",
                "value_low": None,
                "value_high": None,
                "basis": "reported",
                "period": period_local,
                "direction_hint": None,
                "_trigger_source": "text_gate",
            }
            return

        # If we have a guidance cue (change/unchanged) but no numbers in that cue sentence, emit stub
        if prox_hit and prox_hit["rule"].get("type") in {"guidance_change","guidance_unchanged","guidance_reaffirm"}:
            if not cue_sent:
                return
            ss, se, sent_txt = cue_sent
            if not (PERCENT_RANGE.search(sent_txt) or CCY_NUMBER.search(sent_txt)):
                span = prox_hit["meta"]["cue_span"]
                period_local = _period_from_near_span(text, span) or _period_from_doc(doc) or "UNKNOWN"
                yield {
                    "metric": "guidance",
                    "metric_kind": "text",
                    "unit": "text",
                    "currency": None,
                    "value_type": "na",
                    "value_low": None,
                    "value_high": None,
                    "basis": "reported",
                    "period": period_local,
                    "direction_hint": ("flat" if prox_hit["rule"]["type"] in {"guidance_unchanged","guidance_reaffirm"} else None),
                    "_trigger_source": "fwd_cue",
                    "_trigger_label":   cue["label"]   if cue else None,
                    "_trigger_pattern": cue["pattern"] if cue else None,
                    "_trigger_match":   cue["match"]   if cue else None,
                    "_trigger_span":    cue["span"]    if cue else None,
                    "_rule_type":       prox_hit["rule"]["type"],
                    "_rule_name":       prox_hit["rule"]["name"],
                }
                return

        # ---------------- Document context (unchanged) ----------------
        period = _norm_period(text)
        basis = _detect_basis(text) or "reported"

        # ---------------- Metric hints & direction (unchanged) ----------------
        hits = []
        for k, v in FIN_METRICS.items():
            k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
            if k_rx.search(text):
                hits.append((k, v))

        direction_hint = None
        if any_present_morph(text, self.dir_up_terms, min_stem=3):
            direction_hint = "up"
        elif any_present_morph(text, self.dir_dn_terms, min_stem=3):
            direction_hint = "down"
        else:
            direction_hint = "up" if self.rx_dir_up.search(text) else ("down" if self.rx_dir_dn.search(text) else None)

        if prox_hit and prox_hit["rule"]["type"] in {"guidance_unchanged", "guidance_reaffirm"}:
            direction_hint = "flat"

        prov = trig or cue
        trigger_source = "strong" if trig else ("fwd_cue" if prox_hit else ("numeric_gate" if numeric_gate_ok else "none"))

        # ---------------- Extraction (modular, behavior preserved) ----------------
        yielded = False


        allow_doc_fallback = bool(trig or prox_hit or numeric_gate_ok)

                # High-level gate/config state
        _dprint("[guidance_v2 gate]",
                f"trig={bool(trig)}",
                f"prox_hit={bool(prox_hit)}",
                f"numeric_gate_ok={bool(numeric_gate_ok)}",
                f"period_doc={period!r}",
                f"allow_doc_fallback={allow_doc_fallback}",
                f"prox_mode={self.prox_mode}",
        )

        # Show if YAML/config actually loaded
        _dprint("[guidance_v2 cfg]",
                f"fwd_cue_terms={len(self.fwd_cue_terms)}",
                f"metric_terms={len(self.fin_metric_terms)}",
                f"prox_rules={len(self.prox_rules)}",
                f"numeric_rules={len(self.numeric_rules)}",
                f"allow_cross_sentence={self.allow_cross_sentence}",
                f"look_left={self.look_left_sentences}",
                f"look_right={self.look_right_sentences}",
        )

        # Trigger provenance
        if trig:
            _dprint("[guidance_v2 trigger]",
                    f"label={trig['label']}",
                    f"span={trig['span']}",
                    f"match={trig['match']!r}")

        # Proximity provenance
        if prox_hit:
            meta = prox_hit["meta"]
            rule = prox_hit["rule"]
            _dprint("[guidance_v2 prox]",
                    f"rule={rule.get('name')}",
                    f"type={rule.get('type')}",
                    f"cue_term={meta.get('cue_term')!r}",
                    f"cue_span={meta.get('cue_span')}",
                    f"metric_span={meta.get('metric_span')}",
                    f"token_dist={meta.get('token_dist')}",
            )

        # Quick regex visibility: how many numeric things we see in the whole text
        _pcts = list(PERCENT_RANGE.finditer(text))
        _rngs = list(CCY_RANGE.finditer(text))
        _nums = list(CCY_NUMBER.finditer(text))
        _dprint("[guidance_v2 regex_counts]",
                f"pct={len(_pcts)}",
                f"ccy_range={len(_rngs)}",
                f"ccy_num={len(_nums)}")


        # 1) Percent (growth/margins): ranges -> single values (skip covered)
        pct_range_spans = []

        for cand, span in self._extract_pct_range_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            pct_range_spans.append(span)
            yield cand

        for cand in self._extract_pct_number_candidates(
            text=text, sents=sents, head=head, basis=basis,
            period_doc=period, direction_hint=direction_hint,
            prov=prov, trigger_source=trigger_source, doc=doc,
            prox_hit=prox_hit, pct_range_spans=pct_range_spans,
            allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            yield cand

        # 2a) Currency ranges (levels) — track spans to avoid double counting
        range_spans = []
        for cand, span in self._extract_ccy_range_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            range_spans.append(span)
            yield cand

        # 2b) Currency single numbers (levels), skipping anything covered by a range
        for cand in self._extract_ccy_number_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, range_spans=range_spans, allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            yield cand

        # ----------------------------------------------------------------------------------
        # 3) TEXT-ONLY “unchanged guidance” stub — run regardless of numeric gate
        #----------------------------------------------------------------------------------
        for cand in self._extract_unchanged_stub_candidates(
            text=text,
            sents=sents,
            head=head,
            period_doc=period,          # <-- use the computed period
            prov=prov,
            doc=doc,
            basis=basis,
            prox_hit=prox_hit,
            trig=trig,
            cue=cue,
            trigger_source=trigger_source,
            direction_hint=direction_hint,
            hits=hits,
        ):
            yielded = True
            yield cand


    def _detect_v3(self, doc):
        """
        Clean, staged detector that implements the test matrix:
        1) Build text, sentences, tokens
        2) Find a real *gate*: proximity OR numeric OR bare-text ("guidance + period")
        3) Validate cue sentence & suppress (invite/disclaimer/analyst/macro + 'expected' past)
        4) Resolve context (period/basis/direction)
        5) Extract: % ranges → % singles → CCY ranges → CCY singles
        6) Stubs: (a) bare-text gate stub, (b) unchanged/reaffirm/initiate, (c) tight fallback

        Notes
        -----
        - Even with a strong trigger, we require a real gate (prox|numeric|text) to proceed.
        - 'expected' suppression is applied only when the *cue* itself is from the expect-family
        and the sentence is past-looking (rx_expected_past), so present/future “expect” remains.
        """
        import re

        # ---------- 1) Build text, tokenize once ----------
        text = f"{doc.get('title','')}\n{doc.get('body','')}"
        head = f"{doc.get('title','')} {doc.get('body','')[:400]}"
        sents = list(_sentences_with_spans(text))
        toks_spans = [(m.group(0), m.span()) for m in _TOKEN_RX.finditer(norm(text))]
        toks = [t for t, _ in toks_spans]

        def _sent_idx(span):
            for k, (ss, se, _) in enumerate(sents):
                if _span_in_sentence(span, (ss, se)):
                    return k
            return None

        PERIOD_TOKENS = {"fy", "full-year", "h1", "h2", "q1", "q2", "q3", "q4"}

        def _has_number_near(i, window=8):
            lo, hi = max(0, i - window), min(len(toks), i + window + 1)
            seg = " ".join(toks[lo:hi])
            return bool(PERCENT_RANGE.search(seg) or CCY_NUMBER.search(seg))

        def _has_period_near(i, window=8):
            lo, hi = max(0, i - window), min(len(toks), i + window + 1)
            for t in toks[lo:hi]:
                if t in PERIOD_TOKENS:
                    return True
                if len(t) == 4 and t.isdigit() and t.startswith(("20", "19")):
                    return True
            return False

        # ---------- 2) Strong trigger (kept) ----------
        trig = self._find_trigger(text)
        if trig:
            ss, se = trig["span"]
            ctx_s = max(0, ss - 60)
            ctx_e = min(len(text), se + 60)
            if NEGATION_RX.search(text[ctx_s:ctx_e]):
                trig = None

        # ---------- 3) Proximity gate (verbs+markers OR unchanged) ----------
        prox_hit = None   # {"rule": r, "meta": meta}
        best_d = 10**9
        for r in self.prox_rules:
            ok, meta = (False, None)
            if r.get("verbs"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["verbs"], set_b=r["guidance_markers"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )
            elif r.get("unchanged_tokens"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["guidance_markers"], set_b=r["unchanged_tokens"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )
            if not ok:
                continue

            if r.get("require_period_token"):
                if not meta or not meta.get("metric_span"):
                    continue
                m0, _ = meta["metric_span"]
                j = next((i for i, (_, sp) in enumerate(toks_spans) if sp[0] <= m0 < sp[1]), None)
                if j is None or not _has_period_near(j, window=r["window_tokens"]):
                    continue

            if meta and meta.get("token_dist") is not None and meta["token_dist"] < best_d:
                prox_hit = {"rule": r, "meta": meta}
                best_d = meta["token_dist"]

        char_ok = False
        if prox_hit:
            cs, ce = prox_hit["meta"]["cue_span"]; ms, me = prox_hit["meta"]["metric_span"]
            cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
            char_ok = abs(cue_c - met_c) <= self.prox_chars

        # ---------- 4) Numeric gate (statement-only: marker + nearby number) ----------
        numeric_gate_ok = False
        if not trig and not prox_hit and self.numeric_rules:
            for r in self.numeric_rules:
                markers = r.get("guidance_markers") or r.get("metrics") or []
                win = r.get("window_tokens", self.prox_tokens)
                req_num = bool(r.get("require_number", True))
                for i, (tok, _) in enumerate(toks_spans):
                    if any(tok.startswith(m) for m in markers):
                        if (not req_num) or _has_number_near(i, window=win):
                            numeric_gate_ok = True
                            break
                if numeric_gate_ok:
                    break


        # ---------- 5a) Bare text gate: "guidance" + nearby period token ----------
        text_gate_span = None
        ##if not trig and not prox_hit and not numeric_gate_ok:
        for i, (tok, sp) in enumerate(toks_spans):
            if tok.startswith("guidance"):
                has_period = _has_period_near(i, window=8)
                if not has_period:
                    # raw-text year/period fallback (handles "2025." etc)
                    has_period = _period_from_near_span(text, sp) is not None
                if has_period:
                    text_gate_span = sp
                    break

        # ---------- 5b) Marker-only 'guidance interval/range' gate ----------
        text_marker_span: Optional[Tuple[int,int]] = None
        if getattr(self, "text_stub_rules", None):
            for r in self.text_stub_rules:
                m = r["rx"].search(text)
                if not m:
                    continue
                span = m.span()
                # Require a nearby period using raw coordinates (avoid norm(text) offsets)
                if r.get("require_period_token") and not _period_from_near_span(text, span):
                    continue
                text_marker_span = span
                break

        if os.getenv("GUIDANCE_DEBUG") and text_marker_span:
            s, e = text_marker_span
            print("[stub] marker hit:", repr(text[s:e]))
        
        text_gate_ok = bool(text_gate_span or text_marker_span)
        if os.getenv("GUIDANCE_DEBUG"):
            print(f"[gate] text_gate_span={text_gate_span} text_marker_span={text_marker_span}")
        
        # ---------- 6) Legacy cue↔metric fallback ----------
        if not trig and not prox_hit and not numeric_gate_ok and self.fwd_cue_terms and self.fin_metric_terms:
            ok_legacy, meta_legacy = proximity_cooccurs_morph_spans(
                text, self.fwd_cue_terms, self.fin_metric_terms,
                window=self.prox_tokens, min_stem=3
            )
            if ok_legacy and meta_legacy:
                prox_hit = {"rule": {"name": "fwd_cue_near_metric", "type": "guidance_change", "lang": "any"}, "meta": meta_legacy}
                cs, ce = meta_legacy["cue_span"]; ms, me = meta_legacy["metric_span"]
                cue_c = (cs + ce) // 2; met_c = (ms + me) // 2
                char_ok = abs(cue_c - met_c) <= self.prox_chars

        # ---------- 7) Gate decision (require a *real* gate) ----------
        if not trig:
            mode = self.prox_mode
            prox_ok = (prox_hit is not None) if mode == "token" else \
                    (char_ok)              if mode == "char"  else \
                    (prox_hit and char_ok) if mode == "both"  else \
                    (prox_hit or char_ok)
            if not (prox_ok or numeric_gate_ok or text_gate_ok):
                return
            
        # Even if a strong trigger exists, still require a real gate (defensive)
        if not (prox_hit or numeric_gate_ok or text_gate_ok):
            return
        
        if os.getenv("GUIDANCE_DEBUG"):
            print(f"[gate] trig={bool(trig)} prox={bool(prox_hit)} numeric={numeric_gate_ok} text={text_gate_ok}")

        # ---------- 8) Cue provenance + sentence-level suppressors ----------
        cue = None
        cue_sent = None
        if prox_hit:
            cue = {
                "label":   prox_hit["rule"]["name"],
                "pattern": prox_hit["meta"]["cue_term"],
                "match":   text[prox_hit["meta"]["cue_span"][0]:prox_hit["meta"]["cue_span"][1]],
                "span":    prox_hit["meta"]["cue_span"],
            }
            cue_i = _sent_idx(prox_hit["meta"]["cue_span"])
            met_i = _sent_idx(prox_hit["meta"]["metric_span"])
            if cue_i is None or met_i is None:
                return
            if not self.allow_cross_sentence:
                if cue_i != met_i:
                     return
                cue_sent = sents[cue_i]
            else:
                if not (-self.look_left_sentences <= (met_i - cue_i) <= self.look_right_sentences):
                    return
                cue_sent = sents[cue_i]

            ss, se, sent = cue_sent
            
            # Inline/block suppressors
            if self.rx_invite.search(sent) or self.rx_disclaimer_inl.search(sent):
                return
            blk = _find_block_range(text, self.rx_disclaimer_blk)
            if blk and (ss >= blk[0]):
                return

            st = doc.get("source_type")
            if self.rx_sup_macro.search(sent) and st not in {"issuer_pr", "exchange", "regulator"}:
                if not COMPANY_PRONOUNS.search(sent):
                    return
            if self.rx_sup_analyst.search(sent) and st not in {"issuer_pr", "exchange", "regulator"}:
                return

            # Light friction for bare future auxiliary without metrics/keywords
            if re.search(r"\b(will|kommer\s+att)\b", sent, re.I):
                if not (re.search(r"\b(guidance|outlook|forecast|target|aim|mål)\b", sent, re.I)
                        or PERCENT_RANGE.search(sent) or CCY_NUMBER.search(sent)):
                    return

        # ---------- 10) Document context ----------
        period_doc = _norm_period(text)
        basis = _detect_basis(text) or "reported"

        # Metric hints (for text fallback) + direction
        hits = []
        for k, v in FIN_METRICS.items():
            k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
            if k_rx.search(text):
                hits.append((k, v))

        if any_present_morph(text, self.dir_up_terms, min_stem=3):
            direction_hint = "up"
        elif any_present_morph(text, self.dir_dn_terms, min_stem=3):
            direction_hint = "down"
        else:
            direction_hint = "up" if self.rx_dir_up.search(text) else ("down" if self.rx_dir_dn.search(text) else None)

        if prox_hit and prox_hit["rule"]["type"] in {"guidance_unchanged", "guidance_reaffirm"}:
            direction_hint = "flat"

        prov = trig or (cue if prox_hit else None)
        trigger_source = (
            "strong" if trig else
            ("fwd_cue" if prox_hit else
            ("numeric_gate" if numeric_gate_ok else
            ("text_gate" if text_gate_ok else "none")))
        )
        allow_doc_fallback = bool(trig or prox_hit or numeric_gate_ok or text_gate_ok)

        # ---------- 11) Extraction (delegated to existing helpers) ----------
        yielded = False

        pct_range_spans = []
        for cand, span in self._extract_pct_range_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period_doc,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            pct_range_spans.append(span)
            yield cand

        for cand in self._extract_pct_number_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period_doc,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, pct_range_spans=pct_range_spans,
            allow_doc_fallback=allow_doc_fallback
        ):
            yielded = True
            yield cand

        range_spans = []
        for cand, span in self._extract_ccy_range_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period_doc,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, allow_doc_fallback=allow_doc_fallback
        ):

            yielded = True
            range_spans.append(span)
            yield cand

        for cand in self._extract_ccy_number_candidates(
            text=text, sents=sents, head=head, basis=basis, period_doc=period_doc,
            direction_hint=direction_hint, prov=prov, trigger_source=trigger_source,
            doc=doc, prox_hit=prox_hit, range_spans=range_spans,
            allow_doc_fallback=allow_doc_fallback
        ):

            yielded = True
            yield cand

        # ---------- 12) Text-only unchanged/initiate stubs & tight fallback ----------
        for cand in self._extract_unchanged_stub_candidates(
            text=text, sents=sents, head=head, period_doc=period_doc,
            prov=prov, doc=doc, basis=basis, prox_hit=prox_hit, trig=trig, cue=cue,
            trigger_source=trigger_source, direction_hint=direction_hint, hits=hits,
        ):
            yield cand
            return  # only one stub by design
        
        def _period_from_same_sentence_span(span):
            si = _sent_idx(span)
            if si is None:
                return None
            ss, se, _ = sents[si]
            # search for period only inside the sentence bounds
            return (_period_from_near_span(text, (ss, se))
                    or _norm_period(text[ss:se]))  # local norm as a last resort

        # helper: should suppress a *text stub* in this sentence?
        def _suppress_text_stub(sent_txt, source_type):
            # negations like: "no guidance", "not provide guidance", "does not intend to provide guidance"
            RX_NO_GUIDANCE = re.compile(
                r'(?:\bno\s+guidance\b)'
                r'|(?:\b(?:does|do|will)\s+not\s+(?:provide|issue|give)\s+guidance\b)'
                r'|(?:does\s+not\s+intend\s+to\s+provide\s+guidance)',
                re.I
            )
            if self.rx_invite.search(sent_txt) or self.rx_disclaimer_inl.search(sent_txt):
                return True
            if RX_NO_GUIDANCE.search(sent_txt):
                return True
            # analyst/wire style suppression
            if self.rx_sup_analyst.search(sent_txt) and source_type not in {"issuer_pr","exchange","regulator"}:
                return True
            # macro suppression unless issuer/exchange/regulator or with company pronoun
            if self.rx_sup_macro.search(sent_txt) and source_type not in {"issuer_pr","exchange","regulator"}:
                if not COMPANY_PRONOUNS.search(sent_txt):
                    return True
            return False

        # --- A) Stub from text_gate (marker or bare "guidance")
        if text_gate_ok:
            span = text_marker_span or text_gate_span
            si = _sent_idx(span)
            sent_txt = sents[si][2] if si is not None else ""
            st = doc.get("source_type")

            # sentence-level suppressors for stubs
            if _suppress_text_stub(sent_txt, st):
                if os.getenv("GUIDANCE_DEBUG"): print("[stub] skip:suppressor")
                return

            # NOTE: use *same-sentence* period to avoid latching to the previous sentence
            period_local = (_period_from_same_sentence_span(span)
                            or _period_from_doc(doc)
                            or "UNKNOWN")

            yield {
                "metric": "guidance",
                "metric_kind": "text",
                "unit": "text",
                "currency": None,
                "value_type": "na",
                "value_low": None,
                "value_high": None,
                "basis": "reported",
                "period": period_local,
                "direction_hint": None,
                "_trigger_source": "text_gate",
            }
            return
        
        # --- B) Stub from prox-but-no-numbers-in-cue-sentence
        if prox_hit and prox_hit["rule"].get("type") in {"guidance_change","guidance_unchanged","guidance_reaffirm"}:
            if not cue_sent:
                return
            ss, se, sent_txt = cue_sent
            if not (PERCENT_RANGE.search(sent_txt) or CCY_NUMBER.search(sent_txt)):
                st = doc.get("source_type")
                if _suppress_text_stub(sent_txt, st):
                    if os.getenv("GUIDANCE_DEBUG"): print("[stub] skip:suppressor-prox")
                    # do NOT emit stub; keep going (there might be valid numeric elsewhere)
                else:
                    # use *cue sentence* only for the period
                    period_local = (_period_from_near_span(text, (ss, se))
                                    or _norm_period(text[ss:se])
                                    or _period_from_doc(doc)
                                    or "UNKNOWN")
                    yield {
                        "metric": "guidance",
                        "metric_kind": "text",
                        "unit": "text",
                        "currency": None,
                        "value_type": "na",
                        "value_low": None,
                        "value_high": None,
                        "basis": "reported",
                        "period": period_local,
                        "direction_hint": ("flat" if prox_hit["rule"]["type"] in {"guidance_unchanged","guidance_reaffirm"} else None),
                        "_trigger_source": "fwd_cue",
                        "_trigger_label":   cue["label"]   if cue else None,
                        "_trigger_pattern": cue["pattern"] if cue else None,
                        "_trigger_match":   cue["match"]   if cue else None,
                        "_trigger_span":    cue["span"]    if cue else None,
                        "_rule_type":       prox_hit["rule"]["type"],
                        "_rule_name":       prox_hit["rule"]["name"],
                    }
                    return


        # # IF NOTHING then emit a text-only stub from either text gate (with suppressors)
        # if not yielded and (text_gate_span or text_marker_span):
        #     span = text_marker_span or text_gate_span
        #     si = _sent_idx(span)
        #     sent_txt = sents[si][2] if si is not None else ""

        #     # sentence-level suppressors (same logic you apply for prox-hit)
        #     if self.rx_invite.search(sent_txt) or self.rx_disclaimer_inl.search(sent_txt):
        #         return
        #     blk = _find_block_range(text, self.rx_disclaimer_blk)
        #     if blk and si is not None and sents[si][0] >= blk[0]:
        #         return

        #     st = doc.get("source_type")
        #     if self.rx_sup_macro.search(sent_txt) and st not in {"issuer_pr", "exchange", "regulator"}:
        #         if not COMPANY_PRONOUNS.search(sent_txt):
        #             return
        #     if self.rx_sup_analyst.search(sent_txt) and st not in {"issuer_pr", "exchange", "regulator"}:
        #         return

        #     period_local = (
        #         _period_from_near_span(text, span)
        #         or _period_from_doc(doc)
        #         or _norm_period(text)   # lets "… for 2025" become FY2025
        #         or "UNKNOWN"
        #     )
        #     yield {
        #         "metric": "guidance",
        #         "metric_kind": "text",
        #         "unit": "text",
        #         "currency": None,
        #         "value_type": "na",
        #         "value_low": None,
        #         "value_high": None,
        #         "basis": "reported",
        #         "period": period_local,
        #         "direction_hint": None,
        #         "_trigger_source": "text_gate",
        #     }
        #     return

            # helper: period from same sentence as a span
    def _is_expected_past(self, sent_txt: str, cue_term: str | None = None) -> bool:
        # Uses your existing classifier; True only when the sentence is clearly past.
        return self._expected_mode(sent_txt, cue_term=cue_term) == "past"

    # ---------------- Helper extractors (behavior preserved) ----------------
    def _extract_pct_range_candidates(
    self, *, text, sents, head, basis, period_doc, direction_hint,
    prov, trigger_source, doc, prox_hit, allow_doc_fallback
    ):
        """
        Percent *ranges* only. Yields (candidate_dict, span) pairs.
        No sentinels are emitted; the caller collects spans.
        Supports both '12%–13%' and '12–13%'.
        """
        import re
        from event_feed_app.utils.text import tokenize

        # Prepositions linking number <-> metric on the right
        PREP_RX = re.compile(r"\b(of|in|for|i|för|på|av)\b", re.I)

        # Generic compound for margin family, e.g. 'rörelsemarginalen', 'operatingmarginal'
        MARGIN_COMPOUND_RX = re.compile(r"\b(\w+?)marg(?:in|inal)\w*\b", re.I)

        # Trailing-unit form: '12–13%' (vs. global PERCENT_RANGE that matches '12%–13%')
        PERCENT_TRAILING_RANGE = re.compile(r"""
            (?P<a>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
            (?:-|–|to)\s?
            (?P<b>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
            (?P<unit>%|percent|procent)
        """, re.I | re.X)

        # Build a qualifier->metric map for margin family once
        if not hasattr(self, "_margin_qual_map"):
            qual_map = {}
            opm = None
            for _k, v in FIN_METRICS.items():
                metric_name, kind, unit = v
                if unit == "pct" and kind == "margin":
                    if metric_name == "operating_margin":
                        opm = v
                    q = metric_name.replace("_margin", "").replace("margin", "").strip("_ ").lower()
                    if q:
                        qual_map[q] = v
            self._margin_qual_map = qual_map
            self._operating_margin_v = opm

        def _choose_metric_for_number(n0, n1, sent_text, metric_spans_local,
                                    token_window=8, char_window=140):
            """Pick best metric near [n0,n1) within this sentence."""
            if not metric_spans_local:
                return None, None, None

            # Partition by side of the number/range
            left  = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if b <= n0]
            right = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if a >= n1]

            def _tokdist(sub: str) -> int:
                return len(tokenize(sub))

            # Prefer nearest LEFT within tight windows
            if left:
                a,b,v,k = min(left, key=lambda t: n0 - t[1])
                gap_char = n0 - b
                gap_tok  = _tokdist(sent_text[b:n0])
                if gap_char <= char_window and gap_tok <= token_window:
                    return v, k, gap_char

            # RIGHT if linked by a preposition or very close
            if right:
                a,b,v,k = min(right, key=lambda t: t[0] - n1)
                between = sent_text[n1:a]
                gap_char = a - n1
                gap_tok  = _tokdist(between)
                if PREP_RX.search(between) or gap_tok <= token_window or gap_char <= char_window:
                    return v, k, gap_char

            # Fallback: global nearest by char distance
            best = None
            best_d = None
            for a,b,v,k in metric_spans_local:
                d = min(abs(n0 - a), abs(n1 - b))
                if best_d is None or d < best_d:
                    best, best_d = (v, k), d
            if best is None:
                return None, None, None
            v, k = best
            return v, k, best_d

        # Collect both range styles from the full text
        matches: list[re.Match] = []

        # Global PERCENT_RANGE (already imported at module level) matches '12%–13%'
        for m in PERCENT_RANGE.finditer(text):
            # Only true ranges here; singles go to _extract_pct_number_candidates
            if m.group("b"):
                matches.append(m)

        # Trailing-unit style: '12–13%'
        matches.extend(PERCENT_TRAILING_RANGE.finditer(text))

        for m in matches:
            ms, me = m.span()
            # Find the sentence containing this span
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                _dprint("[pct] skip:no_sentence_for_span", f"span=({ms},{me})")
                continue
            ss, se, s = sent
            _dprint("[pct] hit", f"span=({ms},{me})", f"sent={s!r}")

            # # Historical “expected … was …” and general past-tense suppression
            emode = self._expected_mode(s)
            if emode == "past":
                _dprint("[expected] skip: past-tense 'expected' in sentence")
                continue
                
            # # If clearly past-tense (was/were) and no forward cue in the sentence, skip
            # # (rx_fwd_cues should be compiled once from YAML fwd terms in __init__/configure)
            # if re.search(r"\b(was|were)\b", s, re.I) and not self.rx_fwd_cues.search(s):
            #     _dprint("[hist] skip: past-tense without forward cue")
            #     continue

            # Suppressors
            st = doc.get("source_type")
            if (self.rx_sup_ma.search(s)
                or self.rx_sup_debt.search(s)
                or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"})):
                _dprint("[pct] skip:suppressor", f"source_type={st}")
                continue

            # Collect % metrics present in this sentence
            metric_spans_local = []
            for k, v in FIN_METRICS.items():
                if v[2] != "pct":
                    continue
                pat = re.escape(k).replace(r"\ ", r"[\s-]+")
                k_rx = re.compile(r"\b" + pat + r"\b", re.I)
                for mm in k_rx.finditer(s):
                    # Bare "margin" must have a valid qualifier nearby to avoid generic matches
                    if k.strip().lower() == "margin":
                        left = s[max(0, mm.start()-40):mm.start()]
                        if not any(re.search(rf"\b{re.escape(p)}\w*\b", left, re.I)
                                for p in getattr(self, "margin_prefixes",
                                                    ("operating","gross","net","ebit","ebitda","adj","adjusted"))):
                            continue
                    metric_spans_local.append((mm.start(), mm.end(), v, k))

            # Also accept compound '<qualifier>marg(in|inal)' as a margin family mention
            for mm in MARGIN_COMPOUND_RX.finditer(s):
                qual = mm.group(1).lower()
                v = self._margin_qual_map.get(qual, self._operating_margin_v)  # default to operating margin
                if v:
                    metric_spans_local.append((mm.start(), mm.end(), v, f"{qual}margin"))

            if not metric_spans_local:
                _dprint("[pct] skip:no_metric_in_sentence")
                continue

            # Choose best metric span relative to the number/range within this sentence
            n0, n1 = ms - ss, me - ss
            best_v, best_k, best_d = _choose_metric_for_number(n0, n1, s, metric_spans_local)
            if not best_v:
                _dprint("[pct] skip:no_metric_link_for_number")
                continue

            # Determine period (allow doc-level fallback if gate passed)
            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period_doc != "UNKNOWN" and allow_doc_fallback:
                    _dprint("[pct] using doc-level period fallback", f"period_doc={period_doc}")
                    period_local = period_doc
                else:
                    _dprint("[pct] skip:period_unknown_no_fallback")
                    continue

            # Parse numeric values (both regexes expose a/b)
            try:
                a = _to_float(m.group("a"))
                b = _to_float(m.group("b"))
            except Exception as e:
                _dprint("[pct] skip:parse_error", f"err={e}")
                continue

            low, high = (min(a, b), max(a, b))

            metric, kind, unit = best_v
            _dprint("[pct] yield", f"metric={metric}", f"kind={kind}", "unit=pct",
                    f"low={low}", f"high={high}", f"period={period_local}",
                    f"d={best_d}", f"nearest_key={best_k}")

            yield {
                "metric": metric, "metric_kind": kind, "unit": "pct", "currency": None,
                "value_type": "range", "value_low": low, "value_high": high,
                "basis": basis, "period": period_local,
                "direction_hint": direction_hint,
                "_trigger_source": trigger_source,
                "_trigger_label":   prov["label"]   if prov else None,
                "_trigger_pattern": prov["pattern"] if prov else None,
                "_trigger_match":   prov["match"]   if prov else None,
                "_trigger_span":    prov["span"]    if prov else None,
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }, (ms, me)


    def _extract_pct_number_candidates(
        self, *, text, sents, head, basis, period_doc, direction_hint,
        prov, trigger_source, doc, prox_hit, pct_range_spans, allow_doc_fallback
    ):
        import re
        from event_feed_app.utils.text import tokenize

        PREP_RX = re.compile(r"\b(of|in|for|i|för|på|av)\b", re.I)
        MARGIN_COMPOUND_RX = re.compile(r"\b(\w+?)marg(?:in|inal)\w*\b", re.I)

        # single-%; allow approx / approx. / approximately / about / around / circa / c. / ca.
        PCT_NUM_SINGLE = re.compile(
            r"(?:(?P<approx>about|around|approximately|approx\.?|circa|c\.|ca\.?|~)\s*)?"
            r"(?P<a>\d{1,3}(?:[.,]\d+)?)\s*%",
            re.I
        )

        if not hasattr(self, "_margin_qual_map"):
            # ensure the map exists if this function runs first
            qual_map = {}
            opm = None
            for _k, v in FIN_METRICS.items():
                metric_name, kind, unit = v
                if unit == "pct" and kind == "margin":
                    if metric_name == "operating_margin":
                        opm = v
                    q = metric_name.replace("_margin", "").replace("margin", "").strip("_ ").lower()
                    if q:
                        qual_map[q] = v
            self._margin_qual_map = qual_map
            self._operating_margin_v = opm

        def _covered(span):
            s, e = span
            for rs, re_ in pct_range_spans:
                if s >= rs and e <= re_:
                    return True
            return False

        def _choose_metric_for_number(n0, n1, sent_text, metric_spans_local,
                                    token_window=8, char_window=140):
            if not metric_spans_local:
                return None, None, None
            left  = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if b <= n0]
            right = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if a >= n1]

            def _tokdist(sub: str) -> int:
                return len(tokenize(sub))

            if left:
                a,b,v,k = min(left, key=lambda t: n0 - t[1])
                gap_char = n0 - b
                gap_tok  = _tokdist(sent_text[b:n0])
                if gap_char <= char_window and gap_tok <= token_window:
                    return v, k, gap_char

            if right:
                a,b,v,k = min(right, key=lambda t: t[0] - n1)
                between = sent_text[n1:a]
                gap_char = a - n1
                gap_tok  = _tokdist(between)
                if PREP_RX.search(between) or gap_tok <= token_window or gap_char <= char_window:
                    return v, k, gap_char

            best = None
            best_d = None
            for a,b,v,k in metric_spans_local:
                d = min(abs(n0 - a), abs(n1 - b))
                if best_d is None or d < best_d:
                    best, best_d = (v, k), d
            if best is None:
                return None, None, None
            v, k = best
            return v, k, best_d

        for m in PCT_NUM_SINGLE.finditer(text):
            ms, me = m.span()
            if _covered((ms, me)):
                _dprint("[pct] skip:covered_by_range", f"span=({ms},{me})")
                continue

            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                _dprint("[pct] skip:no_sentence_for_span", f"span=({ms},{me})")
                continue
            ss, se, s = sent
            _dprint("[pct] hit_single", f"span=({ms},{me})", f"sent={s!r}")


            #  # Historical “expected … was …” and general past-tense suppression
            emode = self._expected_mode(s)
            if emode == "past":
                _dprint("[expected] skip: past-tense 'expected' in sentence")
                continue
                
            # # If clearly past-tense (was/were) and no forward cue in the sentence, skip
            # # (rx_fwd_cues should be compiled once from YAML fwd terms in __init__/configure)
            # if re.search(r"\b(was|were)\b", s, re.I) and not self.rx_fwd_cues.search(s):
            #     _dprint("[hist] skip: past-tense without forward cue")
            #     continue

            st = doc.get("source_type")
            if (self.rx_sup_ma.search(s)
                or self.rx_sup_debt.search(s)
                or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"})):
                _dprint("[pct] skip:suppressor", f"source_type={st}")
                continue

            # collect % metrics in-sentence (generic)
            metric_spans_local = []
            for k, v in FIN_METRICS.items():
                if v[2] != "pct":
                    continue
                pat = re.escape(k).replace(r"\ ", r"[\s-]+")
                k_rx = re.compile(r"\b" + pat + r"\b", re.I)
                for mm in k_rx.finditer(s):
                    if k.strip().lower() == "margin":
                        left = s[max(0, mm.start()-40):mm.start()]
                        if not any(re.search(rf"\b{re.escape(p)}\w*\b", left, re.I)
                                for p in getattr(self, "margin_prefixes",
                                                    ("operating","gross","net","ebit","ebitda","adj","adjusted"))):
                            continue
                    metric_spans_local.append((mm.start(), mm.end(), v, k))

            # compound margin family
            for mm in MARGIN_COMPOUND_RX.finditer(s):
                qual = mm.group(1).lower()
                v = self._margin_qual_map.get(qual, self._operating_margin_v)
                if v:
                    metric_spans_local.append((mm.start(), mm.end(), v, f"{qual}margin"))

            if not metric_spans_local:
                _dprint("[pct] skip:no_metric_in_sentence")
                continue

            n0, n1 = ms - ss, me - ss
            best_v, best_k, best_d = _choose_metric_for_number(n0, n1, s, metric_spans_local)
            if not best_v:
                _dprint("[pct] skip:no_metric_link_for_number")
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period_doc != "UNKNOWN" and allow_doc_fallback:
                    _dprint("[pct] using doc-level period fallback", f"period_doc={period_doc}")
                    period_local = period_doc
                else:
                    _dprint("[pct] skip:period_unknown_no_fallback")
                    continue

            a = _to_float(m.group("a"))
            approx = bool((m.groupdict() or {}).get("approx"))
            metric, kind, unit = best_v

            if approx:
                band = float(self.thresholds.get(
                    "approx_band_pp_margin" if kind == "margin" else "approx_band_pp_growth",
                    0.5 if kind == "margin" else 1.0
                ))
                low, high = a - band, a + band
                vtype = "range"
            else:
                low = high = a
                vtype = "point"

            _dprint("[pct] yield_single", f"metric={metric}", f"kind={kind}", "unit=pct",
                    f"low={low}", f"high={high}", f"period={period_local}", f"d={best_d}", f"nearest_key={best_k}")
            yield {
                "metric": metric, "metric_kind": kind, "unit": "pct", "currency": None,
                "value_type": vtype, "value_low": low, "value_high": high,
                "basis": basis, "period": period_local,
                "direction_hint": direction_hint,
                "_trigger_source": trigger_source,
                "_trigger_label":   prov["label"]   if prov else None,
                "_trigger_pattern": prov["pattern"] if prov else None,
                "_trigger_match":   prov["match"]   if prov else None,
                "_trigger_span":    prov["span"]    if prov else None,
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }



    def _extract_ccy_range_candidates(
    self, *, text, sents, head, basis, period_doc, direction_hint,
    prov, trigger_source, doc, prox_hit, allow_doc_fallback
):
        import re
        from event_feed_app.utils.text import tokenize

        PREP_RX = re.compile(r"\b(of|in|for|i|för|på|av)\b", re.I)

        # choose the best metric span for a number/range inside *this sentence*
        def _choose_metric_for_number(n0: int, n1: int, sent_text: str,
                                    metric_spans_local: list[tuple[int,int,tuple,str]],
                                    token_window: int = 8, char_window: int = 140):
            if not metric_spans_local:
                return None, None, None  # (metric_tuple, key, dist)

            # partition by side
            left  = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if b <= n0]
            right = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if a >= n1]

            def _tokdist(sub: str) -> int:
                return len(tokenize(sub))

            # 1) prefer nearest LEFT within small windows
            if left:
                a,b,v,k = min(left, key=lambda t: n0 - t[1])  # smallest gap left
                gap_char = n0 - b
                gap_tok  = _tokdist(sent_text[b:n0])
                if gap_char <= char_window and gap_tok <= token_window:
                    return v, k, gap_char

            # 2) RIGHT if linked by preposition or very close
            if right:
                a,b,v,k = min(right, key=lambda t: t[0] - n1)
                between = sent_text[n1:a]
                gap_char = a - n1
                gap_tok  = _tokdist(between)
                if PREP_RX.search(between) or gap_tok <= token_window or gap_char <= char_window:
                    return v, k, gap_char

            # 3) fallback: global nearest by char distance
            best = None
            best_d = None
            for a,b,v,k in metric_spans_local:
                d = min(abs(n0 - a), abs(n1 - b))
                if best_d is None or d < best_d:
                    best, best_d = (v, k), d
            if best is None:
                return None, None, None
            if best is None:
                return None, None, None
            if best is None:
                return None, None, None
            (v, k) = best
            return v, k, best_d

        for m in CCY_RANGE.finditer(text):
            ms, me = m.span()
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                _dprint("[rng] skip:no_sentence_for_span", f"span=({ms},{me})")
                continue
            ss, se, s = sent
            _dprint("[rng] hit", f"span=({ms},{me})", f"sent={s!r}")


            #  # Historical “expected … was …” and general past-tense suppression
            emode = self._expected_mode(s)
            if emode == "past":
                _dprint("[expected] skip: past-tense 'expected' in sentence")
                continue
                
            # # If clearly past-tense (was/were) and no forward cue in the sentence, skip
            # # (rx_fwd_cues should be compiled once from YAML fwd terms in __init__/configure)
            # if re.search(r"\b(was|were)\b", s, re.I) and not self.rx_fwd_cues.search(s):
            #     _dprint("[hist] skip: past-tense without forward cue")
            #     continue

            st = doc.get("source_type")
            # extra safeguard for non-issuer: require company pronoun in the sentence
            if st not in {"issuer_pr","exchange","regulator"} and not COMPANY_PRONOUNS.search(s):
                _dprint("[rng] skip:no_company_pronoun_non_issuer", f"source_type={st}")
                continue
            if (self.rx_sup_ma.search(s)
                or self.rx_sup_debt.search(s)
                or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"})):
                _dprint("[rng] skip:suppressor", f"source_type={st}")
                continue

            # collect ALL ccy metrics in this sentence (local spans)
            metric_spans_local = []  # (start_local, end_local, (metric, kind, unit), key)
            for k, v in FIN_METRICS.items():
                if v[2] != "ccy":
                    continue
                # allow hyphen/space variants; allow Swedish definite/inflected forms
                pat = re.escape(k).replace(r"\ ", r"[\s-]+")
                if re.search(r"[åäö]", k, re.I) or k.lower() in {"omsättning","försäljning"}:
                    pat = pat + r"\w*"  # e.g., 'omsättningen', 'försäljningen'
                k_rx = re.compile(r"\b" + pat + r"\b", re.I)
                for mm in k_rx.finditer(s):
                    metric_spans_local.append((mm.start(), mm.end(), v, k))

            if not metric_spans_local:
                _dprint("[rng] skip:no_metric_in_sentence")
                continue

            # choose best metric near the number/range
            n0, n1 = ms - ss, me - ss
            best_v, best_k, best_d = _choose_metric_for_number(n0, n1, s, metric_spans_local)

            if not best_v:
                _dprint("[rng] skip:no_metric_link_for_number")
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period_doc != "UNKNOWN" and allow_doc_fallback:
                    _dprint("[rng] using doc-level period fallback", f"period_doc={period_doc}")
                    period_local = period_doc
                else:
                    _dprint("[rng] skip:period_unknown_no_fallback")
                    continue

            # ccy   = m.group("ccy").upper()
            # scale = (m.groupdict() or {}).get("scale")
            # a = _apply_scale(_to_float(m.group("a")), scale)
            # b = _apply_scale(_to_float(m.group("b")), scale)
            # low, high = (min(a, b), max(a, b))

            ccy = m.group("ccy").upper()
            scale = (m.group("scale") or "").strip()
            # your comma→dot helper
            lo = _to_float(m.group("a"))   # 2.1
            hi = _to_float(m.group("b"))   # 2.3
            lo_m = _apply_scale(lo, scale) # 2100.0 if mdr
            hi_m = _apply_scale(hi, scale) # 2300.0 if mdr

            metric, kind, unit = best_v
            _dprint("[rng] yield", f"metric={metric}", f"low={lo_m}", f"high={hi_m}",
                    f"ccy={ccy}", f"period={period_local}", f"d={best_d}", f"nearest_key={best_k}")
            yield ({
                "metric": metric, "metric_kind": kind, "unit": "ccy", "currency": ccy,
                "value_type": "range", "value_low": lo_m, "value_high": hi_m,
                "basis": basis, "period": period_local,
                "direction_hint": direction_hint,
                "_trigger_source": trigger_source,
                "_trigger_label":   prov["label"]   if prov else None,
                "_trigger_pattern": prov["pattern"] if prov else None,
                "_trigger_match":   prov["match"]   if prov else None,
                "_trigger_span":    prov["span"]    if prov else None,
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }, (ms, me))



    def _extract_ccy_number_candidates(
    self, *, text, sents, head, basis, period_doc, direction_hint,
    prov, trigger_source, doc, prox_hit, range_spans, allow_doc_fallback
):
        import re
        from event_feed_app.utils.text import tokenize

        PREP_RX = re.compile(r"\b(of|in|for|i|för|på|av)\b", re.I)

        def _in_any_range_span(span):
            s, e = span
            for rs, re_ in range_spans:
                if s >= rs and e <= re_:
                    return True
            return False

        def _choose_metric_for_number(n0: int, n1: int, sent_text: str,
                                    metric_spans_local: list[tuple[int,int,tuple,str]],
                                    token_window: int = 8, char_window: int = 140):
            if not metric_spans_local:
                return None, None, None

            left  = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if b <= n0]
            right = [(a,b,v,k) for (a,b,v,k) in metric_spans_local if a >= n1]

            def _tokdist(sub: str) -> int:
                return len(tokenize(sub))

            if left:
                a,b,v,k = min(left, key=lambda t: n0 - t[1])
                gap_char = n0 - b
                gap_tok  = _tokdist(sent_text[b:n0])
                if gap_char <= char_window and gap_tok <= token_window:
                    return v, k, gap_char

            if right:
                a,b,v,k = min(right, key=lambda t: t[0] - n1)
                between = sent_text[n1:a]
                gap_char = a - n1
                gap_tok  = _tokdist(between)
                if PREP_RX.search(between) or gap_tok <= token_window or gap_char <= char_window:
                    return v, k, gap_char

            best = None
            best_d = None
            for a,b,v,k in metric_spans_local:
                d = min(abs(n0 - a), abs(n1 - b))
                if best_d is None or d < best_d:
                    best, best_d = (v, k), d
            if best is None:
                return None, None, None
            (v, k) = best
            return v, k, best_d

        for m in CCY_NUMBER.finditer(text):
            ms, me = m.span()
            if _in_any_range_span((ms, me)):
                _dprint("[num] skip:covered_by_range", f"span=({ms},{me})")
                continue

            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                _dprint("[num] skip:no_sentence_for_span", f"span=({ms},{me})")
                continue
            ss, se, s = sent
            _dprint("[num] hit", f"span=({ms},{me})", f"sent={s!r}")


            #  # Historical “expected … was …” and general past-tense suppression
            emode = self._expected_mode(s)
            if emode == "past":
                _dprint("[expected] skip: past-tense 'expected' in sentence")
                continue
                
            # # If clearly past-tense (was/were) and no forward cue in the sentence, skip
            # # (rx_fwd_cues should be compiled once from YAML fwd terms in __init__/configure)
            # if re.search(r"\b(was|were)\b", s, re.I) and not self.rx_fwd_cues.search(s):
            #     _dprint("[hist] skip: past-tense without forward cue")
            #     continue

            st = doc.get("source_type")
            # extra safeguard for non-issuer: require company pronoun in the sentence
            if st not in {"issuer_pr","exchange","regulator"} and not COMPANY_PRONOUNS.search(s):
                _dprint("[num] skip:no_company_pronoun_non_issuer", f"source_type={st}")
                continue
            if (self.rx_sup_ma.search(s)
                or self.rx_sup_debt.search(s)
                or (self.rx_sup_analyst.search(s) and st not in {"issuer_pr","exchange","regulator"})):
                _dprint("[num] skip:suppressor", f"source_type={st}")
                continue

            # collect ALL ccy metrics in this sentence (local spans)
            metric_spans_local = []
            for k, v in FIN_METRICS.items():
                if v[2] != "ccy":
                    continue
                pat = re.escape(k).replace(r"\ ", r"[\s-]+")
                if re.search(r"[åäö]", k, re.I) or k.lower() in {"omsättning","försäljning"}:
                    pat = pat + r"\w*"
                k_rx = re.compile(r"\b" + pat + r"\b", re.I)
                for mm in k_rx.finditer(s):
                    metric_spans_local.append((mm.start(), mm.end(), v, k))

            if not metric_spans_local:
                _dprint("[num] skip:no_metric_in_sentence")
                continue

            # choose best metric near the number
            n0, n1 = ms - ss, me - ss
            best_v, best_k, best_d = _choose_metric_for_number(n0, n1, s, metric_spans_local)
            if not best_v:
                _dprint("[num] skip:no_metric_link_for_number")
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                if period_doc != "UNKNOWN" and allow_doc_fallback:
                    _dprint("[num] using doc-level period fallback", f"period_doc={period_doc}")
                    period_local = period_doc
                else:
                    _dprint("[num] skip:period_unknown_no_fallback")
                    continue

            ccy = m.group("ccy").upper()
            scale = (m.groupdict() or {}).get("scale")
            val = _apply_scale(_to_float(m.group("num")), scale)

            metric, kind, unit = best_v
            _dprint("[num] yield", f"metric={metric}", f"val={val}", f"ccy={ccy}",
                    f"period={period_local}", f"d={best_d}", f"nearest_key={best_k}")
            yield {
                "metric": metric, "metric_kind": kind, "unit": "ccy", "currency": ccy,
                "value_type": "point", "value_low": val, "value_high": val,
                "basis": basis, "period": period_local,
                "direction_hint": direction_hint,
                "_trigger_source": trigger_source,
                "_trigger_label":   prov["label"]   if prov else None,
                "_trigger_pattern": prov["pattern"] if prov else None,
                "_trigger_match":   prov["match"]   if prov else None,
                "_trigger_span":    prov["span"]    if prov else None,
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }


    def _extract_unchanged_stub_candidates(
    self,
    *,
    text: str,
    sents,
    head: str,
    period_doc: str,
    prov: dict | None,
    doc: dict,
    basis: str | None = None,
    prox_hit: dict | None = None,
    trig: dict | None = None,
    cue: dict | None = None,
    trigger_source: str | None = None,
    direction_hint: str | None = None,
    hits: list | None = None,
):
        """
        Emit a single text-only 'guidance' stub when either:
        A) We have a prox rule indicating unchanged/reaffirm/initiate and a known period.
        B) Tight text-only fallback (old v1 behavior) with direction, known period, and headline not an invite.

        Suppress on wires/non-issuer sources unless a company pronoun is present in the relevant sentence.
        """
        import re

        # ---------- Helper to derive a period from a span's sentence ----------
        def _period_from_span(span):
            if not span:
                return None
            for ss, se, s in sents:
                if ss <= span[0] < se:
                    # Prefer the sentence itself first
                    return _norm_period(s) or _period_from_near_span(text, (ss, se))
            return None

        # Try to get a locally-derived period first (from prox/trig/cue sentence)
        local_pref = None
        if prox_hit:
            local_pref = _period_from_span((prox_hit.get("meta") or {}).get("cue_span"))
        if not local_pref and trig and isinstance(trig.get("span"), (tuple, list)):
            local_pref = _period_from_span(trig["span"])
        if not local_pref and cue and isinstance(cue.get("span"), (tuple, list)):
            local_pref = _period_from_span(cue["span"])
        
        def _clean_period(p):
            # Treat sentinel/unknowns as absence for local selection only
            return None if p in (None, "UNKNOWN", "PERIOD-UNKNOWN", "FY-UNKNOWN") else p
        
        local_pref = _clean_period(local_pref)

        

        print(
                "ENTER",
                f"title={head!r}",
                f"source_type={doc.get('source_type')}",
                f"period_doc={period_doc!r}",
                f"local_pref={local_pref!r}",
                f"has_hits={bool(hits)}",
                f"direction_hint={direction_hint!r}",
                f"prox_rule={(prox_hit or {}).get('rule',{}).get('type') if prox_hit else None}",
            )


        rule_type = (prox_hit or {}).get("rule", {}).get("type")

        # Only proceed if:
        #  - we have an "unchanged/reaffirm/initiate" prox rule, OR
        #  - we have a text-only directional cue (direction_hint is not None).
        if not (
            rule_type in {"guidance_unchanged", "guidance_reaffirm", "guidance_initiate"}
            or (hits and direction_hint is not None)
        ):
            _dprint("EXIT: skip unchanged-stub",
                    f"rule_type={rule_type}", f"direction_hint={direction_hint!r}")
            return
        
        # # Only suppress if BOTH local and doc-level periods are unknown
        # if (period_doc in (None, "UNKNOWN", "PERIOD-UNKNOWN", "FY-UNKNOWN")) and not local_pref:
        #     return

        st = doc.get("source_type")

        # ---------- A) Prox-based unchanged/reaffirm/initiate ----------
        if prox_hit and prox_hit.get("rule", {}).get("type") in {
            "guidance_unchanged", "guidance_reaffirm", "guidance_initiate"
        }:
             # 1) Always get the cue sentence
            cue_sentence = None
            cue_span = (prox_hit.get("meta") or {}).get("cue_span")
            if cue_span:
                for ss, se, s in sents:
                    if ss <= cue_span[0] < se:
                        cue_sentence = s
                        break

            # For non-issuer sources, require a company pronoun in the cue sentence.
            if st not in {"issuer_pr", "exchange", "regulator"}:
                sent_txt = None
                cue_span = (prox_hit.get("meta") or {}).get("cue_span")
                if cue_span:
                    for ss, se, s in sents:
                        if ss <= cue_span[0] < se:
                            sent_txt = s
                            break
                if not (sent_txt and COMPANY_PRONOUNS.search(sent_txt)):
                    return
            
            cue_period = _clean_period(_norm_period(cue_sentence or ""))
            
            # Prefer local (cue/trig/cue) sentence period; fallback to doc-level
            # 3) Period selection: local -> cue sentence -> headline -> doc
            period_local = (
                local_pref
                or cue_period
                #or _norm_period(head)
                or period_doc
                )
            
            #debug prints
            if prox_hit:
                cs, ce = (prox_hit.get("meta") or {}).get("cue_span") or (None, None)
                print("[unchanged-stub] prox=True",
                    "cue_term=", (prox_hit.get("meta") or {}).get("cue_term"),
                    "cue_txt=", (text[cs:ce] if cs is not None else None))
            else:
                print("[unchanged-stub] prox=False")
            

            # Gather cue info safely
            cs, ce = ((prox_hit.get("meta") or {}).get("cue_span") or (None, None))
            cue_term = (prox_hit.get("meta") or {}).get("cue_term") if prox_hit else None
            cue_txt = text[cs:ce] if cs is not None and ce is not None else None

            cue_sent = None
            if cs is not None:
                for ss, se, s in sents:
                    if ss <= cs < se:
                        cue_sent = s
                        break

            print(
                "[UNCHANGED-DBG]",
                "prox=", bool(prox_hit),
                "cue_span=", (cs, ce),
                "cue_term=", cue_term,
                "cue_txt=", cue_txt,
                "local_pref=", repr(local_pref),
                "period_doc=", repr(period_doc),
                
                "cue_sentence=", repr(cue_sent),
            )

            yield {
                "metric": "guidance",
                "metric_kind": "text",
                "unit": "text",
                "currency": None,
                "value_type": "na",
                "value_low": None,
                "value_high": None,
                "basis": basis or "reported",
                "period": period_local,  # <-- critical change
                "direction_hint": (
                    "flat" if prox_hit["rule"]["type"] in {"guidance_unchanged", "guidance_reaffirm"} else None
                ),
                "_trigger_source": trigger_source or ("fwd_cue" if prox_hit else "none"),
                "_trigger_label":   ((trig or cue) or {}).get("label"),
                "_trigger_pattern": ((trig or cue) or {}).get("pattern"),
                "_trigger_match":   ((trig or cue) or {}).get("match"),
                "_trigger_span":    ((trig or cue) or {}).get("span"),
                "_rule_type":       prox_hit["rule"]["type"],
                "_rule_name":       prox_hit["rule"]["name"],
            }
            return  # only one stub

        # ---------- B) Tight text-only fallback (directional text w/o numbers) ----------
        if (
            hits
            and direction_hint is not None
            and not self.rx_invite.search(head)
        ):
            # Gate text-only on non-issuer wires unless a company pronoun appears near 'guidance/outlook'
            if st not in {"issuer_pr", "exchange", "regulator"}:
                ok_textonly = False
                for _, _, s in sents:
                    if re.search(r"\b(guidance|outlook|forecast|prognos|utsikter)\b", s, re.I) and COMPANY_PRONOUNS.search(s):
                        ok_textonly = True
                        break
                if not ok_textonly:
                    return

            # Prefer trigger/cue sentence period; fallback to doc-level
            fallback_span = None
            if trig and isinstance(trig.get("span"), (tuple, list)):
                fallback_span = trig["span"]
            elif cue and isinstance(cue.get("span"), (tuple, list)):
                fallback_span = cue["span"]
            
              # TO THIS:
            period_local = (
                _period_from_span(fallback_span)  # sentence-first, then nearby window (your helper already does that)
                or local_pref
                #or _norm_period(head)             # <-- headline fallback fixes ma-suppressor
                or period_doc
            )

            yield {
                "metric": "guidance",
                "metric_kind": "text",
                "unit": "text",
                "currency": None,
                "value_type": "na",
                "value_low": None,
                "value_high": None,
                "basis": basis or "reported",
                "period": period_local,  # <-- critical change
                "direction_hint": direction_hint,
                "_trigger_source": trigger_source or ("strong" if trig else ("fwd_cue" if cue else "none")),
                "_trigger_label":   ((trig or cue) or {}).get("label"),
                "_trigger_pattern": ((trig or cue) or {}).get("pattern"),
                "_trigger_match":   ((trig or cue) or {}).get("match"),
                "_trigger_span":    ((trig or cue) or {}).get("span"),
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }
            return

    def _find_trigger(self, text: str) -> Optional[dict]:
        """Return first trigger match with provenance info, else None."""
        for label, rx in self.trigger_regexes:
            m = rx.search(text)
            if m:
                s, e = m.span()
                ctx_s = max(0, s - 40)
                ctx_e = min(len(text), e + 40)
                return {
                    "label": label,
                    "pattern": rx.pattern,
                    "span": (s, e),
                    "match": text[s:e],
                    "excerpt": text[ctx_s:ctx_e],
                }
        return None

    def _find_cue(self, text: str) -> Optional[dict]:
        m = self.rx_fwd_cue.search(text)
        if not m:
            return None
        s, e = m.span()
        ctx_s = max(0, s - 40)
        ctx_e = min(len(text), e + 40)
        return {
            "label": "fwd_cue",
            "pattern": self.rx_fwd_cue.pattern,
            "span": (s, e),
            "match": text[s:e],
            "excerpt": text[ctx_s:ctx_e],
        }

    # def _expected_mode(self, sent: str) -> str | None:
    #     s = sent.lower()
    #     # If no "expect" family token at all, do nothing
    #     if not self.rx_expect_family.search(s):
    #         return None
    #     # If any past marker appears → treat as past (suppress)
    #     if any(tok in s for tok in self.expected_past_terms):
    #         return "past"
    #     # If any forward marker appears → allow as future
    #     if any(tok in s for tok in self.expected_forward_terms):
    #         return "future"
    #     # Ambiguous "expected" → conservative: suppress
    #     return "past"
    

    # def _expected_mode(self, sent: str, cue_term: str | None = None) -> str | None:
    #     s = sent.lower()
    #     # Only evaluate if the cue is an expect-family token; otherwise do nothing
    #     if cue_term is not None and not self.rx_expect_family.search(cue_term):
    #         return None

    #     # If no expect* at all in the sentence, do nothing
    #     if not self.rx_expect_family.search(s):
    #         return None

    #     has_past = bool(self.rx_expected_past.search(s))
    #     has_fwd  = bool(self.rx_expected_forward.search(s))

    #     if has_fwd and not has_past:
    #         return "future"
    #     if has_past and not has_fwd:
    #         return "past"
    #     # Mixed or ambiguous → don't suppress
    #     return None
    def _expected_mode(self, sent: str, cue_term: str | None = None) -> str | None:
        s = sent.lower()

        # Only evaluate if the cue is an expect-family token; otherwise do nothing
        if cue_term is not None and not self.rx_expect_family.search(cue_term):
            return None

        # If no expect* at all in the sentence, do nothing
        if not self.rx_expect_family.search(s):
            return None

        # IMPORTANT: Past beats forward (handles "was expected to ...")
        if self.rx_expected_past.search(s):
            return "past"
        if self.rx_expected_forward.search(s):
            return "future"

        # Mixed/ambiguous without explicit past tokens → don't suppress
        return None
    
    # ------ 3) Normalize & enrich ------
    def normalize(self, cand: dict) -> dict:
        c = dict(cand)
        c["value_origin"] = "numeric" if c.get("unit") in {"pct","ccy"} else "text"
        if not c.get("basis"):
            c["basis"] = "reported"
        if c.get("value_low") is not None and c.get("value_high") is not None:
            if c["value_low"] > c["value_high"]:
                c["value_low"], c["value_high"] = c["value_high"], c["value_low"]
        return c

    # ------ 4) Prior key & compare ------
    def prior_key(self, cand: dict) -> tuple:
        return (cand["company_id"], cand["period"], cand["metric"], cand["basis"])

    def compare(self, cand: dict, prior: dict | None) -> dict:
        out: Dict[str, Any] = {
            "new_mid": None,
            "new_width": None,
            "delta_pp": None,
            "delta_pct": None,
            "range_tightened": None,
            "old_low": None,
            "old_high": None,
            "old_mid": None,
            "old_width": None,
            "change_label": None,
        }

        # Text-only guidance has no numeric comparison context.
        if cand.get("unit") == "text":
            return out

        new_low = cand.get("value_low")
        new_high = cand.get("value_high")
        if not (_is_num(new_low) and _is_num(new_high)):
            return out
        # Explicitly cast after numeric check to satisfy static analyzers
        new_low_f = float(new_low)  # type: ignore[arg-type]
        new_high_f = float(new_high)  # type: ignore[arg-type]
        new_mid = _mid(new_low_f, new_high_f)
        new_width = new_high_f - new_low_f
        out["new_mid"], out["new_width"] = new_mid, new_width

        # No prior or mismatched unit → treat as initiation, no deltas.
        if not prior or prior.get("unit") != cand.get("unit"):
            return out

        old_low = prior.get("value_low")
        old_high = prior.get("value_high")
        if not (_is_num(old_low) and _is_num(old_high)):
            return out
        old_low_f = float(old_low)  # type: ignore[arg-type]
        old_high_f = float(old_high)  # type: ignore[arg-type]
        old_mid = _mid(old_low_f, old_high_f)
        old_width = old_high_f - old_low_f
        out.update({
            "old_low": old_low, "old_high": old_high,
            "old_mid": old_mid, "old_width": old_width,
            "range_tightened": max(0.0, old_width - new_width),
        })

        if cand["unit"] == "pct":
            out["delta_pp"] = new_mid - old_mid
        else:  # currency level
            out["delta_pct"] = math.inf if old_mid == 0 else (new_mid - old_mid) / abs(old_mid) * 100.0

        return out

    # ------ 5) Decide & score ------
    def decide_and_score(self, cand: dict, comp: dict, cfg: dict) -> tuple[bool, float, dict]:
        head = (cand.get("_doc_title","") + " " + cand.get("_doc_body","")[:400]).lower()

        # Prefer numeric inference for verb/change_label; fallback to header verbs
        has_prior = comp.get("old_mid") is not None
        verb = None
        change_label = None

        thr = cfg.get("thresholds", {}) or {}
        # Label thresholds (smaller) vs significance thresholds (existing)
        label_growth_pp_min = float(thr.get("label_delta_pp_growth_min", 1.0))
        label_margin_pp_min = float(thr.get("label_delta_pp_margin_min", 0.3))
        label_level_pct_min = float(thr.get("label_delta_pct_level_min", 2.0))

        if cand.get("unit") != "text":
            if not has_prior:
                verb = "initiate"
                change_label = "new"
            else:
                if cand.get("unit") == "pct":
                    delta = comp.get("delta_pp") or 0.0
                    lim = label_growth_pp_min if cand.get("metric_kind")=="growth" else label_margin_pp_min
                else:
                    delta = comp.get("delta_pct") or 0.0
                    lim = label_level_pct_min
                if abs(delta) < lim:
                    verb = "reaffirm"
                    change_label = "reaffirmation"
                else:
                    if delta > 0:
                        verb, change_label = "raise", "upgrade"
                    elif delta < 0:
                        verb, change_label = "lower", "downgrade"

        # Fallback to header cues if still unknown (text-only etc.)
        if verb is None:
            verb = "raise"
            if   re.search(r"(withdraws|återkallar)", head): verb = "withdraw"
            elif re.search(r"(suspends|suspenderar)", head): verb = "suspend"
            elif re.search(r"(reaffirm|bekräftar)", head):   verb = "reaffirm"
            elif re.search(r"(lowers|sänker)", head):        verb = "lower"
            elif re.search(r"(raises|höjer)", head):         verb = "raise"
            elif re.search(r"((provides|issues|gives).{0,40}(guidance|outlook)|lämnar\s+.*?(prognos|utsikter))", head):
                verb = "initiate"

        score_cfg = cfg.get("score", {})
        base = float(score_cfg.get("base", 0.70))
        source_w = float(score_cfg.get("source_weights", {}).get(cand.get("source_type","wire"), 0.6))
        verb_bonus = float(score_cfg.get("verb_bonus", {}).get(verb, 0.0))

        # Significance thresholds (unchanged)
        growth_pp_min = float(thr.get("delta_pp_growth_min", 3.0))
        margin_pp_min = float(thr.get("delta_pp_margin_min", 1.5))
        level_pct_min = float(thr.get("delta_pct_level_min", 5.0))

        always_sig = (verb in {"withdraw","suspend"}) or (verb == "initiate" and not has_prior)

        magnitude = 0.0
        passed = False
        if cand.get("unit") == "text":
            change_sig = verb in {"raise","lower","withdraw","suspend","initiate"}
            is_sig = always_sig or change_sig
            score = base + 0.20*source_w + 0.15*magnitude + verb_bonus
            comp["change_label"] = change_label
            comp["inferred_verb"] = verb
            feats = {"verb": verb, "magnitude": magnitude, "source_w": source_w, "change_label": change_label, "text_only": True}
            return is_sig, max(0.0, min(1.0, score)), feats

        if cand.get("unit") == "pct":
            delta_pp = comp.get("delta_pp") or 0.0
            magnitude = min(1.0, abs(delta_pp) / (5.0 if cand.get("metric_kind")=="growth" else 3.0))
            limit = growth_pp_min if cand.get("metric_kind")=="growth" else margin_pp_min
            passed = abs(delta_pp) >= limit
        else:  # ccy
            delta_pct = comp.get("delta_pct") or 0.0
            magnitude = min(1.0, abs(delta_pct) / 10.0)
            passed = abs(delta_pct) >= level_pct_min

        is_sig = always_sig or passed
        score = base + 0.20*source_w + 0.15*magnitude + verb_bonus

        comp["change_label"] = change_label
        comp["inferred_verb"] = verb
        feats = {"verb": verb, "magnitude": magnitude, "source_w": source_w, "change_label": change_label}
        return is_sig, max(0.0, min(1.0, score)), feats

    # ------ 6) Build parquet rows (persist provenance) ------
    def build_rows(self, doc: Mapping[str, Any], cand: dict, comp: dict,
                   is_sig: bool, score: float) -> tuple[dict | None, dict | None]:
        versions_row = {
            "company_id": cand["company_id"],
            "period": cand["period"],
            "metric": cand["metric"],
            "metric_kind": cand["metric_kind"],
            "basis": cand["basis"],
            "unit": cand["unit"],
            "currency": cand.get("currency"),
            "value_type": cand["value_type"],
            "value_origin": cand.get("value_origin", "numeric"),
            "value_low": cand["value_low"],
            "value_high": cand["value_high"],
            "observed_at_utc": doc.get("published_utc"),
            "source_event_id": doc.get("doc_id"),
            "source_url": doc.get("source_url"),
        }

        details = {
            "period": cand["period"], "metric": cand["metric"], "metric_kind": cand["metric_kind"],
            "basis": cand["basis"], "unit": cand["unit"], "currency": cand.get("currency"),
            "old_low": comp.get("old_low"), "old_high": comp.get("old_high"),
            "new_low": cand["value_low"], "new_high": cand["value_high"],
            "delta_pp": comp.get("delta_pp"), "delta_pct": comp.get("delta_pct"),
            "range_tightened": comp.get("range_tightened"),
            "change_label": comp.get("change_label"),
            "inferred_verb": comp.get("inferred_verb"),
            "direction_hint": cand.get("direction_hint"),
            # provenance
            "trigger_label":   cand.get("_trigger_label"),
            "trigger_source":  cand.get("_trigger_source"),
            "trigger_pattern": cand.get("_trigger_pattern"),
            "trigger_match":   cand.get("_trigger_match"),
            "trigger_span":    cand.get("_trigger_span"),
        }

        event_id = _event_id([
            "guidance_change", cand["company_id"], cand["period"], cand["metric"], cand["basis"],
            str(doc.get("published_utc")), str(doc.get("doc_id"))
        ])

        features = {
            "plugin": "guidance_change",
            "trigger_label":  cand.get("_trigger_label"),
            "trigger_source": cand.get("_trigger_source"),
        }

        event_row = {
            "event_id": event_id,
            "event_key": "guidance_change",
            "taxonomy_class": "earnings_report",
            "company_id": cand["company_id"],
            "source_type": doc.get("source_type"),
            "published_utc": doc.get("published_utc"),
            "first_seen_utc": datetime.now(timezone.utc),
            "doc_id": doc.get("doc_id"),
            "cluster_id": doc.get("cluster_id"),
            "is_significant": bool(is_sig),
            "sig_score": float(score),
            "decision_version": "sig:v1",
            "evidence_snippet": (doc.get("title","") or "")[:280],
            "features_json": json.dumps(features),
            "details_json": json.dumps(details),
        }
        # --- New: audit row with full cleaned text for external validation file ---
        # Audit row removed from return to satisfy EventPlugin protocol.
        # (If needed elsewhere, expose via separate method.)
        return versions_row, event_row


plugin = GuidanceChangePlugin()
