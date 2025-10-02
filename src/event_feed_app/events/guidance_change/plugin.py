from __future__ import annotations
import re, json, hashlib, math
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional, Pattern, List
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
METRIC_MAP = {
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
    "jämförbar": ("organic_growth","growth","pct"),
    "ebita marginal": ("ebita_margin","margin","pct"),
}

PERCENT_RANGE = re.compile(r"""
    (?P<approx>about|around|approximately|approx\.?|close\sto|near|circa|c\.|~)?\s*
    (?P<a>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
    (?P<unit>%|percent|procent)\s?
    (?:
        (?:-|–|to)\s?                       # dash, en dash, or the word "to"
        (?P<b>[+-]?\d{1,2}(?:[.,]\d{1,2})?)\s?
        (?P=unit)
    )?
""", re.I | re.X)


NUM_GROUP = r"\d{1,3}(?:[ ,.\u00A0\u202F\u2009’']\d{3})*(?:[.,]\d{1,2})?"

CCY_NUMBER = re.compile(
    r"(?P<ccy>SEK|EUR|USD|NOK|DKK)\s?"
    r"(?P<num>\d{1,3}(?:[ .,’]\d{3})*(?:[.,]\d{1,2})?)"
    r"(?:\s*(?P<scale>million|mn|m|billion|bn|thousand|k))?",
    re.I
)

CCY_RANGE = re.compile(
    r"(?P<ccy>SEK|EUR|USD|NOK|DKK)\s?"
    r"(?P<a>\d{1,3}(?:[ .,’]\d{3})*(?:[.,]\d{1,2})?)\s?"
    r"(?:-|–|to)\s?"
    r"(?P<b>\d{1,3}(?:[ .,’]\d{3})*(?:[.,]\d{1,2})?)"
    r"(?:\s*(?P<scale>million|mn|m|billion|bn|thousand|k))?",
    re.I
)

def _apply_scale(value: float, scale: str | None) -> float:
    """Normalize to *millions* of currency."""
    if not scale:
        return value
    s = scale.strip().lower()
    if s in {"m", "mn", "million"}:
        return value                 # already millions
    if s in {"bn", "billion", "b"}:
        return value * 1000.0        # billions -> millions
    if s in {"k", "thousand"}:
        return value / 1000.0        # thousands -> millions
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
    if BASIS_ORGANIC.search(s): return "organic"
    if BASIS_CCFX.search(s): return "cc_fx"
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


def _metric_spans(text: str) -> List[tuple[int,int,str]]:
    spans: List[tuple[int,int,str]] = []
    for k in METRIC_MAP.keys():
        # Skip generic 'margin' if still present in METRIC_MAP
        if k.strip().lower() == "margin":
            continue
        k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
        for m in k_rx.finditer(text):
            spans.append((m.start(), m.end(), k))

    # Guard Swedish 'jämförbar' (require a growth/sales word nearby)
    for m in re.finditer(r"\bjämförbar\b", text, re.I):
        ctx_s = max(0, m.start() - 12)
        ctx_e = min(len(text), m.end() + 12)
        if re.search(r"(tillväxt|försäljning)", text[ctx_s:ctx_e], re.I):
            spans.append((m.start(), m.end(), "jämförbar"))

    # Guard unqualified 'margin' tokens (operate only if still needed)
    for m in re.finditer(r"\bmargin\b", text, re.I):
        ctx_s = max(0, m.start() - 20)
        prefix = text[ctx_s:m.start()]
        if any(re.search(rf"\b{re.escape(p)}\b", prefix, re.I) for p in getattr(plugin, "margin_prefixes", [])):
            spans.append((m.start(), m.end(), "margin"))
    return spans


def _nearest_distance(span: tuple[int,int], spans2: List[tuple[int,int,str]]) -> Optional[int]:
    """Min char distance from span to any in spans2."""
    s1, e1 = span[0], span[1]
    best = None
    for s2, e2, _ in spans2:
        d = 0 if (s1 <= e2 and s2 <= e1) else min(abs(s1 - e2), abs(s2 - e1))
        best = d if best is None or d < best else best
    return best

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

# Simple sentence splitter with spans
_SENT_RX = re.compile(r"[^\.!\?\;\n]+(?:[\.\!\?\;]+|\n+|$)", re.U)
def _sentences_with_spans(text: str) -> list[tuple[int,int,str]]:
    out = []
    for m in _SENT_RX.finditer(text):
        s, e = m.span()
        seg = text[s:e].strip()
        if seg:
            out.append((s, e, seg))
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
        self.margin_prefixes: list[str] = []

        # Metrics: literal phrases to scan for (built from METRIC_MAP)
        self.metric_terms: list[str] = []

        # NEW: parsed rule sets from YAML
        self.prox_rules: list[dict] = []     # patterns.proximity_triggers
        self.numeric_rules: list[dict] = []  # patterns.metric_numeric_triggers

    def configure(self, cfg: Optional[dict]) -> None:
        """
        Ingest YAML config, compile regexes, and normalize proximity/numeric rules.
        """
        self.cfg = cfg or {}
        self.thresholds = self.cfg.get("thresholds", {}) or {}
        patterns_cfg = (self.cfg).get("patterns") or {}

        # ---------------- Strong triggers (regex) ----------------
        # Accept both the new nested shape and legacy flat patterns
        self.trigger_regexes = _compile_trigger_list(patterns_cfg.get("triggers") or patterns_cfg)
        if self.trigger_regexes:
            self.trigger_rx = re.compile(
                "|".join(f"(?:{rx.pattern})" for _, rx in self.trigger_regexes), re.I
            )
        else:
            self.trigger_rx = _rx_match_nothing()

        # ---------------- Proximity tuning ----------------
        prox = (self.thresholds.get("proximity") or {})
        self.prox_mode   = str(prox.get("mode", "token")).lower()
        self.prox_chars  = int(prox.get("fwd_cue_metric_chars",
                                self.thresholds.get("fwd_cue_metric_window_chars", 120)))
        self.prox_tokens = int(prox.get("fwd_cue_metric_tokens", 25))
        self.allow_cross_sentence = bool(prox.get("allow_cross_sentence", False))
        self.look_left_sentences  = int(prox.get("look_left_sentences", 0))
        self.look_right_sentences = int(prox.get("look_right_sentences", 0))

        # Unchanged markers (used by *_unchanged_token rules)
        self.unchanged_markers["en"] = list(prox.get("unchanged_markers_en") or [])
        self.unchanged_markers["sv"] = list(prox.get("unchanged_markers_sv") or [])

        # ---------------- Forward-looking cues (literal) ----------------
        fwd_cfg = (patterns_cfg.get("fwd_cue") or {})
        terms_en = list((fwd_cfg.get("terms") or {}).get("en") or [])
        terms_sv = list((fwd_cfg.get("terms") or {}).get("sv") or [])
        # Allow overrides via top-level terms.fwd_cue_terms
        terms_cfg = (self.cfg.get("terms") or {})
        tune_terms = list(terms_cfg.get("fwd_cue_terms") or [])
        self.fwd_cue_terms = [str(x) for x in (terms_en + terms_sv + tune_terms)]

        # ---------------- Directional / invite / disclaimers (regex unions) ----------------
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

        # ---------------- Term lists (with sensible defaults) ----------------
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

        # ---------------- Metrics (literal phrases) ----------------
        # METRIC_MAP must exist in your module; it's a dict: str -> (metric, kind, unit)
        # Add common alias to be safe
        self.metric_terms = sorted(set(METRIC_MAP.keys()) | {"like for like"})
        # === FIX: unchanged markers (check thresholds first, then prox) ===
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


        # ---------------- Proximity rules (verbs/metrics + unchanged tokens) ----------------
        # helper to resolve @unchanged_markers_en indirection
        def _resolve_tokens(val):
            if isinstance(val, str) and val.startswith("@"):
                key = val[1:]
                return list(self.thresholds.get(key) or prox.get(key) or [])
            return list(val or [])

        self.prox_rules = []
        for rule in (patterns_cfg.get("proximity_triggers") or []):
            r = dict(rule)  # shallow copy
            r["name"]              = r.get("name") or "unnamed_prox_rule"
            r["type"]              = r.get("type") or "guidance_change"
            r["lang"]              = r.get("lang") or "en"
            r["verbs"]             = _resolve_tokens(r.get("verbs"))
            r["metrics"]           = _resolve_tokens(r.get("metrics"))
            r["unchanged_tokens"]  = _resolve_tokens(r.get("unchanged_tokens"))
            r["window_tokens"]     = int(r.get("window_tokens") or self.prox_tokens)
            r["same_sentence"]     = bool(r.get("same_sentence", True))
            r["require_period_token"] = bool(r.get("require_period_token", False))
            self.prox_rules.append(r)

        # ---------------- Numeric (statement-only) rules ----------------
        self.numeric_rules = []
        for rule in (patterns_cfg.get("metric_numeric_triggers") or []):
            r = dict(rule)
            r["name"]          = r.get("name") or "unnamed_numeric_rule"
            r["type"]          = r.get("type") or "guidance_stated"
            r["lang"]          = r.get("lang") or "en"
            r["metrics"]       = _resolve_tokens(r.get("metrics"))
            r["window_tokens"] = int(r.get("window_tokens") or self.prox_tokens)
            r["same_sentence"] = bool(r.get("same_sentence", True))
            r["require_number"]= bool(r.get("require_number", True))
            self.numeric_rules.append(r)

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
    def detect(self, doc: Mapping[str, Any]) -> Iterable[dict]:
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
        prox_hit = None  # {"rule": rule_dict, "meta": proximity_meta}
        best_d = 10**9

        for r in self.prox_rules:
            ok, meta = (False, None)

            # Case A: verbs + metrics (e.g., raise/lower/reaffirm near guidance)
            if r.get("verbs"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["verbs"], set_b=r["metrics"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            # Case B: metric + unchanged markers (verb-less: "guidance unchanged")
            elif r.get("unchanged_tokens"):
                ok, meta = proximity_cooccurs_morph_spans(
                    text, set_a=r["metrics"], set_b=r["unchanged_tokens"],
                    window=r["window_tokens"], min_stem=3,
                    same_sentence=r.get("same_sentence", True)
                )

            if ok:
                # Optional: enforce period token for "initiate" rules
                if r.get("require_period_token"):
                    m0, _ = meta["metric_span"]
                    j = next((i for i, (_,sp) in enumerate(toks_spans) if sp[0] <= m0 < sp[1]), None)
                    if j is None or not _has_period_near(toks, j, window=r["window_tokens"]):
                        continue

                # Keep closest by token distance
                if meta["token_dist"] < best_d:
                    prox_hit = {"rule": r, "meta": meta}
                    best_d = meta["token_dist"]

        # Optional char proximity if you still support char/either/both
        char_ok = False
        if prox_hit:
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
        if not trig and not prox_hit and not numeric_gate_ok and self.fwd_cue_terms and self.metric_terms:
            ok_legacy, meta_legacy = proximity_cooccurs_morph_spans(
                text, self.fwd_cue_terms, self.metric_terms,
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
        for k, v in METRIC_MAP.items():
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
            for k, v in METRIC_MAP.items():
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
            for k, v in METRIC_MAP.items():
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
            for k, v in METRIC_MAP.items():
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
                "_trigger_label":   (trig or cue)["label"]   if (trig or cue) else None,
                "_trigger_pattern": (trig or cue)["pattern"] if (trig or cue) else None,
                "_trigger_match":   (trig or cue)["match"]   if (trig or cue) else None,
                "_trigger_span":    (trig or cue)["span"]    if (trig or cue) else None,
                "_rule_type":       prox_hit["rule"]["type"] if prox_hit else None,
                "_rule_name":       prox_hit["rule"]["name"] if prox_hit else None,
            }



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
        out = {
            "new_mid": None, "new_width": None,
            "delta_pp": None, "delta_pct": None, "range_tightened": None,
            "old_low": None, "old_high": None, "old_mid": None, "old_width": None,
            "change_label": None,
        }

        if cand.get("unit") == "text":
            return out

        new_low, new_high = cand["value_low"], cand["value_high"]
        new_mid = _mid(new_low, new_high)
        new_width = new_high - new_low
        out["new_mid"], out["new_width"] = new_mid, new_width

        if not prior:
            return out
        if prior.get("unit") != cand.get("unit"):
            return out

        old_low, old_high = prior.get("value_low"), prior.get("value_high")
        if not (_is_num(old_low) and _is_num(old_high)):
            return out

        old_mid = _mid(old_low, old_high)
        old_width = old_high - old_low
        out.update({
            "old_low": old_low, "old_high": old_high,
            "old_mid": old_mid, "old_width": old_width,
            "range_tightened": max(0.0, old_width - new_width),
        })

        if cand["unit"] == "pct":
            out["delta_pp"] = new_mid - old_mid
        else:  # ccy
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
        audit_row = {
            "event_id": event_id,
            "event_key": "guidance_change",
            "company_id": cand["company_id"],
            "doc_id": doc.get("doc_id"),
            "cluster_id": doc.get("cluster_id"),
            "published_utc": doc.get("published_utc"),
            "source_type": doc.get("source_type"),
            "source_url": doc.get("source_url"),
            "metric": cand.get("metric"),
            "metric_kind": cand.get("metric_kind"),
            "basis": cand.get("basis"),
            "unit": cand.get("unit"),
            "period": cand.get("period"),
            "direction_hint": cand.get("direction_hint"),
            "sig_score": float(score),
            "is_significant": bool(is_sig),
            "trigger_source": cand.get("_trigger_source"),
            "trigger_label": cand.get("_trigger_label"),
            "trigger_match": cand.get("_trigger_match"),
            # the bits you want to eyeball:
            "title_clean": doc["title_clean"],
            "body_clean": doc["body_clean"],
        }

        return versions_row, event_row,audit_row


plugin = GuidanceChangePlugin()
