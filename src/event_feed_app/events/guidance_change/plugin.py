from __future__ import annotations
import re, json, hashlib, math
from typing import Any, Dict, Iterable, Mapping, Tuple, Optional, Pattern, List
from datetime import datetime, timezone

from ..base import EventPlugin
from event_feed_app.taxonomy.rules.textutils import (
    any_present_morph,
    proximity_cooccurs_morph,
)

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

CCY_NUMBER = re.compile(
    r"(?P<ccy>SEK|EUR|USD|NOK|DKK)\s?(?P<num>\d{1,3}(?:[ .,’]\d{3})*(?:[.,]\d{1,2})?)",
    re.I
)
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
    s = (s.replace("\u00A0","").replace("\u202F","").replace("\u2009","")
           .replace(" ","").replace("’","").replace("'",""))
    sign = 1.0
    if s and s[0] in "+-":
        if s[0] == "-": sign = -1.0
        s = s[1:]
    if "." in s or "," in s:
        dec_pos = max(s.rfind("."), s.rfind(","))
        left = re.sub(r"[.,]", "", s[:dec_pos])
        right = re.sub(r"[.,]", "", s[dec_pos+1:])
        s = left + ("." + right if right else "")
    if not s or s == ".": raise ValueError(f"invalid numeric string: {x!r}")
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
        # Keep both for convenience: a list for provenance, and a union for quick checks
        self.trigger_regexes: List[tuple[str, re.Pattern]] = []
        self.trigger_rx: re.Pattern = _rx_match_nothing()
        self.rx_fwd_cue: re.Pattern = _rx_match_nothing()
        self.rx_dir_up:  re.Pattern = _rx_match_nothing()
        self.rx_dir_dn:  re.Pattern = _rx_match_nothing()
        self.cfg: dict = {}
        self.thresholds: dict = {}
        self.rx_invite:  re.Pattern = _rx_match_nothing()
        self.rx_disclaimer_blk: re.Pattern = _rx_match_nothing()
        self.rx_disclaimer_inl: re.Pattern = _rx_match_nothing()

        # Proximity & terms (as before) ...
        self.prox_mode = "token"
        self.prox_chars = 120
        self.prox_tokens = 25
        self.fwd_cue_terms = []
        self.dir_up_terms = []
        self.dir_dn_terms = []
        self.metric_terms = []

        # NEW: compiled suppressors
        self.rx_sup_analyst = _rx_match_nothing()
        self.rx_sup_macro   = _rx_match_nothing()
        self.rx_sup_ma      = _rx_match_nothing()
        self.rx_sup_debt    = _rx_match_nothing()

    def configure(self, cfg: Optional[dict]) -> None:
        self.cfg = cfg or {}
        self.thresholds = self.cfg.get("thresholds", {}) or {}
        patterns_cfg = (self.cfg).get("patterns") or {}

        # existing unions...
        self.trigger_regexes = _compile_trigger_list(patterns_cfg.get("triggers") or patterns_cfg)
        if self.trigger_regexes:
            union = "|".join(f"(?:{rx.pattern})" for _, rx in self.trigger_regexes)
            self.trigger_rx = re.compile(union, re.I)
        else:
            self.trigger_rx = _rx_match_nothing()

        self.rx_fwd_cue        = _compile_union_group(patterns_cfg, "fwd_cue")
        self.rx_dir_up         = _compile_union_group(patterns_cfg, "dir_up")
        self.rx_dir_dn         = _compile_union_group(patterns_cfg, "dir_dn")
        self.rx_invite         = _compile_union_group(patterns_cfg, "invite")
        self.rx_disclaimer_blk = _compile_union_group(patterns_cfg, "disclaimer_block")
        self.rx_disclaimer_inl = _compile_union_group(patterns_cfg, "disclaimer_inline")

        # NEW: compile suppressors.* (analyst/macro/ma/debt)
        sups = patterns_cfg.get("suppressors") or {}
        self.rx_sup_analyst = _compile_union_from_dict(sups.get("analyst"))
        self.rx_sup_macro   = _compile_union_from_dict(sups.get("macro"))
        self.rx_sup_ma      = _compile_union_from_dict(sups.get("ma"))
        self.rx_sup_debt    = _compile_union_from_dict(sups.get("debt"))

        # Proximity config (unchanged) ...
        prox = (self.thresholds.get("proximity") or {})
        self.prox_mode   = str(prox.get("mode", "token")).lower()
        self.prox_chars  = int(prox.get("fwd_cue_metric_chars",
                                  self.thresholds.get("fwd_cue_metric_window_chars", 120)))
        self.prox_tokens = int(prox.get("fwd_cue_metric_tokens", 25))

        # Term lists (with NEW margin prefixes)
        terms_cfg = (self.cfg.get("terms") or {})

        def _get_terms(key, fallback):
            vals = terms_cfg.get(key)
            return [str(x) for x in vals] if isinstance(vals, list) and vals else fallback

        self.margin_prefixes = _get_terms("margin_allowed_prefixes",
            ["operating","op.","EBIT","EBITDA","EBITA","rörelse","ebit","ebitda","ebita"])

        self.fwd_cue_terms = _get_terms("fwd_cue_terms",
            ["expect","expects","project","aim","target","anticipate","we see","we believe","will","should",
             "förväntar","räknar med","siktar","målsätter","vi ser","kommer att","bör"])
        self.dir_up_terms  = _get_terms("dir_up_terms",
            ["improve","improving","increase","increasing","expand","grow","strengthen","better",
             "förbättra","förbättras","öka","ökande","stärka","bättre"])
        self.dir_dn_terms  = _get_terms("dir_dn_terms",
            ["decline","declining","decrease","decreasing","contract","pressure","headwinds","worse",
             "försämra","försämras","minska","minskande","press","motvind","sämre"])

        self.metric_terms = sorted(set(METRIC_MAP.keys()))

    # ------ 1) Gate: cheap prefilter ------
    def gate(self, doc: Mapping[str, Any]) -> bool:
        # a) fast path via classifier labels
        if _has_earnings_label(doc.get("l1_class")):      return True
        if _has_earnings_label(doc.get("rule_pred_cid")): return True
        if _has_earnings_label(doc.get("rule_label")):    return True

        # b) YAML-driven strong triggers (title + lead)
        head = f"{doc.get('title','')} {doc.get('body','')[:800]}"
        return bool(self.trigger_rx.search(head))

    # ------ 2) Detect numeric/text candidates ------
    def detect(self, doc: Mapping[str, Any]) -> Iterable[dict]:
        text = f"{doc.get('title','')}\n{doc.get('body','')}"
        head = f"{doc.get('title','')} {doc.get('body','')[:400]}"

        trig = self._find_trigger(text)  # strong trigger provenance (may be None)
        if trig:
            ss, se = trig["span"]
            ctx_s = max(0, ss - 60)
            ctx_e = min(len(text), se + 60)
            if NEGATION_RX.search(text[ctx_s:ctx_e]):
                trig = None  # suppress negated trigger like "denies profit warning"
        
        cue  = self._find_cue(text)      # fwd-cue provenance (may be None)

        # Require either strong trigger OR forward-looking cue to proceed
        if not (trig or cue):
            return

        # Cue-only guardrails: proximity via textutils + suppressors
        if cue and not trig:
            ok_tok, prox_pair = proximity_cooccurs_morph(
                text, self.fwd_cue_terms, self.metric_terms,
                window=self.prox_tokens, min_stem=3
            )
            # char proximity (as you had)
            char_ok = False
            metric_spans = _metric_spans(text)
            if metric_spans:
                dist = _nearest_distance(cue["span"], metric_spans)
                char_ok = (dist is not None and dist <= self.prox_chars)

            mode = self.prox_mode
            prox_ok = ok_tok if mode=="token" else char_ok if mode=="char" else (ok_tok and char_ok) if mode=="both" else (ok_tok or char_ok)
            if not prox_ok:
                return

            # same-sentence requirement
            cue_sent = None
            for ss, se, sent in _sentences_with_spans(text):
                if _span_in_sentence(cue["span"], (ss, se)):
                    cue_sent = (ss, se, sent)
                    break
            if not cue_sent:
                return

            ss, se, sent = cue_sent

            # invite / disclaimer inside the cue sentence
            if self.rx_invite.search(sent): 
                return
            if self.rx_disclaimer_inl.search(sent): 
                return
            blk = _find_block_range(text, self.rx_disclaimer_blk)
            if blk and (ss >= blk[0]):  # cue within disclaimer block
                return

            # macro outlook w/o company binding (for non-issuer sources)
            if self.rx_sup_macro.search(sent) and doc.get("source_type") not in {"issuer_pr","exchange","regulator"}:
                if not COMPANY_PRONOUNS.search(sent):
                    return

            # third-party analyst voice (suppress unless issuer source)
            if self.rx_sup_analyst.search(sent) and doc.get("source_type") not in {"issuer_pr","exchange","regulator"}:
                return

            # high-friction for bare "will/kommer att"
            if re.search(r"\b(will|kommer\s+att)\b", sent, re.I):
                if not (re.search(r"\b(guidance|outlook|forecast|target|aim|mål)\b", sent, re.I) or PERCENT_RANGE.search(sent) or CCY_NUMBER.search(sent)):
                    return

        period = _norm_period(text)
        basis = _detect_basis(text) or "reported"

        # metric hints for extraction
        hits: list[Tuple[str, Tuple[str,str,str]]] = []
        for k, v in METRIC_MAP.items():
            k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
            if k_rx.search(text):
                hits.append((k, v))

        # direction: prefer term lists (morph), fallback to regex unions
        direction_hint = None
        if any_present_morph(text, self.dir_up_terms, min_stem=3):
            direction_hint = "up"
        elif any_present_morph(text, self.dir_dn_terms, min_stem=3):
            direction_hint = "down"
        else:
            direction_hint = "up" if self.rx_dir_up.search(text) else ("down" if self.rx_dir_dn.search(text) else None)

        # provenance: prefer strong trigger; else fwd cue
        prov = trig or cue
        trigger_source = "strong" if trig else "fwd_cue"

        yielded = False

        sents = _sentences_with_spans(text)

        # 1) percent (growth/margin)
        for m in PERCENT_RANGE.finditer(text):
            ms, me = m.span()
            # find sentence
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                continue
            ss, se, s = sent

            # M&A / debt / analyst sentences → skip numeric candidates
            if self.rx_sup_ma.search(s) or self.rx_sup_debt.search(s) or (self.rx_sup_analyst.search(s) and doc.get("source_type") not in {"issuer_pr","exchange","regulator"}):
                continue

            # ensure a metric hint exists in the SAME sentence
            local_hits = []
            for k, v in METRIC_MAP.items():
                k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
                if k_rx.search(s):
                    # guard generic 'margin' as above
                    if k.strip().lower() == "margin":
                        prefix_ok = any(re.search(rf"\b{re.escape(p)}\b", s[max(0, s.lower().find('margin')-20):s.lower().find('margin')], re.I) for p in self.margin_prefixes)
                        if not prefix_ok:
                            continue
                    local_hits.append((k, v))
            if not local_hits:
                continue

            # derive period from sentence; require it unless withdraw/suspend strong trigger
            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                head_lower = head.lower()
                is_withdraw_suspend = bool(re.search(r"(withdraw|withdraws|withdrew|withdrawn|suspend|suspends|suspended)", head_lower))
                if not is_withdraw_suspend:
                    continue

            a = _to_float(m.group("a"))
            b = m.group("b")
            approx = bool((m.groupdict() or {}).get("approx"))
            low, high = (min(a, _to_float(b)), max(a, _to_float(b))) if b else (a, a)
            for _, (metric, kind, unit) in local_hits:
                if unit != "pct":
                    continue
                if b is None and approx:
                    band = float(self.thresholds.get("approx_band_pp_margin" if kind=="margin" else "approx_band_pp_growth", 0.5 if kind=="margin" else 1.0))
                    low, high = a - band, a + band
                yield {
                    "metric": metric, "metric_kind": kind, "unit": "pct", "currency": None,
                    "value_type": "range" if (b or approx) else "point",
                    "value_low": low, "value_high": high,
                    "basis": basis, "period": period_local,
                    "direction_hint": direction_hint,
                    "_trigger_source": ("strong" if trig else "fwd_cue"),
                    "_trigger_label":   (trig or cue)["label"] if (trig or cue) else None,
                    "_trigger_pattern": (trig or cue)["pattern"] if (trig or cue) else None,
                    "_trigger_match":   (trig or cue)["match"] if (trig or cue) else None,
                    "_trigger_span":    (trig or cue)["span"] if (trig or cue) else None,
                }
                yielded = True

        # 2) currency (levels)
        for m in CCY_NUMBER.finditer(text):
            ms, me = m.span()
            sent = next(((ss, se, s) for ss, se, s in sents if _span_in_sentence((ms, me), (ss, se))), None)
            if not sent:
                continue
            ss, se, s = sent

            if self.rx_sup_ma.search(s) or self.rx_sup_debt.search(s) or (self.rx_sup_analyst.search(s) and doc.get("source_type") not in {"issuer_pr","exchange","regulator"}):
                continue

            # require a 'level' metric in the SAME sentence (e.g., revenue/sales/EBITDA)
            local_hits = []
            for k, v in METRIC_MAP.items():
                k_rx = re.compile(r"\b" + re.escape(k).replace(r"\ ", r"[\s-]+") + r"\b", re.I)
                if k_rx.search(s):
                    if v[2] == "ccy":
                        local_hits.append((k, v))
            if not local_hits:
                continue

            period_local = _norm_period_sentence(s)
            if period_local == "UNKNOWN":
                continue  # require a concrete period for level guidance

            ccy = m.group("ccy").upper()
            raw_num = m.group("num")
            try:
                num = _to_float(raw_num)
            except Exception:
                continue

            for _, (metric, kind, unit) in local_hits:
                if unit != "ccy":
                    continue
                yield {
                    "metric": metric, "metric_kind": kind, "unit": "ccy", "currency": ccy,
                    "value_type": "point",
                    "value_low": num, "value_high": num,
                    "basis": basis, "period": period_local,
                    "direction_hint": direction_hint,
                    "_trigger_source": ("strong" if trig else "fwd_cue"),
                    "_trigger_label":   (trig or cue)["label"] if (trig or cue) else None,
                    "_trigger_pattern": (trig or cue)["pattern"] if (trig or cue) else None,
                    "_trigger_match":   (trig or cue)["match"] if (trig or cue) else None,
                    "_trigger_span":    (trig or cue)["span"] if (trig or cue) else None,
                }
                yielded = True

        # ---- Text-only fallback (tight): need metric + concrete period + direction; suppress invites/disclaimers
        if (not yielded
            and hits
            and period != "UNKNOWN"
            and direction_hint is not None
            and not self.rx_invite.search(head)):
            if not (cue and not trig and (self.rx_disclaimer_blk.search(text) or self.rx_disclaimer_inl.search(text))):
                # for wires/news: require a sentence that includes (guidance/outlook) + company pronoun
                if doc.get("source_type") not in {"issuer_pr","exchange","regulator"}:
                    ok_textonly = False
                    for _, _, s in _sentences_with_spans(text):
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
                    # provenance
                    "_trigger_source":  trigger_source,
                    "_trigger_label":   prov["label"]   if prov else None,
                    "_trigger_pattern": prov["pattern"] if prov else None,
                    "_trigger_match":   prov["match"]   if prov else None,
                    "_trigger_span":    prov["span"]    if prov else None,
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
