# event_feed_app/taxonomy/rules/constants.py
from __future__ import annotations
import re
from event_feed_app.utils.text import rx  # our compiled alternation helper

# ------------------------
# Confidence & locking config
# ------------------------

RULE_CONF_BASE = {
    "legal regulatory compliance events":      0.90,
    "agency + rating action/noun":     0.90,  # credit ratings
    "MAR Article 19 / PDMR":           0.95,
    "debt security + currency/amount": 0.90,
    "orders/contracts + context":      0.85,
    "role + move + name":              0.90,  # personnel
    "agm notice/outcome":              0.90,
    "mna":             0.90,  # M&A with sentence-level ORG gating
    "invitation routing":              0.80,
    "keywords+exclusions":             0.60,  # adjusted by hits/competition
    "recall/cyber cues":               0.80,
    "fallback":                        0.20,
}

LOCK_RULES = {
    "legal_regulatory_compliance",
    "regulatory approval/patent",
    "agency + rating action/noun",
    "MAR Article 19 / PDMR",
    "debt security + currency/amount",
    "role + move + name",
    "agm notice/outcome",
    "mna",
}

# ------------------------
# Morph/inflection control
# ------------------------

# Restricted verb suffix groups used by any_present_inflected()
EN_VERB_SUFFIX = r"(?:|s|ed|ing)"
SV_VERB_SUFFIX = r"(?:|ar|ade|at|er|te|t)"

# ------------------------
# High-signal building blocks
# ------------------------

# ----- Exchanges & listing/delisting -----
# --- Exchanges / venues (Nordics + common) ---
EXCHANGE_CUES = re.compile(
    r"(?ix)\b("
    r"nasdaq(?:\s+(?:stockholm|helsinki|copenhagen|iceland))?"
    r"|first\s+north(?:\s+(?:growth|premier|growth\s+market))?"
    r"|euronext(?:\s+(?:growth|oslo|paris|amsterdam|brussels|lisbon))?"
    r"|oslo\s+b[øo]rs"
    r"|stockholmsb[öo]rsen"
    r"|spotlight(?:\s+stock\s+market)?"
    r"|ngm(?:\s+equity)?"
    r"|aim\b|london\s+stock\s+exchange|lse\b"
    r"|cboe(?:\s+(?:bxz|byx|edga|edgx))?"
    r"|xetra|otc(?:qb|qx)?\b"
    r")\b"
)

# --- Strong listing cues ---
LIST_STRICT = re.compile(
    r"(?ix)\b("
    # EN
    r"admission\s+to\s+trading|admitted\s+to\s+trading|"
    r"(?:first|1st)\s+day\s+of\s+trading|"
    r"(?:to\s+)?commenc(?:e|ing)\s+trading|begins?\s+trading\s+on|"
    r"listing\s+on|listing\s+of|to\s+be\s+listed\s+on|will\s+list\s+on|"
    r"uplist(?:ing)?|transfer\s+to\s+main\s+market|move\s+to\s+main\s+market|"
    # SV
    r"upptas\s+till\s+handel|tas\s+upp\s+till\s+handel|"
    r"f[öo]rsta\s+handelsdag|notering|b[öo]rsnotering|listning|"
    # DA/NO
    r"optag(?:et|else)\s+til\s+handel|f[øo]rste\s+handelsdag|b[øo]rsnotering|"
    r"opptak\s+til\s+handel|f[øo]rste\s+handelsdag|"
    # FI
    r"listautuminen|listalleotto|ensimm(?:ä|a)inen\s+kaupank[äa]yntip[äa]iv[äa]|"
    r"kaupank[äa]ynnin\s+aloittaminen"
    r")\b"
)

# --- Strong delisting / trading-status cues ---
DELIST_STRICT = re.compile(
    r"(?ix)\b("
    # EN
    r"delist(?:ing)?|removed\s+from\s+trading|last\s+day\s+of\s+trading|"
    r"trading\s+(?:halt|halted|suspend(?:ed|sion))|"
    r"to\s+be\s+delisted|will\s+be\s+delisted|"
    # SV
    r"avnotering|sista\s+handelsdag|handelsstopp|"
    # DA/NO
    r"handel\s+oph[øo]rt|handelsstop|handelsstans|"
    # FI
    r"poistaminen\s+kaupank[äa]ynnist[äa]|viimeinen\s+kaupank[äa]yntip[äa]iv[äa]|"
    r"kaupank[äa]ynnin\s+keskeytys"
    r")\b"
)

# --- Weak phrases to explicitly ignore when standing alone ---
# (Avoid historic statements like "is listed on ... since 2020")
WEAK_LISTED_STATUS = re.compile(
    r"(?i)\b(is|are|has been|have been|remains)\s+listed\s+on\b"
)

#------AGM/EGM-----
# Strong “notice” cues (AGM + EGM, EN + Nordic)
AGM_EGM_NOTICE = re.compile(
    r"(?i)\b("
    r"notice of (?:annual|extraordinary) general meeting|"
    r"notice to (?:annual|extraordinary) general meeting|"
    r"kallelse till (?:års|extra)\s*stämma|"
    r"indkaldelse(?:\s+til)? (?:generalforsamling|ekstraordinær generalforsamling)|"
    r"kokouskutsu"
    r")\b"
)

# Outcome / proceedings cues
AGM_EGM_OUTCOME = re.compile(
    r"(?i)\b("
    r"(?:resolutions|proceedings|minutes|decisions)\s+(?:at|from|of)\s+the\s+(?:annual|extraordinary)\s+general\s+meeting|"
    r"beslut (?:vid|från) (?:års|extra)\s*stämman|"
    r"vedtagelser (?:på|fra) (?:generalforsamlingen|ekstraordinær generalforsamling)"
    r")\b"
)

# Abbreviation backoff (soft): titles like “AGM 2025 – Resolutions”
AGM_EGM_ABBR_SOFT = re.compile(
    r"(?i)\b(?:agm|egm)\b.*\b(?:resolutions?|minutes?|proceedings?|outcome)\b"
)

#------------------------DEBT------------------------
DEBT_TERMS = re.compile(r"(?i)\b(bond issue|notes|note issue|senior unsecured|convertible|loan agreement|tap issue|prospectus)\b")


#------Invite routing-----
# Robust invite patterns (title-only)
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


# ----- Credit ratings -----
# Include both curly and straight apostrophes (norm() also NFKC-normalizes).
AGENCIES = [
    "moody’s", "moody's", "moody's ratings",
    "fitch", "fitch ratings",
    "s&p", "standard & poor’s", "standard & poor's", "s&p global ratings",
]
RATING_NOUNS = [
    "credit rating", "kreditbetyg", "issuer rating", "bond rating", "instrument rating",
    "ratingförändring", "rating outlook", "watch placement", "rating action",
    "long-term issuer rating", "lt issuer rating",
    "corporate family rating", "cfr",
    "probability of default rating", "pdr",
    "insurance financial strength rating", "ifs rating", "ifsr",
    "senior unsecured rating", "subordinated rating", "hybrid notes rating",
    "mtn programme rating", "euro medium term notes rating",
    "mtn program rating", "emtn program rating",
    "stable outlook", "negative outlook", "positive outlook",
    "outlook stable", "outlook negative", "outlook positive",
    "issuer credit rating", "issue rating", "recovery rating",
    "rating_action", "icr", "national scale rating", "nsr",
    "periodic review", "credit opinion", "rating report",
    "creditwatch negative", "creditwatch positive", "creditwatch developing",
]
RATING_ACTION_STEMS_EN = [
    "affirm", "downgrade", "upgrade", "revise", "assign", "maintain", "withdraw",
    "reaffirm", "confirm", "raise", "lower", "rate",
]
RATING_ACTION_STEMS_SV = [
    "bekräft", "höj", "sänk", "ändr", "revider", "tilldel", "behåll", "återkall", "bibehåll",
]
FIXED_ACTION_PHRASES = [
    "placed on watch", "place on watch", "put on watch",
    "outlook changed to", "changed the outlook to",
    "outlook maintained", "outlook remains",
    "placed on creditwatch", "removed from creditwatch",
    "on creditwatch negative", "remains on creditwatch", "kept on creditwatch",
    "outlook revised to", "outlook unchanged", "unchanged outlook", "outlook remains unchanged",
]
RATING_TERMS_FOR_PROX = (
    FIXED_ACTION_PHRASES + RATING_ACTION_STEMS_EN + RATING_ACTION_STEMS_SV + RATING_NOUNS
)

GUIDANCE_WORDS = ["guidance", "prognos", "utsikter", "outlook for"]

# ----- Orders & Contracts -----
ORDER_ACTION_STEMS_EN = ["win", "secure", "receive", "book", "award", "sign", "enter", "ink", "land"]
ORDER_FIXED_PHRASES_EN = [
    "wins order", "wins contract", "secures order", "secures contract",
    "received an order", "received a contract", "books order", "booked order",
    "awarded a contract", "contract awarded", "purchase order",
    "call-off order", "call off order", "call-off from framework",
    "enter into agreement", "signs contract", "signed a contract",
]
ORDER_ACTION_STEMS_NORDIC = [
    "tilldel", "teckn", "ingå", "avrop", "beställ", "erhåll", "vinn",
    "tildel", "indgå", "rammeaftal", "rammeavtal", "vind", "inngå",
    "voitt", "saa tilauk", "puitesopim", "sopimukse",
]
ORDER_OBJECT_NOUNS = [
    "order", "contract",
    "framework agreement", "ramavtal", "rammeaftale", "rammeavtale", "puitesopimus",
    "call-off", "call-off order", "purchase order",
    "tender", "upphandling", "udbud", "anbud", "tilaus", "sopimus",
]
ORDER_CONTEXT_TERMS = [
    "order value", "ordervärde", "kontraktsvärde", "valued at", "worth", "value of",
    "deliver", "delivery", "deliveries", "supply", "supplies", "ship", "install", "commission",
    "customer", "client", "end customer", "end-client", "project", "site", "factory", "plant",
    "units", "buses", "trucks", "locomotives", "turbines", "mw", "mwh", "kv", "tonnes", "barrels", "bbl",
]
ORDER_EXCLUSION_PHRASES = [
    "tender offer", "public tender offer", "takeover bid", "recommended cash offer",
]

# ----- Personnel -----
PERSONNEL_ROLES = [
    "ceo", "chief executive officer", "group ceo",
    "cfo", "chief financial officer",
    "coo", "chief operating officer",
    "cto", "chief technology officer",
    "cio", "chief information officer",
    "chair", "chairperson", "chairman", "chair of the board",
    "board member", "board director", "non-executive director", "non-executive director",
    "managing director", "md",
    "vd", "verkställande direktör",
    "finanschef", "ekonomichef", "finansdirektör",
    "styrelseledamot", "styrelseordförande", "ordförande",
]

# EN: keep coverage; only remove raw "step" (too broad) and handle via phrases.
PERSONNEL_MOVE_STEMS_EN = [
    "appoint", "resign", "leav", "join", "retir", "promot",
    "elect", "reelect", "re-elect", "dismiss", "terminat",
    "succeed", "replac", "assum", "depart"
]

# NORDIC: keep yours, add only high-signal succession/acting terms; no broad stems.
PERSONNEL_MOVE_STEMS_NORDIC = [
    # Swedish
    "utse", "utnämn", "tillsätt", "avgå", "lämn", "frånträd", "tillträ",
    "rekryter", "anställ", "befordr", "väljs", "omväljs",
    "ersätt", "efterträd", "konstituer",
    # Danish
    "udnævn", "fratræd", "træder", "tiltræd", "ansæt", "afgå",
    # Norwegian
    "utnev", "fratr", "tiltr", "ansett", "erstatt", "etterfølg", "konstituer",
    # Finnish
    "nimit", "ero", "siirty"
]

PERSONNEL_FIXED_PHRASES = [
    "appointed as", "elected as", "joins as", "steps down as",
    "to step down as", "leaves the board", "leaves board",
    "assumes the role of", "assumes the position of",
    "new ceo", "interim ceo", "acting ceo", "interim cfo", "acting cfo",
]
PERSONNEL_PROPOSAL_NOUNS = [
    "nomination committee", "valberedningen", "valberedning",
    "valgkomité", "valgkomite", "valgkomiteen",
]
PERSONNEL_PROPOSAL_STEMS = ["propos", "föresl"]
AGM_TERMS = ["annual general meeting", "extraordinary general meeting", "årsstämma", "extra bolagsstämma"]

# ----- M&A -----

MNA_CORE = re.compile(
    r"(?i)\b("
    r"acquisition|acquire[sd]?|to acquire|buy(?:ing)?|takeover|merger|merge[sd]?|"
    r"combination|amalgamation|scheme of arrangement|reverse takeover|"
    r"(?:public\s+)?(?:cash\s+)?offer\s+(?:to|for)\s+(?:the\s+)?shareholders|tender\s+offer|"
    r"recommended\s+(?:cash\s+)?offer|offer document|merger agreement|"
    r"letters?\s+of\s+intent|binding\s+offer|SPA\b|share purchase agreement|"
    r"sale and purchase agreement"
    r")\b"
)

BUYBACK_EXCLUDE = re.compile(
    r"(?i)\b("
    r"buy-?back|repurchas(?:e|ing)|own shares|treasury shares|"
    r"share repurchase programme|authorization to repurchase"
    r")\b"
)

MNA_ACTION_STEMS_EN = ["acquir", "merg", "combin", "offer", "bid", "purchas", "sell", "divest", "dispos", "spin", "agree"]
MNA_ACTION_STEMS_NORDIC = [
    "förvärv", "köp", "uppköp", "lägg", "bud", "försälj", "sälj",
    "avyttr", "fusion", "samgå", "sammanslag",
    "opkøb", "overtagelsestilbud", "overtakelsestilbud",
    "oppkjøp", "fusjon", "sammenslå",
]
MNA_FIXED_PHRASES = [
    "public tender offer", "recommended public offer", "recommended public cash offer",
    "takeover bid", "mandatory offer", "cash offer", "exchange offer",
    "merger agreement", "business combination agreement", "combination agreement",
    "share purchase agreement", "asset purchase agreement", "sale and purchase agreement",
    "scheme of arrangement",
    "offer to the shareholders of", "to acquire all shares", "to acquire the remaining shares",
    "sale of", "spin-off", "spins off", "split-off", "carve-out",
]
MNA_OBJECT_NOUNS = [
    "acquisition", "merger", "business combination", "combination",
    "offer", "tender offer", "takeover bid", "public offer", "cash offer", "exchange offer",
    "divestment", "disposal", "sale", "spin-off", "demerger", "split-off", "carve-out",
    "minority stake", "majority stake",
    "förvärv", "uppköp", "offentligt uppköpserbjudande", "kontantbud", "budpliktsbud",
    "försäljning", "avyttring", "fusion", "samgående", "sammanslagning",
]
MNA_CONTEXT_TERMS = [
    "offer price", "equity value", "enterprise value", "ev", "valuation", "implies a premium",
    "premium of", "cash consideration", "share consideration", "mixed consideration",
    "acceptance period", "acceptance level", "acceptance rate", "unconditional", "settlement",
    "timetable", "completion", "closing", "conditions to completion", "regulatory approvals",
    "merger clearance", "antitrust approval", "competition approval",
    "per share", "per aktie",
]
PER_SHARE_RX = re.compile(r"\b(per[-\s]?share|per\s+aktie)\b", re.IGNORECASE)

#----------PRODUCT LAUNCH/PARTNERSHIP----------

# ---------- Positive signals (morph-aware) ----------
import sys
from typing import List, Tuple
LAUNCH_PHRASES: List[str] = [
    # EN
    "launch", "roll out", "roll-out", "release", "go live", "general availability",
    "market launch", "commercial launch", "availability in",
    "introduce", "introduction",
    # Nordics (stems work via morph)
    "lanser", "introducer", "utrull", "lanserer",
]

PRODUCT_TERMS: List[str] = [
    # EN
    "product", "solution", "platform", "service", "offering",
    "module", "feature", "firmware", "software", "version", "update", "upgrade",
    "app", "application", "sdk", "api",
    # Nordics (stems picked up by morph)
    "produkt", "lösning", "plattform", "tjänst", "erbjudande",   # SV
    "modul", "funktion", "firmware", "mjukvara", "uppdatering",  # SV
    "platform", "service", "opdatering", "firmware", "software", # DA
    "tuote", "alusta", "palvelu", "ohjelmisto", "laiteohjelmisto", "päivitys"  # FI
]

DEAL_NOUNS: List[str] = [
    # EN
    "partnership", "collaboration", "alliance",
    "agreement", "supply agreement", "distribution agreement",
    "commercialization agreement", "commercialisation agreement",
    "supply and commercialization agreement", "licensing agreement",
    "reseller agreement", "oem agreement",
    "co-development", "joint development", "cooperation agreement",
    "go-to-market agreement", "integration partnership",
    "distribution partner", "channel partner",
    # Nordics
    "distributionsavtal", "licensavtal", "kommersialisering", "samarbetsavtal",
]

SUPPORT_CTX: List[str] = [
    "pilot", "pilot project", "deployment", "rollout", "roll-out",
    "customer deployment", "go-to-market", "gtm",
    "integration", "embedded", "powered by",
    "product", "platform", "solution", "service", "offering",
]

# Verbs with restricted endings (avoid 'significant' / 'enterprise')
EN_VERB_STEMS: List[str] = ["sign", "enter", "form", "agree"]
SVNO_VERB_STEMS: List[str] = ["ingå", "teckn", "signer", "underteckn", "inngå"]
EN_ENDINGS = r"(?:|s|ed|ing)"
SVNO_ENDINGS = r"(?:|r|de|te|t|er)"

# ---------- Hard vetoes (expanded) ----------
_HARD_VETO_RE = re.compile(
    r"("
    # Ops/advisories
    r"\bcustomer advisory\b|"
    r"\bsituation update\b|"
    r"\boperational update\b|"

    r"\b(buyback programme?|share repurchase programme?)\b|"

    # Labor/workforce & transport cancellations
    r"\bindustrial action\b|"
    r"\bstrike[s]?\b|"
    r"\blockout\b|"
    r"\bwalkout\b|"
    r"\bwork stoppage\b|"
    r"\bunion\b|"
    r"\bcollective bargaining agreement\b|"
    r"\blabou?r agreement\b|"
    r"\bunion agreement\b|"
    r"\bcancel(?:led|ing)? flights?\b|"
    r"\bflight cancellation[s]?\b|"

    # Governance / meetings
    r"\bnotice of (?:the )?annual general meeting\b|"
    r"\bannual general meeting\b|\bextraordinary general meeting\b|\bagm\b|\begm\b|"
    r"\bkallelse\b|\bårsstämma\b|\bbolagsstämma\b|\bgeneralforsamling\b|\byhtiökokous\b|"

    # Regulatory-only & IP
    r"\bfda approval\b|\bema approval\b|\bchmp opinion\b|"
    r"\bmarketing authori[sz]ation\b|\bce mark\b|"
    r"\bpatent (?:granted|approval)\b|"

    # Transparency / holdings
    r"\btr-1\b|\bnotification of major holdings\b|\bshareholding notification\b|"
    r"\bflaggningsanmälan\b|\bchapter 9\b|\bsecurities market act\b|"

    # M&A (route elsewhere)
    r"\bmerger\b|\bacquisition\b|\btender offer\b|\btakeover bid\b|\bbusiness combination\b|"
    r"\bmerger agreement\b|\bbusiness combination agreement\b|\bsale and purchase agreement\b|"
    r"\bagreement to acquire\b"
    r")",
    re.IGNORECASE,
)

# Soft veto: generic “advisory/update” when no strong positives
_SOFT_VETO_HINTS = re.compile(
    r"\b(advisory|service update|travel update|weather update)\b",
    re.IGNORECASE
)


#--------------------R&D/CLINICAL UPDATE--------------------

# ---------- Positive vocab ----------

CLINICAL_ANCHORS: Tuple[str, ...] = (
    # EN
    "clinical trial", "trial", "study", "randomized", "double blind", "placebo",
    "phase i", "phase ii", "phase iii", "phase 1", "phase 2", "phase 3",
    "preclinical", "in vivo", "in vitro", "biomarker", "endpoint", "cohort", "patients", "arm",
    "nct",
    # SV
    "klinisk studie", "studie", "prövning", "fas i", "fas ii", "fas iii", "patienter",
    # FI
    "kliininen tutkimus", "tutkimus", "koe", "vaiheen i", "vaiheen ii", "vaiheen iii", "potilaat",
    # DA
    "klinisk undersøgelse", "undersøgelse", "forsøg", "fase i", "fase ii", "fase iii", "patienter",
)

DISSEMINATION_OR_RESULTS: Tuple[str, ...] = (
    # EN
    "results", "topline", "readout", "data", "findings",
    "present", "will present", "presentation", "presented",
    "abstract", "poster", "oral presentation", "publication", "published",
    "accepted for presentation", "accepted for publication",
    "first patient in", "last patient in", "first patient dosed", "completed enrollment",
    # SV
    "resultat", "presenterar", "presentation", "abstrakt", "poster", "publikation",
    "första patienten", "sista patienten", "inkludering slutförd",
    # FI
    "tulokset", "esittelee", "esitys", "abstrakti", "posteri", "julkaistu", "julkaisu",
    "ensimmäinen potilas", "viimeinen potilas", "rekrytointi valmis",
    # DA
    "resultater", "præsenterer", "præsentation", "abstract", "plakat", "publiceret", "publicering",
    "første patient", "sidste patient", "rekruttering fuldført",
)

# Medical/statistical context that strengthens “we will present data” without an explicit anchor
BIOMED_CONTEXT: Tuple[str, ...] = (
    # EN
    "efficacy", "safety", "tolerability", "endpoint", "primary endpoint", "secondary endpoint",
    "hazard ratio", "p-value", "confidence interval", "dose", "dosing", "mg", "μg", "microgram",
    "biomarker", "pharmacokinetic", "pharmacodynamic",
    # Nordics (broad cues)
    "effekt", "säkerhet", "tolerans", "endpunkt", "primär endpoint", "sekundär endpoint",
    "annos", "dosi", "dosis", "mg", "ug",
)

# Congress/meeting lexicon (generic + common LS acronyms)
CONGRESS_TERMS: Tuple[str, ...] = (
    # EN generic
    "congress", "conference", "annual meeting", "scientific meeting", "symposium",
    # SV/FI/DA generic
    "kongress", "konferens", "årsmöte", "vetenskapligt möte", "symposium",
    "kongressi", "konferenssi", "vuosikokous", "tieteellinen kokous",
    "kongres", "konference", "årsmøde", "videnskabeligt møde",
    # Common biomed acronyms
    "asco", "esmo", "esc", "aha", "ada", "aacr", "ash", "asn", "easl", "ectrims", "ean", "ers",
)

# ---------- Structural vetoes (light, robust) ----------

REPORT_TITLE_TERMS: Tuple[str, ...] = (
    # EN
    "interim report", "half-year report", "half year report", "annual report", "year-end report",
    # SV
    "delårsrapport", "halvårsrapport", "årsredovisning", "bokslutskommuniké",
    # FI
    "puolivuosikatsaus", "osavuosikatsaus", "tilinpäätös",
    # DA
    "halvårsrapport", "årsrapport",
)

MAJOR_HOLDINGS_VETOS: Tuple[str, ...] = (
    # EN
    "tr-1", "major holdings", "notification of major holdings", "securities market act",
    "section 30", "chapter 9",
    # SV
    "flaggningsanmälan", "flaggningsmeddelande", "värdepappersmarknadslagen",
    # FI
    "arvopaperimarkkinalain", "arvopaperimarkkinalaki", "omistusosuuden muutoksesta", "liputusilmoitus",
    # DA
    "flagning", "storaktionær", "værdipapirhandelsloven",
)

REGULATORY_VETO = (
    "marketing authorization", "marketing authorisation",
    "ema approval", "fda approval", "ec approval",
    "chmp opinion", "ce mark", "clearance", "510(k)", "de novo",
    "mhra approval", "market authorisation", "market authorization"
)


ORDER_CONTRACT_HEAD_TERMS: Tuple[str, ...] = (
    "first commercial order", "commercial order", "purchase order", "contract", "framework agreement",
)

# ------------------------
# Legal / flagging / incidents / money
# ------------------------

FLAGGING_TERMS = [
    "major shareholder", "shareholding notification", "notification of major holdings",
    "flaggningsanmälan", "notification under chapter 9", "securities market act",
    "värdepappersmarknadslagen", "section 30", "chapter 9", "voting rights", "threshold", "tr-1",
]
FLAG_NUMERIC = re.compile(r"\b(\d{1,2}(?:\.\d+)?)\s?%\b")

CURRENCY_AMOUNT = re.compile(
    r"\b(?:(?:m|bn)\s?)?(sek|nok|dkk|eur|usd|msek|mnok|mdkk|meur|musd)\b"
    r"|\b\d+(?:[\.,]\d+)?\s?(m|bn)\s?(sek|eur|usd)\b",
    re.IGNORECASE,
)

EXCHANGE_CUES = rx([
    "admission to trading on", "listing on nasdaq", "first north",
    "euronext", "otc", "first day of trading",
])

RECALL = rx(["product recall", "recall of", "återkallar", "tilbagekaldelse"])
CYBER  = rx(["cyber attack", "ransomware", "databrud", "intrång", "data breach"])

PATENT_NUMBER = re.compile(r"\b(?:us|ep)\s?\d{6,}\b", re.IGNORECASE)


#-----------------earnings report-----------------

# ---------- Strong earnings phrases (title-anchored) ----------
EARNINGS_TITLE_TERMS: List[str] = [
    # English
    "interim report", "quarterly report", "half-year report", "half year report",
    "year-end report", "full-year report", "annual report",
    "q1 report", "q2 report", "q3 report", "q4 report",
    "q1 results", "q2 results", "q3 results", "q4 results",
    "half-year results", "half year results", "full-year results", "annual results",
    "h1 report", "h2 report", "h1 results", "h2 results",
    # Swedish
    "delårsrapport", "kvartalsrapport", "halvårsrapport", "bokslutskommuniké", "årsrapport",
    # Danish / Norwegian (spelling overlaps), Finnish
    "delårsrapport", "halvårsrapport", "årsrapport",
    "osavuosikatsaus", "puolivuosikatsaus", "tilinpäätös",
]

# Financial context cues (for body corroboration)
FIN_CTX: List[str] = [
    "net sales", "revenue", "revenues", "turnover",
    "ebit", "ebitda", "operating profit", "operating income",
    "profit", "loss", "guidance", "outlook",
    "per share", "eps", "cash flow", "order intake",
    # Nordic common words
    "omsättning", "rörelseresultat", "resultat", "vinst", "förlust",
    "liikevaihto", "liikevoitto", "tappio", "tulos",
    "indtægter", "omsætning", "driftsresultat", "overskud", "underskud",
]


# ------------------------
# Priority & competitor sets (for fallback scorer)
# ------------------------
"""
CID_TIEBREAK_ORDER:
  Only used by the 'keywords+exclusions' fallback to break score ties
  between categories (higher precedence earlier in the list).
  This is NOT the same as rule execution priority.
"""
CID_TIEBREAK_ORDER = [
    "legal_regulatory_compliance",
    "credit_ratings",
    "pdmr_managers_transactions",   # run PDMR early
    "mna",                          # M&A before softer ops news
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
    "personnel_management_change",
    "labor_workforce",
    "incidents_controversies",
]
# Back-compat alias so existing imports still work
#PRIORITY = CID_TIEBREAK_ORDER

# Suppression rule:
# If cat and any suppressor in SUPPRESSORS_BY_CATEGORY[cat] have gate_passed=True,
# suppress cat (set score=0, mark suppressed_by).
SUPPRESSORS_BY_CATEGORY = {
    "earnings_report": [
        "dividend", "share_buyback", "capital_raise_rights_issue", "credit_ratings",
        "debt_bond_issue", "product_launch_partnership",
    ],
    "dividend": ["earnings_report", "share_buyback", "capital_raise_rights_issue"],
    "share_buyback": ["earnings_report", "dividend", "capital_raise_rights_issue", "equity_actions_non_buyback"],
    "credit_ratings": ["earnings_report", "dividend", "share_buyback"],
    "capital_raise_rights_issue": ["equity_actions_non_buyback", "share_buyback", "dividend"],
    "equity_actions_non_buyback": ["capital_raise_rights_issue", "share_buyback"],
    "admission_listing": ["admission_delisting"],
    "admission_delisting": ["admission_listing"],
    "debt_bond_issue": ["earnings_report", "legal_regulatory_compliance"],
    "product_launch_partnership": ["legal_regulatory_compliance", "earnings_report","rd_clinical_update"],
    "legal_regulatory_compliance": ["credit_ratings", "product_launch_partnership", "debt_bond_issue"],
    "orders_contracts": ["labor_workforce"],
    "labor_workforce": ["orders_contracts"],
    "mna": ["personnel_management_change"],
    "personnel_management_change": ["mna", "earnings_report", "agm_egm_governance"],
    "agm_egm_governance": [],
    "rd_clinical_update": ["product_launch_partnership",
                     "legal_regulatory_compliance",
                     "orders_contracts",
                     "incidents_controversies","share_buyback", 
                     "earnings_report", "admission_listing",
                     "dividend","agm_egm_governance"],
}
