# event_taxonomy/keyword_rules.py
# Lowercased for matching
from typing import Dict, List, Mapping

keywords: Dict[str, Mapping[str, List[str]]] = {

    # ---------- Earnings / Capital / Shareholder ----------
    "earnings_report": {
        "must_any": [
            "interim report", "quarterly report", "quarterly results",
            "half-year", "half-year report", "year-end", "annual report", "full-year report",
            "delårsrapport", "halvårsrapport", "kvartalsrapport", "bokslutskommuniké",
            "tilinpäätös", "puolivuosikatsaus", "osavuosikatsaus", "årsrapport"
        ],
        "any": ["q1", "q2", "q3", "q4", "first quarter", "second quarter", "third quarter", "fourth quarter"],
        "context": ["guidance", "outlook", "net sales", "ebit", "ebitda", "profit", "loss"],
        "exclude": [],
    },

    "share_buyback": {
        "must_any": [
            "share repurchase", "buy-back", "buyback", "buy-back programme",
            "repurchase of own shares", "acquisition of own shares",
            "återköp", "aktieåterköp",
            "tilbagekøb",
            "omien osakkeiden osto", "osto omia osakkeita"
        ],
        "any": [
            "launches buyback", "buyback program", "repurchase program",
            "authorization to repurchase", "authorised to repurchase", "authorizes repurchase"
        ],
        "context": ["maximum amount", "up to", "no more than", "daily purchases", "daily repurchases"],
        "exclude": ["rights issue", "directed issue", "private placement", "dividend"],
    },

    "dividend": {
        "must_any": ["dividend", "utdelning", "osinko", "udbytte", "dividend proposal", "dividend policy"],
        "any": ["record date", "ex-dividend", "payment date", "pay-out", "pay out"],
        "context": ["per share", "per aktie", "ordinary dividend", "special dividend", "interim dividend", "final dividend"],
        "exclude": ["buyback", "share repurchase"],
    },

    "capital_raise_rights_issue": {
        "must_any": [
            "rights issue", "directed issue", "private placement",
            "equity issue", "share issue", "emission", "nyemission", "företrädesemission",
            "kapitalforhøjelse", "emisjon", "osakeanti"
        ],
        "any": ["subscription price", "issue price", "units", "warrants", "prospectus", "gross proceeds", "net proceeds"],
        "context": ["offer period", "subscription period", "bookbuilding", "book-building", "allocation", "settlement"],
        "exclude": ["share repurchase", "buyback", "dividend"],
    },

    "equity_actions_non_buyback": {
        "must_any": [
            "share split", "reverse split", "consolidation of shares", "conversion of shares",
            "change of isin", "name change", "aktiesplit", "sammanläggning",
            "omdöpning", "konvertering", "total number of shares and votes"
        ],
        "any": ["change of name", "new name", "ticker change", "isin change"],
        "context": [],
        "exclude": ["rights issue", "buyback", "dividend"],
    },

    # ---------- Governance / PDMR ----------
    "agm_egm_governance": {
        "must_any": [
            "annual general meeting", "extraordinary general meeting",
            "agm", "egm", "yhtiökokous", "generalforsamling", "bolagsstämma", "årsstämma"
        ],
        "any": ["resolutions at the annual", "proceedings of annual", "notice of annual general meeting"],
        "context": ["board election", "auditor", "nomination committee", "valberedning", "revisor"],
        "exclude": [],
    },

    "pdmr_managers_transactions": {
        "must_any": [
        "managers' transaction", "managers’ transaction",
        "pdmr", "person discharging managerial responsibilities",
        "insynshandel", "ledningens transaktioner"
    ],
    "any": [
        "article 19 mar", "mar article 19", "article 19 of mar"
    ],
    "context": [
        "isin", "shares", "aktier", "transaction date", "price", "volume"
    ],
    "exclude": [
        # keep this for truly generic disclaimers if needed; optional
        # "this information is such that", "in accordance with the market abuse regulation"
    ],
    },

    # ---------- Listings / Debt ----------
    "admission_listing": {
        "must_any": [
            "listing", "admission", "admission to trading", "first day of trading",
            "up-listing", "uplist", "ipo", "introduktion", "notering",
            "opnotering", "børsnotering", "listautuminen"
        ],
        "any": ["euronext", "nasdaq", "first north", "otc", "spotlight"],
        "context": ["ticker", "isin", "market place", "exchange"],
        "exclude": ["delisting", "trading halt", "suspension", "trading suspended", "avnotering", "handelsstopp"],
    },

    "admission_delisting": {
        "must_any": [
            "delisting", "trading halt", "suspension", "trading suspended",
            "avnotering", "handelsstopp", "handelssuspendering", "handel ophørt"
        ],
        "any": ["terminated from trading", "to be delisted", "removal from trading"],
        "context": ["exchange", "market place", "ticker", "isin"],
        "exclude": ["admission to trading", "first day of trading", "listing", "ipo"],
    },

    "debt_bond_issue": {
        "must_any": ["bond issue", "note issue", "notes", "loan agreement", "convertible", "prospectus"],
        "any": ["senior unsecured", "tap issue", "maturity", "coupon", "convertible notes", "green bond"],
        "context": ["eurobond", "minibond", "sustainability-linked", "debenture", "framework"],
        "exclude": [],
    },

    # ---------- Orders / Product ----------
    "orders_contracts": {
        "must_any": [
            "order", "contract awarded", "framework agreement", "tender",
            "purchase order", "call-off order"
        ],
        "any": [
            "wins order", "wins contract", "secures order", "received an order",
            "contract awarded", "signs contract", "enter into agreement", "call-off from framework"
        ],
        "context": [
            "order value", "ordervärde", "kontraktsvärde",
            "delivery", "deliveries", "supply", "customer", "factory", "plant", "site",
            "units", "buses", "trucks", "locomotives", "turbines", "mw", "kv", "per share"  # leave if you want; harmless
        ],
        "exclude": ["tender offer", "public tender offer", "takeover bid", "recommended cash offer"],
    },

    "product_launch_partnership": {
        "must_any": ["launch", "product release", "partnership", "collaboration", "agreement"],
        "any": ["pilot", "pilot project", "memorandum of understanding", "strategic partnership"],
        "context": ["customer", "deployment", "rollout", "market launch", "commercialization", "joint venture"],
        "exclude": ["fda approval", "ema approval", "chmp opinion", "patent granted", "patent approval"],
    },

    # ---------- Personnel ----------
    "personnel_management_change": {
        "must_any": [
            # roles
            "ceo", "chief executive officer", "group ceo",
            "cfo", "chief financial officer",
            "coo", "chief operating officer",
            "cto", "chief technology officer",
            "cio", "chief information officer",
            "chair", "chairperson", "chairman", "chair of the board",
            "board member", "board director", "non-executive director", "managing director",
            "vd", "verkställande direktör", "finanschef", "ekonomichef", "finansdirektör",
            "styrelseledamot", "styrelseordförande", "ordförande"
        ],
        "any": [
            # moves
            "appointed", "appointment", "resigns", "resignation", "steps down", "leaves board",
            "joins as", "elected as", "assumes the role of", "new ceo", "interim ceo", "acting ceo",
            "avgår", "utträder", "tillsätts", "udnævnelse", "fratræder"
        ],
        "context": ["effective from", "effective date", "starts on", "with effect from"],
        "exclude": ["nomination committee", "valberedning", "agm", "annual general meeting", "proposes", "föreslår"],
    },

    # ---------- M&A ----------
    "mna": {
        "must_any": [
            "acquisition", "merger", "business combination", "combination",
            "offer", "tender offer", "takeover bid", "public offer", "cash offer", "exchange offer",
            "divestment", "disposal", "sale", "spin-off", "demerger", "split-off", "carve-out",
            "förvärv", "uppköp", "offentligt uppköpserbjudande", "kontantbud", "budpliktsbud",
            "försäljning", "avyttring", "fusion", "samgående", "sammanslagning"
        ],
        "any": [
            "to acquire", "to acquire all shares", "to acquire the remaining shares",
            "merger agreement", "business combination agreement", "share purchase agreement",
            "asset purchase agreement", "sale and purchase agreement", "scheme of arrangement",
            "bid", "offer price", "public tender offer"
        ],
        "context": [
            "offer price", "equity value", "enterprise value", "valuation", "implies a premium",
            "acceptance period", "acceptance level", "unconditional", "settlement",
            "completion", "closing", "conditions to completion", "regulatory approvals",
            "per share", "per aktie"
        ],
        "exclude": ["purchase order", "call-off order"],  # avoid orders mixups
    },

    # ---------- Legal / Ratings / Labor / Incidents ----------
    "legal_regulatory_compliance": {
        "must_any": [
            # approvals / patents
            "patent granted", "patent approval", "uspto", "epo", "ce mark",
            "fda approval", "ema approval", "chmp opinion", "medicare reimbursement",
            "market authorisation", "market authorization",
            # flagging/TR-1
            "major shareholder", "shareholding notification", "notification of major holdings",
            "flaggningsanmälan", "notification under chapter 9", "securities market act",
            "värdepappersmarknadslagen", "section 30", "chapter 9", "voting rights", "threshold", "tr-1"
        ],
        "any": ["approval", "authorisation", "authorization", "granted", "patent number"],
        "context": [],
        "exclude": [],  # we rely on rule priority + competitor penalties
    },

    "credit_ratings": {
        "must_any": [
            "moody’s", "moody's", "moody's ratings",
            "fitch", "fitch ratings",
            "s&p", "standard & poor’s", "standard & poor's", "s&p global ratings"
        ],
        "any": [
            "credit rating", "issuer rating", "bond rating", "instrument rating",
            "rating action", "rating outlook", "watch placement",
            "affirmed", "downgrade", "upgrade", "outlook revised", "outlook stable",
            "placed on watch", "creditwatch"
        ],
        "context": ["stable outlook", "negative outlook", "positive outlook"],
        "exclude": ["outlook for"],  # earnings guidance wording
    },

    "labor_workforce": {
        "must_any": [
            "layoff", "redundancies", "restructure workforce", "plant closure", "furlough",
            "varsla", "varsel", "uppsägning", "permittering", "personalneddragning",
            "strike", "lockout", "strejk"
        ],
        "any": ["headcount reduction", "positions", "fte", "job cuts"],
        "context": ["factory", "plant", "site", "closure", "close down"],
        "exclude": [],
    },

    "incidents_controversies": {
        "must_any": [
            "accident", "fire", "explosion", "breach", "cyber attack",
            "data leak", "data breach", "sanctions", "recall",
            "olycka", "brand", "intrång", "skandal",
            "ulykke", "databrud", "hackerangreb", "angrep"
        ],
        "any": ["investigation", "probe", "incident", "security incident", "ransomware"],
        "context": ["customer data", "production halted", "facilities damaged"],
        "exclude": [],
    },

    # ---------- Other ----------
    "other_corporate_update": {
        "must_any": [
            "certified adviser", "liquidity provider", "capital markets day",
            "new office opening", "corporate housekeeping"
        ],
        "any": ["end of day stock quote", "weekly summary alert", "reminder", "save the date"],
        "context": [],
        "exclude": [],
    },
}

__all__ = ["keywords"]
