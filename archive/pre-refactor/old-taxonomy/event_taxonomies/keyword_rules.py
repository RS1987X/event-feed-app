# Taxonomy keyword dictionaries for rule-based pre-classifier
# Lowercased for matching

keywords = {
    "earnings_report": [
        # English
        "interim report", "quarterly report", "quarterly results",
        "half-year", "half-yearly results", "half-year report",
        "year-end", "annual report", "full-year report",
        # Swedish
        "delårsrapport", "halvårsrapport", "kvartalsrapport",
        "bokslutskommuniké", "substansvärde",
        # Finnish
        "tilinpäätös", "puolivuosikatsaus", "osavuosikatsaus",
        # Danish
        "delårsrapport", "årsrapport"
    ],
    "share_buyback": [
        "share repurchase", "buy-back", "buyback", "buy-back programme",
        "återköp", "aktieåterköp",
        "tilbagekøb",
        "osto omia osakkeita", "omien osakkeiden osto",
        "repurchase of own shares", "acquisition of own shares"
    ],
    "dividend": [
        # English
        "dividend", "dividend proposal", "dividend policy",
        "record date", "ex-dividend", "payment date",
        # Swedish
        "utdelning", "utdelningsförslag", "utdelningspolicy",
        # Finnish
        "osinko", "osinkoehdotus",
        # Danish
        "udbetaling af udbytte", "udbyttebetaling"
    ],
    "agm_egm_governance": [
        # English
        "annual general meeting", "extraordinary general meeting",
        "agm", "egm", "board election", "auditor",
        "resolutions at the annual", "proceedings of annual",
        # Swedish
        "årsstämma", "bolagsstämma", "styrelseförändring", "revisor", "valberedning",
        # Danish
        "generalforsamling", "revisor",
        # Finnish
        "yhtiökokous"
    ],
    "capital_raise_rights_issue": [
        # English
        "rights issue", "directed issue", "private placement",
        "equity issue", "emission", "subscription price", "units", "warrants", "prospectus",
        # Swedish
        "nyemission", "företrädesemission",
        # Danish
        "kapitalforhøjelse", "emisjon",  # NO often uses "emisjon"
        # Finnish
        "osakeanti"
    ],
    "pdmr_managers_transactions": [
        "managers’ transaction", "managers' transaction", "insider transaction",
        "mar article 19", "mar 19", "ledningens transaktioner", "insynshandel"
    ],
    "admission_listing": [
        "listing", "admission", "admission to trading",
        "first day of trading", "up-listing", "uplist", "ipo",
        # Swedish
        "introduktion", "notering",
        # Danish
        "opnotering", "børsnotering",
        # Finnish
        "listautuminen"
    ],
    "debt_bond_issue": [
        "bond issue", "note issue", "notes", "loan agreement",
        "convertible", "senior unsecured", "obligation",
        "prospectus",
        # Swedish
        "låneavtal",
        # Finnish
        "lainaemissio",
        # Danish
        "realkredit", "indfrielse"
    ],
    "mna": [
        "acquisition", "merger", "divestment", "spin-off",
        # Swedish
        "förvärv", "samgående", "avyttring", "fusion",
        # Norwegian
        "oppkjøp", "fusjon",
        # Finnish
        "yrityskauppa"
    ],
    "orders_contracts": [
        "order", "contract awarded", "framework agreement", "tender",
        # Swedish
        "avtal", "ordervärde", "beställning", "kontrakt",
        # Danish
        "tilbud",
        # Finnish
        "hankinta"
    ],
    "product_launch_partnership": [
        "launch", "product release", "partnership", "collaboration", "agreement",
        "pilot", "data presentation", "poster", "abstract",
        # Swedish
        "samarbete", "lansering", "samverkan",
        # Finnish
        "yhteistyö", "uutuus",
        # Clinical
        "phase 1", "phase 2", "phase 3", "clinical trial", "study results"
    ],
    "personnel_management_change": [
        "appointed ceo", "resigns as cfo", "leaves board", "appointment", "resignation",
        # Swedish
        "tillsätts", "avgår", "styrelseledamot", "utträder",
        # Danish
        "udnævnelse", "fratræder"
    ],
    "legal_regulatory_compliance": [
        # Patents / approvals
        "patent granted", "patent approval", "uspto", "epo", "ce mark",
        "fda approval", "ema approval", "chmp opinion", "medicare reimbursement",
        "hta", "market authorisation", "market authorization",
        # Flagging
        "major shareholder", "shareholding notification", "notification of major holdings",
        "flaggningsanmälan", "notification under chapter 9", "securities market act",
        "värdepappersmarknadslagen", "section 30", "chapter 9", "voting rights", "threshold"
    ],
    "credit_ratings": [
        "moody’s", "moody's", "fitch", "s&p", "standard & poor’s", "standard & poor's",
        "credit rating", "issuer rating", "bond rating", "instrument rating",
        "ratingförändring", "rating outlook", "watch placement", "rating action",
        "affirmed", "downgrade", "upgrade",
        "outlook revised", "outlook to positive", "outlook to negative",
        "outlook stable", "placed on watch"
    ],
    "equity_actions_non_buyback": [
        "share split", "reverse split", "consolidation of shares", "conversion of shares",
        "change of isin", "name change", "aktiesplit", "sammanläggning",
        "omdöpning", "konvertering", "total number of shares and votes"
    ],
    "labor_workforce": [
        "layoff", "redundancies", "restructure workforce", "plant closure", "furlough",
        "strike", "lockout", "varsla", "varsel", "uppsägning", "permittering",
        "personalneddragning", "strejk"
    ],
    "incidents_controversies": [
        "accident", "fire", "explosion", "breach", "cyber attack",
        "data leak", "data breach", "sanctions", "recall",
        # Swedish
        "olycka", "brand", "intrång", "skandal",
        # Danish/Norwegian
        "ulykke", "databrud", "hackerangreb", "hacker attack", "angrep"
    ],
    "admission_delisting": [
        "delisting", "trading halt", "suspension", "trading suspended",
        # Swedish
        "handelsstopp", "avnotering",
        # Danish
        "handelssuspendering", "handel ophørt",
        # Norwegian
        "avnotering"
    ],
    "other_corporate_update": [
        "certified adviser", "liquidity provider", "capital markets day",
        "end of day stock quote", "weekly summary alert",
        "new office opening", "corporate housekeeping"
    ]
}
