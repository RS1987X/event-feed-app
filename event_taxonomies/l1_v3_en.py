
from typing import List, Tuple

VERSION = "3.0.0"


#removed the guidance/outlook category as it was too noisy and overlapping with earnings reports and mna
#removed sustainability/esg as it was too noisy and overlapping with other categories

# =========================
# LEVEL-1 TAXONOMY
# =========================
L1_CATEGORIES: List[Tuple[str, str, str]] = [
    ("earnings_report", "Earnings report",
     "Periodic financial results: interim/quarter/half-year/year-end reports; may include KPIs, guidance commentary, and statutory statements."),
    ("share_buyback", "Share buyback",
     "Company repurchases of its own shares: program launches, extensions, updates, completions."),
    ("dividend", "Dividend",
     "Dividend declarations, proposals, changes, schedules (ex-date, record date, payment date)."),
    ("agm_egm_governance", "AGM/EGM & governance",
     "Annual/extraordinary general meeting notices and outcomes; board/nominations; bylaws changes; auditor matters."),
    ("capital_raise_rights_issue", "Capital raise / Rights issue",
     "Equity financing events: rights issues, directed issues, private placements, units/warrants; subscription details and use of proceeds."),
    ("pdmr_managers_transactions", "Managers' transactions (PDMR)",
     "Insider dealings by persons discharging managerial responsibilities (PDMR): purchases/sales/options exercises."),
    ("admission_listing", "Admission / Listing",
     "Listings and admissions to trading: first day of trading, uplistings, venue changes."),
    ("debt_bond_issue", "Debt / Bond issue",
     "Debt financing: bonds/notes issuance, convertibles, syndicated loans, green/sustainability-linked bonds."),
    ("mna", "M&A",
     "Mergers, acquisitions, divestments, spin-offs, asset sales; includes deal terms, valuations, conditions, and timelines."),
    ("orders_contracts", "Orders / Contracts",
     "Order wins, contract awards, framework agreements, renewals, significant tender results."),
    ("product_launch_partnership", "Product / Launch / Partnership",
     "Product/service launches and upgrades; partnerships, collaborations, joint ventures, pilot programs."),
    ("personnel_management_change", "Personnel / Management change",
     "Executive and board changes: appointments, resignations, interim roles, succession plans."),
    ("legal_regulatory_compliance", "Legal / Regulatory / Compliance",
     "Litigation, investigations, regulatory approvals/denials, fines/penalties, compliance disclosures."),
    ("credit_ratings", "Credit & Ratings",
     "Issuer/instrument rating actions and outlooks by rating agencies; watch placements."),
    ("equity_actions_non_buyback", "Equity actions (non-buyback)",
     "Share splits/reverse splits, ticker/name changes, conversions, warrant exercises, share class changes."),
    ("labor_workforce", "Labor / Workforce",
     "Workforce actions and relations: layoffs, plant closures, strikes/union matters, large hiring plans."),
    ("incidents_controversies", "Incidents / Controversies",
     "Operational incidents and controversies: safety/accidents, cybersecurity breaches, data/privacy incidents, sanctions."),
    ("admission_delisting", "Delisting / Trading halt",
     "Delistings, trading suspensions/halts, market notices affecting trading status."),
    ("other_corporate_update", "Other / Corporate update",
     "General corporate updates that do not fit the above categories; housekeeping announcements.")
]