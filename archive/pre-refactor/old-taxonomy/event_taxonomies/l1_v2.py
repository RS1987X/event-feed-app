

from typing import List, Tuple

VERSION = "2.0.0"

L1_CATEGORIES: List[Tuple[str, str, str]] = [
    ("results_outlook", "Results & Outlook",
     "Covers periodic financial reporting (quarter/interim/H1/9M/FY), preliminary/flash results, trading updates, "
     "and forward-looking guidance (initiate/raise/cut/confirm). Signals: revenue/EBIT/EBITDA/net income, EPS, margins, "
     "organic growth, one-offs, KPI tables, IFRS/GAAP notes, guidance ranges/withdrawals, profit warnings. "
     "Includes: restatements and selected KPI-only updates if framed as results or outlook. "
     "Excludes: one-off order wins (Orders & Contracts), product news (Products & Partnerships), or financing details (Capital Markets)."),

    ("capital_markets", "Capital Markets (Equity/Debt)",
     "Primary market financing actions. Equity: rights issues, directed issues/placements, private placements/PIPEs, units with warrants; "
     "details on size, price, discount, subscription period, use of proceeds, underwriting. Debt: bonds/notes/convertibles, sustainability-linked/green bonds, "
     "term sheets, coupons, maturities, covenants, loans/RCFs. Signals: ‘issue’, ‘placement’, ‘prospectus’, ‘bookbuilding’, ‘subscription’, ‘ISIN’. "
     "Excludes: share splits/ticker/name changes (Equity Actions), routine dividend/buyback updates (Capital Returns)."),

    ("capital_returns", "Dividends & Buybacks",
     "Distributions to shareholders. Dividends: declarations, changes, schedules (ex-date, record date, payment date), interim/special dividends. "
     "Buybacks: program launches, increases, extensions, daily/weekly execution updates, completions, cancellations. "
     "Signals: ‘ex-dividend’, ‘record date’, ‘repurchase authorization’, ‘maximum consideration’, ‘treasury shares’. "
     "Excludes: capital raises (Capital Markets)."),

    ("listings_trading_status", "Listings & Trading Status",
     "Admission and trading venue changes plus status interventions. Includes IPO/first day of trading, uplistings/venue transfers, "
     "new instrument admissions, observation/monitoring status, trading halts/suspensions/resumptions, delistings, certified adviser changes (for SME venues). "
     "Signals: ‘admitted to trading’, ‘first North’, ‘transfer to main market’, ‘suspension’, ‘delisting’, ‘ISIN change due to venue’. "
     "Excludes: financing terms (Capital Markets) unless the admission is purely about a new instrument listing."),

    ("mna", "M&A / Strategic Transactions",
     "Corporate structural moves: mergers, acquisitions, disposals/divestments, carve-outs, spin-offs, joint ventures with ownership changes, asset sales. "
     "Includes agreed and indicative offers, competing offers, fairness opinions, earn-outs, regulatory/antitrust milestones, long stop dates, closings. "
     "Signals: ‘SPA’, ‘binding agreement’, ‘LOI’, ‘consideration’, ‘EV/EBITDA’, ‘completion’, ‘closing conditions’. "
     "Excludes: non-equity partnerships without ownership change (Products & Partnerships)."),

    ("orders_contracts", "Orders & Contracts",
     "Revenue-generating commercial wins and material contract events: major orders, awards, framework/master service agreements, renewals/extensions, "
     "large tender outcomes, multi-year supply or distribution deals with value or strategic importance. "
     "Signals: ‘order value’, ‘PO received’, ‘award’, ‘framework agreement’, ‘call-off’, ‘won/lost tender’. "
     "Excludes: product announcements with no commercial commitment (Products & Partnerships)."),

    ("products_partnerships", "Products & Partnerships",
     "Market-facing product/service lifecycle and non-ownership collaborations: launches, updates, feature releases, certifications; "
     "MoUs, pilots/POCs, technology partnerships, co-marketing, distribution partnerships without equity change. "
     "Signals: ‘launch’, ‘rollout’, ‘pilot with’, ‘integration’, ‘certified by’, ‘beta/GA’. "
     "Excludes: JV/equity stake changes (M&A), confirmed revenue orders (Orders & Contracts)."),

    ("governance_agm", "Governance & Meetings",
     "AGM/EGM notices, agendas, proposals and outcomes; board composition and nomination committee proposals; changes to bylaws/articles; "
     "auditor appointment/fees; remuneration policies; share authorization mandates granted by shareholders. "
     "Signals: ‘notice of AGM/EGM’, ‘minutes/resolutions adopted’, ‘authorization to issue/repurchase’, ‘articles of association’. "
     "Excludes: actual execution of buybacks or capital raises (those belong in Capital Returns/Capital Markets respectively)."),

    ("management_changes", "Management & Personnel",
     "C-suite and boardroom moves: CEO/CFO/CTO/Chair appointments, resignations, terminations, interim designations, successions; "
     "key leadership at subsidiary level if market-moving. Often includes start dates, handover, search process. "
     "Signals: ‘appointed as’, ‘steps down’, ‘interim’, ‘effective on’, ‘succession’. "
     "Excludes: broad workforce actions (Labor & Workforce)."),

    ("pdmr_transactions", "PDMR / Insider Transactions",
     "Transactions by persons discharging managerial responsibilities (and closely associated persons): purchases, sales, option exercises, "
     "share grants/vesting under LTIP/RSU/PSU, subscription in offerings. Typically MAR-compliant notices with volumes, prices, dates. "
     "Signals: ‘PDMR notification’, ‘Article 19 MAR’, ‘closely associated person’. "
     "Excludes: general employee grants without PDMR involvement (may be Other/Corporate)."),

    ("equity_actions", "Equity Actions (Non-financing)",
     "Mechanical/security identity changes: share splits/reverse splits, bonus issues, ISIN/ticker/name changes, reclassification of share classes, "
     "conversions, reductions of share capital without distribution, technical consolidations. "
     "Signals: ‘split ratio’, ‘par value’, ‘change of ISIN’, ‘ticker change’, ‘reclassification’. "
     "Excludes: actions that raise capital (Capital Markets) or distribute capital (Capital Returns)."),

    ("credit_ratings", "Credit & Ratings",
     "Official ratings actions and outlooks from agencies (S&P, Moody’s, Fitch, etc.) and local equivalents: initiations, upgrades/downgrades, "
     "outlook changes, watch placements, confirmations after review; issuer or specific instrument ratings. "
     "Signals: ‘affirmed at BBB with stable outlook’, ‘placed on watch negative’, ‘downgraded to Ba1’. "
     "Excludes: internal risk scores or lender covenants without agency action (Legal & Regulatory if it’s a breach)."),

    ("legal_regulatory", "Legal & Regulatory",
     "Matters involving courts, regulators, or compliance regimes: litigation filed/settled, investigations, administrative fines/penalties, "
     "regulatory approvals/denials (e.g., market authorizations), consent decrees, significant compliance disclosures (e.g., covenant breaches). "
     "Signals: ‘filed suit’, ‘settled for’, ‘authority approved/denied’, ‘penalty of’, ‘infringement’, ‘consent order’. "
     "Excludes: ordinary-course permits with no materiality (Other/Corporate) unless market-moving."),

    ("labor_workforce", "Labor & Workforce",
     "Company-wide or material site-level workforce developments: layoffs/redundancies, plant/office closures or relocations, strike/union actions, "
     "collective bargaining outcomes, large-scale hiring plans. "
     "Signals: ‘restructuring affecting X FTE’, ‘closure of’, ‘strike action’, ‘collective agreement’. "
     "Excludes: single executive moves (Management & Personnel)."),

    ("incidents_disruptions", "Incidents & Disruptions",
     "Acute negative events impacting operations, safety, or continuity: accidents, fatalities, environmental spills, cyber intrusions/data breaches, "
     "material outages, sanctions/export restrictions, major recalls. Often includes containment, remediation, insurance, and estimated impact. "
     "Signals: ‘incident’, ‘breach’, ‘attack’, ‘fire/explosion’, ‘recall’, ‘sanctions’. "
     "Excludes: reputational controversies without operational impact (Other/Corporate unless legal/regulatory is triggered)."),

    ("other_corporate", "Other / Corporate Update",
     "Legitimate company updates that don’t cleanly fit elsewhere: branding/identity changes (non-security), office moves without workforce impact, "
     "minor awards/certifications, CSR/ESG narrative updates without financing/regulatory hooks, website launches, housekeeping notices. "
     "Use sparingly; prefer a specific category if any material financing, legal, governance, order, listing, or result element is present.")
]
