# Signal Development Roadmap

This document tracks planned signal types for the event-feed-app alert system beyond the currently implemented `guidance_change` signal.

## Status Legend

- 🎯 **Planned** - Documented, not yet started
- 🚧 **In Development** - Active implementation
- ✅ **Complete** - Implemented and deployed

## Implemented Signals

### ✅ Guidance Change (`guidance_change`)

**Status**: Complete  
**Description**: Detects changes to company earnings guidance and financial outlook  
**Triggers**:
- Guidance raises/upgrades
- Guidance cuts/downgrades
- Guidance withdrawals/suspensions
- Guidance initiations for new periods
- Guidance reaffirmations/unchanged statements

**Documentation**: See `EARNINGS_GUIDANCE_ALERT_STRATEGY.md`

---

## Planned Signals

### 🎯 Dividend Change (`dividend_change`)

**Status**: Planned  
**Priority**: High  
**Description**: Detects changes to dividend announcements and policies

**Key Use Cases**:
- **Dividend cancellations** - Company cancels previously announced dividend
- **Dividend suspensions** - Company temporarily suspends dividend payments
- **Dividend cuts** - Reduction in dividend per share vs. prior period
- **Dividend increases** - Raise in dividend per share vs. prior period
- **Dividend policy changes** - Changes to payout ratio, frequency, or policy

**Detection Patterns**:
```yaml
triggers:
  cancellation:
    - "cancel dividend"
    - "dividend cancelled"
    - "withdraw dividend"
    
  suspension:
    - "suspend dividend"
    - "dividend suspended"
    - "discontinue dividend"
    
  cuts:
    - "reduce dividend"
    - "lower dividend"
    - "cut dividend"
    
  increases:
    - "raise dividend"
    - "increase dividend"
    - "higher dividend"
```

**Expected Metrics**:
- `dividend_per_share`: Numeric value in currency
- `change_direction`: cancel | suspend | cut | raise | initiate
- `prior_dividend`: Previous DPS for comparison
- `ex_date`: Ex-dividend date
- `record_date`: Record date
- `payment_date`: Payment date
- `dividend_type`: ordinary | interim | special | final

**Similar To**: Guidance change detection but for dividend events  
**Implementation Notes**:
- Leverage existing plugin architecture from `guidance_change`
- Create `src/event_feed_app/events/dividend/plugin.py`
- Add configuration to `significant_events.yaml`
- High investor relevance for dividend-focused portfolios

---

## Future Signal Ideas

### 🎯 Executive Transitions Database (`executive_changes`)

**Priority**: High  
**Description**: Build database of people mentioned in press releases who are newly appointed or leaving positions, enabling pattern detection and relationship mapping

**Key Use Cases**:
- **Track executive turnover patterns** - Identify companies with high C-suite churn
- **Career trajectory mapping** - Follow executives moving between companies/industries
- **Serial executives** - Identify executives with track records at multiple companies
- **Sudden departures** - Flag unexpected resignations, terminations, retirements
- **Appointment quality signals** - Track if new hires come from stronger/weaker peers
- **Board composition trends** - Monitor independent director appointments and departures

**Detection Patterns**:
```yaml
triggers:
  appointments:
    - "appoint[ed|s] [NAME] as [TITLE]"
    - "named [NAME] [new|as] [TITLE]"
    - "announced [NAME] will join as"
    - "promoted [NAME] to [TITLE]"
    - "[NAME] has been elected"
    
  departures:
    - "[NAME] [is stepping down|will step down|resigned]"
    - "[NAME] [departed|left] the company"
    - "announced [the departure|resignation] of [NAME]"
    - "[NAME] will be leaving"
    - "terminated [NAME]"
    - "[NAME] retired"
```

**Database Schema**:
- `person_name`: Executive/director name
- `company`: Company where event occurred
- `event_type`: appointment | departure | promotion | termination | retirement
- `role_title`: Position title (CEO, CFO, Director, etc.)
- `prior_company`: Previous employer (if disclosed)
- `prior_role`: Previous position
- `event_date`: Date of announcement
- `reason`: stated reason (new opportunity, retirement, personal, restructuring, etc.)
- `interim_flag`: Whether position is interim/temporary

**Signal Opportunities**:
- **Repeat patterns**: Same executive hired by multiple companies (success signal)
- **Quick departures**: Short tenure suggesting problems
- **Industry migration**: Talent moving to/from specific sectors
- **Company clusters**: Multiple executives rotating among same set of companies

**Implementation Notes**:
- Requires NER (Named Entity Recognition) for person names
- Title extraction and normalization needed
- Build historical database over time for pattern recognition
- High value for understanding management quality and stability
- Can be combined with company performance data for predictive insights

---

### 🎯 Company Co-mention Network (`company_relationships`)

**Priority**: High  
**Description**: Build database/graph of companies mentioned together in press releases to identify business relationships, partnerships, and supply chain connections

**Key Use Cases**:
- **Partnership detection** - Identify strategic alliances and collaborations
- **Supply chain mapping** - Track supplier/customer relationships
- **M&A signals** - Companies mentioned together may indicate deal discussions
- **Competitive dynamics** - Track which companies are mentioned as competitors
- **Industry clustering** - Identify ecosystem players and relationships
- **Risk propagation** - Understand which companies are operationally linked

**Detection Patterns**:
```yaml
triggers:
  partnerships:
    - "[COMPANY_A] and [COMPANY_B] announced partnership"
    - "collaboration between [COMPANY_A] and [COMPANY_B]"
    - "joint venture with [COMPANY_B]"
    
  supply_chain:
    - "[COMPANY_A] selected [COMPANY_B] as supplier"
    - "awarded contract to [COMPANY_B]"
    - "customer [COMPANY_B]"
    
  competitive:
    - "competitors including [COMPANY_B]"
    - "outperformed [COMPANY_B]"
    
  transactions:
    - "acquired by [COMPANY_B]"
    - "merger with [COMPANY_B]"
    - "in talks with [COMPANY_B]"
```

**Database Schema**:
- `company_a`: First company
- `company_b`: Second company (or list of co-mentioned companies)
- `relationship_type`: partnership | supplier | customer | competitor | acquirer | target | joint_venture
- `co_mention_count`: Frequency of co-mentions over time
- `relationship_strength`: Strong | medium | weak (based on context and frequency)
- `first_mention_date`: When relationship first detected
- `latest_mention_date`: Most recent co-mention
- `context_keywords`: Key terms describing relationship
- `announcement_type`: Material event or routine mention

**Signal Opportunities**:
- **New relationships**: First-time partnerships with strategic implications
- **Relationship intensity**: Increasing co-mention frequency signals deepening ties
- **Network centrality**: Companies mentioned with many others = ecosystem hubs
- **Broken relationships**: Sudden drop in co-mentions = potential issues
- **Cluster analysis**: Identify industry sub-segments and competitive sets
- **Contagion risk**: Map how problems at one company could spread to partners

**Implementation Notes**:
- Requires robust company name entity recognition
- Need entity disambiguation (IBM vs. IBM Cloud vs. International Business Machines)
- Build graph database for relationship queries
- Track relationship evolution over time
- Can be visualized as network graphs
- Extremely valuable for understanding competitive dynamics and dependencies
- Combine with other signals (e.g., executive moves between linked companies)

---

### 🎯 Insider Trading Activity (`insider_trading`)

**Priority**: High  
**Description**: Track significant insider buying/selling and options trades by executives, directors, and major shareholders as disclosed in press releases and regulatory filings

**Key Use Cases**:
- **Significant insider buying** - Track when executives/directors buy meaningful amounts
- **Clustered insider purchases** - Multiple insiders buying in short timeframe (strong signal)
- **Options exercises and holds** - Executives exercising options and retaining shares
- **10b5-1 plan disclosures** - Automated selling plans vs. discretionary trades
- **Form 4 filing announcements** - Large transactions by insiders
- **Director buying** - Independent directors investing own capital
- **Options grants vs. purchases** - Distinguish compensation from open market buys

**Detection Patterns**:
```yaml
triggers:
  insider_buying:
    - "[NAME].*purchased [NUMBER] shares"
    - "[NAME].*acquired [NUMBER] shares"
    - "insider purchase.*[NUMBER] shares"
    - "[TITLE] bought [NUMBER] shares"
    - "open market purchase"
    
  significant_volume:
    - "purchased.*million shares"
    - "acquired.*[0-9]+% of outstanding"
    - "significant stake"
    - "substantial investment"
    
  options_activity:
    - "exercised [NUMBER] options"
    - "option exercise.*retained shares"
    - "stock option.*[NUMBER] shares"
    - "converted options to shares"
    
  filing_disclosures:
    - "filed Form 4"
    - "SEC filing disclosed"
    - "insider transaction report"
```

**Database Schema**:
- `insider_name`: Name of insider
- `insider_role`: CEO | CFO | Director | COO | Other C-Suite | 10% Owner
- `transaction_type`: purchase | sale | option_exercise | option_grant | gift
- `shares_transacted`: Number of shares bought/sold
- `transaction_value`: Dollar value of transaction
- `share_price`: Price per share
- `shares_owned_after`: Total shares owned after transaction
- `ownership_percentage`: Percentage of company owned
- `transaction_date`: Date of transaction
- `filing_date`: Date disclosed/filed
- `transaction_code`: D (disposition) | A (acquisition) | M (option exercise)
- `form_type`: Form 4 | Form 3 | Form 5
- `plan_type`: 10b5-1 | discretionary | compensation
- `shares_retained_after_exercise`: For option exercises, shares kept vs sold

**Signal Opportunities**:
- **Heavy insider buying**: Multiple large purchases = strong confidence signal
- **Options exercise + hold**: Executives converting and retaining = bullish
- **Director purchases**: Independent directors using own capital = high conviction
- **Clustered buying**: Multiple insiders buying within days = coordinated confidence
- **Buy vs sell ratio**: Track insider sentiment over time
- **Size significance**: Purchases >1% of shares outstanding or >$1M
- **Timing analysis**: Buying before positive news = potential information advantage
- **10b5-1 violations**: Discretionary buys outside plans = stronger signal

**Implementation Notes**:
- Parse transaction amounts, prices, and share counts
- Distinguish between compensation (grants/awards) and open market purchases
- Track cumulative insider activity over rolling periods (30/90 days)
- Flag transactions above materiality thresholds (% of salary, % of ownership)
- Cross-reference with stock price to identify buying near lows
- Particularly valuable when combined with other signals (e.g., insider buying before guidance raise)
- May integrate with SEC EDGAR for comprehensive filing coverage
- Options activity especially informative when executives hold shares post-exercise

**Example High-Value Scenarios**:
1. **CEO buying $5M+ in open market** - Personal capital at risk = high confidence
2. **Multiple directors buying within 1 week** - Coordinated signal of undervaluation
3. **CFO exercises 100K options and retains all shares** - Bullish on stock vs selling for liquidity
4. **10% owner increases stake from 12% to 15%** - Strategic accumulation

---
---

### 🎯 Corporate Events Calendar (`future_events_calendar`)

**Priority**: High  
**Description**: Build a forward-looking calendar database of important future dates extracted from press releases, enabling anticipatory signals and pressure point analysis

**Key Use Cases**:
- **Rights issue lifecycle tracking** - Monitor subscription periods, ex-rights dates, and trading cessation
- **Earnings calendar** - Track announced earnings release dates and guidance updates
- **Shareholder meetings** - AGM/EGM dates, voting deadlines, record dates
- **Debt maturity tracking** - Bond redemptions, covenant test dates, refinancing deadlines
- **Product launches** - Announced release dates for new products/services
- **Regulatory deadlines** - Filing deadlines, license renewals, compliance dates
- **Lock-up expirations** - Post-IPO or secondary offering lock-up end dates
- **Dividend schedule** - Ex-dividend, record, and payment dates
- **Incentive program deadlines** - Stock option vesting dates, performance measurement period ends
- **Competitor events** - Peer earnings dates, product launches for competitive analysis

**Detection Patterns**:
```yaml
triggers:
  rights_issues:
    - "rights issue.*ex-rights date [DATE]"
    - "subscription period.*[START_DATE] to [END_DATE]"
    - "rights will cease trading on [DATE]"
    - "record date for rights [DATE]"
    
  earnings_dates:
    - "will report [Q1|Q2|Q3|Q4|full year] results on [DATE]"
    - "earnings release scheduled for [DATE]"
    - "plans to announce.*on [DATE]"
    
  shareholder_meetings:
    - "annual general meeting.*[DATE]"
    - "AGM scheduled for [DATE]"
    - "shareholder vote on [DATE]"
    - "record date.*[DATE]"
    
  debt_events:
    - "bonds mature on [DATE]"
    - "debt refinancing.*[DATE]"
    - "covenant test date [DATE]"
    
  lock_ups:
    - "lock-up period expires [DATE]"
    - "lock-up ends [DATE]"
    
  incentive_programs:
    - "performance period ends [DATE]"
    - "measurement period.*[END_DATE]"
    - "incentive program.*expires [DATE]"
    - "performance targets.*measured on [DATE]"
    - "vesting date [DATE]"
```

**Database Schema**:
- `company`: Company name/ticker
- `event_type`: rights_issue | earnings | agm | debt_maturity | lock_up | dividend | product_launch | regulatory
- `event_subtype`: ex_rights_date | subscription_end | trading_cessation | payment_date | etc.
- `event_date`: Future date of the event
- `announcement_date`: When the event was announced
- `event_description`: Brief description of the event
- `key_terms`: Important terms (e.g., subscription price, rights ratio)
- `related_events`: Links to related calendar entries (e.g., subscription period start/end)
- `market_impact_likelihood`: high | medium | low
- `status`: upcoming | completed | cancelled

**Signal Opportunities**:
- **Rights issue pressure relief**: When rights cease trading, selling pressure often disappears - BUY signal
- **Pre-earnings positioning**: Track companies approaching earnings dates for volatility plays
- **Lock-up expiration risk**: Anticipate potential selling pressure when lock-ups expire
- **Debt maturity refinancing risk**: Monitor companies approaching debt maturities without announcements
- **Dividend capture strategies**: Ex-dividend date calendar for income strategies
- **AGM catalysts**: Shareholder activism opportunities around meeting dates
- **Competitor calendar clustering**: When multiple peers report earnings in same week
- **Event gaps**: Missing expected events (e.g., no earnings date announced) = potential issue
- **Incentive program manipulation**: Watch for stock price clustering/pumping before performance measurement dates - management may try to drive stock up to hit targets before period ends

**Implementation Notes**:
- Requires date extraction and normalization (various date formats)
- Build calendar that updates as new announcements occur
- Track event completion and compare announced vs actual dates
- Generate alerts X days before important events
- Cross-reference with market data for post-event analysis
- Particularly valuable for event-driven trading strategies
- Can generate daily/weekly "upcoming events" reports
- Combine with price action to identify pre-event positioning

**Example Use Case - Rights Issue Lifecycle**:
1. **Announcement**: Rights issue announced with subscription period and ex-rights date
2. **Subscription period**: Track participation rate if disclosed
3. **Ex-rights date**: Stock trades without rights attached
4. **Rights trading cessation**: Rights stop trading separately - **KEY SIGNAL**: selling pressure from rights arbitrage disappears, often marks bottom
5. **Post-event**: Monitor stock recovery post-pressure relief

---

### 🎯 Management Changes

**Priority**: Medium  
**Description**: C-suite and board changes  
**Triggers**: CEO/CFO appointments, resignations, terminations

**Key Use Cases**:
- Track executive leadership transitions
- Monitor board composition changes
- Detect sudden departures or terminations
- Identify interim appointments

### 🎯 Product Recalls (`product_recall`)

**Priority**: High  
**Description**: Detect major product recalls, particularly for medical device, pharmaceutical, automotive, and consumer product companies where recalls have significant financial and reputational impact

**Key Use Cases**:
- **Medical device recalls** - FDA Class I/II recalls for safety issues (highest priority)
- **Pharmaceutical recalls** - Drug recalls due to contamination, labeling, or efficacy issues
- **Automotive recalls** - Vehicle safety recalls affecting large volumes
- **Consumer product recalls** - CPSC recalls for safety hazards
- **Food recalls** - FDA/USDA recalls for contamination or allergens
- **Voluntary vs. mandatory recalls** - Distinguish severity and regulatory pressure
- **Recall expansion** - Track when initial recalls are expanded in scope

**Detection Patterns**:
```yaml
triggers:
  recall_announcements:
    - "recall[ing|ed] [NUMBER] [units|devices|vehicles]"
    - "voluntary recall"
    - "FDA.*recall"
    - "CPSC.*recall"
    - "safety recall"
    - "product recall"
    
  medical_device:
    - "Class [I|II|III] recall"
    - "medical device recall"
    - "FDA clearance.*withdrawn"
    - "safety alert"
    
  severity_indicators:
    - "serious health consequences"
    - "death or serious injury"
    - "safety hazard"
    - "contamination"
    - "manufacturing defect"
    
  scope:
    - "expanding recall"
    - "recall affects [NUMBER] units"
    - "worldwide recall"
    - "all lots recalled"
```

**Database Schema**:
- `product_name`: Name of recalled product
- `product_category`: medical_device | pharmaceutical | automotive | consumer | food
- `recall_reason`: Safety hazard, contamination, defect, labeling error, etc.
- `recall_class`: Class I | Class II | Class III (for FDA recalls)
- `recall_type`: voluntary | mandatory | market_withdrawal
- `units_affected`: Number of units/lots recalled
- `geographic_scope`: US only | worldwide | specific regions
- `regulatory_agency`: FDA | CPSC | NHTSA | USDA | etc.
- `health_risk_level`: critical | serious | moderate | low
- `recall_initiation_date`: When recall began
- `announcement_date`: When publicly announced
- `estimated_financial_impact`: Dollar cost if disclosed
- `recall_expansion`: Boolean flag if recall was later expanded

**Signal Opportunities**:
- **Class I medical device recalls**: Highest severity - potential for significant stock decline
- **Recall expansion signals**: Initial recall expanded = worse than first disclosed
- **Multiple recalls in short period**: Pattern of quality issues = systemic problem
- **Competitor advantage**: Recall creates market share opportunity for competitors
- **Recall cost disclosure**: Track companies that disclose financial impact vs. those that don't
- **Repeat offenders**: Companies with history of recalls in same product line
- **Post-recall monitoring**: Stock often overshoots on bad news, potential buy opportunity after stabilization

**Implementation Notes**:
- Particularly critical for medtech companies (Class I recalls can destroy stock)
- Parse unit counts and scope to assess materiality
- Track if company provides financial impact estimates
- Monitor for follow-up announcements (recall expansions, lawsuits)
- Cross-reference with regulatory databases (FDA, CPSC) for validation
- Severity varies dramatically by industry and class
- Medical device recalls often have outsized impact vs. consumer product recalls

**Example High-Impact Scenarios**:
1. **Class I FDA recall for implantable device** - Immediate trading halt risk, stock down 20-50%
2. **Recall affecting >50% of product line revenue** - Material earnings impact
3. **Second recall of same product within 6 months** - Quality control failure pattern
4. **Worldwide recall + regulatory investigation** - Potential for additional penalties

---

### 🎯 Regulatory Investigations (`regulatory_investigation`)

**Priority**: High  
**Description**: Detect when major regulatory bodies announce investigations into fraud, corruption, money laundering, terrorist financing, sanctions violations, or other serious misconduct

**Key Use Cases**:
- **Fraud investigations** - SEC, DOJ, FTC investigating fraudulent practices
- **Anti-money laundering (AML) probes** - FinCEN, banking regulators investigating AML failures
- **Corruption/bribery** - DOJ FCPA investigations, international corruption probes
- **Terrorist financing** - OFAC, Treasury investigations into terror finance links
- **Sanctions violations** - OFAC sanctions breach investigations
- **Market manipulation** - SEC/FINRA investigating pump-and-dump, insider trading
- **Accounting fraud** - SEC investigating financial statement irregularities
- **Regulatory enforcement actions** - Consent orders, fines, penalties announced

**Detection Patterns**:
```yaml
triggers:
  investigation_announcements:
    - "SEC.*investigat[ing|ion]"
    - "DOJ.*probe"
    - "regulatory investigation"
    - "under investigation by [AGENCY]"
    - "received subpoena from"
    - "cooperating with authorities"
    
  fraud_corruption:
    - "fraud investigation"
    - "accounting irregularities"
    - "FCPA.*investigation"
    - "bribery.*probe"
    - "corruption.*investigation"
    
  aml_sanctions:
    - "anti-money laundering.*investigation"
    - "AML.*violations"
    - "sanctions violation"
    - "OFAC.*investigation"
    - "terrorist financing"
    - "FinCEN.*action"
    
  enforcement:
    - "consent order"
    - "cease and desist"
    - "civil penalty.*[AMOUNT]"
    - "settlement.*[AMOUNT]"
    - "enforcement action"
    
  severity_indicators:
    - "criminal investigation"
    - "class action lawsuit"
    - "shareholder lawsuit"
    - "whistleblower complaint"
```

**Database Schema**:
- `investigating_agency`: SEC | DOJ | FinCEN | OFAC | FTC | FDA | State AG | etc.
- `investigation_type`: fraud | aml | corruption | sanctions | market_manipulation | accounting
- `allegation_summary`: Brief description of alleged misconduct
- `investigation_status`: preliminary | formal | settled | ongoing
- `announcement_date`: When investigation disclosed
- `criminal_vs_civil`: criminal | civil | both
- `potential_penalties`: Estimated or disclosed penalty range
- `management_response`: cooperating | disputing | settled
- `prior_violations`: History of similar issues (if any)
- `international_scope`: Domestic only | international
- `related_parties`: Other entities or individuals under investigation

**Signal Opportunities**:
- **Criminal investigations**: Far more serious than civil - potential for executive liability
- **Multiple agency involvement**: DOJ + SEC = higher severity, potential criminal charges
- **Repeat offenders**: Companies with history of compliance failures face harsher penalties
- **Whistleblower involvement**: Often indicates substantial evidence exists
- **Class action following investigation**: Investigation triggers lawsuits = compounding risk
- **Settlement amounts**: Track disclosed vs. eventual settlement to gauge severity
- **Stock response patterns**: Often sharp initial decline, then gradual recovery (or further decline if expanded)
- **Leadership changes post-announcement**: CEO/CFO departures following investigation = serious

**Implementation Notes**:
- Parse agency names and investigation types carefully
- Track investigation lifecycle from announcement to resolution
- Monitor for updates, expansions, or additional charges
- Cross-reference with enforcement databases (SEC, DOJ)
- Distinguish between voluntary disclosure vs. agency announcement
- Financial institutions particularly vulnerable to AML/sanctions probes
- International investigations (FCPA, EU regulators) often involve multiple years

**Example High-Impact Scenarios**:
1. **DOJ criminal investigation + SEC civil probe** - Dual track investigation = serious
2. **Money laundering investigation at major bank** - Multi-billion dollar penalty risk
3. **FCPA bribery probe** - Can result in >$500M settlements, executive prosecutions
4. **Accounting fraud with class action lawsuit** - Stock down 30-60%, potential delisting
5. **OFAC sanctions violation for defense/energy companies** - Business model disruption risk

**Risk Indicators**:
- **Immediate risk**: Stock typically drops 10-30% on investigation announcement
- **Long-term risk**: Overhang until resolution (can be years)
- **Execution risk**: Management distraction, legal costs, business disruption
- **Reputational damage**: Customer/partner reluctance, contract losses
- **Regulatory restrictions**: May lose licenses, certifications, or operating permissions

---

### 🎯 Production Facility Delays (`facility_delay`)

**Priority**: Medium  
**Description**: Postponements and delays in major production facilities, plants, and capital projects  
**Triggers**: 
- Construction/commissioning delays
- Project timeline extensions
- Facility startup postponements
- Production ramp-up delays
- Plant closure or mothballing announcements

**Key Use Cases**:
- Track delays in major capital projects (factories, mines, power plants)
- Monitor production facility startups and commissioning
- Detect postponements in expansion projects
- Identify supply chain impact from facility delays

**Similar To**: Guidance change but for operational milestones rather than financial metrics

---

### 🎯 Significant Ownership Stakes (`ownership_stake`)

**Priority**: High  
**Description**: Detects when strategic/industrial buyers with operations acquire significant ownership stakes (>5%) in a company, signaling potential acquisition interest

**Key Use Cases**:
- **Strategic buyer entry** - Competitors or industry players taking positions
- **Industrial consolidation signals** - Operating companies acquiring stakes in peers
- **Pre-acquisition positioning** - Strategic investors building toehold positions
- **Cross-border industrial investments** - Foreign industrial players entering markets
- **Vertical integration moves** - Suppliers or customers taking ownership stakes

**Detection Patterns**:
```yaml
triggers:
  strategic_stakes:
    - "strategic investor"
    - "industrial partner"
    - "operating company"
    - "competitor [acquired|purchased|disclosed]"
    
  ownership_disclosure:
    - "disclosed [a-z ]+ stake"
    - "filed [13D|13G|Schedule 13]"
    - "acquired [0-9]+% stake"
    - "reported ownership of"
    
  strategic_intent:
    - "strategic investment"
    - "commercial partnership"
    - "potential acquisition"
    - "industry consolidation"
```

**Expected Metrics**:
- `ownership_percentage`: Percentage of shares owned (e.g., 5.1%, 9.8%)
- `buyer_name`: Name of the strategic/industrial buyer
- `buyer_type`: strategic | industrial | competitor | supplier | customer
- `buyer_industry`: Industry/sector of the acquiring entity
- `filing_type`: 13D | 13G | Form 4 | other
- `stake_size_shares`: Number of shares held
- `intent`: strategic | acquisition_interest | partnership
- `prior_ownership`: Previous ownership percentage (if disclosed)

**Implementation Notes**:
- Focus on operating companies (not financial investors/funds)
- High signal value for potential M&A catalyst events
- Consider cross-referencing buyer's industry with target company
- May require integration with SEC filing data for validation
- Particularly valuable for identifying early-stage acquisition interest

**Similar To**: Event-driven M&A signals rather than financial metrics

---

## Implementation Checklist

When implementing a new signal type:

- [ ] Add configuration to `src/event_feed_app/configs/significant_events.yaml`
- [ ] Create plugin in `src/event_feed_app/events/{signal_name}/plugin.py`
- [ ] Register in `src/event_feed_app/events/registry.py`
- [ ] Create GCS partition `signal_type={signal_name}/`
- [ ] Add unit tests in `tests/{signal_name}_tests/`
- [ ] Update web viewer for signal-specific fields
- [ ] Add to `docs/FEEDBACK_SYSTEM.md` schema
- [ ] Document detection strategy (like `EARNINGS_GUIDANCE_ALERT_STRATEGY.md`)
- [ ] Test end-to-end with real press releases
- [ ] Deploy and monitor initial performance

---

## Contributing

To propose a new signal type:

1. Add entry to "Future Signal Ideas" section above
2. Document key use cases and triggers
3. Estimate priority based on investor value
4. Consider implementation complexity vs. existing infrastructure
