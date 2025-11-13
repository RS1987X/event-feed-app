# Guidance Change Signal: Scope & Edge Cases

**Version:** 1.0  
**Last Updated:** 2025-11-13  
**Owner:** Event Feed Alert System

---

## 1. Core Objective

The `guidance_change` signal aims to capture **material changes in forward-looking financial or operational expectations** communicated by companies in press releases, specifically:

### What We Want to Capture
- **Explicit guidance revisions** (raise/lower/narrow/widen)
- **Guidance withdrawals or suspensions** (always material)
- **New guidance initiation** for upcoming periods
- **Quantitative changes** in key metrics:
  - Revenue / Net Sales / Omsättning
  - EBITDA / EBIT / EBITA
  - Operating Margin / EBIT Margin / Rörelseresultat
  - Organic Growth / Like-for-Like (LFL)
  - Profit Warnings (always material)

### Materiality Thresholds (Current)
```yaml
withdraw_or_suspend: always     # Always alert
initiate_new_period: always     # Always alert
delta_pct_level_min: 5%         # Revenue/EBITDA absolute % change
delta_pp_growth_min: 3pp        # Growth rate change (percentage points)
delta_pp_margin_min: 1.5pp      # Margin change (percentage points)
```

---

## 2. Current Detection Mechanisms

### 2.1 Pattern-Based Detection

**Guidance Change Verbs (EN):**
- raise, lower, revise, update
- withdraw, suspend
- narrow, widen

**Guidance Change Verbs (SV):**
- höjer, sänker, reviderar, uppdaterar
- drar tillbaka, upphäver
- snävar, vidgar

**Guidance Markers:**
- "guidance", "outlook", "forecast"
- "full-year guidance", "FY guidance", "guidance range"
- Swedish: "prognos", "utsikter"

**Proximity Rules:**
- Verb + marker within 10 tokens, same sentence
- Example: "raises full-year guidance" ✅
- Example: "guidance. We raised targets." ❌ (cross-sentence not allowed)

### 2.2 Numeric Extraction

**Supported Formats:**
- Percentage ranges: `15-18%`, `≈15%`, `around 20%`
- Currency amounts: `SEK 500M`, `$2.5B`, `€100m`
- Proximity to metrics: within 8 tokens of financial metric keywords

**Change Detection:**
- Compares current guidance to stored `guidance_prior_state`
- Calculates delta (absolute or percentage point)
- Filters by materiality thresholds

### 2.3 Unchanged Guidance Detection

**Verbal Markers (EN):**
- reaffirm, maintain, keep, leave (unchanged)

**Verbal Markers (SV):**
- bekräftar, bibehåller, upprepar

**Token Markers:**
- "unchanged", "in line", "no change", "as previously"
- Swedish: "oförändrad", "i linje med", "som tidigare"

---

## 3. Edge Cases & Challenges

### 3.1 Ambiguous Language

**Challenge:** Vague forward-looking statements that don't constitute formal guidance.

| Example | Classification | Reasoning |
|---------|---------------|-----------|
| "We expect strong growth this year" | ❌ Not guidance | No specific metric or range |
| "We target 15% EBIT margin" | ✅ Guidance | Quantitative target |
| "Outlook remains positive" | ❌ Not guidance | Qualitative, non-specific |
| "We see EBITDA margin of 18-20%" | ✅ Guidance | Range with specific metric |

**Current Limitation:** Requires both a guidance marker AND numeric value or change verb. Purely qualitative outlook statements are missed.

### 3.2 Reaffirmation vs. Change

**Challenge:** Distinguishing true changes from restatements.

| Example | Should Detect? | Current Behavior |
|---------|----------------|------------------|
| "We maintain our guidance of 15% margin" | No (unchanged) | ✅ Correctly filtered |
| "We narrow guidance to 15-16% from 14-17%" | Yes (precision change) | ✅ Detected via "narrow" verb |
| "Guidance unchanged at 15%" | No | ✅ Filtered by unchanged markers |
| "We reiterate 15% margin guidance" | No | ⚠️ May trigger if "reiterate" not in unchanged list |

**Risk:** False positives from earnings reports that repeat Q1 guidance given in January.

### 3.3 Conditional or Scenario-Based Guidance

**Challenge:** Guidance tied to conditions or multiple scenarios.

| Example | Classification | Current Handling |
|---------|---------------|------------------|
| "If market conditions improve, we expect 10% growth" | Uncertain guidance | ✅ Extracted but low significance |
| "Base case: 15% margin; optimistic: 18%" | Multiple scenarios | ⚠️ May extract only one value |
| "Subject to no material disruptions" | Qualified guidance | ✅ Extracted (qualifier ignored) |

**Gap:** No modeling of uncertainty or scenario conditionality.

### 3.4 Metric Ambiguity

**Challenge:** Different metrics with similar names or implied metrics.

| Example | Issue | Current Handling |
|---------|-------|------------------|
| "margin improves to 15%" | Which margin? | ⚠️ Default to operating_margin |
| "EBIT margin 12%" vs "EBITA margin 12%" | Different metrics | ✅ Distinct FIN_METRICS entries |
| "growth of 10%" | Organic or reported? | ⚠️ Default to organic_growth |
| "sales increase 5%" | Level or growth? | ⚠️ Interpreted as growth |

**Gap:** Context-dependent metric disambiguation not implemented.

### 3.5 Period Mismatches

**Challenge:** Guidance for different time horizons.

| Example | Period | Handling |
|---------|--------|----------|
| "Q2 EBIT €50M" | Current quarter | ✅ Detected |
| "FY2025 revenue SEK 10B" | Full year | ✅ Detected |
| "2025-2027 CAGR 8%" | Multi-year | ⚠️ Extracted but period not normalized |
| "Next 12 months margin 15%" | Rolling period | ⚠️ Period extraction incomplete |

**Gap:** No period normalization or comparison across different guidance horizons.

### 3.6 Profit Warnings

**Challenge:** Always material but often lack numeric specifics.

| Example | Current Behavior | Desired |
|---------|-----------------|---------|
| "We issue a profit warning" | ✅ Trigger via "profit warning" pattern | ✅ Correct |
| "Results will be materially below expectations" | ⚠️ May miss without "warning" keyword | Should detect |
| "Significant impairment charge expected" | ✅ Trigger via "impairment" pattern | ✅ Correct |

**Gap:** Synonym coverage for adverse guidance could be broader.

### 3.7 Language & Translation Issues

**Challenge:** Swedish press releases with mixed language or translation artifacts.

| Example | Issue | Current Handling |
|---------|-------|------------------|
| "Prognosen höjs till SEK 500M" | Pure Swedish | ✅ Detected |
| "Guidance höjs till 15%" | Mixed EN/SV | ⚠️ May work if markers align |
| "We expect omsättning of SEK 1B" | Metric in SV, context EN | ✅ FIN_METRICS covers both |

**Gap:** Full document language detection and routing to language-specific rules not implemented.

### 3.8 Comparative Statements (Competitor Guidance)

**Challenge:** Distinguishing own-company guidance from competitor references.

| Example | Should Alert? | Current Behavior |
|---------|--------------|------------------|
| "We raise our guidance to 15%" | Yes | ✅ Detected |
| "Competitor X raised their guidance" | No | ⚠️ May trigger if proximity rules match |
| "Industry outlook improves" | No | ❌ Not detected (no specific guidance) |

**Gap:** No entity resolution to filter out third-party guidance mentions.

### 3.9 Non-Financial Operational Guidance

**Challenge:** Operational KPIs (customer growth, production capacity) vs. financial metrics.

| Example | Current Handling | Should We Track? |
|---------|-----------------|------------------|
| "We expect 50,000 new customers" | ❌ Not detected (not in FIN_METRICS) | Maybe (sector-specific) |
| "Production capacity to increase 30%" | ❌ Not detected | Maybe (for manufacturing) |
| "Subscriber growth 10-12%" | ⚠️ Detected as "growth" if near % | Potentially (for SaaS/telco) |

**Gap:** No operational metrics beyond financial ratios.

### 3.10 Class Action Lawsuits Referencing Guidance

**Challenge:** Legal announcements that discuss past failures to disclose guidance changes (not new guidance).

| Example | Should Alert? | Current Behavior |
|---------|--------------|------------------|
| "Class action alleges company failed to disclose revenue guidance reduction in Q2" | No (historical reference) | ⚠️ May trigger on "guidance reduction" |
| "Lawsuit claims undisclosed profit warning" | No (legal context, not company statement) | ⚠️ May trigger on "profit warning" |
| "Settlement regarding missed EBITDA targets" | No (litigation, not guidance update) | ⚠️ May trigger if numbers mentioned |

**Risk:** False positives from legal boilerplate discussing historical guidance failures.

**Gap:** No detection of legal/litigation context to suppress guidance extraction from third-party claims.

### 3.11 M&A Announcements with Target Guidance

**Challenge:** Acquirer press releases mentioning target company's expected financials (not acquirer's own guidance).

| Example | Should Alert? | Current Behavior |
|---------|--------------|------------------|
| "Target company projects €50M EBITDA for FY2025" | No (target's guidance, not acquirer's) | ⚠️ May trigger if proximity rules match |
| "The acquisition target expects 15% revenue growth" | No (not our company) | ⚠️ May trigger on "expects" + metric |
| "Combined entity guidance: SEK 2B revenue" | Yes (new combined guidance) | ✅ Should detect (post-merger guidance) |

**Risk:** False positives from M&A announcements where target company guidance is mentioned but not relevant to the acquirer's own outlook.

**Gap:** No entity resolution to distinguish acquirer vs. target guidance. Combined/pro-forma guidance post-merger is legitimate but needs distinct handling.

### 3.12 Boilerplate & Operational Use of "Guidance"

**Challenge:** Non-financial uses of the word "guidance" (regulatory guidance, strategic guidance, corporate governance).

| Example | Should Alert? | Current Behavior |
|---------|--------------|------------------|
| "In accordance with regulatory guidance from the FSA" | No (regulatory context) | ⚠️ Unlikely if no financial metrics nearby |
| "Board provides strategic guidance to management" | No (governance context) | ⚠️ Unlikely if no numbers |
| "Under the guidance of our new CEO" | No (leadership context) | ⚠️ Unlikely if no proximity to metrics |
| "Following IFRS guidance for revenue recognition" | No (accounting standards) | ⚠️ Unlikely if no forward-looking verb |

**Risk:** Low false positive rate due to proximity requirements, but edge cases exist.

**Gap:** No semantic filtering for non-financial senses of "guidance". Current reliance on metric proximity works but isn't robust.

---

## 4. Current Implementation Strengths

✅ **Multi-language support** (EN/SV with extensible patterns)  
✅ **Materiality filtering** (thresholds prevent noise)  
✅ **State tracking** (`guidance_prior_state` for change detection)  
✅ **Unchanged guidance filtering** (reduces false positives)  
✅ **Proximity-based extraction** (reduces spurious matches)  
✅ **Approximate value handling** (`≈15%`, `around 20%`)  
✅ **Profit warning triggers** (always-on alerts)  

---

## 5. Known Limitations & Improvement Opportunities

### 5.1 Missing Context
- **Limitation:** No consideration of who is speaking (CEO quote vs. analyst commentary in PR)
- **Impact:** Potential false positives from third-party statements embedded in releases
- **Fix:** Named entity recognition + speaker attribution

### 5.2 Temporal Reasoning
- **Limitation:** No understanding of guidance update frequency (monthly vs. annual)
- **Impact:** May alert on routine updates that aren't newsworthy
- **Fix:** Time-decay significance scoring based on last update

### 5.3 Sector-Specific Metrics
- **Limitation:** Generic FIN_METRICS don't cover biotech (trial milestones), real estate (NOI), SaaS (ARR)
- **Impact:** Misses sector-critical guidance
- **Fix:** Configurable metric dictionaries per industry taxonomy

### 5.4 Magnitude vs. Direction
- **Limitation:** 1pp margin increase treated equally whether from 2%→3% or 15%→16%
- **Impact:** Relative magnitude not captured
- **Fix:** Proportional significance scoring (e.g., 50% relative change)

### 5.5 Multi-Sentence Context
- **Limitation:** `allow_cross_sentence: false` means "We raise guidance. The new range is 15-18%" may fail
- **Impact:** Incomplete extractions when guidance spans sentences
- **Fix:** Configurable cross-sentence window for specific verbs

### 5.6 Uncertainty Quantification
- **Limitation:** No differentiation between "definitely 15%" and "possibly 15% depending on X"
- **Impact:** Equal significance assigned to firm and contingent guidance
- **Fix:** Hedge word detection + significance downweighting

---

## 6. Recommended Enhancements (Priority Order)

### High Priority
1. **Add "reiterate"/"upprepar" to unchanged markers** (quick win to reduce FPs)
2. **Cross-sentence guidance extraction** for specific verbs (raise/lower/revise)
3. **Entity resolution** to filter competitor guidance mentions
4. **Legal/litigation context detection** to suppress class action lawsuit references
5. **M&A context detection** to distinguish acquirer vs. target guidance
6. **Period normalization** (Q1/Q2/FY/multi-year) for accurate comparisons

### Medium Priority
5. **Relative magnitude scoring** (1pp from 2% more significant than from 15%)
6. **Hedge word detection** ("expect", "subject to", "if") to downweight uncertain guidance
7. **Operational metrics** for key sectors (subscribers, production, customers)

### Low Priority
8. **Speaker attribution** (CEO vs. analyst)
9. **Language routing** (detect SV vs. EN and apply language-specific rules exclusively)
10. **Machine learning augmentation** for ambiguous cases (train on labeled guidance/non-guidance corpus)

---

## 7. Testing & Validation Strategy

### Unit Tests
- ✅ Pattern matching (verb + marker proximity)
- ✅ Numeric extraction (ranges, approximations)
- ✅ Unchanged detection
- ⚠️ Need: Edge case coverage (cross-sentence, competitor mentions, conditional)

### Integration Tests
- ✅ End-to-end with real press releases
- ⚠️ Need: Regression suite with labeled FP/FN cases

### Metrics to Track
- **Precision:** % of alerts that are truly material guidance changes
- **Recall:** % of known guidance changes detected
- **False Positive Rate:** Non-guidance alerts / total alerts
- **Latency:** Time from publication to alert delivery

---

## 8. Example Scenarios (Annotated)

### Scenario 1: Clear Guidance Raise
**Text:** "We raise our full-year revenue guidance to SEK 5.2-5.5B from SEK 5.0-5.3B"  
**Detection:** ✅ Change verb + marker + numeric delta  
**Significance:** High (10% increase in midpoint)  
**Alert:** ✅ Yes

### Scenario 2: Reaffirmation
**Text:** "We reaffirm our Q4 EBIT margin guidance of 12%"  
**Detection:** ⚠️ "reaffirm" should be in unchanged markers  
**Significance:** N/A (no change)  
**Alert:** ❌ Should be suppressed (needs fix)

### Scenario 3: Vague Outlook
**Text:** "We expect continued strong performance in the second half"  
**Detection:** ❌ No numeric guidance or formal marker  
**Significance:** N/A  
**Alert:** ❌ Correct (not actionable)

### Scenario 4: Conditional Guidance
**Text:** "Assuming no further lockdowns, we target 15% EBITDA margin"  
**Detection:** ✅ Numeric + metric + target verb  
**Significance:** Medium (conditional)  
**Alert:** ⚠️ Yes, but significance should be lower (hedge not modeled)

### Scenario 5: Profit Warning
**Text:** "We issue a profit warning due to supply chain disruptions"  
**Detection:** ✅ Trigger pattern "profit warning"  
**Significance:** Always high  
**Alert:** ✅ Yes

### Scenario 6: Competitor Reference
**Text:** "Competitor ABC raised their guidance, we maintain ours at 10%"  
**Detection:** ⚠️ May trigger on "raised...guidance" before "we maintain"  
**Significance:** Should be zero (not our guidance)  
**Alert:** ❌ Should be suppressed (needs entity resolution)

### Scenario 7: Cross-Sentence Guidance
**Text:** "We revise our outlook. The new EBIT margin range is 14-16%."  
**Detection:** ⚠️ Likely missed (cross-sentence disabled)  
**Significance:** High (if detected)  
**Alert:** ❌ Currently missed (needs cross-sentence support)

### Scenario 8: Class Action Lawsuit Reference
**Text:** "Class action lawsuit alleges failure to disclose Q2 revenue guidance reduction from SEK 5B to SEK 4.5B"  
**Detection:** ⚠️ May trigger on "guidance reduction" + numbers  
**Significance:** Zero (historical legal claim, not new guidance)  
**Alert:** ❌ Should be suppressed (needs legal context filter)

### Scenario 9: M&A Target Guidance
**Text:** "The acquisition target projects 2025 EBITDA of €120M, representing 18% margin"  
**Detection:** ⚠️ May trigger on "projects" + metric + numbers  
**Significance:** Zero (target company, not acquirer)  
**Alert:** ❌ Should be suppressed (needs entity resolution)

### Scenario 10: Boilerplate Regulatory Guidance
**Text:** "In compliance with IFRS guidance, we recognize revenue over time"  
**Detection:** ❌ No financial metrics or forward-looking verbs nearby  
**Significance:** N/A  
**Alert:** ❌ Correct (not financial guidance)

---

## 9. Success Criteria

### For Alert Recipients
- **0 false positives** from routine unchanged guidance restatements
- **< 5% miss rate** on material guidance changes in coverage universe
- **< 15 min latency** from publication to Telegram alert

### For System Operators
- **> 95% precision** on validation set
- **Explainable** every alert (evidence spans + delta shown)
- **Tunable** thresholds per user preference

---

## 10. Open Questions for Review

1. **Should we alert on guidance narrowing without magnitude change?**  
   Example: "15-20%" → "17-18%" (same midpoint, tighter range)  
   Current: ✅ Yes via "narrow" verb  
   Question: Is this always material?

2. **How to handle multi-year guidance vs. annual?**  
   Example: "2025-2027 CAGR 8%" vs. "FY2025 growth 8%"  
   Current: Both extracted, but not distinguished  
   Question: Should we normalize to annual or flag multi-year separately?

3. **Operational metrics: sector-specific opt-in or universal?**  
   Example: "Subscriber growth 10%" (material for SaaS, less for manufacturing)  
   Question: Add operational metrics with sector filtering, or keep financial-only?

4. **Hedge word weighting: binary or graduated?**  
   Example: "expect" vs. "commit to" vs. "subject to regulatory approval"  
   Current: No differentiation  
   Question: Simple flag or confidence score (0.5 for hedged, 1.0 for firm)?

5. **Competitor guidance: filter always or make configurable?**  
   Some users may want competitor guidance changes as separate signal  
   Question: Hard-coded filter or user preference?

---

**Next Steps:**
- [ ] Review this spec with stakeholders
- [ ] Prioritize enhancements (quick wins vs. long-term improvements)
- [ ] Create regression test suite with edge cases
- [ ] Document decision log for open questions
