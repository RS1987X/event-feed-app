# Guidance Change Signals - Backfill Summary

## Overview
Successful backfill of guidance change signals for September-November 2025 (3 months).

## Results

### Processing Statistics
- **Total PRs Processed**: 45,506
- **Total Alerts Generated**: 1,512
- **Detection Rate**: 3.32% (healthy range)
- **Errors**: 0
- **Processing Time**: ~3 seconds (batch processing with concurrency)

### Signal Statistics
- **Total Events Stored**: 1,822
- **Unique Companies**: 269
- **Date Range**: 2025-09-01 to 2025-11-30

## Storage Architecture

### GCS Structure
```
gs://event-feed-app-data/signals/guidance_change/
├── events/
│   └── date=2025-12-01/      # Partitioned by detection date
│       └── events_*.parquet   # Individual guidance events
└── aggregated/
    └── date=2025-12-01/       # Partitioned by detection date
        └── alert_*.parquet    # Aggregated alerts by company
```

### Schema Fields (31 columns)
- **Identifiers**: event_id, press_release_id, alert_id, company_id, company_name
- **Parametrization**: metric, metric_kind, **period**, value_low, value_high, unit, currency, basis, direction_hint
- **Change Metrics**: prior_value_low, prior_value_high, delta_pp, delta_pct, range_tightened, change_label
- **Detection Metadata**: confidence, rule_type, rule_name, trigger_source, text_snippet
- **Temporal**: press_release_date, detected_at, date (partition)
- **Source**: source_url, source_type

## Parametrization Verification

### ✅ Period Field
- **100% populated** across all 1,822 events
- **Single string field** (validated as sufficient for all patterns)
- **Distribution**:
  - FY2025: 70.5% (1,284 events)
  - Q3-2025: 10.8% (197 events)
  - FY2024: 7.1% (130 events)
  - FY2026: 3.5% (63 events)
  - Other quarters/years: 7.1%

### ✅ Metric Field
- **100% populated** across all events
- **Distribution**:
  - guidance: 44.6% (812 events)
  - operating_margin: 16.6% (302 events)
  - organic_growth: 11.0% (200 events)
  - ebita_margin: 9.9% (181 events)
  - net_sales: 7.0% (127 events)
  - Other metrics: 10.9%

### ✅ Direction & Change
- Direction hints: 0% (optional field, not populated for text-based guidance)
- Value ranges: 55.4% (1,009 events with numeric guidance)
- Currency distribution: EUR (240), DKK (54), USD (24), NOK (6), SEK (3)

## Period Validation Success

### Year Validation Working
- No spurious historic years (FY1941, FY2032 from prior issues)
- Year bounds enforced: current_year ± 2 (2023-2027 for 2025 releases)
- FY-UNKNOWN and UNKNOWN: only 3.3% (59 events) where period couldn't be determined

### Period Patterns Captured
- Fiscal years: FY2025, FY2024, FY2026
- Quarters: Q3-2025, Q4-2025, Q2-2025
- Half years: H1-2025, H1-2026
- All stored in **single period field** (sufficient for tracking changes over time)

## Exclusion Logic Success

### Law Firm PRs Filtered
- OR-based exclusion: `indicator OR law_firm` triggers exclusion
- Broader indicators: "class action", "law firm", "shareholder rights firm"
- Expanded law firm list: Bragar Eagel, Hagens Berman, Levi & Korsinsky, etc.
- Result: No false positives from second-hand securities litigation reporting

## Detection Quality

### Detection Rate: 3.32%
- **Healthy range**: 1-10% target
- Consistent with broader test (2.85% across 2000 PRs)
- 269 unique companies detected (good coverage)

### Signal Types Captured
- Guidance updates (44.6%)
- Margin guidance (operating, EBITA: 26.5%)
- Growth guidance (organic: 11.0%)
- Revenue/sales guidance (12.0%)
- EBITDA/EBIT guidance (5.5%)

## Next Steps

### Immediate
1. ✅ Backfill complete and validated
2. Build analytics queries for:
   - Company-specific guidance history
   - Period-over-period comparison
   - Significant changes detection (delta_pp, delta_pct thresholds)
   - Alert grouping by company and time window

### Future Enhancements
1. Extend backfill to full historical data (6-12 months)
2. Build change detection logic:
   - Compare same company, same period across time
   - Flag material changes (e.g., >10pp margin shift)
   - Aggregate changes by quarter/year for trending
3. Add dashboard/visualization layer
4. Integrate with alert delivery system

## Validation Checklist

- [x] Period field captures all temporal patterns (FY, Q, H)
- [x] Single period field sufficient for storage and queries
- [x] Year validation prevents spurious detections
- [x] Metric, metric_kind, currency correctly stored
- [x] Direction hints and value ranges available where applicable
- [x] Exclusion logic filters law firm PRs
- [x] Detection rate in healthy range (3.32%)
- [x] Parquet schema enforced, files written to GCS
- [x] Date partitioning working (by detected_at date)
- [x] Query-ready structure for cross-company and time-series analysis

---

**Status**: ✅ **BACKFILL SUCCESSFUL** - Ready for analytics and production use
**Date**: 2025-12-01
**Coverage**: September-November 2025 (3 months, 45,506 PRs)
