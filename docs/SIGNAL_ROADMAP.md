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

### 🎯 Management Changes

**Priority**: Medium  
**Description**: C-suite and board changes  
**Triggers**: CEO/CFO appointments, resignations, terminations

**Key Use Cases**:
- Track executive leadership transitions
- Monitor board composition changes
- Detect sudden departures or terminations
- Identify interim appointments

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
