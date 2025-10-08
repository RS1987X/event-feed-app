from event_feed_app.events.guidance_change.plugin import CCY_RANGE, CCY_NUMBER, _to_float

import pytest
# def test_tokmanni_like_case(plugin, doc_factory):
#     title = "Tokmanni Half-Year Report"
#     body = (
#         "GUIDANCE FOR 2025\n"
#         "In 2025, Tokmanni Group expects its revenue to be EUR 1,700–1,790 million. "
#         "Comparable EBIT is expected to be EUR 85–105 million."
#     )
#     doc = doc_factory(title, body, source_type="issuer_pr")

#     cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
#     got = {(c["metric"], c["value_low"], c["value_high"], c["unit"], c["period"]) for c in cands}
#     assert {
#         ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
#         ("ebit", 85.0, 105.0, "ccy", "FY2025"),
#     } <= got

#     # provenance sanity
#     m = {c["metric"]: c.get("_trigger_match","").lower() for c in cands}
#     assert "expects" in m.get("revenue","")
#     assert "expected" in m.get("ebit","")

# @pytest.mark.parametrize("title,body,source_type,expect_empty", [
#     (
#         "Sector outlook",
#         "Industry analysts expect the market to grow in FY2025. Sector outlook remains positive.",
#         "wire",
#         True,  # macro/analyst should be suppressed
#     ),
#     (
#         "Update",
#         "FY2025 guidance unchanged. The company maintains its outlook.",
#         "issuer_pr",
#         False,  # should yield a text-only unchanged stub
#     ),
# ])
# @pytest.fixture(autouse=True)
# def force_detect_v2(plugin):
#     # whichever knob your plugin exposes:
#     if hasattr(plugin, "detect_version"):
#         plugin.detect_version = "v2"
#     else:
#         setattr(plugin, "_use_v2", True)
#     yield
# import pytest

# =============== POSITIVES: CORE DETECTION ===============

# Core multi-metric range (sanity)
POS_CORE_RANGES = [
    pytest.param(
        "Tokmanni Half-Year Report",
        (
            "GUIDANCE FOR 2025\n"
            "In 2025, Tokmanni Group expects its revenue to be EUR 1,700–1,790 million. "
            "Comparable EBIT is expected to be EUR 85–105 million."
        ),
        "issuer_pr",
        {
            ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
            ("ebit", 85.0, 105.0, "ccy", "FY2025"),
        },
        True,
        id="tokmanni-revenue+ebit-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "The company expects revenue to be EUR 1,750 million in FY2025.",
        "issuer_pr",
        {("revenue", 1750.0, 1750.0, "ccy", "FY2025")},
        True,
        id="single-ccy-point-with-period",
    ),
]

# Percent ranges & single-point with banding
POS_PERCENT = [
    pytest.param(
        "FY2025 guidance: margin approx 12–13%",
        "Operating margin expected to be approximately 12%–13%.",
        "issuer_pr",
        {("operating_margin", 12.0, 13.0, "pct", "FY2025")},
        True,
        id="pct-margin-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "Organic growth expected to be approximately 3%.",
        "issuer_pr",
        {("organic_growth", 2.0, 4.0, "pct", "FY2025")},
        True,
        id="organic-growth-approx-single",
    ),
    pytest.param(
        "FY2025 outlook",
        "Operating margin expected to be approx. 12%.",
        "issuer_pr",
        {("operating_margin", 11.5, 12.5, "pct", "FY2025")},
        True,
        id="margin-approx-single",
    ),
]

# Numeric formatting variants (thousands separators etc.)
POS_FORMAT_VARIANTS = [
    pytest.param(
        "FY2025 guidance",
        "Company expects revenue of EUR 1.700–1.790 million and EBIT of EUR 80–90 million.",
        "issuer_pr",
        {
            ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
            ("ebit", 80.0, 90.0, "ccy", "FY2025"),
        },
        True,
        id="european-thousands-dot",
    ),
    pytest.param(
        "FY2025 guidance",
        "We expect revenue to be EUR 1’700–1’790 million and EBIT to be EUR 80–90 million.",
        "issuer_pr",
        {
            ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
            ("ebit", 80.0, 90.0, "ccy", "FY2025"),
        },
        True,
        id="apostrophe-thousands",
    ),
    pytest.param(
        "FY2025 guidance",
        "We expect revenue to be EUR 1 700–1 790 million. EBIT to be EUR 80–90 million.",
        "issuer_pr",
        {
            ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
            ("ebit", 80.0, 90.0, "ccy", "FY2025"),
        },
        True,
        id="space-thousands",
    ),
    pytest.param(
        "FY2025 outlook",
        "The company expects revenue to be EUR1,750 million in FY2025.",
        "issuer_pr",
        {("revenue", 1750.0, 1750.0, "ccy", "FY2025")},
        True,
        id="nospace-after-ccy",
    ),
]

# Unit scaling / normalization
POS_SCALING = [
    pytest.param(
        "FY2025 guidance",
        "We expect revenue to be SEK 2.3–2.5 bn in FY2025.",
        "issuer_pr",
        {("revenue", 2300.0, 2500.0, "ccy", "FY2025")},
        True,
        id="bn-scaling",
    ),
    pytest.param(
        "FY2025 guidance",
        "We expect revenue to be SEK 2,3–2,5 billion in FY2025.",
        "issuer_pr",
        {("revenue", 2300.0, 2500.0, "ccy", "FY2025")},
        True,
        id="bn-scaling-decimal-comma",
    ),
    pytest.param(
        "FY2025 guidance",
        "Revenue expected at USD 950–1,000 m.",
        "issuer_pr",
        {("revenue", 950.0, 1000.0, "ccy", "FY2025")},
        True,
        id="m-mn-scaling",
    ),
    pytest.param(
        "FY2025 guidance",
        "EBIT expected at USD 850–950 thousand.",
        "issuer_pr",
        {("ebit", 0.85, 0.95, "ccy", "FY2025")},
        True,
        id="thousand-scaling",
    ),
    pytest.param(
        "FY2025 guidance",
        "EBITDA expected at USD 900k.",
        "issuer_pr",
        {("ebitda", 0.9, 0.9, "ccy", "FY2025")},
        True,
        id="k-scaling-point",
    ),
]

# Period handling (doc/title fallback; explicit H2/Q; FY-UNKNOWN)
POS_PERIODS = [
    pytest.param(
        "FY2025 guidance",
        "Revenue expected to be EUR 900–950 million. EBIT expected at EUR 75–85 million.",
        "issuer_pr",
        {
            ("revenue", 900.0, 950.0, "ccy", "FY2025"),
            ("ebit", 75.0, 85.0, "ccy", "FY2025"),
        },
        True,
        id="title-period-doc-fallback-two-metrics",
    ),
    pytest.param(
        "Full-year guidance",
        "We expect revenue to be EUR 1,200–1,250 million.",
        "issuer_pr",
        {("revenue", 1200.0, 1250.0, "ccy", "FY-UNKNOWN")},
        True,
        id="full-year-unknown-year",
    ),
    pytest.param(
        "Outlook",
        "H2 2025 revenue expected to be EUR 800–900 million.",
        "issuer_pr",
        {("revenue", 800.0, 900.0, "ccy", "H2-2025")},
        True,
        id="h2-period",
    ),
    pytest.param(
        "Outlook",
        "Revenue guidance for Q3 2025 is EUR 400–450 million.",
        "issuer_pr",
        {("revenue", 400.0, 450.0, "ccy", "Q3-2025")},
        True,
        id="q3-period",
    ),
]

# Text-only stubs (unchanged/verb-based/initiation; plus M&A suppression with stub)
POS_TEXT_STUBS = [
    pytest.param(
        "Update",
        "FY2025 guidance unchanged. The company maintains its outlook.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="unchanged-text-stub",
    ),
    pytest.param(
        "FY2025 update",
        "The company reaffirms its guidance for FY2025.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="reaffirms-unchanged",
    ),
    pytest.param(
        "Outlook",
        "We provide FY2025 guidance: revenue EUR 1,000–1,100 million.",
        "issuer_pr",
        {("revenue", 1000.0, 1100.0, "ccy", "FY2025")},
        True,
        id="initiate-with-period",
    ),
    pytest.param(
        "FY2025 guidance",
        "We agreed to an acquisition; purchase price is EUR 900–950 million. Guidance unchanged.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="ma-suppressor",
    ),
    # Optional directional stubs
    pytest.param(
        "FY2025 update",
        "We raise our guidance for FY2025.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="raise-text-stub",
    ),
    pytest.param(
        "FY2025 update",
        "We lower our guidance for FY2025.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="lower-text-stub",
    ),
]

# Multilingual (Swedish)
POS_MULTILINGUAL = [
    pytest.param(
        "Prognos FY2025",
        "Vi förväntar oss att rörelsemarginalen blir 10–12%.",
        "issuer_pr",
        {("operating_margin", 10.0, 12.0, "pct", "FY2025")},
        True,
        id="sv-roe-marginal",
    ),
    pytest.param(
        "Prognos FY2025",
        "Bolaget förväntar sig omsättning på EUR 1,200–1,250 million.",
        "issuer_pr",
        {("revenue", 1200.0, 1250.0, "ccy", "FY2025")},
        True,
        id="sv-omsattning",
    ),
]

# English cue variants (forecast/projected/will be/guiding/target; guidance interval stub)
POS_EN_VARIANTS = [
    pytest.param(
        "FY2025 outlook",
        "We forecast revenue of EUR 900–950 million in FY2025.",
        "issuer_pr",
        {("revenue", 900.0, 950.0, "ccy", "FY2025")},
        True,
        id="forecast-range-ccy",
    ),
    pytest.param(
        "FY2025 outlook",
        "EBITDA projected at USD 900 m in FY2025.",
        "issuer_pr",
        {("ebitda", 900.0, 900.0, "ccy", "FY2025")},
        True,
        id="projected-single-ccy",
    ),
    pytest.param(
        "FY2025 outlook",
        "Revenue will be EUR 1,700–1,790 million in FY2025.",
        "issuer_pr",
        {("revenue", 1700.0, 1790.0, "ccy", "FY2025")},
        True,
        id="will-be-range-ccy",
    ),
    pytest.param(
        "FY2025 guidance",
        "We are guiding to an operating margin of 12–13% for FY2025.",
        "issuer_pr",
        {("operating_margin", 12.0, 13.0, "pct", "FY2025")},
        True,
        id="guiding-to-pct-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "We target organic growth of around 3% in FY2025.",
        "issuer_pr",
        {("organic_growth", 2.0, 4.0, "pct", "FY2025")},
        True,
        id="target-around-growth-single",
    ),
    pytest.param(
        "FY2025 update",
        "…thus within the targeted guidance interval for 2025.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="within-guidance-interval-text-stub",
    ),
]

PROGRESS_CASES = [# --- Expected variants & doc-period fallback (new) ---
    pytest.param(
        "FY2025 outlook",
        "EBIT is expected to be EUR 800–900 million in FY2025.",
        "issuer_pr",
        {("ebit", 800.0, 900.0, "ccy", "FY2025")},
        True,
        id="expected-passive-future-ccy-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "We expect operating margin to be about 12–13% in FY2025.",
        "issuer_pr",
        {("operating_margin", 12.0, 13.0, "pct", "FY2025")},
        True,
        id="expect-active-future-pct-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "Revenue is expected at around EUR 950–1,000 million in FY2025.",
        "issuer_pr",
        {("revenue", 950.0, 1000.0, "ccy", "FY2025")},
        True,
        id="expected-passive-around-ccy-range",
    ),
    pytest.param(
        "FY2025 guidance",
        "Revenue expected to be EUR 900–950 million.",
        "issuer_pr",
        {("revenue", 900.0, 950.0, "ccy", "FY2025")},  # period from title (doc-level fallback)
        True,
        id="doc-period-fallback-still-works",
    )]

# =============== NEGATIVES: SUPPRESSORS / PRECISION GUARDS ===============

# Macro/analyst/competitor (wire) suppressors
NEG_SUPPRESSORS_ENTITY = [
    pytest.param(
        "Sector outlook",
        "Industry analysts expect the market to grow in FY2025. Sector outlook remains positive.",
        "wire",
        set(),
        False,
        id="macro-analyst-suppress",
    ),
    pytest.param(
        "Sector outlook",
        "Analysts expect revenue to be EUR 2,000–2,200 million in FY2025.",
        "wire",
        set(),
        False,
        id="analyst-wire-suppressed",
    ),
    pytest.param(
        "Sector outlook",
        "Analysts reaffirm FY2025 guidance range of EUR 1,700–1,790 million.",
        "wire",
        set(),
        False,
        id="analyst-wire-reaffirm-suppressed",
    ),
    pytest.param(
        "Competitor news",
        "RivalCo expects revenue to be EUR 2,000–2,200 million in FY2025.",
        "wire",
        set(),
        False,
        id="competitor-wire-suppressed",
    ),
]

# Finance-context & disclaimers
NEG_FINANCE_CONTEXT = [
    pytest.param(
        "Share issue",
        "Issue price set at EUR 10–12 per share.",
        "issuer_pr",
        set(),
        False,
        id="issue-price-suppressed",
    ),
    pytest.param(
        "Debt financing",
        "New bond to carry a coupon of 6–7%.",
        "issuer_pr",
        set(),
        False,
        id="coupon-suppressed",
    ),
    pytest.param(
        "CMD invite",
        "Join our Capital Markets Day where we will update our outlook.",
        "issuer_pr",
        set(),
        False,
        id="invite-suppressed",
    ),
    pytest.param(
        "Update",
        "Forward-looking statements: Operating margin could be around 12–13%.",
        "issuer_pr",
        set(),
        False,
        id="disclaimer-block-suppressed",
    ),
]

# Period/format guards
NEG_PERIOD_FORMAT = [
    pytest.param(
        "Guidance",
        "Revenue will be EUR 1,700–1,790 million.",  # no FY/period anywhere
        "issuer_pr",
        set(),
        False,
        id="unknown-period-no-fallback-suppressed",
    ),
    pytest.param(
        "FY2025 outlook",
        "Revenue expected to be 1,700–1,790.",  # no currency code
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        False,
        id="no-currency-keep",
    ),
]

# Historical / retrospective
NEG_HISTORICAL = [
    pytest.param(
        "Q4 2024 results",
        "Operating margin was 12% in Q4 2024.",
        "issuer_pr",
        set(),
        False,
        id="historical-margin-not-guidance",
    ),
    pytest.param(
        "FY2024 results",
        "Revenue was EUR 1,750 million in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="historical-revenue-not-guidance",
    ),
    pytest.param(
        "Q2 2025 results",
        "Results were impacted by an impairment charge. Adjusted EBITA margin was 13%.",
        "issuer_pr",
        set(),
        False,
        id="impairment-historical-suppressed",
    ),
    pytest.param(
        "Update",
        "The company does not intend to provide guidance for FY2025.",
        "issuer_pr",
        set(),
        False,
        id="negation-no-guidance",
    ),
]

# Sentence boundary / non-metrics
NEG_SENTENCE_NONMETRIC = [
    pytest.param(
        "Outlook",
        "We expect. Revenue will be EUR 1,700–1,790 million.",
        "issuer_pr",
        set(),
        False,
        id="cross-sentence-suppressed",
    ),
    pytest.param(
        "FY2025 outlook",
        "Margin expected to be 12–13%.",
        "issuer_pr",
        set(),
        False,
        id="generic-margin-suppressed",
    ),
    pytest.param(
        "FY2025 outlook",
        "Market share expected to be 12–13%.",
        "issuer_pr",
        set(),
        False,
        id="non-metric-percent-suppressed",
    ),
]

# Multilingual (Swedish) suppressors on wire
NEG_MULTILINGUAL = [
    pytest.param(
        "Sektorutsikter",
        "Analytiker förväntar sig omsättning på EUR 1,8–2,0 miljarder under 2025.",
        "wire",
        set(),
        False,
        id="sv-analytiker-wire-suppressed",
    ),
]

# =============== FUTURE (XFAIL) – enable when implemented ===============

XF_CASES = [
    pytest.param(
        "FY2025 outlook",
        "Revenue is expected to be between EUR 1,000 million and EUR 1,100 million in FY2025.",
        "issuer_pr",
        {("revenue", 1000.0, 1100.0, "ccy", "FY2025")},
        True,
        id="between-and-ccy-range",
        marks=pytest.mark.xfail(reason="Worded range parse not implemented yet"),
    ),
    pytest.param(
        "FY2025 outlook",
        "EBIT is not expected to exceed EUR 100 million in FY2025.",
        "issuer_pr",
        set(),  # or encode as an upper bound once you add inequality support
        False,
        id="inequality-not-exceed",
        marks=pytest.mark.xfail(reason="Inequality parse not implemented yet"),
    ),
]

PAST_EXPECTED_NEGATIVES = [
    # Passive, past tense + historical period
    pytest.param(
        "FY2024 results",
        "EBIT was expected to be EUR 800–900 million in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="was-expected-passive-historical-ccy",
    ),
    # Active, past perfect
    pytest.param(
        "FY2024 results",
        "We had expected revenue of EUR 1,750 million in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="had-expected-active-historical-ccy",
    ),
    # Progressive past
    pytest.param(
        "FY2024 results",
        "We were expecting an operating margin of 12–13% in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="were-expecting-historical-pct",
    ),
    # Idiomatic “as expected” (historical)
    pytest.param(
        "Q2 2024 results",
        "As expected, revenue was EUR 900 million in Q2 2024.",
        "issuer_pr",
        set(),
        False,
        id="as-expected-historical-ccy",
    ),
    # Results in/against expectations
    pytest.param(
        "FY2024 results",
        "Results were in line with expectations for FY2024.",
        "issuer_pr",
        set(),
        False,
        id="in-line-with-expectations-historical",
    ),
    pytest.param(
        "FY2024 results",
        "Results were ahead of expectations in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="ahead-of-expectations-historical",
    ),
    # “Previously expected …” (historical reference, should not emit)
    pytest.param(
        "Update",
        "We previously expected FY2024 revenue of EUR 1,7–1,9 billion.",
        "issuer_pr",
        set(),
        False,
        id="previously-expected-historical-bn",
    ),
    # Third-party wording but from issuer source (still historical)
    pytest.param(
        "FY2024 results",
        "The company had been expected to post revenue of EUR 1,700 million in FY2024.",
        "issuer_pr",
        set(),
        False,
        id="had-been-expected-passive-historical",
    ),
    # Historical expectation about a future period (still retrospective)
    pytest.param(
        "Strategy update",
        "Back in 2023, FY2025 revenue was expected to reach EUR 1,900–2,100 million.",
        "issuer_pr",
        set(),
        False,
        id="historical-expectation-about-future-period",
    ),
    # Mixed case: past-tense expected (historical) + real guidance stub
    pytest.param(
        "Update",
        "EBIT was expected to be EUR 800–900 million in FY2024. FY2025 guidance unchanged.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},  # only the stub should survive
        True,
        id="suppress-past-expected-allow-unchanged-stub",
    ),
]

# =============== FINAL AGGREGATION ===============

CASES = (
    POS_CORE_RANGES
    + POS_PERCENT
    + POS_FORMAT_VARIANTS
    + POS_SCALING
    + POS_PERIODS
    + POS_TEXT_STUBS
    + POS_MULTILINGUAL
    + POS_EN_VARIANTS
    + PROGRESS_CASES
    + XF_CASES  # keep at end; they’re xfail until implemented
)

NEGATIVE_CASES = (
    NEG_SUPPRESSORS_ENTITY
    + NEG_FINANCE_CONTEXT
    + NEG_PERIOD_FORMAT
    + NEG_HISTORICAL
    + NEG_SENTENCE_NONMETRIC
    + NEG_MULTILINGUAL
    + PAST_EXPECTED_NEGATIVES
)

CASES = CASES + NEGATIVE_CASES

#CASES = CASES + EXTRA_CASES + NEGATIVE_CASES

@pytest.mark.parametrize("title, body, source_type, expected, allow_extras", CASES)
def test_guidance_parametrized(plugin, doc_factory, title, body, source_type, expected, allow_extras):
    doc = doc_factory(title, body, source_type=source_type)
    cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
    got = {(c["metric"], c.get("value_low"), c.get("value_high"), c["unit"], c["period"]) for c in cands}

    if not expected:  # expecting suppression
        assert got == set(), f"Expected no candidates, but got: {sorted(got)}"
        return

    if allow_extras:
        missing = set(expected) - got
        assert not missing, f"Missing expected: {sorted(missing)}; Got: {sorted(got)}"
    else:
        assert got == set(expected), f"Expected exactly {sorted(expected)}; Got: {sorted(got)}"

# def test_more_realistic_variants(plugin, doc_factory, title, body, source_type, expect_empty):
#     doc = doc_factory(title, body, source_type=source_type)
#     cands = list(plugin.detect(doc) or [])
#     if expect_empty:
#         assert cands == []
#     else:
#         assert cands  # at least one candidate


def test_ccy_range_revenue_and_ebit(plugin, doc_factory):
    title = "Tokmanni Half-Year Report"
    body = (
        "GUIDANCE FOR 2025\n"
        "In 2025, Tokmanni Group expects its revenue to be EUR 1,700–1,790 million. "
        "Comparable EBIT is expected to be EUR 85–105 million."
    )
    doc = doc_factory(title, body, source_type="issuer_pr")
     #--- TEMP DEBUG: what do our money regexes see? ---
    ranges = [m.group(0) for m in CCY_RANGE.finditer(doc["body"])]
    numbers = [m.group(0) for m in CCY_NUMBER.finditer(doc["body"])]
    # Use asserts so the info shows on failure (no -s needed)
    assert any("1,700" in r and "1,790" in r for r in ranges), f"Revenue range NOT matched; CCY_RANGE saw: {ranges}"
    assert any("85" in r and "105" in r for r in ranges), f"EBIT range NOT matched; CCY_RANGE saw: {ranges}"
    assert "revenue" in doc["body"].lower(), "The word 'revenue' isn't in the body (?)"
    assert "ebit" in doc["body"].lower(), "The word 'EBIT' isn't in the body (?)"
    # --- END TEMP DEBUG ---

    cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
    want = ("revenue", 1700.0, 1790.0, "ccy", "FY2025")
    got  = {(c["metric"], c["value_low"], c["value_high"], c["unit"], c["period"]) for c in cands}
    assert want in got, f"Missing revenue range {want}. Got: {sorted(got)}"
    
    # # Assert EBIT is captured
    # assert any(
    #     c["metric"] == "ebit" and c["unit"] == "ccy" and c["value_low"] == 85.0 and c["value_high"] == 105.0
    #     and c["period"] == "FY2025"
    #     for c in cands
    # )

    # # Assert REVENUE is captured (desired behavior)
    # assert any(
    #     c["metric"] == "revenue" and c["unit"] == "ccy" and c["value_low"] == 1700.0 and c["value_high"] == 1790.0
    #     and c["period"] == "FY2025"
    #     for c in cands
    # )

    # got = {(c["metric"], c["value_low"], c["value_high"], c["unit"], c["period"]) for c in cands}
    # assert {
    #     ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
    #     ("ebit", 85.0, 105.0, "ccy", "FY2025"),
    # } <= got
