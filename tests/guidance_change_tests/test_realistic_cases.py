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

# Each expected tuple is: (metric, low, high, unit, period)
CASES = [
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
        True,   # allow extras (e.g., if margins also get picked up)
        id="tokmanni-revenue+ebit-range"
    ),
    pytest.param(
        "FY2025 guidance: margin approx 12–13%",
        "Operating margin expected to be approximately 12%–13%.",
        "issuer_pr",
        {("operating_margin", 12.0, 13.0, "pct", "FY2025")},
        True,
        id="pct-margin-range"
    ),
    pytest.param(
        "FY2025 outlook",
        "The company expects revenue to be EUR 1,750 million in FY2025.",
        "issuer_pr",
        {("revenue", 1750.0, 1750.0, "ccy", "FY2025")},
        True,
        id="single-ccy-point-with-period"
    ),
    pytest.param(
        "FY2025 guidance",
        "Company expects revenue of EUR 1.700–1.790 million and EBIT of EUR 80–90 million.",
        "issuer_pr",
        {
            ("revenue", 1700.0, 1790.0, "ccy", "FY2025"),
            ("ebit", 80.0, 90.0, "ccy", "FY2025"),
        },
        True,
        id="european-thousands-dot"
    ),
    pytest.param(
        "FY2025 guidance",
        "We expect revenue to be SEK 2.3–2.5 bn in FY2025.",
        "issuer_pr",
        {("revenue", 2300.0, 2500.0, "ccy", "FY2025")},  # bn -> thousands of "million"
        True,
        id="bn-scaling"
    ),
    pytest.param(
        "Sector outlook",
        "Industry analysts expect the market to grow in FY2025. Sector outlook remains positive.",
        "wire",
        set(),  # should be suppressed (macro/analyst)
        False,
        id="macro-analyst-suppress"
    ),
    pytest.param(
        "Update",
        "FY2025 guidance unchanged. The company maintains its outlook.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},  # text-only unchanged stub
        True,
        id="unchanged-text-stub"
    ),
]


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

def test_more_realistic_variants(plugin, doc_factory, title, body, source_type, expect_empty):
    doc = doc_factory(title, body, source_type=source_type)
    cands = list(plugin.detect(doc) or [])
    if expect_empty:
        assert cands == []
    else:
        assert cands  # at least one candidate


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
