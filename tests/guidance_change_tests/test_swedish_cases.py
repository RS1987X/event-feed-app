import pytest

# This mirrors the style of your existing realistic_cases tests:
# - uses the plugin + doc_factory fixtures from your suite
# - compares normalized candidates as (metric, low, high, unit, period)

CASES = [
    # --- ALREADY HANDLED (should PASS) ---

    # Swedish guidance words + issuer PR + period in title
    pytest.param(
        "FY2025 utsikter",
        "Prognosen oförändrad.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-guidance-unchanged-basic",
    ),

    # Swedish forward-looking verb (förväntas) + margin range in procent
    pytest.param(
        "FY2025 utsikter",
        "Rörelsemarginalen förväntas bli 11–13 procent.",
        "issuer_pr",
        {("operating_margin", 11.0, 13.0, "pct", "FY2025")},
        True,
        id="sv-margin-range-procent",
    ),

    # Swedish approx + % (make sure splitter doesn't break on 'ca.')
    pytest.param(
        "FY2025 utsikter",
        "Rörelsemarginalen väntas bli ca. 12 %.",
        "issuer_pr",
        {("operating_margin", 11.5, 12.5, "pct", "FY2025")},
        True,
        id="sv-margin-approx-ca",
    ),

    # Number formatting: thin space + en dash, no currency → keep as guidance stub
    pytest.param(
        "FY2025 utsikter",
        "Omsättningen väntas bli 1\u202f700–1\u202f790.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-revenue-range-no-ccy-guidance-stub",
    ),

    # Non-issuer source without company pronoun → suppress
    pytest.param(
        "FY2025 utsikter",
        "Prognosen oförändrad.",
        "analyst_wire",
        set(),                       # expect suppression (no candidates)
        False,
        id="sv-nonissuer-no-pronoun-suppress",
    ),

    # Non-issuer source WITH company pronoun → allow stub
    pytest.param(
        "FY2025 utsikter",
        "Bolaget upprepar prognosen.",
        "analyst_wire",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-nonissuer-with-pronoun-allow",
    ),

    # Swedish currency scale in billions: "SEK 2,1–2,3 mdr" => millions 2100–2300
    pytest.param(
        "FY2025 prognos",
        "Omsättningen förväntas bli SEK 2,1–2,3 mdr.",
        "issuer_pr",
        {("revenue", 2100.0, 2300.0, "ccy", "FY2025")},
        True,
        id="sv-revenue-sek-mdr-range",
    ),

    # --- DESIRED / NOT FULLY HANDLED YET (mark as XFAIL until implemented) ---

    # Passive/assessed variants (väntas/bedöms) — if not fully covered, xfail for now
    pytest.param(
        "FY2025 utsikter",
        "Rörelsemarginalen bedöms uppgå till 10–12 procent.",
        "issuer_pr",
        {("operating_margin", 10.0, 12.0, "pct", "FY2025")},
        True,
        marks=pytest.mark.xfail(reason="Add 'bedöms' passive verb support to guidance/numeric rules"),
        id="sv-margin-range-bedoms-xfail",
    ),

    # Period phrasing “för helåret 2025” → map to FY2025
    pytest.param(
        "Utsikter",
        "Prognosen för helåret 2025 är oförändrad.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        marks=pytest.mark.xfail(reason="Extend _norm_period to map 'helåret 2025' -> FY2025"),
        id="sv-period-helaret-2025-xfail",
    ),

    # Trailing currency tokens like MSEK (instead of leading 'SEK ...') — plan later
    pytest.param(
        "FY2025 prognos",
        "Omsättningen väntas bli 1 700–1 900 MSEK.",
        "issuer_pr",
        {("revenue", 1700.0, 1900.0, "ccy", "FY2025")},
        True,
        marks=pytest.mark.xfail(reason="Add trailing currency patterns (e.g., MSEK/MDKK/MNOK)"),
        id="sv-revenue-trailing-MSEK-xfail",
    ),
       # Issuer PR, active form, up-direction (no numbers) -> text-only guidance stub
    pytest.param(
        "FY2025 utsikter",
        "Vi förväntar oss nu att omsättningen växer.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-up-active-text-stub",
    ),

    # Issuer PR, passive form, up-direction
    pytest.param(
        "FY2025 utsikter",
        "Omsättningen förväntas öka under perioden.",
        "issuer_pr",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-up-passive-text-stub",
    ),

    # Non-issuer without company pronoun -> suppressed
    pytest.param(
        "FY2025 utsikter",
        "Förväntas att omsättningen växer.",
        "analyst_wire",
        set(),
        False,
        id="sv-nonissuer-no-pronoun-suppress",
    ),

    # Non-issuer with company pronoun -> allowed
    pytest.param(
        "FY2025 utsikter",
        "Bolaget förväntas att omsättningen växer.",
        "analyst_wire",
        {("guidance", None, None, "text", "FY2025")},
        True,
        id="sv-nonissuer-with-pronoun-allow",
    ),
]



@pytest.mark.parametrize("title, body, source_type, expected, allow_extras", CASES)
def test_swedish_parametrized(plugin, doc_factory, title, body, source_type, expected, allow_extras):
    # Build doc the same way as the other test suite
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
