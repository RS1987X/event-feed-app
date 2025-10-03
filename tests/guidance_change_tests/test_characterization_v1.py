import pytest
from event_feed_app.events.guidance_change.plugin import plugin as _shared_plugin


@pytest.fixture(autouse=True)
def enable_v2(plugin):
    plugin._use_v2 = True
    yield
    
# Canonicalize candidates so order/noise doesn't affect equality
def summarize(cands):
    def tup(c):
        return (
            c["metric"],
            c["unit"],
            c.get("period"),
            None if c.get("value_low") is None else float(c["value_low"]),
            None if c.get("value_high") is None else float(c["value_high"]),
        )
    return sorted(tup(c) for c in cands)

# Each expected tuple is: (metric, unit, period, low, high)
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
            ("revenue", "ccy", "FY2025", 1700.0, 1790.0),
            ("ebit",    "ccy", "FY2025",   85.0,  105.0),
        },
        id="tokmanni-revenue+ebit-range",
    ),
    pytest.param(
        "FY2025 guidance: margin approx 12–13%",
        "Operating margin expected to be approximately 12%–13%.",
        "issuer_pr",
        {("operating_margin", "pct", "FY2025", 12.0, 13.0)},
        id="pct-margin-range",
    ),
    pytest.param(
        "FY2025 outlook",
        "The company expects revenue to be EUR 1,750 million in FY2025.",
        "issuer_pr",
        {("revenue", "ccy", "FY2025", 1750.0, 1750.0)},
        id="single-ccy-point-with-period",
    ),
    pytest.param(
        "Sector outlook",
        "Industry analysts expect the market to grow in FY2025. Sector outlook remains positive.",
        "wire",
        set(),  # current behavior: suppressed (macro/analyst)
        id="macro-analyst-suppress",
    ),
    pytest.param(
        "Update",
        "FY2025 guidance unchanged. The company maintains its outlook.",
        "issuer_pr",
        {("guidance", "text", "FY2025", None, None)},
        id="unchanged-text-stub",
    ),
]

@pytest.mark.parametrize("title, body, source_type, expected", CASES)
def test_detect_v1_characterization(plugin, doc_factory, title, body, source_type, expected):
    """
    Locks in current behavior of detect() + normalize().
    If you refactor, these tests should stay green unless you intentionally change outputs.
    """
    doc = doc_factory(title, body, source_type=source_type)
    cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
    got = set(summarize(cands))

    # For suppression cases we expect exactly empty.
    if expected == set():
        assert got == set(), f"Expected no candidates, but got: {sorted(got)}"
        return

    # For positive cases, require exact match (freezes behavior).
    exp = set(expected)
    assert got == exp, f"Mismatch.\nMissing: {sorted(exp - got)}\nExtra: {sorted(got - exp)}\nGot: {sorted(got)}"

def test_candidate_shape_invariants(plugin, doc_factory):
    """Basic schema check so refactors don't drop required keys."""
    doc = doc_factory(
        "FY2025 outlook",
        "The company expects revenue to be EUR 1,750 million in FY2025.",
        source_type="issuer_pr",
    )
    cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
    for c in cands:
        for key in ("metric", "metric_kind", "unit", "value_type", "period", "basis"):
            assert key in c, f"Missing key {key} in candidate: {c}"
