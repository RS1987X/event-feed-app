# tests/guidance_change_tests/test_detect_versions_strict.py
import pytest

#from .test_detect_versions import CASES, _summarize  # reuse helpers
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

def _summarize(cands):
    """Normalize to a hashable shape for easy set comparisons."""
    out = []
    for c in cands:
        out.append((
            c["metric"],
            c["unit"],
            c["period"],
            c.get("value_low"),
            c.get("value_high"),
        ))
    return out

@pytest.mark.parametrize("title, body, source_type, _expected", CASES)
def test_v1_equals_v2(plugin, doc_factory, title, body, source_type, _expected):
    doc = doc_factory(title, body, source_type=source_type)

    # run v1
    if hasattr(plugin, "detect_version"):
        plugin.detect_version = "v1"
    else:
        setattr(plugin, "_use_v2", False)
    got_v1 = set(_summarize([plugin.normalize(c) for c in (plugin.detect(doc) or [])]))

    # run v2
    if hasattr(plugin, "detect_version"):
        plugin.detect_version = "v2"
    else:
        setattr(plugin, "_use_v2", True)
    got_v2 = set(_summarize([plugin.normalize(c) for c in (plugin.detect(doc) or [])]))

    assert got_v1 == got_v2, (
        f"Outputs differ.\nOnly in v1: {sorted(got_v1 - got_v2)}\nOnly in v2: {sorted(got_v2 - got_v1)}"
    )
