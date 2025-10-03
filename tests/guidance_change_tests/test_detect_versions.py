# tests/guidance_change_tests/test_detect_versions.py
import pytest

# Minimal shared cases you want to exercise in both versions.
# Each tuple: (title, body, source_type, expected_set)
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

@pytest.mark.parametrize("ver", ["v1", "v2"])
@pytest.mark.parametrize("title, body, source_type, expected", CASES)
def test_detect_runs_both_versions(plugin, doc_factory, ver, title, body, source_type, expected):
    # Flip the plugin to the requested version
    if hasattr(plugin, "detect_version"):
        plugin.detect_version = ver
    else:
        # fallback if your plugin uses a boolean toggle instead of a string
        setattr(plugin, "_use_v2", (ver == "v2"))

    doc = doc_factory(title, body, source_type=source_type)
    cands = [plugin.normalize(c) for c in (plugin.detect(doc) or [])]
    got = set(_summarize(cands))
    exp = set(expected)

    # Suppression cases: both versions must produce exactly nothing.
    if not exp:
        assert got == set(), f"[{ver}] Expected no candidates, but got: {sorted(got)}"
        return

    if ver == "v1":
        # Freeze the old behavior exactly (characterization).
        assert got == exp, (
            f"[v1] Mismatch.\nMissing: {sorted(exp - got)}\n"
            f"Extra: {sorted(got - exp)}\nGot: {sorted(got)}"
        )
    else:
        # v2 may add improvements; require at least the old signals.
        assert exp <= got, (
            f"[v2] Missing expected: {sorted(exp - got)}; "
            f"Got: {sorted(got)}"
        )
