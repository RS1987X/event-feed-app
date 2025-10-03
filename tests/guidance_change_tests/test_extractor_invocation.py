# tests/guidance_change_tests/test_extractor_invocation.py
import pytest

@pytest.fixture(autouse=True)
def force_v2(plugin):
    # Make sure detect() uses the v2 path
    if hasattr(plugin, "detect_version"):
        plugin.detect_version = "v2"
    else:
        setattr(plugin, "_use_v2", True)
    return plugin

def _spy_generator(counter_key, counters, orig_fn):
    def wrapper(**kwargs):
        counters[counter_key] += 1
        # IMPORTANT: keep behavior identical by yielding from the original
        yield from orig_fn(**kwargs)
    return wrapper

@pytest.mark.parametrize("title, body, source_type", [
    ("FY2025 guidance",
     "Company expects revenue of EUR 1.700–1.790 million and EBIT of EUR 80–90 million.",
     "issuer_pr"),
    ("FY2025 guidance",
     "We expect revenue to be SEK 2.3–2.5 bn in FY2025.",
     "issuer_pr"),
])
def test_gate_blocks_extractors_for_problem_cases(plugin, doc_factory, monkeypatch, title, body, source_type):
    counters = {"pct": 0, "rng": 0, "num": 0}

    # Grab originals
    orig_pct = plugin._extract_percent_candidates
    orig_rng = plugin._extract_ccy_range_candidates
    orig_num = plugin._extract_ccy_number_candidates

    # Patch with spies
    monkeypatch.setattr(plugin, "_extract_percent_candidates",
                        _spy_generator("pct", counters, orig_pct))
    monkeypatch.setattr(plugin, "_extract_ccy_range_candidates",
                        _spy_generator("rng", counters, orig_rng))
    monkeypatch.setattr(plugin, "_extract_ccy_number_candidates",
                        _spy_generator("num", counters, orig_num))

    doc = doc_factory(title, body, source_type=source_type)
    cands = list(plugin.detect(doc) or [])

    # If your current behavior returns nothing, assert extractors were never invoked
    assert cands == [], f"Expected no candidates for this characterization; got: {cands}"
    assert counters == {"pct": 0, "rng": 0, "num": 0}, f"Extractors were invoked: {counters}"