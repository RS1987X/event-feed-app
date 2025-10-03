# tests/guidance_change_tests/test_title_body_integration.py
import re
import pytest
from event_feed_app.events.guidance_change.plugin import (
    CCY_RANGE,
    _sentences_with_spans,
    _norm_period,
)

@pytest.mark.parametrize("title, body", [
    (
        "FY2025 guidance",
        "Company expects revenue of EUR 1.700–1.790 million and EBIT of EUR 80–90 million.",
    ),
    (
        "FY2025 guidance",
        "We expect revenue to be SEK 2.3–2.5 bn in FY2025.",
    ),
])
def test_title_trigger_body_numbers_preconditions(plugin, title, body):
    """
    Preconditions for the 'title carries period; body carries numbers' pattern.
    We don't require a *strong* trigger explicitly — either a strong trigger
    OR a cue/metric proximity can open the gate in your design.
    """
    text = f"{title}\n{body}"

    # 1) The document-level period must be parsed from the title.
    period = _norm_period(text)
    assert period == "FY2025", f"Doc period not parsed from title: {period}"

    # 2) The body must contain a currency range matched by our regex.
    m = next(iter(CCY_RANGE.finditer(text)), None)
    assert m is not None, f"CCY_RANGE didn't match this body: {body!r}"

    # 3) The sentence containing the range should also mention a level metric (e.g., revenue/EBIT).
    ss, se, sent = next(s for s in _sentences_with_spans(text) if s[0] <= m.start() < s[1])
    assert (
        re.search(r"\brevenue\b", sent, re.I) or re.search(r"\bebit\b", sent, re.I)
    ), f"Expected a level metric keyword in the matched sentence; got: {sent!r}"

    # Optional (looser) cue presence check: helpful but not strictly required for this test
    assert re.search(r"\b(expect|expects|expected|we expect)\b", sent, re.I), \
        "Helpful cue verb not found near the metric; proximity gate may fail."
